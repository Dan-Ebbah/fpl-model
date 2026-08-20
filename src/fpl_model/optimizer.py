"""Squad optimizer: pick the best legal 15-man squad from the projections.

Integer program over three sets of binary variables:

    squad[p]      player p is in the 15
    start[p, gw]  p starts in gameweek gw
    cap[p, gw]    p is captain in gw (captain scores double)

maximizing  sum over gw, p of  xP[p,gw] * (start + cap + bench_weight*(squad-start))
subject to the FPL rulebook: budget, 2/5/5/3 squad, max 3 per club, and a
legal formation each week. Unlike the Excel Solver version this lets the XI
and captain change week to week while the 15 stay fixed, so squad depth that
covers bad fixtures is valued correctly.

The optimizer exploits any prediction error ruthlessly — treat its output as
a proposal to check against team news, not an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


@dataclass(frozen=True)
class SquadRules:
    budget: float = 100.0
    squad_size: dict[str, int] = field(
        default_factory=lambda: {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    )
    max_per_club: int = 3
    xi_size: int = 11
    formation: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    )


@dataclass
class SquadSolution:
    squad: pd.DataFrame     # one row per picked player, with per-window totals
    lineups: pd.DataFrame   # one row per (gw, player) for starters, captain flagged
    expected_points: float
    cost: float
    status: str


def select_pool(
    players: pd.DataFrame,
    window_xp: pd.Series,
    per_position: int = 80,
    cheapest: int = 15,
    keep: set[int] | None = None,
) -> pd.DataFrame:
    """Trim the candidate pool to keep the MILP fast.

    Keeps, per position: the top `per_position` by projected points and the
    `cheapest` lowest-priced (bench-enabler candidates), plus anything in
    `keep` (locked players must survive the cut).
    """
    pool_codes: set[int] = set(keep or ())
    xp = window_xp.reindex(players.index).fillna(0.0)
    for pos in POSITION_ORDER:
        at_pos = players[players["POS"] == pos]
        pool_codes |= set(xp.loc[at_pos.index].nlargest(per_position).index)
        pool_codes |= set(at_pos["price"].nsmallest(cheapest).index)
    return players.loc[sorted(pool_codes & set(players.index))]


def optimize_squad(
    players: pd.DataFrame,
    xp: pd.DataFrame,
    rules: SquadRules | None = None,
    bench_weight: float = 0.1,
    locks: set[int] | None = None,
    excludes: set[int] | None = None,
) -> SquadSolution:
    """Solve for the optimal squad.

    `players`: indexed by player code, columns FPL_Name, TEAM, POS, price.
    `xp`: expected points, indexed by code, one column per gameweek.
    `locks`/`excludes`: player codes forced in or out of the 15.
    """
    rules = rules or SquadRules()
    players = players[players["price"].notna() & (players["price"] > 0)]
    if excludes:
        players = players.drop(index=set(excludes) & set(players.index))
    xp = xp.reindex(players.index).fillna(0.0)
    codes = list(players.index)
    gws = list(xp.columns)
    missing_locks = set(locks or ()) - set(codes)
    if missing_locks:
        raise ValueError(f"locked players not in candidate pool: {missing_locks}")

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", codes, cat="Binary")
    start = pulp.LpVariable.dicts("start", (codes, gws), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", (codes, gws), cat="Binary")

    xpd = {gw: xp[gw].to_dict() for gw in gws}
    prob += pulp.lpSum(
        xpd[gw][p] * ((1 - bench_weight) * start[p][gw] + cap[p][gw] + bench_weight * squad[p])
        for gw in gws
        for p in codes
    )

    price = players["price"].to_dict()
    prob += pulp.lpSum(price[p] * squad[p] for p in codes) <= rules.budget

    pos_of = players["POS"].to_dict()
    for pos, n in rules.squad_size.items():
        prob += pulp.lpSum(squad[p] for p in codes if pos_of[p] == pos) == n

    for club, members in players.groupby("TEAM").groups.items():
        prob += pulp.lpSum(squad[p] for p in members) <= rules.max_per_club

    for gw in gws:
        prob += pulp.lpSum(start[p][gw] for p in codes) == rules.xi_size
        prob += pulp.lpSum(cap[p][gw] for p in codes) == 1
        for pos, (lo, hi) in rules.formation.items():
            n_started = pulp.lpSum(start[p][gw] for p in codes if pos_of[p] == pos)
            prob += n_started >= lo
            prob += n_started <= hi
        for p in codes:
            prob += start[p][gw] <= squad[p]
            prob += cap[p][gw] <= start[p][gw]

    for p in locks or ():
        prob += squad[p] == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        raise RuntimeError(f"solver finished with status {status_name}")

    picked = [p for p in codes if squad[p].value() > 0.5]
    lineup_rows = [
        {
            "gw": gw,
            "Code": p,
            "started": start[p][gw].value() > 0.5,
            "captain": cap[p][gw].value() > 0.5,
        }
        for gw in gws
        for p in picked
    ]
    lineups = pd.DataFrame(lineup_rows).merge(
        players[["FPL_Name", "TEAM", "POS"]], left_on="Code", right_index=True
    )

    squad_df = players.loc[picked, ["FPL_Name", "TEAM", "POS", "price"]].copy()
    squad_df["window_xp"] = xp.loc[picked].sum(axis=1)
    squad_df["starts"] = lineups[lineups["started"]].groupby("Code").size().reindex(picked).fillna(0).astype(int)
    squad_df["captaincies"] = lineups[lineups["captain"]].groupby("Code").size().reindex(picked).fillna(0).astype(int)
    squad_df["pos_rank"] = squad_df["POS"].map({p: i for i, p in enumerate(POSITION_ORDER)})
    squad_df = squad_df.sort_values(["pos_rank", "window_xp"], ascending=[True, False]).drop(
        columns="pos_rank"
    )

    return SquadSolution(
        squad=squad_df,
        lineups=lineups,
        expected_points=pulp.value(prob.objective),
        cost=float(squad_df["price"].sum()),
        status=status_name,
    )
