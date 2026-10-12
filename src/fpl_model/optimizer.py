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

Transfer mode (`current` given): the budget becomes bank + money raised by
selling, each player leaving the current 15 is a transfer, transfers beyond
the free ones cost `hit_cost` points off the objective, and `max_transfers`
caps them. A banked free transfer has option value, so each free one used
costs `ft_value` points in the objective (not in the reported xP) — except
one when you already hold the maximum, since that one is lost anyway. The 15
still stay fixed across the window, so this answers "what should I do at
*this* deadline given the next N fixtures" — it does not plan a sequence of
transfers.

The optimizer exploits any prediction error ruthlessly — treat its output as
a proposal to check against team news, not an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from .rules import (
    BUDGET,
    DEFAULT_BENCH_WEIGHT,
    DEFAULT_FT_VALUE,
    HIT_COST,
    MAX_FREE_TRANSFERS,
    MAX_PER_CLUB,
    POSITION_ORDER,
    POSITIONS,
    SQUAD_SHAPE,
    XI_SIZE,
)


@dataclass(frozen=True)
class SquadRules:
    budget: float = BUDGET
    squad_size: dict[str, int] = field(default_factory=lambda: dict(SQUAD_SHAPE))
    max_per_club: int = MAX_PER_CLUB
    xi_size: int = XI_SIZE
    formation: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    )


@dataclass
class SquadSolution:
    squad: pd.DataFrame     # one row per picked player, with per-window totals
    lineups: pd.DataFrame   # one row per (gw, picked player) with `started` and `captain` flags
    expected_points: float  # window xP of the chosen squad/XIs (gross of hits)
    cost: float             # sum of the 15 buy prices — in transfer mode too, so not what you paid
    status: str
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hit_points: float = 0.0

    @property
    def n_transfers(self) -> int:
        return len(self.transfers_out)


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
    for pos in POSITIONS:
        at_pos = players[players["POS"] == pos]
        pool_codes |= set(xp.loc[at_pos.index].nlargest(per_position).index)
        pool_codes |= set(at_pos["price"].nsmallest(cheapest).index)
    return players.loc[sorted(pool_codes & set(players.index))]


def _prepare(
    players: pd.DataFrame,
    xp: pd.DataFrame,
    locks: set[int] | None,
    excludes: set[int] | None,
    current: dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Trim the pool to priced players, apply excludes, check locks/current are present.

    Returns (players, xp aligned to players, forced_out). An excluded player
    you own must be sold, so he stays in the model as a forced out instead of
    being dropped.
    """
    players = players[players["price"].notna() & (players["price"] > 0)]
    forced_out = set(excludes or ()) & set(current)
    if excludes:
        players = players.drop(index=(set(excludes) - forced_out) & set(players.index))
    xp = xp.reindex(players.index).fillna(0.0)
    codes = set(players.index)
    missing_locks = set(locks or ()) - codes
    if missing_locks:
        raise ValueError(f"locked players not in candidate pool: {missing_locks}")
    missing_current = set(current) - codes
    if missing_current:
        raise ValueError(f"current squad players not in candidate pool: {missing_current}")
    return players, xp, forced_out


def _transfer_terms(
    prob: pulp.LpProblem,
    objective: pulp.LpAffineExpression,
    squad: dict,
    codes: list[int],
    price: dict[int, float],
    current: dict[int, float],
    bank: float,
    free_transfers: int,
    hit_cost: float,
    max_transfers: int | None,
    ft_value: float,
) -> pulp.LpAffineExpression:
    """Transfer-mode money constraint and transfer penalties; returns the netted objective.

    Adds to `prob`: buys <= bank + sales, the `paid` (hit) and `charged`
    (free-transfer-spent) counters, and the optional transfer cap.
    """
    # money in from sales + bank must cover the buys
    raised = pulp.lpSum(current[p] * (1 - squad[p]) for p in current)
    spent = pulp.lpSum(price[p] * squad[p] for p in codes if p not in current)
    prob += spent <= bank + raised
    n_out = pulp.lpSum(1 - squad[p] for p in current)
    paid = pulp.LpVariable("paid_transfers", lowBound=0, cat="Integer")
    prob += paid >= n_out - free_transfers
    objective = objective - hit_cost * paid
    # free transfers used = n_out - paid; the first is on the house when
    # the bank is full (it would expire), the rest carry ft_value
    forfeit = 1 if free_transfers >= MAX_FREE_TRANSFERS else 0
    charged = pulp.LpVariable("charged_free_transfers", lowBound=0, cat="Integer")
    prob += charged >= n_out - paid - forfeit
    objective = objective - ft_value * charged
    if max_transfers is not None:
        prob += n_out <= max_transfers
    return objective


def _rulebook_constraints(
    prob: pulp.LpProblem,
    squad: dict,
    start: dict,
    cap: dict,
    codes: list[int],
    gws: list,
    players: pd.DataFrame,
    rules: SquadRules,
    locks: set[int] | None,
    forced_out: set[int],
) -> None:
    """Squad shape, club cap, a legal XI and one starting captain per week, locks and forced sales."""
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
    for p in forced_out:
        prob += squad[p] == 0


def _read_solution(
    prob: pulp.LpProblem,
    status_name: str,
    players: pd.DataFrame,
    xp: pd.DataFrame,
    xpd: dict,
    codes: list[int],
    gws: list,
    squad: dict,
    start: dict,
    cap: dict,
    bench_weight: float,
    current: dict[int, float],
    free_transfers: int,
    hit_cost: float,
) -> SquadSolution:
    """Turn the solved variables into the squad and lineup tables and the transfer lists."""
    picked = [p for p in codes if squad[p].value() > 0.5]
    window_points = sum(
        xpd[gw][p] * (
            (1 - bench_weight) * start[p][gw].value() + cap[p][gw].value() + bench_weight
        )
        for gw in gws
        for p in picked
    )
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
    squad_df["pos_rank"] = squad_df["POS"].map(POSITION_ORDER)
    squad_df = squad_df.sort_values(["pos_rank", "window_xp"], ascending=[True, False]).drop(
        columns="pos_rank"
    )

    transfers_out = [p for p in current if p not in set(picked)]
    transfers_in = [p for p in picked if p not in current] if current else []
    hit_points = hit_cost * max(len(transfers_out) - free_transfers, 0) if current else 0.0
    return SquadSolution(
        squad=squad_df,
        lineups=lineups,
        expected_points=float(window_points),
        cost=float(squad_df["price"].sum()),
        status=status_name,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        hit_points=hit_points,
    )


def optimize_squad(
    players: pd.DataFrame,
    xp: pd.DataFrame,
    rules: SquadRules | None = None,
    bench_weight: float = DEFAULT_BENCH_WEIGHT,
    locks: set[int] | None = None,
    excludes: set[int] | None = None,
    current: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
    hit_cost: float = HIT_COST,
    max_transfers: int | None = None,
    ft_value: float = DEFAULT_FT_VALUE,
) -> SquadSolution:
    """Solve for the optimal squad.

    `players`: indexed by player code, columns FPL_Name, TEAM, POS, price.
    `xp`: expected points, indexed by code, one column per gameweek.
    `locks`/`excludes`: player codes forced in or out of the 15.
    `current`: {code: selling price} of the squad you already own — switches
    on transfer mode (see module docstring); `price` is then the buy price.
    """
    rules = rules or SquadRules()
    current = dict(current or {})
    players, xp, forced_out = _prepare(players, xp, locks, excludes, current)
    codes = list(players.index)
    gws = list(xp.columns)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", codes, cat="Binary")
    start = pulp.LpVariable.dicts("start", (codes, gws), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", (codes, gws), cat="Binary")

    xpd = {gw: xp[gw].to_dict() for gw in gws}
    objective = pulp.lpSum(
        xpd[gw][p] * ((1 - bench_weight) * start[p][gw] + cap[p][gw] + bench_weight * squad[p])
        for gw in gws
        for p in codes
    )

    price = players["price"].to_dict()
    if current:
        objective = _transfer_terms(
            prob, objective, squad, codes, price, current,
            bank, free_transfers, hit_cost, max_transfers, ft_value,
        )
    else:
        prob += pulp.lpSum(price[p] * squad[p] for p in codes) <= rules.budget
    prob += objective

    _rulebook_constraints(prob, squad, start, cap, codes, gws, players, rules, locks, forced_out)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus[status]
    if status_name != "Optimal":
        raise RuntimeError(f"solver finished with status {status_name}")

    return _read_solution(
        prob, status_name, players, xp, xpd, codes, gws, squad, start, cap,
        bench_weight, current, free_transfers, hit_cost,
    )


def formation(xi: pd.DataFrame) -> str:
    """The XI's shape as "DEF-MID-FWD", e.g. "4-5-1", from its POS column."""
    return "-".join(str((xi["POS"] == p).sum()) for p in ["DEF", "MID", "FWD"])


def captains_by_gw(lineups: pd.DataFrame) -> pd.DataFrame:
    """The captain row of each gameweek's lineup, in gameweek order."""
    return lineups[lineups["captain"]].sort_values("gw")
