"""Deadline plan for *your* team: transfers, XI, captain, bench order.

Glue between the projections (`ppts`), your reconstructed squad (`team.py`)
and the transfer-mode optimizer. Two solves per plan: the squad as it stands
(no transfers) gives the baseline, the free solve gives the proposal, and the
difference is what the transfers are worth over the window — net of hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .optimizer import SquadSolution, optimize_squad, select_pool
from .rules import (
    DEFAULT_BENCH_WEIGHT,
    DEFAULT_FT_VALUE,
    DEFAULT_HORIZON,
    HIT_COST,
    LAST_GW,
    POSITION_ORDER,
    SQUAD_SIZE,
)
from .team import MyTeam

# bench sort key: the GK sits apart in slot 1, the outfield three share a pool
# (ordered by xP among themselves) — deliberately NOT rules.POSITION_ORDER
BENCH_ORDER_POS = {"GK": 0, "DEF": 1, "MID": 1, "FWD": 1}


@dataclass
class TeamPlan:
    team: MyTeam
    from_gw: int
    to_gw: int
    current: pd.DataFrame        # your 15 with window xP and sell prices
    baseline: SquadSolution      # best XI/captain with no transfers
    proposal: SquadSolution      # best squad allowing transfers
    # one dict per transfer: out, out_code, out_team, pos, out_sell, out_xp,
    # in, in_code, in_team, in_price, in_xp, xp_gain — plus `why` (list[str])
    # once explain.explain_plan has run
    transfers: list[dict]
    xi: pd.DataFrame             # next-GW starters of the proposal, captain/vice flagged
    bench: pd.DataFrame          # next-GW bench in recommended order
    bank_after: float
    warnings: list[str] = field(default_factory=list)
    verdict: list[str] = field(default_factory=list)  # filled by explain.explain_plan

    @property
    def gain(self) -> float:
        """Window xP gained by the proposal over standing pat, net of hits."""
        return self.proposal.expected_points - self.proposal.hit_points - self.baseline.expected_points


def squad_frame(ppts: pd.DataFrame, prices: pd.Series, from_gw: int, to_gw: int):
    """(players, xp) in the optimizer's shape for a gameweek window."""
    window = ppts[(ppts["gw"] >= from_gw) & (ppts["gw"] <= to_gw)]
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")
    xp = xp.reindex(columns=range(from_gw, to_gw + 1)).fillna(0.0)
    players = (
        window.groupby("Code")[["FPL_Name", "TEAM", "POS"]].first()
        .join(prices.rename("price"))
    )
    return players[players["price"].notna()], xp


def _ensure_owned(players: pd.DataFrame, xp: pd.DataFrame, team: MyTeam, warnings: list[str]):
    """Owned players missing from the model get a zero projection, not an error."""
    rows = []
    for p in team.picks:
        if p.code not in players.index:
            rows.append({"Code": p.code, "FPL_Name": p.name, "TEAM": p.team, "POS": p.pos, "price": p.now})
            warnings.append(f"{p.name} has no projection (not in the model) — treated as 0 xP.")
    if rows:
        players = pd.concat([players, pd.DataFrame(rows).set_index("Code")])
        xp = xp.reindex(players.index).fillna(0.0)
    return players, xp


def _next_gw_lineup(sol: SquadSolution, xp: pd.DataFrame, gw: int):
    """(xi, bench) for gameweek `gw` of a solution, each with that week's xP.

    The XI is sorted by position then xP and flags the vice-captain (best
    starter after the captain); the bench is in auto-sub order.
    """
    first = sol.lineups[sol.lineups["gw"] == gw].copy()
    first["xp"] = first["Code"].map(xp[gw])
    xi = first[first["started"]].copy()
    # vice-captain: best remaining starter by that week's xP
    others = xi[~xi["captain"]].sort_values("xp", ascending=False)
    xi["vice"] = xi["Code"] == (others["Code"].iloc[0] if len(others) else -1)
    xi["pos_rank"] = xi["POS"].map(POSITION_ORDER)
    xi = xi.sort_values(["pos_rank", "xp"], ascending=[True, False]).drop(columns="pos_rank")
    bench = first[~first["started"]].copy()
    # FPL bench: the GK sits apart; outfield order = who you want auto-subbed first
    bench["slot"] = bench["POS"].map(BENCH_ORDER_POS)
    bench = bench.sort_values(["slot", "xp"], ascending=[True, False]).drop(columns="slot")
    return xi, bench


def _current_table(
    players: pd.DataFrame, xp: pd.DataFrame, team: MyTeam, sell: dict[int, float]
) -> pd.DataFrame:
    """Your 15 with window xP, sell/purchase prices and availability, by position then xP."""
    current = players.loc[sorted(team.codes)].copy()
    current["window_xp"] = xp.loc[current.index].sum(axis=1)
    current["sell"] = current.index.map(sell)
    current["purchase"] = current.index.map({p.code: p.purchase for p in team.picks})
    current["chance"] = current.index.map({p.code: p.chance for p in team.picks})
    current["pos_rank"] = current["POS"].map(POSITION_ORDER)
    return current.sort_values(["pos_rank", "window_xp"], ascending=[True, False]).drop(columns="pos_rank")


def _pair_transfers(
    proposal: SquadSolution, players: pd.DataFrame, xp: pd.DataFrame, sell: dict[int, float]
) -> list[dict]:
    """Pair each player sold with one bought in the same position.

    Ins and outs are position-balanced by the squad rules, so pairing is
    well-defined; outs go weakest first and take the next unpaired in.
    """
    wxp = xp.sum(axis=1)
    transfers: list[dict] = []
    ins_by_pos: dict[str, list[int]] = {}
    for c in proposal.transfers_in:
        ins_by_pos.setdefault(players.loc[c, "POS"], []).append(c)
    for out_code in sorted(proposal.transfers_out, key=lambda c: wxp[c]):
        pos = players.loc[out_code, "POS"]
        in_code = ins_by_pos[pos].pop(0)
        transfers.append({
            "out": players.loc[out_code, "FPL_Name"], "out_code": int(out_code),
            "out_team": players.loc[out_code, "TEAM"], "pos": pos,
            "out_sell": float(sell[out_code]), "out_xp": float(wxp[out_code]),
            "in": players.loc[in_code, "FPL_Name"], "in_code": int(in_code),
            "in_team": players.loc[in_code, "TEAM"],
            "in_price": float(players.loc[in_code, "price"]), "in_xp": float(wxp[in_code]),
            "xp_gain": float(wxp[in_code] - wxp[out_code]),
        })
    return transfers


def _plan_warnings(plan: TeamPlan, ft: int, wildcard: bool) -> list[str]:
    """Rolling-is-fine when FTs go unused; the net worth of any hit taken."""
    warnings: list[str] = []
    if ft >= 2 and not plan.transfers and not wildcard:
        warnings.append(f"You hold {ft} free transfers and the model sees nothing worth using them on — rolling is fine.")
    if plan.proposal.hit_points:
        warnings.append(
            f"Proposal takes a {plan.proposal.hit_points:.0f}-point hit; it is still "
            f"+{plan.gain:.1f} xP "
            f"net over GW{plan.from_gw}-{plan.to_gw}."
        )
    return warnings


def plan_transfers(
    ppts: pd.DataFrame,
    prices: pd.Series,
    team: MyTeam,
    to_gw: int | None = None,
    bench_weight: float = DEFAULT_BENCH_WEIGHT,
    free_transfers: int | None = None,
    bank: float | None = None,
    max_transfers: int | None = None,
    hit_cost: float = HIT_COST,
    ft_value: float = DEFAULT_FT_VALUE,
    wildcard: bool = False,
    locks: set[int] | None = None,
    excludes: set[int] | None = None,
) -> TeamPlan:
    """Plan the next deadline for `team` over the window up to `to_gw`.

    The window runs from the team's next deadline to `to_gw` (default: the
    next `rules.DEFAULT_HORIZON` gameweeks, capped at the season's end).
    `free_transfers`/`bank` override the reconstructed values; `wildcard`
    makes every transfer free and uncapped. `locks`/`excludes` pass through
    to the optimizer — excluding an owned player forces his sale.
    """
    from_gw = team.next_gw
    to_gw = to_gw or min(from_gw + DEFAULT_HORIZON, LAST_GW)
    if to_gw < from_gw:
        raise ValueError(f"--to-gw {to_gw} is before the next deadline (GW{from_gw})")
    warnings: list[str] = list(team.notes)
    ft = team.free_transfers if free_transfers is None else free_transfers
    bank = team.bank if bank is None else bank
    if wildcard:
        # "unlimited" free transfers = one per squad slot: nothing to hit, nothing to value
        ft, hit_cost, max_transfers, ft_value = SQUAD_SIZE, 0.0, None, 0.0

    players, xp = squad_frame(ppts, prices, from_gw, to_gw)
    players, xp = _ensure_owned(players, xp, team, warnings)
    owned = team.codes
    pool = select_pool(players, xp.sum(axis=1), keep=owned | set(locks or ()))
    sell = team.selling_prices

    baseline = optimize_squad(
        pool, xp, bench_weight=bench_weight, current=sell, bank=bank,
        free_transfers=ft, hit_cost=hit_cost, max_transfers=0,
    )
    proposal = optimize_squad(
        pool, xp, bench_weight=bench_weight, current=sell, bank=bank,
        free_transfers=ft, hit_cost=hit_cost, max_transfers=max_transfers,
        ft_value=ft_value, locks=locks, excludes=excludes,
    )

    current = _current_table(players, xp, team, sell)
    transfers = _pair_transfers(proposal, players, xp, sell)
    raised = sum(t["out_sell"] for t in transfers)
    spent = sum(t["in_price"] for t in transfers)
    bank_after = round(bank + raised - spent, 1)

    xi, bench = _next_gw_lineup(proposal, xp, from_gw)
    plan = TeamPlan(
        team=team, from_gw=from_gw, to_gw=to_gw, current=current,
        baseline=baseline, proposal=proposal, transfers=transfers,
        xi=xi, bench=bench, bank_after=bank_after, warnings=warnings,
    )
    plan.warnings.extend(_plan_warnings(plan, ft, wildcard))
    return plan
