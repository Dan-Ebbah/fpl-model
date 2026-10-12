"""Terminal reports: the text `fpl squad`, `fpl team`, `fpl scorers --team`,
`fpl style --team` and `fpl backtest` print.

Pure formatting — every function takes the objects a command already built
(an optimizer solution, a transfer plan, the pipeline, the backtest tables)
and writes to stdout. The CLI stays argparse plus thin commands; nothing here
fetches or computes.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from . import backtest as bt
from .optimizer import SquadSolution, captains_by_gw, formation
from .pipeline import Pipeline
from .rules import BUDGET, OUTFIELD, POSITIONS

_2DP = "{:.2f}".format  # the tables' float format


def _print_xi(xi: pd.DataFrame, label: Callable) -> None:
    """One line per position, `label(row)` for each starter."""
    for pos in POSITIONS:
        names = [label(r) for r in xi[xi["POS"] == pos].itertuples()]
        print(f"  {pos:3s} {', '.join(names)}")


def _print_captains(lineups: pd.DataFrame) -> None:
    print("\nCaptain by gameweek:")
    for r in captains_by_gw(lineups).itertuples():
        print(f"  GW{r.gw}: {r.FPL_Name} ({r.TEAM})")


def print_squad(solution: SquadSolution, from_gw: int, to_gw: int, live: bool) -> None:
    """`fpl squad`: the 15, the first gameweek's XI and bench, the captain per GW."""
    print(
        f"Optimal squad, GW{from_gw}-{to_gw}"
        f"{' (live prices/availability)' if live else ' (workbook snapshot)'}:"
    )
    print(f"cost £{solution.cost:.1f}m of £{BUDGET:.1f}m — expected points {solution.expected_points:.1f}\n")
    print(solution.squad.to_string(index=False, float_format=_2DP))

    first_gw = solution.lineups[solution.lineups["gw"] == from_gw]
    xi = first_gw[first_gw["started"]]
    bench = first_gw[~first_gw["started"]]
    print(f"\nGW{from_gw} XI ({formation(xi)}):")
    _print_xi(xi, lambda r: r.FPL_Name + (" (C)" if r.captain else ""))
    print(f"  Bench: {', '.join(bench['FPL_Name'])}")
    _print_captains(solution.lineups)


def print_plan(plan) -> None:
    """`fpl team`: team state, the current 15, the transfer proposal with its
    reasons, the first gameweek's XI/bench, the captain per GW and the verdict."""
    my = plan.team
    rank = f"{my.overall_rank:,}" if my.overall_rank else "n/a"
    print(f"{my.team_name} ({my.manager}, entry {my.entry_id}) — {my.total_points} pts, overall rank {rank}")
    print(
        f"Planning GW{plan.from_gw}-{plan.to_gw}. Bank £{my.bank:.1f}m, squad value £{my.squad_value:.1f}m "
        f"(selling), {my.free_transfers} free transfer(s), chips left: "
        f"{', '.join(my.chips_available) or 'none'}"
    )
    for w in plan.warnings:
        print(f"  ! {w}")

    print("\nYour squad (window xP, selling price):")
    cols = ["FPL_Name", "TEAM", "POS", "purchase", "sell", "chance", "window_xp"]
    print(plan.current[cols].to_string(index=False, float_format=_2DP))
    print(f"\nNo transfers: {plan.baseline.expected_points:.1f} xP over the window")

    if plan.transfers:
        hit = f" minus a {plan.proposal.hit_points:.0f}-pt hit" if plan.proposal.hit_points else ""
        print(f"Recommended: {len(plan.transfers)} transfer(s), {plan.proposal.expected_points:.1f} xP"
              f"{hit} — net gain {plan.gain:+.1f}, bank after £{plan.bank_after:.1f}m")
        for t in plan.transfers:
            print(f"  OUT {t['out']} ({t['out_team']} {t['pos']}, £{t['out_sell']:.1f}m, {t['out_xp']:.1f} xP)"
                  f"  ->  IN {t['in']} ({t['in_team']}, £{t['in_price']:.1f}m, {t['in_xp']:.1f} xP)"
                  f"   {t['xp_gain']:+.1f}")
            for line in t.get("why", []):
                print(f"      - {line}")
    else:
        print("Recommended: no transfers — the current 15 is the best use of your money this window.")

    xi, bench = plan.xi, plan.bench
    print(f"\nGW{plan.from_gw} XI ({formation(xi)}), {xi['xp'].sum():.1f} xP:")
    _print_xi(xi, lambda r: r.FPL_Name + (" (C)" if r.captain else " (V)" if r.vice else ""))
    print("  Bench: " + ", ".join(f"{i}. {r.FPL_Name}" for i, r in enumerate(bench.itertuples(), 1)))

    _print_captains(plan.proposal.lineups)
    if plan.verdict:
        print()
        for line in plan.verdict:
            print(line)


def print_style(pipe: Pipeline, team: str) -> None:
    """One team's playing style: venue tilt and which positions the chances go through."""
    st = pipe.style
    if st is None or team not in st.venue.index:
        return
    v = st.venue.loc[team]
    print(f"\n{team} playing style ({int(v['home_games'])} home, {int(v['away_games'])} away so far — factors are shrunk toward 1.0):")
    print(f"  venue, relative to a typical side: attack x{v['att_home']:.2f} home / x{v['att_away']:.2f} away "
          f"(raw tilt {v['att_tilt_raw']:.2f}); goals conceded x{v['def_home']:.2f} home / x{v['def_away']:.2f} away "
          f"(raw tilt {v['def_tilt_raw']:.2f})")
    for label, frame, cols in [
        ("chances fall to", st.attack, ("xg_share_raw", "xa_share_raw")),
        ("chances conceded to", st.concede, ("xg_share_raw", "xa_share_raw")),
    ]:
        bits = []
        for pos in OUTFIELD:
            r, lg = frame.loc[(team, pos)], st.league.loc[pos]
            bits.append(f"{pos} {r[cols[0]]:.0%} of xG (league {lg['xg_share']:.0%}), {r[cols[1]]:.0%} of xA (league {lg['xa_share']:.0%})")
        print(f"  {label}: " + "; ".join(bits))


def print_skipped_gameweeks(skipped: dict[int, str]) -> None:
    """`fpl backtest`: the gameweeks the replay could not score, and why."""
    for gw, why in sorted(skipped.items()):
        print(f"  ! GW{gw} skipped: {why}")


def print_backtest(table: pd.DataFrame, ev: pd.DataFrame, n_boot: int) -> None:
    """`fpl backtest`: what was replayed, each layer's scores with bootstrap
    intervals against the bare port, the verdicts, and start-chance calibration."""
    used = table.drop_duplicates("gw").sort_values("gw")
    print("Replayed " + ", ".join(f"GW{r.gw} (snapshot {r.snapshot})" for r in used.itertuples())
          + f" — {int(ev['n'].iloc[0])} player-gameweeks in the relevant pool.")
    print("Each variant saw only what was known before that deadline. priors.yaml is off.\n")
    cols = ["mae", "bias", "rank", "top10", "start_brier", "goal_brier", "assist_brier"]
    print(ev[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nmae/brier: lower is better. rank (Spearman within position) and top10 (actual points of the")
    print("ten best-projected per position) are what matter for picking a team: higher is better.\n")

    print(f"Difference from 'port', 90% paired-bootstrap interval ({n_boot} draws):")
    for name, r in ev.iterrows():
        if pd.isna(r.get("mae_lo")):
            continue
        print(f"  {name:8s}" + "  ".join(f"{k} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]" for k in ["mae", "rank", "top10", "start_brier"]))
    print()
    for line in bt.verdicts(ev):
        print(line)
    n_gw = int(ev["gws"].iloc[0])
    if n_gw < 8:
        print(f"\n{n_gw} gameweeks: treat this as a smoke test of the harness, not a ranking of the layers. "
              "The intervals resample players, not weeks, so they understate the real uncertainty.")

    for variant in ["port", "all"]:
        print(f"\nStart-chance calibration, '{variant}' (predicted vs how often they started):")
        print(bt.calibration(table, variant).to_string(float_format=_2DP))
    print(f"\nPer-player table saved to {bt.RESULTS_PATH}")
