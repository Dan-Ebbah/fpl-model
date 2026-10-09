"""Command-line interface for querying the model's projections.

Runs the full pipeline off the workbook reference snapshot and prints a
ranked table. Examples:

    uv run fpl predict                          # top 20, GW1-10
    uv run fpl predict --from-gw 1 --to-gw 5    # shorter window
    uv run fpl predict --pos DEF --max-price 6  # budget defenders
    uv run fpl predict --team ARS --top 30      # one team
    uv run fpl predict --sort value             # points per £ per game
    uv run fpl ppts --player Haaland            # per-gameweek breakdown
    uv run fpl squad --to-gw 8 --refresh        # optimal 15 with live prices/news
    uv run fpl squad --lock Haaland --exclude Salah
    uv run fpl team --entry 1234567             # YOUR team: transfers, XI, captain
    uv run fpl team --to-gw 12 --max-transfers 1  # (entry id is remembered)
    uv run fpl team --wildcard                  # rebuild the 15 from your bank + sales
    uv run fpl form                             # team form multipliers in use
    uv run fpl style --team NEW                 # home/away tilt, where chances go
    uv run fpl scorers --gw 5 --team NEW        # anytime scorer / assist odds
    uv run fpl backtest                         # replay past deadlines, score each layer
    uv run fpl archive                          # weekly elite-manager archive pass
"""

from __future__ import annotations

import argparse

from . import api, archive, explain, reference, report, team as team_mod, web
from .optimizer import optimize_squad, select_pool
from .pipeline import build_pipeline, build_ppts, player_prices, resolve_codes
from .plan import plan_transfers
from .predict import aggregate_window, gameweek_window, with_fixture, with_value
from .rules import (
    DEFAULT_BENCH_WEIGHT, DEFAULT_FT_VALUE, FIRST_GW, HIT_COST, OUTFIELD, POSITIONS,
)
from .scorers import scorer_table

# --- defaults the flags fall back to -----------------------------------------
DEFAULT_TO_GW = 10  # predict / ppts / squad window end
DEFAULT_PREDICT_TOP = 20
DEFAULT_SCORERS_TOP = 25
DEFAULT_BOOT_DRAWS = 300  # paired-bootstrap draws behind the backtest intervals
DEFAULT_ARCHIVE_TOP = 250  # overall-league entries the archive adds per pass

_2DP = "{:.2f}".format


def cmd_predict(args: argparse.Namespace) -> None:
    pred = with_value(
        aggregate_window(build_ppts(), args.from_gw, args.to_gw), reference.load("API")
    )
    if args.pos:
        pred = pred[pred["POS"] == args.pos.upper()]
    if args.team:
        pred = pred[pred["TEAM"] == args.team.upper()]
    if args.max_price:
        pred = pred[pred["price"] <= args.max_price]
    pred = pred.sort_values(
        {"pts": "true_pts", "value": "value", "price": "price"}[args.sort],
        ascending=False,
    )
    cols = ["FPL_Name", "TEAM", "POS", "price", "games", "if_start_pts", "true_pts", "value"]
    print(f"GW{args.from_gw}-{args.to_gw}, sorted by {args.sort}:")
    print(pred[cols].head(args.top).to_string(index=False, float_format=_2DP))


def cmd_ppts(args: argparse.Namespace) -> None:
    ppts = build_ppts()
    rows = ppts[ppts["FPL_Name"].str.contains(args.player, case=False, na=False)]
    names = rows["FPL_Name"].unique()
    if len(names) == 0:
        raise SystemExit(f"no player matching {args.player!r}")
    if len(names) > 1:
        raise SystemExit(f"ambiguous — matches: {', '.join(names)}")
    rows = with_fixture(gameweek_window(rows, args.from_gw, args.to_gw))
    cols = ["gw", "VS", "8", "7", "9", "10", "11", "6", "Start %", "IF START TOTAL", "TRUE TOTAL"]
    header = {"8": "goals", "7": "assists", "9": "CS", "10": "2+GC", "11": "DC", "6": "bonus"}
    print(f"{names[0]} — per-gameweek projection (uppercase VS = home):")
    print(rows[cols].rename(columns=header).to_string(index=False, float_format=_2DP))


def cmd_squad(args: argparse.Namespace) -> None:
    ppts = build_ppts(refresh=args.refresh)
    window = gameweek_window(ppts, args.from_gw, args.to_gw)
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")

    players = (
        window.groupby("Code")[["FPL_Name", "TEAM", "POS"]].first().join(player_prices(args.refresh).rename("price"))
    )
    players = players[players["price"].notna()]

    locks = resolve_codes(args.lock or [], players)
    excludes = resolve_codes(args.exclude or [], players)
    pool = select_pool(players, xp.sum(axis=1), keep=locks)

    solution = optimize_squad(
        players=pool,
        xp=xp,
        bench_weight=args.bench_weight,
        locks=locks,
        excludes=excludes,
    )
    report.print_squad(solution, args.from_gw, args.to_gw, live=args.refresh)


def cmd_team(args: argparse.Namespace) -> None:
    entry_id = team_mod.resolve_entry(args.entry)
    boot = api.fetch_bootstrap()
    my = team_mod.load_my_team(entry_id, boot)
    if args.entry:
        team_mod.remember_entry(entry_id)  # only once it resolved to a real team
    pipe = build_pipeline(refresh=True, boot=boot, use_inseason=not args.no_inseason)
    ppts = pipe.ppts
    prices = player_prices(True, boot)
    players_all = ppts.groupby("Code")[["FPL_Name"]].first()
    plan = plan_transfers(
        ppts, prices, my, to_gw=args.to_gw, bench_weight=args.bench_weight,
        free_transfers=args.ft, bank=args.bank, max_transfers=args.max_transfers,
        hit_cost=args.hit_cost, ft_value=args.ft_value, wildcard=args.wildcard,
        locks=resolve_codes(args.lock or [], players_all),
        excludes=resolve_codes(args.exclude or [], players_all),
    )
    explain.explain_plan(plan, pipe, prices, ft_value=args.ft_value)
    report.print_plan(plan)


def cmd_form(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    f = pipe.team_form.sort_values("att_mult", ascending=False)
    print("Team form this season vs what pre-season strength predicted for the fixtures played.")
    print("raw = actual/expected; mult = what the model applies after shrinking toward 1.0.")
    print("Defence columns are goals conceded: below 1 is good.\n")
    cols = ["games", "xg", "exp_xg", "att_raw", "att_mult", "xgc", "exp_xgc", "def_raw", "def_mult"]
    print(f[cols].to_string(float_format=_2DP))


def cmd_scorers(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    gw = args.gw or team_mod.current_and_next_gw(pipe.boot)[1]
    table = scorer_table(pipe.ppts, gw, args.team)
    key = {"goal": "P(goal)", "assist": "P(assist)", "either": "P(either)"}[args.sort]
    cols = ["FPL_Name", "TEAM", "POS", "VS", "start", "xG", "P(goal)", "xA", "P(assist)", "P(either)"]
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    fmt = {c: pct for c in ["start", "P(goal)", "P(assist)", "P(either)"]}
    fmt.update({"xG": _2DP, "xA": _2DP})
    print(f"GW{gw} scorer / assist predictor{f' — {args.team.upper()}' if args.team else ''}, sorted by {key} (uppercase VS = home):")
    print(table.sort_values(key, ascending=False).head(args.top)[cols].to_string(index=False, formatters=fmt))
    if args.team:
        tot = table[["xG", "xA"]].sum()
        print(f"\nTeam total: {tot['xG']:.2f} expected goals, {tot['xA']:.2f} expected FPL assists.")
        report.print_style(pipe, args.team.upper())


def cmd_style(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    if args.team:
        report.print_style(pipe, args.team.upper())
        return
    st = pipe.style
    a = st.attack["xg_share_raw"].unstack()[list(OUTFIELD)].add_prefix("xG% ")
    c = st.concede["xg_share_raw"].unstack()[list(OUTFIELD)].add_prefix("conc% ")
    out = st.venue[["att_tilt_raw", "att_home", "def_tilt_raw", "def_home"]].join(a).join(c)
    print("Venue tilt (home/away ratio vs a typical side; att_home/def_home are the shrunk factors applied,")
    print("away is the reciprocal) and the share of xG each position takes / is conceded to. --team for detail.\n")
    print(out.to_string(float_format=_2DP))


def cmd_backtest(args: argparse.Namespace) -> None:
    from . import backtest as bt

    table, skipped, ev = bt.run(gws=args.gw, n_boot=args.boot)
    report.print_skipped_gameweeks(skipped)
    if table.empty:
        raise SystemExit("nothing to score yet")
    report.print_backtest(table, ev, n_boot=args.boot)


def cmd_archive(args: argparse.Namespace) -> None:
    dest = archive.run(top_n=args.top)
    print(f"archive at {dest}")


def cmd_web(args: argparse.Namespace) -> None:
    web.serve(port=args.port, open_browser=not args.no_browser)


# --- parsers ------------------------------------------------------------------


def _window_args() -> argparse.ArgumentParser:
    """`--from-gw` / `--to-gw`, shared by the projection commands."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--from-gw", type=int, default=FIRST_GW)
    common.add_argument("--to-gw", type=int, default=DEFAULT_TO_GW)
    return common


def _add_predict_parser(sub, common: argparse.ArgumentParser) -> None:
    pp = sub.add_parser("predict", parents=[common], help="ranked projections over a GW window")
    pp.add_argument("--top", type=int, default=DEFAULT_PREDICT_TOP)
    pp.add_argument("--pos", choices=[*POSITIONS, *(p.lower() for p in POSITIONS)])
    pp.add_argument("--team", help="team short name, e.g. ARS")
    pp.add_argument("--max-price", type=float)
    pp.add_argument("--sort", choices=["pts", "value", "price"], default="pts")
    pp.set_defaults(func=cmd_predict)


def _add_ppts_parser(sub, common: argparse.ArgumentParser) -> None:
    pg = sub.add_parser("ppts", parents=[common], help="per-gameweek breakdown for one player")
    pg.add_argument("--player", required=True, help="name substring, e.g. Haaland")
    pg.set_defaults(func=cmd_ppts)


def _add_squad_parser(sub, common: argparse.ArgumentParser) -> None:
    ps = sub.add_parser("squad", parents=[common], help="solve for the optimal 15-man squad")
    ps.add_argument("--bench-weight", type=float, default=DEFAULT_BENCH_WEIGHT)
    ps.add_argument("--lock", action="append", help="force a player in (repeatable)")
    ps.add_argument("--exclude", action="append", help="force a player out (repeatable)")
    ps.add_argument(
        "--refresh", action="store_true",
        help="pull live prices and availability from the FPL API",
    )
    ps.set_defaults(func=cmd_squad)


def _add_team_parser(sub) -> None:
    pt = sub.add_parser("team", help="transfers, XI and captain for YOUR team (live data)")
    pt.add_argument("--entry", type=int, help="your FPL entry id (remembered in data/me.json)")
    pt.add_argument("--to-gw", type=int, help="end of the planning window (default: next GW + 4)")
    pt.add_argument("--bench-weight", type=float, default=DEFAULT_BENCH_WEIGHT)
    pt.add_argument("--ft", type=int, help="override the free-transfer count")
    pt.add_argument("--bank", type=float, help="override the bank (£m)")
    pt.add_argument("--max-transfers", type=int, help="cap transfers (0 = just pick the XI)")
    pt.add_argument("--hit-cost", type=float, default=HIT_COST, help="points per extra transfer")
    pt.add_argument(
        "--ft-value", type=float, default=DEFAULT_FT_VALUE,
        help="xP a free transfer must beat to be worth using (0 = use them on any gain)",
    )
    pt.add_argument("--wildcard", action="store_true", help="unlimited free transfers, budget = bank + sales")
    pt.add_argument("--lock", action="append", help="force a player in (repeatable)")
    pt.add_argument("--exclude", action="append", help="force a player out (repeatable)")
    pt.add_argument(
        "--no-inseason", action="store_true",
        help="pre-season model only: skip team form and the current-season rate blend",
    )
    pt.set_defaults(func=cmd_team)


def _add_scorers_parser(sub) -> None:
    psc = sub.add_parser("scorers", help="anytime goalscorer / assist probabilities for a gameweek")
    psc.add_argument("--gw", type=int, help="gameweek (default: the next one)")
    psc.add_argument("--team", help="one team, with its style profile, e.g. NEW")
    psc.add_argument("--top", type=int, default=DEFAULT_SCORERS_TOP)
    psc.add_argument("--sort", choices=["goal", "assist", "either"], default="goal")
    psc.set_defaults(func=cmd_scorers)


def _add_style_parser(sub) -> None:
    pst = sub.add_parser("style", help="team playing style: home/away tilt and positional channels")
    pst.add_argument("--team", help="one team in detail, e.g. NEW")
    pst.set_defaults(func=cmd_style)


def _add_backtest_parser(sub) -> None:
    pb = sub.add_parser("backtest", help="replay past deadlines from the snapshots and score each model layer")
    pb.add_argument("--gw", type=int, action="append", help="only this gameweek (repeatable; default: all finished)")
    pb.add_argument("--boot", type=int, default=DEFAULT_BOOT_DRAWS, help="bootstrap draws for the intervals")
    pb.set_defaults(func=cmd_backtest)


def _add_form_parser(sub) -> None:
    pf = sub.add_parser("form", help="schedule-adjusted team form the model is applying")
    pf.set_defaults(func=cmd_form)


def _add_web_parser(sub) -> None:
    pw = sub.add_parser("web", help="local browser UI for predictions and squad solving")
    pw.add_argument("--port", type=int, default=web.DEFAULT_PORT)
    pw.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    pw.set_defaults(func=cmd_web)


def _add_archive_parser(sub) -> None:
    pa = sub.add_parser("archive", help="archive overall-league leaders' picks/transfers/chips")
    pa.add_argument(
        "--top", type=int, default=DEFAULT_ARCHIVE_TOP,
        help=f"overall-league entries added to the cohort each pass (default {DEFAULT_ARCHIVE_TOP})",
    )
    pa.set_defaults(func=cmd_archive)


def main() -> None:
    p = argparse.ArgumentParser(prog="fpl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = _window_args()
    _add_predict_parser(sub, common)
    _add_ppts_parser(sub, common)
    _add_squad_parser(sub, common)
    _add_team_parser(sub)
    _add_scorers_parser(sub)
    _add_style_parser(sub)
    _add_backtest_parser(sub)
    _add_form_parser(sub)
    _add_web_parser(sub)
    _add_archive_parser(sub)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
