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
    uv run fpl archive                          # weekly elite-manager archive pass
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import api, archive, reference
from .config import DEFAULT_PARAMS
from .fixtures import build_fixture_slots, opponent_grid
from .optimizer import optimize_squad, select_pool
from .ppts import compute_ppts
from .predict import aggregate_window, with_value


def _live_players() -> pd.DataFrame:
    """Current bootstrap players, keyed by code: price, availability, status."""
    live = pd.DataFrame(api.fetch_bootstrap()["elements"])
    live["price"] = live["now_cost"] / 10
    # chance_of_playing_next_round is null when fully fit
    live["chance"] = live["chance_of_playing_next_round"].fillna(100) / 100
    return live.set_index("code")[["price", "chance", "status", "web_name"]]


def build_ppts(refresh: bool = False) -> pd.DataFrame:
    model = reference.load_model()
    if refresh:
        live = _live_players()
        codes = model["Code"]
        # players gone from the game (transfers abroad, unregistered) drop out
        model = model[codes.isin(live.index)].copy()
        model["Chance of Playing"] = model["Code"].map(live["chance"])
    fixtures = reference.load("FIXTURES")[["event", "team_h", "team_a"]].apply(
        pd.to_numeric, errors="coerce"
    )
    teams = reference.load("TEAMS")
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)
    la_xg, la_xgc = reference.league_average_xg()
    dcscore, dc_avg = reference.load_dcscore()
    return compute_ppts(
        model=model,
        team_strength=reference.load_team_strength(),
        dcscore=dcscore,
        dc_league_avg=dc_avg,
        fixture_grid=opponent_grid(build_fixture_slots(fixtures, teams), gws=range(1, 39)),
        league_avg_xg=la_xg,
        league_avg_xgc=la_xgc,
        params=DEFAULT_PARAMS,
        manual_start_pct=reference.load_manual_start_pct(),
    )


def _prices(refresh: bool) -> pd.Series:
    """Player price by code — live if refreshing, else the workbook snapshot."""
    if refresh:
        live = _live_players()
        return live.loc[live["status"] != "u", "price"]
    ref = reference.load("API")
    codes = pd.to_numeric(ref["code"], errors="coerce")
    price = pd.to_numeric(ref["now_cost"], errors="coerce") / 10
    mask = codes.notna() & price.notna()
    return pd.Series(price[mask].values, index=codes[mask].astype(int).values)


def _resolve_codes(names: list[str], players: pd.DataFrame) -> set[int]:
    codes: set[int] = set()
    for name in names:
        hits = players[players["FPL_Name"].str.contains(name, case=False, na=False)]
        if hits.empty:
            raise SystemExit(f"no player matching {name!r}")
        if len(hits) > 1:
            raise SystemExit(f"{name!r} is ambiguous — matches: {', '.join(hits['FPL_Name'])}")
        codes.add(int(hits.index[0]))
    return codes


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
    print(
        pred[cols]
        .head(args.top)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


def cmd_ppts(args: argparse.Namespace) -> None:
    ppts = build_ppts()
    rows = ppts[ppts["FPL_Name"].str.contains(args.player, case=False, na=False)]
    names = rows["FPL_Name"].unique()
    if len(names) == 0:
        raise SystemExit(f"no player matching {args.player!r}")
    if len(names) > 1:
        raise SystemExit(f"ambiguous — matches: {', '.join(names)}")
    rows = rows[(rows["gw"] >= args.from_gw) & (rows["gw"] <= args.to_gw)]
    rows = rows[rows["VS"].str.len() >= 3]  # drop blank fixture slots
    cols = ["gw", "VS", "8", "7", "9", "10", "11", "6", "Start %", "IF START TOTAL", "TRUE TOTAL"]
    header = {"8": "goals", "7": "assists", "9": "CS", "10": "2+GC", "11": "DC", "6": "bonus"}
    print(f"{names[0]} — per-gameweek projection (uppercase VS = home):")
    print(
        rows[cols]
        .rename(columns=header)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )


def cmd_squad(args: argparse.Namespace) -> None:
    ppts = build_ppts(refresh=args.refresh)
    window = ppts[(ppts["gw"] >= args.from_gw) & (ppts["gw"] <= args.to_gw)]
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")

    players = (
        window.groupby("Code")[["FPL_Name", "TEAM", "POS"]].first().join(_prices(args.refresh).rename("price"))
    )
    players = players[players["price"].notna()]

    locks = _resolve_codes(args.lock or [], players)
    excludes = _resolve_codes(args.exclude or [], players)
    pool = select_pool(players, xp.sum(axis=1), keep=locks)

    solution = optimize_squad(
        players=pool,
        xp=xp,
        bench_weight=args.bench_weight,
        locks=locks,
        excludes=excludes,
    )

    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    print(
        f"Optimal squad, GW{args.from_gw}-{args.to_gw}"
        f"{' (live prices/availability)' if args.refresh else ' (workbook snapshot)'}:"
    )
    print(f"cost £{solution.cost:.1f}m of £100.0m — expected points {solution.expected_points:.1f}\n")
    print(solution.squad.to_string(index=False, float_format=fmt))

    first_gw = solution.lineups[solution.lineups["gw"] == args.from_gw]
    xi = first_gw[first_gw["started"]]
    bench = first_gw[~first_gw["started"]]
    formation = "-".join(
        str((xi["POS"] == p).sum()) for p in ["DEF", "MID", "FWD"]
    )
    print(f"\nGW{args.from_gw} XI ({formation}):")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        names = [
            r.FPL_Name + (" (C)" if r.captain else "")
            for r in xi[xi["POS"] == pos].itertuples()
        ]
        print(f"  {pos:3s} {', '.join(names)}")
    print(f"  Bench: {', '.join(bench['FPL_Name'])}")

    captains = (
        solution.lineups[solution.lineups["captain"]]
        .sort_values("gw")[["gw", "FPL_Name", "TEAM"]]
    )
    print("\nCaptain by gameweek:")
    for r in captains.itertuples():
        print(f"  GW{r.gw}: {r.FPL_Name} ({r.TEAM})")


def cmd_archive(args: argparse.Namespace) -> None:
    dest = archive.run(top_n=args.top)
    print(f"archive at {dest}")


def main() -> None:
    p = argparse.ArgumentParser(prog="fpl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--from-gw", type=int, default=1)
    common.add_argument("--to-gw", type=int, default=10)

    pp = sub.add_parser("predict", parents=[common], help="ranked projections over a GW window")
    pp.add_argument("--top", type=int, default=20)
    pp.add_argument("--pos", choices=["GK", "DEF", "MID", "FWD", "gk", "def", "mid", "fwd"])
    pp.add_argument("--team", help="team short name, e.g. ARS")
    pp.add_argument("--max-price", type=float)
    pp.add_argument("--sort", choices=["pts", "value", "price"], default="pts")
    pp.set_defaults(func=cmd_predict)

    pg = sub.add_parser("ppts", parents=[common], help="per-gameweek breakdown for one player")
    pg.add_argument("--player", required=True, help="name substring, e.g. Haaland")
    pg.set_defaults(func=cmd_ppts)

    ps = sub.add_parser("squad", parents=[common], help="solve for the optimal 15-man squad")
    ps.add_argument("--bench-weight", type=float, default=0.1)
    ps.add_argument("--lock", action="append", help="force a player in (repeatable)")
    ps.add_argument("--exclude", action="append", help="force a player out (repeatable)")
    ps.add_argument(
        "--refresh", action="store_true",
        help="pull live prices and availability from the FPL API",
    )
    ps.set_defaults(func=cmd_squad)

    pa = sub.add_parser("archive", help="archive overall-league leaders' picks/transfers/chips")
    pa.add_argument(
        "--top", type=int, default=250,
        help="overall-league entries added to the cohort each pass (default 250)",
    )
    pa.set_defaults(func=cmd_archive)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
