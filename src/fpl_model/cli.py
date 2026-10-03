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
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import api, archive, explain, inseason, matchlog, priors, reference, team as team_mod
from .config import DEFAULT_PARAMS
from .fixtures import build_fixture_slots, opponent_grid
from .model_build import build_model
from .optimizer import optimize_squad, select_pool
from .plan import plan_transfers
from .ppts import compute_ppts
from .style import DEFAULT_STYLE, TeamStyle, fixture_adjustments, team_style
from .predict import aggregate_window, with_value


def _live_players(boot: dict | None = None) -> pd.DataFrame:
    """Current bootstrap players, keyed by code: price, availability, status."""
    live = pd.DataFrame((boot or api.fetch_bootstrap())["elements"])
    live["price"] = live["now_cost"] / 10
    # chance_of_playing_next_round is null when fully fit
    live["chance"] = live["chance_of_playing_next_round"].fillna(100) / 100
    return live.set_index("code")[["price", "chance", "status", "web_name"]]


@dataclass
class Pipeline:
    """Projections plus the intermediate tables the explainer reads."""

    ppts: pd.DataFrame
    model: pd.DataFrame
    team_form: pd.DataFrame | None = None  # None when the in-season layer is off
    style: TeamStyle | None = None
    boot: dict | None = None


def build_pipeline(
    refresh: bool = False, boot: dict | None = None, fixtures_raw: list[dict] | None = None,
    use_inseason: bool = True, layers: inseason.Layers | None = None, persist: bool = True,
    log: pd.DataFrame | None = None, base_model: pd.DataFrame | None = None,
    gws: range = range(1, 39),
) -> Pipeline:
    """The full projection pipeline.

    The last five arguments exist for the backtester. `layers` picks the
    adjustments and their constants (default: all on; `use_inseason=False` is
    shorthand for the bare port). `persist=False` stops a replay of an old
    snapshot being stored as today's. `log` supplies the match log — a replay
    passes one cut off before the gameweek it predicts, since fetching would
    leak the future. `base_model` reuses build_model's output across variants
    of the same snapshot; `gws` limits the fixture grid.
    """
    layers = layers or (inseason.ALL_LAYERS if use_inseason else inseason.PORT_ONLY)
    la_xg, la_xgc = reference.league_average_xg()
    team_strength = reference.load_team_strength()
    form = style = fixture_adj = None
    if refresh:
        # Rebuild the MODEL player table from the live bootstrap: current
        # clubs, positions, injury flags, and new players — the frozen inputs
        # (last-season history) stay as extracted. Fixtures also come live so
        # rescheduled games land in the right gameweek.
        boot = boot or api.fetch_bootstrap()
        fixtures_raw = fixtures_raw or api.fetch_fixtures()
        if persist:
            # every fresh pull is persisted: the dated snapshot series is the
            # backtester's only point-in-time record
            dest = api.snapshot(bootstrap=boot, fixtures=fixtures_raw)
            print(f"(live data pulled and stored as snapshot {dest.name})")
        teams = pd.DataFrame(boot["teams"])
        api_df = pd.DataFrame(boot["elements"])
        if base_model is None:
            inputs = reference.load_model_inputs()
            inputs["api"], inputs["teams"] = api_df, teams
            base_model = build_model(points=DEFAULT_PARAMS.points, **inputs)
        model = base_model
        fixtures = pd.DataFrame(fixtures_raw)[["event", "team_h", "team_a"]].apply(
            pd.to_numeric, errors="coerce"
        )
        # current-season evidence — only meaningful on live data; the frozen
        # path below stays the faithful port
        if layers.form or layers.starts:
            measured = inseason.team_form(boot, fixtures_raw, team_strength, la_xgc, DEFAULT_PARAMS, layers.params)
            if layers.form:
                form = measured
                team_strength = inseason.adjust_team_strength(team_strength, form)
        if layers.style:
            # playing style needs per-match rows (venue, opponent, position)
            if log is None:
                log = matchlog.load_match_log(boot, fixtures_raw)
            if len(log):
                style = team_style(log, team_strength, la_xgc, DEFAULT_PARAMS, layers.style_params or DEFAULT_STYLE)
                fixture_adj = fixture_adjustments(style, list(team_strength["ABVR"].str.upper()))
        if layers.rates:
            model = inseason.adjust_player_rates(
                model, api_df, DEFAULT_PARAMS.points, layers.params,
                team_attack_rel=inseason.team_attack_relative(team_strength, la_xg, DEFAULT_PARAMS),
                channel_rel=style.attack if style is not None else None,
            )
        if layers.starts:
            model = inseason.adjust_start_rates(model, api_df, measured, layers.params)
    else:
        model = reference.load_model()
        teams = reference.load("TEAMS")
        fixtures = reference.load("FIXTURES")[["event", "team_h", "team_a"]].apply(
            pd.to_numeric, errors="coerce"
        )
    teams = teams.copy()
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)

    manual_start_pct = reference.load_manual_start_pct()
    if layers.priors:
        # hand-entered priors for players with no PL history (data/priors.yaml)
        model, prior_starts = priors.apply_priors(model, priors.load_priors(), DEFAULT_PARAMS.points)
        for code, pct in prior_starts.items():
            manual_start_pct[code] = pct

    dcscore, dc_avg = reference.load_dcscore()
    ppts = compute_ppts(
        model=model,
        team_strength=team_strength,
        dcscore=dcscore,
        dc_league_avg=dc_avg,
        fixture_grid=opponent_grid(build_fixture_slots(fixtures, teams), gws=gws),
        league_avg_xg=la_xg,
        league_avg_xgc=la_xgc,
        params=DEFAULT_PARAMS,
        manual_start_pct=manual_start_pct,
        fixture_adj=fixture_adj,
    )
    return Pipeline(ppts=ppts, model=model, team_form=form, boot=boot if refresh else None, style=style)


def build_ppts(
    refresh: bool = False, boot: dict | None = None, fixtures_raw: list[dict] | None = None,
    use_inseason: bool = True,
) -> pd.DataFrame:
    return build_pipeline(refresh, boot, fixtures_raw, use_inseason).ppts


def _prices(refresh: bool, boot: dict | None = None) -> pd.Series:
    """Player price by code — live if refreshing, else the workbook snapshot."""
    if refresh:
        live = _live_players(boot)
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


def cmd_team(args: argparse.Namespace) -> None:
    entry_id = team_mod.resolve_entry(args.entry)
    boot = api.fetch_bootstrap()
    my = team_mod.load_my_team(entry_id, boot)
    if args.entry:
        team_mod.remember_entry(entry_id)  # only once it resolved to a real team
    pipe = build_pipeline(refresh=True, boot=boot, use_inseason=not args.no_inseason)
    ppts = pipe.ppts
    prices = _prices(True, boot)
    players_all = ppts.groupby("Code")[["FPL_Name"]].first()
    plan = plan_transfers(
        ppts, prices, my, to_gw=args.to_gw, bench_weight=args.bench_weight,
        free_transfers=args.ft, bank=args.bank, max_transfers=args.max_transfers,
        hit_cost=args.hit_cost, ft_value=args.ft_value, wildcard=args.wildcard,
        locks=_resolve_codes(args.lock or [], players_all),
        excludes=_resolve_codes(args.exclude or [], players_all),
    )
    explain.explain_plan(plan, pipe, prices, ft_value=args.ft_value)
    print_plan(plan)


def print_plan(plan) -> None:
    my, fmt = plan.team, (lambda x: f"{x:.2f}")
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
    print(plan.current[cols].to_string(index=False, float_format=fmt))
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
    formation = "-".join(str((xi["POS"] == p).sum()) for p in ["DEF", "MID", "FWD"])
    print(f"\nGW{plan.from_gw} XI ({formation}), {xi['xp'].sum():.1f} xP:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        names = [
            r.FPL_Name + (" (C)" if r.captain else " (V)" if r.vice else "")
            for r in xi[xi["POS"] == pos].itertuples()
        ]
        print(f"  {pos:3s} {', '.join(names)}")
    print("  Bench: " + ", ".join(f"{i}. {r.FPL_Name}" for i, r in enumerate(bench.itertuples(), 1)))

    captains = plan.proposal.lineups[plan.proposal.lineups["captain"]].sort_values("gw")
    print("\nCaptain by gameweek:")
    for r in captains.itertuples():
        print(f"  GW{r.gw}: {r.FPL_Name} ({r.TEAM})")
    if plan.verdict:
        print()
        for line in plan.verdict:
            print(line)


def cmd_form(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    f = pipe.team_form.sort_values("att_mult", ascending=False)
    print("Team form this season vs what pre-season strength predicted for the fixtures played.")
    print("raw = actual/expected; mult = what the model applies after shrinking toward 1.0.")
    print("Defence columns are goals conceded: below 1 is good.\n")
    cols = ["games", "xg", "exp_xg", "att_raw", "att_mult", "xgc", "exp_xgc", "def_raw", "def_mult"]
    print(f[cols].to_string(float_format=lambda x: f"{x:.2f}"))


def scorer_table(ppts: pd.DataFrame, gw: int, team: str | None = None) -> pd.DataFrame:
    """Anytime scorer / assist probabilities for one gameweek.

    Poisson on the fixture's expected goals and FPL assists (already scaled
    for opponent, venue tilt, positional channel and minutes): P = 1 - e^-x.
    A double gameweek sums the expectations before converting.

    Start probabilities are per-player estimates and don't know a team fields
    exactly ten outfielders: a rebuilt squad can sum to 7, a deep one to 13.
    For a probability table that has to reconcile with the team's goals, each
    team's outfield expectations are rescaled toward ten starters (capped —
    the points model itself is left alone).
    """
    rows = ppts[(ppts["gw"] == gw) & (ppts["VS"].astype(str).str.len() >= 3)].copy()
    outfield = rows[rows["POS"] != "GK"]
    n_fix = outfield.groupby("TEAM")["slot"].nunique()
    scale = (10 * n_fix / outfield.groupby("TEAM")["Start %"].sum()).clip(0.8, 1.5)
    k = rows["TEAM"].map(scale).fillna(1.0).where(rows["POS"] != "GK", 1.0)
    rows["exp_goals"], rows["exp_assists"] = rows["exp_goals"] * k, rows["exp_assists"] * k
    rows["Start %"] = (rows["Start %"] * k).clip(upper=0.98)
    if team:
        rows = rows[rows["TEAM"].str.upper() == team.upper()]
    g = rows.groupby("Code").agg(
        FPL_Name=("FPL_Name", "first"), TEAM=("TEAM", "first"), POS=("POS", "first"),
        VS=("VS", "+".join), start=("Start %", "mean"),
        xG=("exp_goals", "sum"), xA=("exp_assists", "sum"),
    )
    g["P(goal)"] = 1 - np.exp(-g["xG"])
    g["P(assist)"] = 1 - np.exp(-g["xA"])
    g["P(either)"] = 1 - np.exp(-(g["xG"] + g["xA"]))
    return g


def cmd_scorers(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    gw = args.gw or team_mod.current_and_next_gw(pipe.boot)[1]
    table = scorer_table(pipe.ppts, gw, args.team)
    key = {"goal": "P(goal)", "assist": "P(assist)", "either": "P(either)"}[args.sort]
    cols = ["FPL_Name", "TEAM", "POS", "VS", "start", "xG", "P(goal)", "xA", "P(assist)", "P(either)"]
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    fmt = {c: pct for c in ["start", "P(goal)", "P(assist)", "P(either)"]}
    fmt.update({"xG": "{:.2f}".format, "xA": "{:.2f}".format})
    print(f"GW{gw} scorer / assist predictor{f' — {args.team.upper()}' if args.team else ''}, sorted by {key} (uppercase VS = home):")
    print(table.sort_values(key, ascending=False).head(args.top)[cols].to_string(index=False, formatters=fmt))
    if args.team:
        tot = table[["xG", "xA"]].sum()
        print(f"\nTeam total: {tot['xG']:.2f} expected goals, {tot['xA']:.2f} expected FPL assists.")
        _print_style(pipe, args.team.upper())


def _print_style(pipe: Pipeline, team: str) -> None:
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
        for pos in ["DEF", "MID", "FWD"]:
            r, lg = frame.loc[(team, pos)], st.league.loc[pos]
            bits.append(f"{pos} {r[cols[0]]:.0%} of xG (league {lg['xg_share']:.0%}), {r[cols[1]]:.0%} of xA (league {lg['xa_share']:.0%})")
        print(f"  {label}: " + "; ".join(bits))


def cmd_style(args: argparse.Namespace) -> None:
    pipe = build_pipeline(refresh=True)
    if args.team:
        _print_style(pipe, args.team.upper())
        return
    st = pipe.style
    a = st.attack["xg_share_raw"].unstack()[["DEF", "MID", "FWD"]].add_prefix("xG% ")
    c = st.concede["xg_share_raw"].unstack()[["DEF", "MID", "FWD"]].add_prefix("conc% ")
    out = st.venue[["att_tilt_raw", "att_home", "def_tilt_raw", "def_home"]].join(a).join(c)
    print("Venue tilt (home/away ratio vs a typical side; att_home/def_home are the shrunk factors applied,")
    print("away is the reciprocal) and the share of xG each position takes / is conceded to. --team for detail.\n")
    print(out.to_string(float_format=lambda x: f"{x:.2f}"))


def cmd_backtest(args: argparse.Namespace) -> None:
    from . import backtest as bt

    table, skipped, ev = bt.run(gws=args.gw, n_boot=args.boot)
    for gw, why in sorted(skipped.items()):
        print(f"  ! GW{gw} skipped: {why}")
    if table.empty:
        raise SystemExit("nothing to score yet")
    used = table.drop_duplicates("gw").sort_values("gw")
    print("Replayed " + ", ".join(f"GW{r.gw} (snapshot {r.snapshot})" for r in used.itertuples())
          + f" — {int(ev['n'].iloc[0])} player-gameweeks in the relevant pool.")
    print("Each variant saw only what was known before that deadline. priors.yaml is off.\n")
    cols = ["mae", "bias", "rank", "top10", "start_brier", "goal_brier", "assist_brier"]
    print(ev[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nmae/brier: lower is better. rank (Spearman within position) and top10 (actual points of the")
    print("ten best-projected per position) are what matter for picking a team: higher is better.\n")

    print(f"Difference from 'port', 90% paired-bootstrap interval ({args.boot} draws):")
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
        print(bt.calibration(table, variant).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nPer-player table saved to {bt.RESULTS_PATH}")


def cmd_archive(args: argparse.Namespace) -> None:
    dest = archive.run(top_n=args.top)
    print(f"archive at {dest}")


def cmd_web(args: argparse.Namespace) -> None:
    from . import web

    web.serve(port=args.port, open_browser=not args.no_browser)


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

    pt = sub.add_parser("team", help="transfers, XI and captain for YOUR team (live data)")
    pt.add_argument("--entry", type=int, help="your FPL entry id (remembered in data/me.json)")
    pt.add_argument("--to-gw", type=int, help="end of the planning window (default: next GW + 4)")
    pt.add_argument("--bench-weight", type=float, default=0.1)
    pt.add_argument("--ft", type=int, help="override the free-transfer count")
    pt.add_argument("--bank", type=float, help="override the bank (£m)")
    pt.add_argument("--max-transfers", type=int, help="cap transfers (0 = just pick the XI)")
    pt.add_argument("--hit-cost", type=float, default=4.0, help="points per extra transfer")
    pt.add_argument(
        "--ft-value", type=float, default=1.5,
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

    psc = sub.add_parser("scorers", help="anytime goalscorer / assist probabilities for a gameweek")
    psc.add_argument("--gw", type=int, help="gameweek (default: the next one)")
    psc.add_argument("--team", help="one team, with its style profile, e.g. NEW")
    psc.add_argument("--top", type=int, default=25)
    psc.add_argument("--sort", choices=["goal", "assist", "either"], default="goal")
    psc.set_defaults(func=cmd_scorers)

    pst = sub.add_parser("style", help="team playing style: home/away tilt and positional channels")
    pst.add_argument("--team", help="one team in detail, e.g. NEW")
    pst.set_defaults(func=cmd_style)

    pb = sub.add_parser("backtest", help="replay past deadlines from the snapshots and score each model layer")
    pb.add_argument("--gw", type=int, action="append", help="only this gameweek (repeatable; default: all finished)")
    pb.add_argument("--boot", type=int, default=300, help="bootstrap draws for the intervals")
    pb.set_defaults(func=cmd_backtest)

    pf = sub.add_parser("form", help="schedule-adjusted team form the model is applying")
    pf.set_defaults(func=cmd_form)

    pw = sub.add_parser("web", help="local browser UI for predictions and squad solving")
    pw.add_argument("--port", type=int, default=8765)
    pw.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    pw.set_defaults(func=cmd_web)

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
