"""The projection pipeline: inputs -> model table -> per-gameweek expected points.

`build_pipeline` is the one entry point every consumer shares — the CLI, the
web UI and the backtester — and `Pipeline` is what it returns: the PPts table
plus the intermediate frames the explainer reads. The small helpers after it
(`player_prices`, `resolve_codes`) are the pieces of the same data layer that
callers need next to the projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import api, inseason, matchlog, priors, reference
from .config import DEFAULT_PARAMS
from .fixtures import build_fixture_slots, opponent_grid
from .model_build import build_model
from .ppts import compute_ppts
from .style import DEFAULT_STYLE, TeamStyle, fixture_adjustments, team_style


def live_players(boot: dict | None = None) -> pd.DataFrame:
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


def player_prices(refresh: bool, boot: dict | None = None) -> pd.Series:
    """Player price by code — live if refreshing, else the workbook snapshot."""
    if refresh:
        live = live_players(boot)
        return live.loc[live["status"] != "u", "price"]
    ref = reference.load("API")
    codes = pd.to_numeric(ref["code"], errors="coerce")
    price = pd.to_numeric(ref["now_cost"], errors="coerce") / 10
    mask = codes.notna() & price.notna()
    return pd.Series(price[mask].values, index=codes[mask].astype(int).values)


def resolve_codes(names: list[str], players: pd.DataFrame) -> set[int]:
    codes: set[int] = set()
    for name in names:
        hits = players[players["FPL_Name"].str.contains(name, case=False, na=False)]
        if hits.empty:
            raise SystemExit(f"no player matching {name!r}")
        if len(hits) > 1:
            raise SystemExit(f"{name!r} is ambiguous — matches: {', '.join(hits['FPL_Name'])}")
        codes.add(int(hits.index[0]))
    return codes
