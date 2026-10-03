"""Per-player, per-gameweek expected points — port of the workbook's PPts sheet.

One row per (player, gameweek, fixture slot). Each FPL scoring category is an
expected-points column, numbered as the workbook numbers them:

    1  appearance (start)      5  red cards          9  clean sheet
    2  playing 60+ mins        6  bonus             10  2+ goals conceded
    3  saves (GK)              7  assists           11  defensive contribution
    4  yellow cards            8  goals

Player rates come from the MODEL aggregates; opponent strength scales the
goal/assist/CS/save terms via long/short-form team xG blends relative to the
league average. The final TRUE TOTAL marginalizes over start probability:
start points plus non-start (sub) points weighted by Start %.

Faithful-port notes (kept bug-for-bug until the backtester can judge changes):
- the home/away multiplier is applied twice on the sub-appearance path
  (inside IF NOT START TOTAL and again on TRUE TOTAL), as in the sheet;
- the opponent DC adjustment (DC/90 POS2 X) is computed but unused, as in
  the sheet;
- team-code lookups are case-insensitive (Excel MATCH semantics) — the
  lowercase away codes hit the uppercase table keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ControlParams
from .fixtures import NO_FIXTURE

# MODEL-sheet columns consumed, by workbook header
MODEL_COLS = [
    "Code", "FPL_Name", "TEAM", "POS", "POS2",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "Goal", "G+A PSxTS Mn/Start", "G+A PSxTS Mn/Sub",
    "Chance of Playing", "PS Sts/SqdsMd", "St/SqdsMd",
]

_NUMERIC_MODEL_COLS = MODEL_COLS[5:]


def _pos_map(values: dict[str, float], pos: pd.Series) -> pd.Series:
    return pos.map(values).fillna(0.0)


def compute_ppts(
    model: pd.DataFrame,
    team_strength: pd.DataFrame,
    dcscore: pd.DataFrame,
    dc_league_avg: pd.Series,
    fixture_grid: pd.DataFrame,
    league_avg_xg: float,
    league_avg_xgc: float,
    params: ControlParams,
    manual_start_pct: pd.Series | None = None,
    fixture_adj: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute expected points per (player, gw, slot).

    `team_strength` needs columns ABVR, LF xG/90, SF xG/90, LF xGC/90,
    SF xGC/90, LFxSF US xGC FIX. `dcscore` is indexed by team ABVR with one
    column per position group (CB..CF); `dc_league_avg` is the league-average
    row of the same. `manual_start_pct` maps player code -> override.

    `fixture_adj` is the one deliberate extension point beyond the sheet:
    per (TEAM, vs_key, is_home, POS) multipliers `goal_mult`, `assist_mult`,
    `conceded_mult` from style.fixture_adjustments (team venue tilt and
    positional concession channels). None — the default, and what the parity
    tests run — leaves every number exactly as the workbook computes it.
    """
    m = model[MODEL_COLS].copy()
    for c in _NUMERIC_MODEL_COLS:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    df = m.merge(fixture_grid, left_on="TEAM", right_on="team", how="inner")

    vs = df["vs"].astype(str)
    vs_key = vs.str.upper()  # Excel MATCH is case-insensitive
    has_fixture = vs != NO_FIXTURE

    ts = team_strength.set_index("ABVR")
    w_lf, w_sf = params.fdr_long_form_weight, params.fdr_short_form_weight
    vs_xg = (
        w_lf * vs_key.map(ts["LF xG/90"]) + w_sf * vs_key.map(ts["SF xG/90"])
    ).fillna(0.0)
    vs_xgc = (
        w_lf * vs_key.map(ts["LF xGC/90"]) + w_sf * vs_key.map(ts["SF xGC/90"])
    ).fillna(0.0)

    # opponent-strength multipliers relative to league average:
    # attack_mult > 1 means the opponent's attack is weaker than average
    attack_mult = 1 + (1 - vs_xg / league_avg_xg)      # "VS xG/90/LA"
    defence_mult = vs_xgc / league_avg_xgc             # "VS xGC/90/LA"

    goal_adj = assist_adj = conceded_adj = 1.0
    if fixture_adj is not None:
        key = pd.DataFrame({"TEAM": df["TEAM"].str.upper(), "vs_key": vs_key, "is_home": vs == vs_key, "POS": df["POS"]})
        adj = key.merge(fixture_adj, on=["TEAM", "vs_key", "is_home", "POS"], how="left")
        goal_adj = adj["goal_mult"].fillna(1.0).set_axis(df.index)
        assist_adj = adj["assist_mult"].fillna(1.0).set_axis(df.index)
        conceded_adj = adj["conceded_mult"].fillna(1.0).set_axis(df.index)

    is_gk_def = df["POS"].isin(["GK", "DEF"])
    pts = params.points

    c2 = df["2"]
    c3 = df["3"] * (vs_xg / league_avg_xg)
    c4 = df["4"]
    c5 = df["5"]
    c6 = df["6"] * np.where(is_gk_def, attack_mult, defence_mult)
    xa_per_start = df["7"] / pts.assist
    c7 = xa_per_start * defence_mult * assist_adj * (1 + params.fantasy_assist_boost) * pts.assist
    xg_per_start = (df["8"] / df["Goal"]).replace([np.inf, -np.inf], 0).fillna(0.0)
    c8 = xg_per_start * defence_mult * goal_adj * _pos_map(pts.goal, df["POS"])

    own_xgc = df["TEAM"].map(ts["LFxSF US xGC FIX"])
    cs_lambda = (own_xgc * vs_xg / league_avg_xg * conceded_adj).fillna(0.0)  # expected goals conceded
    cs_pct = np.exp(-cs_lambda)                                 # Poisson P(0 conceded)
    c9 = _pos_map(pts.clean_sheet, df["POS"]) * cs_pct
    pct_2plus_gc = 1 - cs_pct * (1 + cs_lambda)                 # Poisson P(>=2 conceded)
    c10 = _pos_map(pts.two_plus_conceded, df["POS"]) * pct_2plus_gc

    dc_per_start = df["11"] / 2
    c11 = _pos_map(pts.defensive_contribution, df["POS"]) * dc_per_start

    home_away = np.where(vs == vs_key, params.home_multiplier, params.away_multiplier)

    category_sum = c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11
    if_start_total = np.where(
        has_fixture, (has_fixture.astype(float) + category_sum) * home_away, 0.0
    )

    starts_per_squad = (
        params.starts_rate_prev_season_weight * df["PS Sts/SqdsMd"]
        + params.starts_rate_this_season_weight * df["St/SqdsMd"]
    )
    manual = (
        df["Code"].map(manual_start_pct)
        if manual_start_pct is not None
        else pd.Series(np.nan, index=df.index)
    )
    start_pct = manual.fillna(df["Chance of Playing"] * starts_per_squad)
    # appearance expectation: a manual start override guarantees the appearance
    appearance = np.where(manual.notna(), manual + (1 - manual) * df["1"], df["1"])

    sub_minutes_ratio = df["G+A PSxTS Mn/Sub"] / df["G+A PSxTS Mn/Start"]
    if_not_start_total = (c4 + c5 + c6 + c7 + c8 + c10) * sub_minutes_ratio * home_away

    true_total = np.where(
        has_fixture,
        (appearance + category_sum * start_pct + if_not_start_total * (1 - start_pct))
        * home_away,
        0.0,
    )

    # expected goals / FPL assists in the fixture, marginalised over minutes —
    # not in the sheet, and no points flow through them: the scorer predictor
    # reads these. Unlike the sheet's sub path, a non-starter only contributes
    # if he actually comes on.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_sub = ((appearance - start_pct) / (1 - start_pct)).clip(0, 1).fillna(0.0)
    minutes_factor = np.where(has_fixture, start_pct + (1 - start_pct) * p_sub * sub_minutes_ratio.fillna(0.0), 0.0)
    exp_goals = (c8 / _pos_map(pts.goal, df["POS"]).replace(0, np.nan)).fillna(0.0) * minutes_factor
    exp_assists = c7 / pts.assist * minutes_factor

    out = df[["Code", "FPL_Name", "TEAM", "POS", "POS2", "gw", "slot"]].copy()
    out["VS"] = vs
    out["H/A"] = home_away
    for name, col in [
        ("2", c2), ("3", c3), ("4", c4), ("5", c5), ("6", c6), ("7", c7),
        ("8", c8), ("9", c9), ("10", c10), ("11", c11),
        ("CS%", cs_pct), ("% 2+ GC", pct_2plus_gc),
        ("Start %", start_pct), ("IF NOT START TOTAL", if_not_start_total),
        ("exp_goals", exp_goals), ("exp_assists", exp_assists),
    ]:
        out[name] = col
    out["IF START TOTAL"] = if_start_total
    out["TRUE TOTAL"] = true_total
    return out
