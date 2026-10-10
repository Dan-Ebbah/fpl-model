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
- the sheet computes an opponent DC adjustment (DC/90 POS2 X) but never uses
  it, so the port does not compute it at all;
- team-code lookups are case-insensitive (Excel MATCH semantics) — the
  lowercase away codes hit the uppercase table keys;
- home detection is the sheet's `vs == UPPER(vs)`, which also fires on the
  blank "-" slots (harmless: their totals are forced to 0).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .config import ControlParams, PositionPoints
from .fixtures import NO_FIXTURE

# MODEL-sheet columns consumed, by workbook header
MODEL_COLS = [
    "Code", "FPL_Name", "TEAM", "POS", "POS2",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "Goal", "G+A PSxTS Mn/Start", "G+A PSxTS Mn/Sub",
    "Chance of Playing", "PS Sts/SqdsMd", "St/SqdsMd",
]

_NUMERIC_MODEL_COLS = MODEL_COLS[5:]

# per-category expected points, keyed by output column
_CATEGORY_COLS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]


def _pos_map(values: dict[str, float], pos: pd.Series) -> pd.Series:
    """Position -> points table applied to a POS column; unknown positions score 0."""
    return pos.map(values).fillna(0.0)


class _Opponent(NamedTuple):
    """Opponent strength per fixture row."""

    xg: pd.Series  # blended opponent xG/90 ("VS xG/90")
    attack_mult: pd.Series  # "VS xG/90/LA": > 1 means the opponent's attack is weaker than average
    defence_mult: pd.Series  # "VS xGC/90/LA": > 1 means the opponent concedes more than average


class _FixtureAdjust(NamedTuple):
    """Style-layer multipliers (1.0 everywhere when the layer is off)."""

    goal: pd.Series | float
    assist: pd.Series | float
    conceded: pd.Series | float


def _opponent_strength(
    vs_key: pd.Series,
    ts: pd.DataFrame,
    league_avg_xg: float,
    league_avg_xgc: float,
    params: ControlParams,
) -> _Opponent:
    """Long/short-form blend of the opponent's xG and xGC per 90, relative to the league."""
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
    return _Opponent(vs_xg, attack_mult, defence_mult)


def _fixture_adjustments(
    df: pd.DataFrame,
    vs_key: pd.Series,
    is_home: pd.Series,
    fixture_adj: pd.DataFrame | None,
) -> _FixtureAdjust:
    """Join the style layer's (TEAM, vs_key, is_home, POS) multipliers; 1.0 when absent."""
    if fixture_adj is None:
        return _FixtureAdjust(1.0, 1.0, 1.0)
    key = pd.DataFrame({"TEAM": df["TEAM"].str.upper(), "vs_key": vs_key, "is_home": is_home, "POS": df["POS"]})
    adj = key.merge(fixture_adj, on=["TEAM", "vs_key", "is_home", "POS"], how="left")
    return _FixtureAdjust(
        adj["goal_mult"].fillna(1.0).set_axis(df.index),
        adj["assist_mult"].fillna(1.0).set_axis(df.index),
        adj["conceded_mult"].fillna(1.0).set_axis(df.index),
    )


def _category_points(
    df: pd.DataFrame,
    ts: pd.DataFrame,
    pts: PositionPoints,
    params: ControlParams,
    opp: _Opponent,
    adj: _FixtureAdjust,
    goal_pts: pd.Series,
    league_avg_xg: float,
) -> dict[str, pd.Series]:
    """Per-start expected points for categories 2..11, plus CS% and % 2+ GC."""
    is_gk_def = df["POS"].isin(["GK", "DEF"])

    c2 = df["2"]
    c3 = df["3"] * (opp.xg / league_avg_xg)
    c4 = df["4"]
    c5 = df["5"]
    c6 = df["6"] * np.where(is_gk_def, opp.attack_mult, opp.defence_mult)
    xa_per_start = df["7"] / pts.assist
    c7 = xa_per_start * opp.defence_mult * adj.assist * (1 + params.fantasy_assist_boost) * pts.assist
    xg_per_start = (df["8"] / df["Goal"]).replace([np.inf, -np.inf], 0).fillna(0.0)
    c8 = xg_per_start * opp.defence_mult * adj.goal * goal_pts

    own_xgc = df["TEAM"].map(ts["LFxSF US xGC FIX"])
    cs_lambda = (own_xgc * opp.xg / league_avg_xg * adj.conceded).fillna(0.0)  # expected goals conceded
    cs_pct = np.exp(-cs_lambda)                                 # Poisson P(0 conceded)
    c9 = _pos_map(pts.clean_sheet, df["POS"]) * cs_pct
    pct_2plus_gc = 1 - cs_pct * (1 + cs_lambda)                 # Poisson P(>=2 conceded)
    c10 = _pos_map(pts.two_plus_conceded, df["POS"]) * pct_2plus_gc

    dc_per_start = df["11"] / 2
    c11 = _pos_map(pts.defensive_contribution, df["POS"]) * dc_per_start

    return {
        "2": c2, "3": c3, "4": c4, "5": c5, "6": c6, "7": c7,
        "8": c8, "9": c9, "10": c10, "11": c11,
        "CS%": cs_pct, "% 2+ GC": pct_2plus_gc,
    }


def _start_probability(
    df: pd.DataFrame,
    manual_start_pct: pd.Series | None,
    params: ControlParams,
) -> tuple[pd.Series, np.ndarray]:
    """Start % (manual override wins) and the appearance expectation it implies."""
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
    return start_pct, appearance


def _scorer_expectations(
    c7: pd.Series,
    c8: pd.Series,
    appearance: np.ndarray,
    start_pct: pd.Series,
    sub_minutes_ratio: pd.Series,
    has_fixture: pd.Series,
    goal_pts: pd.Series,
    assist_pts: float,
) -> tuple[pd.Series, pd.Series]:
    """Expected goals / FPL assists in the fixture, marginalised over minutes.

    Not in the sheet, and no points flow through them: the scorer predictor
    reads these. Unlike the sheet's sub path, a non-starter only contributes
    if he actually comes on.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p_sub = ((appearance - start_pct) / (1 - start_pct)).clip(0, 1).fillna(0.0)
    minutes_factor = np.where(has_fixture, start_pct + (1 - start_pct) * p_sub * sub_minutes_ratio.fillna(0.0), 0.0)
    exp_goals = (c8 / goal_pts.replace(0, np.nan)).fillna(0.0) * minutes_factor
    exp_assists = c7 / assist_pts * minutes_factor
    return exp_goals, exp_assists


def _assemble_output(
    df: pd.DataFrame,
    vs: pd.Series,
    home_away: np.ndarray,
    cat: dict[str, pd.Series],
    start_pct: pd.Series,
    if_not_start_total: pd.Series,
    exp_goals: pd.Series,
    exp_assists: pd.Series,
    if_start_total: np.ndarray,
    true_total: np.ndarray,
) -> pd.DataFrame:
    """The PPts frame in the sheet's column order."""
    out = df[["Code", "FPL_Name", "TEAM", "POS", "POS2", "gw", "slot"]].copy()
    out["VS"] = vs
    out["H/A"] = home_away
    for name in _CATEGORY_COLS + ["CS%", "% 2+ GC"]:
        out[name] = cat[name]
    for name, col in [
        ("Start %", start_pct), ("IF NOT START TOTAL", if_not_start_total),
        ("exp_goals", exp_goals), ("exp_assists", exp_assists),
    ]:
        out[name] = col
    out["IF START TOTAL"] = if_start_total
    out["TRUE TOTAL"] = true_total
    return out


def compute_ppts(
    model: pd.DataFrame,
    team_strength: pd.DataFrame,
    fixture_grid: pd.DataFrame,
    league_avg_xg: float,
    league_avg_xgc: float,
    params: ControlParams,
    manual_start_pct: pd.Series | None = None,
    fixture_adj: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute expected points per (player, gw, slot).

    `team_strength` needs columns ABVR, LF xG/90, SF xG/90, LF xGC/90,
    SF xGC/90, LFxSF US xGC FIX. `manual_start_pct` maps player code -> override.

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
    is_home = vs == vs_key  # the sheet's venue test (not the grid's is_home; blank slots read as home)

    ts = team_strength.set_index("ABVR")
    opp = _opponent_strength(vs_key, ts, league_avg_xg, league_avg_xgc, params)
    adj = _fixture_adjustments(df, vs_key, is_home, fixture_adj)

    pts = params.points
    goal_pts = _pos_map(pts.goal, df["POS"])
    cat = _category_points(df, ts, pts, params, opp, adj, goal_pts, league_avg_xg)

    home_away = np.where(is_home, params.home_multiplier, params.away_multiplier)

    category_sum = (
        cat["2"] + cat["3"] + cat["4"] + cat["5"] + cat["6"]
        + cat["7"] + cat["8"] + cat["9"] + cat["10"] + cat["11"]
    )
    if_start_total = np.where(
        has_fixture, (has_fixture.astype(float) + category_sum) * home_away, 0.0
    )

    start_pct, appearance = _start_probability(df, manual_start_pct, params)

    sub_minutes_ratio = df["G+A PSxTS Mn/Sub"] / df["G+A PSxTS Mn/Start"]
    if_not_start_total = (
        cat["4"] + cat["5"] + cat["6"] + cat["7"] + cat["8"] + cat["10"]
    ) * sub_minutes_ratio * home_away

    true_total = np.where(
        has_fixture,
        (appearance + category_sum * start_pct + if_not_start_total * (1 - start_pct))
        * home_away,
        0.0,
    )

    exp_goals, exp_assists = _scorer_expectations(
        cat["7"], cat["8"], appearance, start_pct, sub_minutes_ratio, has_fixture, goal_pts, pts.assist
    )

    return _assemble_output(
        df, vs, home_away, cat, start_pct, if_not_start_total,
        exp_goals, exp_assists, if_start_total, true_total,
    )
