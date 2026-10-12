"""Anytime goalscorer / assist probabilities, derived from the PPts table."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
