"""Aggregate per-gameweek projections over a planning window — the PREDICT sheet.

This is the frame the squad optimizer consumes: one row per player with
expected points summed over the chosen gameweeks, plus price and value.
"""

from __future__ import annotations

import pandas as pd


def aggregate_window(ppts: pd.DataFrame, gw_from: int, gw_to: int) -> pd.DataFrame:
    """Sum projections over [gw_from, gw_to] inclusive, one row per player."""
    window = ppts[(ppts["gw"] >= gw_from) & (ppts["gw"] <= gw_to)]
    has_fixture = window["VS"].str.len() >= 3
    agg = (
        window.assign(games=has_fixture.astype(int))
        .groupby(["Code", "FPL_Name", "TEAM", "POS"], as_index=False)
        .agg(
            games=("games", "sum"),
            if_start_pts=("IF START TOTAL", "sum"),
            true_pts=("TRUE TOTAL", "sum"),
        )
    )
    return agg.sort_values("true_pts", ascending=False, ignore_index=True)


def with_value(predictions: pd.DataFrame, players_api: pd.DataFrame) -> pd.DataFrame:
    """Attach price and ownership from the FPL players table; value = pts/£/game."""
    api = players_api[["code", "now_cost", "selected_by_percent"]].copy()
    api["code"] = pd.to_numeric(api["code"], errors="coerce")
    api["price"] = pd.to_numeric(api["now_cost"], errors="coerce") / 10
    out = predictions.merge(
        api[["code", "price", "selected_by_percent"]],
        left_on="Code", right_on="code", how="left",
    ).drop(columns="code")
    out["value"] = (out["true_pts"] / out["price"] / out["games"]).where(
        (out["price"] > 0) & (out["games"] > 0), 0.0
    )
    return out
