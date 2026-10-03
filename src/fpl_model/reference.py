"""Loaders for the workbook-extracted reference data in data/reference/.

These parquets are a frozen pre-season (2026-27 GW0) snapshot of the Excel
model's sheets — both its inputs and its computed outputs. They serve two
purposes: parity fixtures for tests, and the input data for running the
ported model before the live-API data layer takes over.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


def load(sheet: str) -> pd.DataFrame:
    return pd.read_parquet(REFERENCE_DIR / f"{sheet}.parquet")


def load_model() -> pd.DataFrame:
    """MODEL sheet rows with a real player behind them."""
    m = load("MODEL")
    m = m[pd.to_numeric(m["Code"], errors="coerce").notna()].copy()
    m["Code"] = m["Code"].astype(int)
    return m


def load_team_strength() -> pd.DataFrame:
    """TABLES sheet: the 20 team rows (rows below are other tables)."""
    t = load("TABLES")
    t = t[t["ABVR"].notna() & (t["ABVR"] != "ABVR")].head(20).copy()
    for c in [
        "LF xG/90", "SF xG/90", "LF xGC/90", "SF xGC/90",
        "LFxSF US xGC FIX", "LF US xGC FIX", "SF US xGC FIX",
    ]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def load_control_values() -> pd.Series:
    """CONTROL sheet cell values (cell coordinate -> value, as text)."""
    return load("CONTROL_values").set_index("cell")["value"]


def load_model_inputs() -> dict:
    """Everything build_model() needs, from the frozen workbook extracts.

    The API/teams entries are the workbook's snapshot — the live pipeline
    swaps those two for a fresh bootstrap pull and keeps the rest.
    """
    model = load_model()
    return {
        "api": load("API")[pd.to_numeric(load("API")["code"], errors="coerce").notna()],
        "teams": load("TEAMS"),
        "psapi": load("PSAPI"),
        "pt": load("PT"),
        "pspt": load("PSPT"),
        "pivots": {name: load(f"{name}_delta") for name in ["MINS", "xG", "xA", "DC"]},
        "team_strength": load_team_strength(),
        "control": load_control_values(),
        "name_map": model.drop_duplicates("Code").set_index("Code")[
            ["FBRef_Name", "Alt Name 2", "POS2"]
        ],
    }


def load_dcscore() -> tuple[pd.DataFrame, pd.Series]:
    dc = load("DCSCORE").set_index("ABVR")
    avg = load("DCSCORE_league_avg").iloc[0]
    return dc, avg


def league_average_xg() -> tuple[float, float]:
    """Previous-season league-average xG/90 and xGC/90 (PSTABLE bottom row)."""
    ps = load("PSTABLE")
    xg = pd.to_numeric(ps["xG/90"], errors="coerce").dropna().iloc[-1]
    xgc = pd.to_numeric(ps["xGC/90"], errors="coerce").dropna().iloc[-1]
    return float(xg), float(xgc)


def load_manual_start_pct() -> pd.Series:
    """PREDICT sheet manual start-% overrides, keyed by player code."""
    pr = load("PREDICT")
    codes = pd.to_numeric(pr["HARD CODE"], errors="coerce")
    pct = pd.to_numeric(pr["Manual Start %"], errors="coerce")
    mask = codes.notna() & pct.notna()
    return pd.Series(pct[mask].values, index=codes[mask].astype(int).values)


def load_reference_ppts() -> pd.DataFrame:
    """The workbook's own computed PPts rows (the parity target)."""
    pp = load("PPts")
    pp = pp[pd.to_numeric(pp["Code"], errors="coerce").notna()].copy()
    pp["Code"] = pp["Code"].astype(int)
    pp["GW"] = pp["GW"].astype(int)
    pp["GW2"] = pp["GW2"].astype(int)
    return pp
