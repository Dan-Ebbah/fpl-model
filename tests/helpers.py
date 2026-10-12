"""Builders shared by the in-season and style tests (synthetic inputs)."""

from __future__ import annotations

import pandas as pd

LA = 1.5  # league-average xG (and xGC) per 90 in the synthetic leagues


def strength_table(att: dict[str, float], dfn: dict[str, float]) -> pd.DataFrame:
    """A team-strength table with every xG/xGC column at the given per-team
    attack and defence values (long- and short-form alike)."""
    rows = []
    for t in att:
        rows.append({"ABVR": t, "LF xG/90": att[t], "SF xG/90": att[t],
                     "LF xGC/90": dfn[t], "SF xGC/90": dfn[t], "LFxSF US xGC FIX": dfn[t],
                     "LF US xGC FIX": dfn[t], "SF US xGC FIX": dfn[t]})
    return pd.DataFrame(rows)
