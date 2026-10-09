"""Parity: window aggregation must match the workbook's PREDICT sheet (GW1-10)."""

import numpy as np
import pandas as pd

from fpl_model import reference
from fpl_model.predict import aggregate_window


def test_true_pts_matches_predict_sheet(frozen_ppts):
    ours = aggregate_window(frozen_ppts, gw_from=1, gw_to=10)

    theirs = reference.load("PREDICT")
    theirs = theirs[pd.to_numeric(theirs["HARD CODE"], errors="coerce").notna()].copy()
    theirs["Code"] = theirs["HARD CODE"].astype(float).astype(int)
    theirs["TRUE PTS"] = pd.to_numeric(theirs["TRUE PTS"], errors="coerce")

    merged = ours.merge(theirs[["Code", "TRUE PTS"]], on="Code", how="inner")
    both = merged["true_pts"].notna() & merged["TRUE PTS"].notna()
    assert both.sum() > 400
    diff = (merged.loc[both, "true_pts"] - merged.loc[both, "TRUE PTS"]).abs()
    assert np.isclose(diff.max(), 0, atol=1e-9), (
        f"max diff {diff.max():.3e}:\n{merged.loc[diff.idxmax()]}"
    )
