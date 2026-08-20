"""Parity: window aggregation must match the workbook's PREDICT sheet (GW1-10)."""

import numpy as np
import pandas as pd
import pytest

from fpl_model import reference
from fpl_model.config import DEFAULT_PARAMS
from fpl_model.fixtures import build_fixture_slots, opponent_grid
from fpl_model.ppts import compute_ppts
from fpl_model.predict import aggregate_window


@pytest.fixture(scope="module")
def ppts():
    model = reference.load_model()
    fixtures = reference.load("FIXTURES")
    teams = reference.load("TEAMS")
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)
    fixtures = fixtures[["event", "team_h", "team_a"]].apply(pd.to_numeric, errors="coerce")
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


def test_true_pts_matches_predict_sheet(ppts):
    ours = aggregate_window(ppts, gw_from=1, gw_to=10)

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
