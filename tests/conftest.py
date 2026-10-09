"""Shared fixtures: the frozen workbook PPts build the parity suites compare against."""

import pandas as pd
import pytest

from fpl_model import reference
from fpl_model.config import DEFAULT_PARAMS
from fpl_model.fixtures import build_fixture_slots, opponent_grid
from fpl_model.ppts import compute_ppts


@pytest.fixture(scope="session")
def frozen_ppts() -> pd.DataFrame:
    """compute_ppts over the workbook's own inputs: GW1-38, default params, no priors or in-season layer."""
    model = reference.load_model()
    fixtures = reference.load("FIXTURES")
    teams = reference.load("TEAMS")
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)
    fixtures = fixtures[["event", "team_h", "team_a"]].apply(pd.to_numeric, errors="coerce")

    slots = build_fixture_slots(fixtures, teams)
    grid = opponent_grid(slots, gws=range(1, 39))
    la_xg, la_xgc = reference.league_average_xg()
    dcscore, dc_avg = reference.load_dcscore()

    return compute_ppts(
        model=model,
        team_strength=reference.load_team_strength(),
        dcscore=dcscore,
        dc_league_avg=dc_avg,
        fixture_grid=grid,
        league_avg_xg=la_xg,
        league_avg_xgc=la_xgc,
        params=DEFAULT_PARAMS,
        manual_start_pct=reference.load_manual_start_pct(),
    )
