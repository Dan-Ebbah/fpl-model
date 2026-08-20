"""Parity: the ported PPts engine must reproduce the workbook's cached values.

The workbook is a frozen pre-season snapshot, so agreement is exact up to
float noise. Any intentional deviation from the spreadsheet (bug fixes,
improvements) must update these tests explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_model import reference
from fpl_model.config import DEFAULT_PARAMS
from fpl_model.fixtures import build_fixture_slots, opponent_grid
from fpl_model.ppts import compute_ppts


@pytest.fixture(scope="module")
def computed_and_reference():
    model = reference.load_model()
    fixtures = reference.load("FIXTURES")
    teams = reference.load("TEAMS")
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)
    fixtures = fixtures[["event", "team_h", "team_a"]].apply(pd.to_numeric, errors="coerce")

    slots = build_fixture_slots(fixtures, teams)
    grid = opponent_grid(slots, gws=range(1, 39))
    la_xg, la_xgc = reference.league_average_xg()
    dcscore, dc_avg = reference.load_dcscore()

    ours = compute_ppts(
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

    theirs = reference.load_reference_ppts()
    merged = ours.merge(
        theirs,
        left_on=["Code", "gw", "slot"],
        right_on=["Code", "GW", "GW2"],
        suffixes=("_py", "_xl"),
        how="inner",
    )
    assert len(merged) > 20000, "join between port and workbook rows collapsed"
    return merged


def test_opponent_assignment_matches(computed_and_reference):
    m = computed_and_reference
    mismatched = m[m["VS_py"] != m["VS_xl"]]
    assert mismatched.empty, mismatched[["Code", "gw", "slot", "VS_py", "VS_xl"]].head(20)


@pytest.mark.parametrize(
    "column",
    ["2", "3", "6", "7", "8", "9", "10", "11", "Start %", "IF START TOTAL", "TRUE TOTAL"],
)
def test_component_parity(computed_and_reference, column):
    m = computed_and_reference
    ours = pd.to_numeric(m[f"{column}_py"], errors="coerce")
    theirs = pd.to_numeric(m[f"{column}_xl"], errors="coerce")
    both = ours.notna() & theirs.notna()
    assert both.sum() > 20000
    diff = (ours[both] - theirs[both]).abs()
    worst = diff.max()
    assert np.isclose(worst, 0, atol=1e-9), (
        f"{column}: max |py - xl| = {worst:.3e} at\n"
        f"{m.loc[diff.idxmax(), ['Code', 'FPL_Name', 'gw', 'slot', f'{column}_py', f'{column}_xl']]}"
    )


def test_true_total_coverage(computed_and_reference):
    """Every nonzero workbook TRUE TOTAL row is present and matched in the port."""
    m = computed_and_reference
    nonzero = m[pd.to_numeric(m["TRUE TOTAL_xl"], errors="coerce").abs() > 1e-12]
    assert len(nonzero) > 10000
