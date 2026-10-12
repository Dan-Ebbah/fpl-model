"""Parity: the ported PPts engine must reproduce the workbook's cached values.

The workbook is a frozen pre-season snapshot, so agreement is exact up to
float noise. Any intentional deviation from the spreadsheet (bug fixes,
improvements) must update these tests explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_model import reference


@pytest.fixture(scope="module")
def computed_and_reference(frozen_ppts):
    theirs = reference.load_reference_ppts()
    merged = frozen_ppts.merge(
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
