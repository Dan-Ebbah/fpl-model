"""Parity: build_model() vs the workbook's own MODEL sheet.

Feeds the builder the workbook's extracted inputs (its API snapshot, PSAPI,
FBRef tables, pivot deltas, CONTROL values) and requires every ported column
to match the sheet's cached values at 1e-9 for all real player rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_model import reference
from fpl_model.config import DEFAULT_PARAMS
from fpl_model.model_build import build_model

NUMERIC_COLS = [
    "PS Starts", "PS Subs", "PS UnSub", "PS Mn/St", "PS Mn/Sub",
    "PS Mn/St %", "PS Mn/Sub %", "PS Sts/SqdsMd", "PS UnSub/SqdsMd",
    "Starts", "Subs", "UnSub", "Mn/St", "Mn/Sub", "Mn/St %", "Mn/Sub %",
    "St/SqdsMd", "UnSub/SqdsMd", "PS Mins", "Mins",
    "PS Saves/90", "Saves/90", "PS YC/St", "YC/90", "PS RC/90", "RC/90",
    "PS BPS/Start", "PS Bonus/Start", "BPS/Start", "Bonus/Start",
    "PS A/St", "A/St", "PS xA/St", "xA/St", "Chance of Playing",
    "G+A PSxTS Mn/Start", "G+A PSxTS Mn/Sub",
    "LF xA/St", "LF xG/St", "SF xA/St", "SF xG/St",
    "LF xGC/90", "SF xGC/90", "LF % CS/St", "SF % CS/90",
    "LF % 2+ GC", "SF % 2+ GC", "LF DC/St", "SF DC/St", "LF DC %", "SF DC %",
    "PS %1", "%1", "PS %2", "%2", "Goal",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "TOTAL",
]
STRING_COLS = ["FPL_Name", "TEAM", "POS", "POS2"]


@pytest.fixture(scope="module")
def built_and_ref() -> tuple[pd.DataFrame, pd.DataFrame]:
    inp = reference.load_model_inputs()
    built = build_model(points=DEFAULT_PARAMS.points, **inp)
    ref = reference.load_model()
    b = built.drop_duplicates("Code").set_index("Code")
    r = ref.drop_duplicates("Code").set_index("Code")
    common = b.index.intersection(r.index)
    # every live API row must have a workbook row to compare against
    assert len(common) == len(b)
    return b.loc[common], r.loc[common]


@pytest.mark.parametrize("col", NUMERIC_COLS)
def test_numeric_parity(built_and_ref, col):
    b, r = built_and_ref
    bv = pd.to_numeric(b[col], errors="coerce")
    rv = pd.to_numeric(r[col], errors="coerce")
    assert (bv.isna() == rv.isna()).all(), f"{col}: NaN pattern differs"
    diff = (bv - rv).abs().max()
    assert diff <= 1e-9, f"{col}: max abs diff {diff}"


@pytest.mark.parametrize("col", STRING_COLS)
def test_string_parity(built_and_ref, col):
    b, r = built_and_ref
    mismatch = (b[col].astype(str) != r[col].astype(str)).sum()
    assert mismatch == 0, f"{col}: {mismatch} rows differ"
