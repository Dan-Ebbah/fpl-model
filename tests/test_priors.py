"""Priors layer: injecting hand-entered rates for players without PL history."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_model.config import PositionPoints
from fpl_model.priors import apply_priors, load_priors


@pytest.fixture
def model() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Code": [1, 2],
            "FPL_Name": ["Christos Tzolis", "Old Hand"],
            "web_name": ["Tzolis", "Hand"],
            "TEAM": ["ARS", "CHE"],
            "POS": ["MID", "MID"],
            "Goal": [5.0, 5.0],
            "G+A PSxTS Mn/Start": [0.66, 0.95],
            "7": [0.0, 0.9],
            "8": [0.0, 1.2],
        }
    )


def test_apply_prior_sets_categories_and_start(model):
    priors = [
        {"player": "Tzolis", "team": "ARS", "xg90": 0.5, "xa90": 0.4,
         "league_factor": 0.6, "start_pct": 0.5}
    ]
    out, starts = apply_priors(model, priors, PositionPoints())
    row = out[out["Code"] == 1].iloc[0]
    assert row["8"] == pytest.approx(0.5 * 0.6 * 0.66 * 5.0)
    assert row["7"] == pytest.approx(0.4 * 0.6 * 0.66 * 3.0)
    assert starts == {1: 0.5}
    # untouched player keeps his numbers
    assert out.loc[out["Code"] == 2, "8"].iloc[0] == 1.2


def test_missing_player_skips_ambiguous_raises(model, capsys):
    out, starts = apply_priors(model, [{"player": "Nobody", "xg90": 1.0}], PositionPoints())
    assert starts == {} and out["8"].tolist() == [0.0, 1.2]
    assert "skipped" in capsys.readouterr().err
    with pytest.raises(ValueError, match="ambiguous"):
        apply_priors(model, [{"player": "o", "xg90": 1.0}], PositionPoints())


def test_partial_entry_only_touches_given_fields(model):
    out, starts = apply_priors(model, [{"player": "Tzolis", "xa90": 0.3}], PositionPoints())
    row = out[out["Code"] == 1].iloc[0]
    assert row["8"] == 0.0                       # xg90 not given -> untouched
    assert row["7"] == pytest.approx(0.3 * 0.66 * 3.0)
    assert starts == {}


def test_repo_priors_file_parses():
    for entry in load_priors():
        assert "player" in entry
        for k in ["xg90", "xa90", "league_factor", "start_pct"]:
            if entry.get(k) is not None:
                float(entry[k])
