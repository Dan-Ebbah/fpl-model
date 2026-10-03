"""Manual priors for players without Premier League history.

New signings from other leagues project near zero because every input the
model has for them is empty. This layer injects hand-entered priors from
data/priors.yaml: last-season per-90 attacking rates from the player's
previous league (times a league-strength discount) and a start probability.

Deliberately manual: the numbers come from a human reading FBRef, because the
biggest error term for a new signing is minutes, which is a team-news
judgment no foreign stat fixes. Automating the fetch and fitting the league
coefficients is future work gated on the backtester.

YAML entry fields:
    player         name substring, resolved against FPL_Name/web_name
    xg90, xa90     per-90 rates from the previous league (either optional)
    league_factor  strength discount applied to both rates (default 1.0)
    start_pct      start probability override (optional), 0..1
    team, note     documentation only (team is sanity-checked if present)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import PositionPoints

PRIORS_PATH = Path(__file__).resolve().parents[2] / "data" / "priors.yaml"


def load_priors(path: Path = PRIORS_PATH) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("players", []) or []


def apply_priors(
    model: pd.DataFrame,
    priors: list[dict],
    points: PositionPoints,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Overwrite the xG/xA category columns for matched players.

    Returns the modified model plus {code: start_pct} overrides to merge into
    the manual start-% series. A prior whose player isn't in the model (not
    registered, or running off the workbook snapshot) is skipped with a
    warning — priors describe live reality and must not crash a frozen run.
    """
    model = model.copy()
    start_overrides: dict[int, float] = {}
    for entry in priors:
        name = str(entry["player"])
        hits = model[
            model["FPL_Name"].str.contains(name, case=False, na=False)
            | model["web_name"].str.contains(name, case=False, na=False)
        ] if "web_name" in model.columns else model[
            model["FPL_Name"].str.contains(name, case=False, na=False)
        ]
        if hits.empty:
            print(f"priors: no player matching {name!r} — skipped", file=sys.stderr)
            continue
        if hits["Code"].nunique() > 1:
            raise ValueError(
                f"priors: {name!r} is ambiguous — matches {', '.join(hits['FPL_Name'].unique())}"
            )
        idx = hits.index
        row = hits.iloc[0]
        if entry.get("team") and row["TEAM"] != entry["team"]:
            print(
                f"priors: {name!r} is at {row['TEAM']}, prior says {entry['team']} — "
                "applying anyway, update the file",
                file=sys.stderr,
            )
        factor = float(entry.get("league_factor", 1.0))
        # category columns are per-start expectations: per-90 rate scaled by
        # the player's minutes-per-start share (0.66 default for no history)
        blend = float(row["G+A PSxTS Mn/Start"])
        if entry.get("xg90") is not None:
            model.loc[idx, "8"] = float(entry["xg90"]) * factor * blend * float(row["Goal"])
        if entry.get("xa90") is not None:
            model.loc[idx, "7"] = float(entry["xa90"]) * factor * blend * points.assist
        if entry.get("start_pct") is not None:
            start_overrides[int(row["Code"])] = float(entry["start_pct"])
    return model, start_overrides
