"""FPL API client and snapshot store.

The official API is unauthenticated. Snapshots are written to data/snapshots/
as dated parquet/JSON so the model's inputs are reproducible — prices,
ownership, and injury news change daily, and price-change prediction needs
the history.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api"
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"


def _get(path: str) -> dict | list:
    r = requests.get(f"{BASE}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_bootstrap() -> dict:
    """Players, teams, events, and game settings."""
    return _get("bootstrap-static/")


def fetch_fixtures() -> list[dict]:
    return _get("fixtures/")


def fetch_player_history(player_id: int) -> dict:
    """Per-gameweek history and remaining fixtures for one player."""
    return _get(f"element-summary/{player_id}/")


def snapshot(out_dir: Path | None = None) -> Path:
    """Pull the current bootstrap + fixtures and store them under today's date."""
    out_dir = out_dir or SNAPSHOT_DIR
    stamp = dt.date.today().isoformat()
    dest = out_dir / stamp
    dest.mkdir(parents=True, exist_ok=True)

    bootstrap = fetch_bootstrap()
    (dest / "bootstrap.json").write_text(json.dumps(bootstrap))
    pd.DataFrame(bootstrap["elements"]).to_parquet(dest / "players.parquet")
    pd.DataFrame(bootstrap["teams"]).to_parquet(dest / "teams.parquet")
    pd.DataFrame(fetch_fixtures()).to_parquet(dest / "fixtures.parquet")
    return dest


def load_snapshot(date: str | None = None, snap_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load a stored snapshot (latest by default) as dataframes."""
    snap_dir = snap_dir or SNAPSHOT_DIR
    if date is None:
        dates = sorted(p.name for p in snap_dir.iterdir() if p.is_dir())
        if not dates:
            raise FileNotFoundError(f"no snapshots in {snap_dir}; run snapshot() first")
        date = dates[-1]
    dest = snap_dir / date
    return {
        "players": pd.read_parquet(dest / "players.parquet"),
        "teams": pd.read_parquet(dest / "teams.parquet"),
        "fixtures": pd.read_parquet(dest / "fixtures.parquet"),
    }


if __name__ == "__main__":
    print(f"snapshot written to {snapshot()}")
