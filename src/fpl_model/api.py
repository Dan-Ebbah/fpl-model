"""FPL API client and snapshot store.

The official API is unauthenticated. Snapshots are written to data/snapshots/
as dated parquet/JSON so the model's inputs are reproducible — prices,
ownership, and injury news change daily, and price-change prediction needs
the history.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from . import paths

BASE = "https://fantasy.premierleague.com/api"
SNAPSHOT_DIR = paths.SNAPSHOT_DIR
TIMEOUT_S = 30
# rate-limited or a server-side hiccup: worth a retry after a backoff
RETRY_STATUSES = (429, 500, 502, 503)


def _get(path: str, retries: int = 4) -> dict | list:
    # transient TLS/connection drops from the FPL API used to kill the weekly
    # launchd run outright, so retry connection-level failures too
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{path}", timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            continue
        if r.status_code in RETRY_STATUSES and attempt < retries - 1:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise AssertionError("unreachable")


def fetch_bootstrap() -> dict:
    """Players, teams, events, and game settings."""
    return _get("bootstrap-static/")


def fetch_fixtures() -> list[dict]:
    """Every fixture of the season: gameweek, home/away team ids, kick-off,
    finished flag and (once played) the match stats."""
    return _get("fixtures/")


def team_short_names(boot: dict) -> dict[int, str]:
    """Bootstrap team id -> upper-case short name (the model's ABVR key)."""
    return {int(t["id"]): str(t["short_name"]).upper() for t in boot["teams"]}


def fetch_player_history(player_id: int) -> dict:
    """Per-gameweek history and remaining fixtures for one player."""
    return _get(f"element-summary/{player_id}/")


# --- manager (entry) endpoints: public, no login needed ---------------------
# What is NOT public: the pre-deadline team (my-team/) and purchase prices.
# Both are reconstructed in team.py from the picks + transfers history.


def fetch_entry(entry_id: int) -> dict:
    """Manager summary: name, bank/value at the last deadline, overall rank."""
    return _get(f"entry/{entry_id}/")


def fetch_entry_picks(entry_id: int, gw: int) -> dict | None:
    """The 15 picked for a gameweek (visible once its deadline has passed);
    None if the entry had no team that week."""
    try:
        return _get(f"entry/{entry_id}/event/{gw}/picks/")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def fetch_entry_history(entry_id: int) -> dict:
    """Per-gameweek totals, chips played, past seasons."""
    return _get(f"entry/{entry_id}/history/")


def fetch_entry_transfers(entry_id: int) -> list[dict]:
    """Every transfer made this season (element in/out and the prices paid)."""
    return _get(f"entry/{entry_id}/transfers/")


def snapshot(
    out_dir: Path | None = None,
    bootstrap: dict | None = None,
    fixtures: list[dict] | None = None,
) -> Path:
    """Store bootstrap + fixtures under today's date (fetching them if not given).

    Every fresh pull should land here: the snapshot series is the raw
    material for the in-season data layer (per-GW deltas of the cumulative
    season totals, the same construction as the workbook's pivot sheets).
    A second snapshot on the same day overwrites the first — latest wins.
    """
    out_dir = out_dir or SNAPSHOT_DIR
    stamp = dt.date.today().isoformat()
    dest = out_dir / stamp

    # fetch before creating the folder: a failed pull used to leave an empty
    # dated directory behind, which reads as "a snapshot exists" to anything
    # listing the series (the backtester)
    bootstrap = bootstrap if bootstrap is not None else fetch_bootstrap()
    fixtures = fixtures if fixtures is not None else fetch_fixtures()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bootstrap.json").write_text(json.dumps(bootstrap))
    pd.DataFrame(bootstrap["elements"]).to_parquet(dest / "players.parquet")
    pd.DataFrame(bootstrap["teams"]).to_parquet(dest / "teams.parquet")
    pd.DataFrame(fixtures).to_parquet(dest / "fixtures.parquet")
    return dest



if __name__ == "__main__":
    print(f"snapshot written to {snapshot()}")
