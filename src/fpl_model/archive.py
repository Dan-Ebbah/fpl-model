"""Elite-manager season archiver.

The FPL API wipes all manager picks/transfers/chips at the July rollover
(verified 2026-08-20: every entry's prior-season picks endpoint 404s), so a
winner's season can only be studied if it was captured while the season ran.
This module archives the overall-league leaders every week:

- the cohort is the union of every entry ever seen in the top N of the
  overall league (league 314), so the eventual winner is guaranteed in it
- each run backfills any gameweek picks a cohort member is missing, so a
  missed week self-heals on the next run
- raw JSON is stored per entry under data/archive/<season>/, cheap to keep
  and the point-in-time record a study of elite squads (or a future
  backtest of them) would need; nothing reads it back yet

Run weekly (any time after a gameweek finishes):

    uv run fpl archive             # top 250, ~5 min on the first real run
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import requests

from . import paths
from .api import BASE, RETRY_STATUSES, TIMEOUT_S, fetch_bootstrap

ARCHIVE_DIR = paths.ARCHIVE_DIR
OVERALL_LEAGUE = 314
PAGE_SIZE = 50  # fixed by the API
REQUEST_GAP_S = 0.25
DEFAULT_TOP_N = 250  # cohort intake per run: the overall league's top N
PROGRESS_EVERY = 50  # entries archived between progress lines


def _get(path: str, retries: int = 3) -> dict | list | None:
    """GET with backoff; None for 404 (entry gone / GW not played)."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{path}", timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code in RETRY_STATUSES:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def season_label(bootstrap: dict) -> str:
    """Derive '2026-27' from the first event deadline."""
    first = dt.datetime.fromisoformat(bootstrap["events"][0]["deadline_time"])
    return f"{first.year}-{str(first.year + 1)[2:]}"


def completed_gws(bootstrap: dict) -> list[int]:
    """Ids of the gameweeks the API marks finished — the ones whose picks can
    be archived (a round in progress is not yet settled)."""
    return [e["id"] for e in bootstrap["events"] if e["finished"]]


def fetch_top_entries(n: int) -> list[dict]:
    """Top-n rows of the overall league (entry id, rank, total)."""
    rows: list[dict] = []
    for page in range(1, -(-n // PAGE_SIZE) + 1):
        data = _get(f"leagues-classic/{OVERALL_LEAGUE}/standings/?page_standings={page}")
        results = (data or {}).get("standings", {}).get("results", [])
        if not results:
            break
        rows.extend(results)
        if not data["standings"]["has_next"]:
            break
        time.sleep(REQUEST_GAP_S)
    return rows[:n]


def _load_cohort(season_dir: Path) -> dict[str, dict]:
    path = season_dir / "cohort.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _archive_entry(entry_id: int, gws: list[int], entries_dir: Path) -> int:
    """Fetch history/transfers (cumulative, overwritten) and any missing GW picks."""
    dest = entries_dir / str(entry_id)
    dest.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for name in ("history", "transfers"):
        data = _get(f"entry/{entry_id}/{name}/")
        if data is not None:
            (dest / f"{name}.json").write_text(json.dumps(data))
            fetched += 1
        time.sleep(REQUEST_GAP_S)
    for gw in gws:
        out = dest / f"picks_gw{gw:02d}.json"
        if out.exists():
            continue
        picks = _get(f"entry/{entry_id}/event/{gw}/picks/")
        if picks is not None:  # 404 = entry joined after this GW; nothing to store
            out.write_text(json.dumps(picks))
            fetched += 1
        time.sleep(REQUEST_GAP_S)
    return fetched


def run(top_n: int = DEFAULT_TOP_N, out_dir: Path | None = None) -> Path:
    """One archive pass: refresh cohort from the overall league, backfill picks."""
    bootstrap = fetch_bootstrap()
    season_dir = (out_dir or ARCHIVE_DIR) / season_label(bootstrap)
    season_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    gws = completed_gws(bootstrap)

    standings = fetch_top_entries(top_n)
    (season_dir / f"standings_{stamp}.json").write_text(json.dumps(standings))

    cohort = _load_cohort(season_dir)
    for row in standings:
        member = cohort.setdefault(str(row["entry"]), {"first_seen": stamp})
        member.update(last_seen=stamp, last_rank=row["rank"], player_name=row["player_name"])
    (season_dir / "cohort.json").write_text(json.dumps(cohort, indent=1))

    if not gws:
        print(f"season {season_label(bootstrap)}: no finished gameweeks yet — "
              f"snapshotted standings ({len(standings)} entries), nothing to backfill")
        return season_dir

    entries_dir = season_dir / "entries"
    requests_made = 0
    for i, entry_id in enumerate(cohort, 1):
        requests_made += _archive_entry(int(entry_id), gws, entries_dir)
        if i % PROGRESS_EVERY == 0:
            print(f"  {i}/{len(cohort)} entries archived")
    print(f"season {season_label(bootstrap)}: cohort {len(cohort)} entries, "
          f"GWs 1-{max(gws)}, {requests_made} new files")
    return season_dir


if __name__ == "__main__":
    run()
