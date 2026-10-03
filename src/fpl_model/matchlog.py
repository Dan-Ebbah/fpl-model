"""Per-match player log for the current season.

The bootstrap only has season totals; anything about *how* a team plays —
home vs away, which positions its chances fall to, who it concedes to —
needs per-match rows. Those live behind one element-summary call per player,
so the log is cached and only players whose season minutes moved since the
cached copy are refetched (a weekly run costs ~300 small requests, a rerun
the same day costs none).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from . import api

CACHE_PATH = api.SNAPSHOT_DIR / "matchlog.parquet"
ELEMENT_TYPE_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
COLUMNS = [
    "element", "code", "pos", "fixture", "round", "team", "opponent", "was_home",
    "minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "points", "bonus",
]


def rows_from_history(player: dict, history: list[dict], fixtures: dict[int, dict], short: dict[int, str]) -> list[dict]:
    """One row per match played. The player's team *in that match* comes from
    the fixture, not the bootstrap — a mid-season mover's old games must stay
    with his old club."""
    rows = []
    for h in history:
        fx = fixtures.get(int(h["fixture"]))
        if fx is None or not h.get("minutes"):
            continue
        home = bool(h["was_home"])
        rows.append({
            "element": int(player["id"]), "code": int(player["code"]),
            "pos": ELEMENT_TYPE_POS.get(int(player["element_type"]), "-"),
            "fixture": int(h["fixture"]), "round": int(h["round"]),
            "team": short[int(fx["team_h"] if home else fx["team_a"])],
            "opponent": short[int(h["opponent_team"])], "was_home": home,
            "minutes": int(h["minutes"]), "starts": int(h.get("starts", 0)),
            "goals": int(h["goals_scored"]), "assists": int(h["assists"]),
            "xg": float(h["expected_goals"]), "xa": float(h["expected_assists"]),
            "xgc": float(h["expected_goals_conceded"]),
            "points": int(h.get("total_points", 0)), "bonus": int(h.get("bonus", 0)),
        })
    return rows


def load_match_log(
    boot: dict,
    fixtures_raw: list[dict],
    cache_path: Path | None = CACHE_PATH,
    fetch: Callable[[int], dict] = api.fetch_player_history,
    workers: int = 8,
) -> pd.DataFrame:
    short = {int(t["id"]): str(t["short_name"]).upper() for t in boot["teams"]}
    fixtures = {int(f["id"]): f for f in fixtures_raw if f.get("id") is not None}

    cached = pd.DataFrame(columns=COLUMNS)
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if set(COLUMNS) - set(cached.columns):  # written by an older version: refetch
            cached = pd.DataFrame(columns=COLUMNS)
    have = cached.groupby("element")["minutes"].sum() if len(cached) else pd.Series(dtype=float)

    played = {int(p["id"]): int(p["minutes"]) for p in boot["elements"] if int(p.get("minutes") or 0) > 0}
    # fresh = cached minutes agree with the bootstrap total. Everything else
    # is refetched or dropped — which also clears last season's rows, since
    # element and fixture ids are reused across seasons
    fresh_ids = {e for e, mins in played.items() if have.get(e, -1) == mins}
    stale = [p for p in boot["elements"] if int(p["id"]) in played and int(p["id"]) not in fresh_ids]
    log = cached[cached["element"].isin(fresh_ids)]
    if stale:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(lambda p: fetch(int(p["id"])), stale))
        rows = [
            r for p, s in zip(stale, summaries)
            for r in rows_from_history(p, s.get("history", []), fixtures, short)
        ]
        log = pd.concat([log, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    if cache_path is not None and (stale or len(log) != len(cached)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(cache_path)
    log = log.reset_index(drop=True)
    log["was_home"] = log["was_home"].astype(bool)
    return log
