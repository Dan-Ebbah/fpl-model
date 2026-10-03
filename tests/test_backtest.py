"""Backtester: snapshot selection, no leakage, metrics and honest verdicts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpl_model import api, backtest as bt, reference
from fpl_model.inseason import PORT_ONLY


def _events(next_gw: int, n: int = 4) -> list[dict]:
    return [{"id": i, "is_next": i == next_gw, "finished": i < next_gw} for i in range(1, n + 1)]


def test_snapshot_for_uses_the_snapshots_own_view_of_the_deadline():
    snaps = [
        bt.Snapshot("2026-08-20", {"events": _events(1)}, []),
        bt.Snapshot("2026-08-24", {"events": _events(2)}, []),   # mid-week: GW2 upcoming
        bt.Snapshot("2026-08-28", {"events": _events(2)}, []),   # deadline day, still before it
        bt.Snapshot("2026-09-07", {"events": _events(4)}, []),   # nothing captured while GW3 was next
    ]
    assert bt.snapshot_for(snaps, 2).date == "2026-08-28"        # latest that qualifies
    assert bt.snapshot_for(snaps, 3) is None                     # never borrowed from a later one
    assert bt.snapshot_for(snaps, 4).date == "2026-09-07"


def test_load_snapshots_skips_a_failed_pulls_empty_folder(tmp_path):
    (tmp_path / "2026-09-15").mkdir()
    good = tmp_path / "2026-09-17"
    good.mkdir()
    (good / "bootstrap.json").write_text('{"events": []}')
    pd.DataFrame([{"id": 1, "team_h": 1, "team_a": 2, "stats": None}]).to_parquet(good / "fixtures.parquet")
    (tmp_path / "matchlog.parquet").write_text("not a folder")
    snaps = bt.load_snapshots(tmp_path)
    assert [s.date for s in snaps] == ["2026-09-17"]
    assert "stats" not in snaps[0].fixtures[0]


def test_failed_snapshot_leaves_no_folder(tmp_path, monkeypatch):
    def boom():
        raise ConnectionError("TLS EOF")

    monkeypatch.setattr(api, "fetch_bootstrap", boom)
    with pytest.raises(ConnectionError):
        api.snapshot(out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def world():
    """A replayable season built from the frozen workbook extracts: GW1 played,
    GW2 upcoming in the snapshot, and a log that already knows GW2 and GW3."""
    players = reference.load("API")
    players = players[pd.to_numeric(players["code"], errors="coerce").notna()].drop_duplicates("code")
    teams = reference.load("TEAMS")
    short = {int(t.id): str(t.short_name).upper() for t in teams.itertuples()}
    fx = reference.load("FIXTURES")[["id", "event", "team_h", "team_a"]].astype(int)
    fixtures = [{**r, "finished": r["event"] < 2} for r in fx.to_dict("records")]
    boot = {"elements": players.to_dict("records"), "teams": teams.to_dict("records"), "events": _events(2)}

    # one forward and one defender per side per match; the forward takes 70% of the xG
    by_team = {t: g for t, g in players.groupby(pd.to_numeric(players["team"]).astype(int))}
    rows = []
    for f in fixtures:
        if f["event"] > 3:
            continue
        for team, opp, home in [(f["team_h"], f["team_a"], True), (f["team_a"], f["team_h"], False)]:
            for etype, pos, share in [(4, "FWD", 0.7), (2, "DEF", 0.3)]:
                p = by_team[team][pd.to_numeric(by_team[team]["element_type"]) == etype].iloc[0]
                rows.append({"element": int(p["id"]), "code": int(p["code"]), "pos": pos, "fixture": f["id"],
                             "round": f["event"], "team": short[team], "opponent": short[opp], "was_home": home,
                             "minutes": 90, "starts": 1, "goals": int(pos == "FWD"), "assists": 0,
                             "xg": 1.5 * share, "xa": 1.0 * share, "xgc": 1.5, "points": 6 if pos == "FWD" else 2, "bonus": 0})
    return bt.Snapshot("2026-08-28", boot, fixtures), pd.DataFrame(rows)


VARIANT = {"all": replace(PORT_ONLY, form=True, rates=True, starts=True, style=True, priors=False)}


def test_replay_cannot_see_the_gameweek_it_predicts_or_later(world):
    snap, log = world
    table, skipped = bt.replay([snap], log, VARIANT, gws=[2], finished={1, 2})
    assert not skipped and set(table["gw"]) == {2} and table["snapshot"].iloc[0] == "2026-08-28"

    future = log.copy()
    hot = future["round"] >= 2
    future.loc[hot, ["xg", "xa"]] *= 50          # what happened in GW2/3 must be invisible...
    future.loc[hot & (future["pos"] == "DEF"), "xg"] = 40.0
    leaked, _ = bt.replay([snap], future, VARIANT, gws=[2], finished={1, 2})
    pd.testing.assert_series_equal(leaked["xp"], table["xp"])

    past = log.copy()                             # ...while GW1 is fair game, and moves the numbers
    one = (past["round"] == 1) & (past["team"] == past["team"].iloc[0])  # style is relative: change one side
    past.loc[one & (past["pos"] == "FWD"), "xg"], past.loc[one & (past["pos"] == "DEF"), "xg"] = 0.1, 1.4
    moved, _ = bt.replay([snap], past, VARIANT, gws=[2], finished={1, 2})
    assert (moved["xp"] - table["xp"]).abs().max() > 0.01


def test_replay_joins_actuals_and_reports_skips(world):
    snap, log = world
    table, skipped = bt.replay([snap], log, VARIANT, gws=[2, 3], finished={1, 2})
    assert skipped == {3: "gameweek not finished"}
    _, skipped = bt.replay([snap], log, VARIANT, gws=[3], finished={1, 2, 3})
    assert skipped == {3: "no snapshot taken before the deadline"}
    played = table[table["minutes"] > 0]
    # the frozen fixture list has a double gameweek in GW2: two sides play twice, and score twice
    assert len(played) == 40 and set(played["points"]) == {2, 4, 6, 12}
    assert (played["points"] == played["minutes"] / 90 * played["POS"].map({"FWD": 6, "DEF": 2})).all()
    assert (table.loc[table["minutes"] == 0, "points"] == 0).all()
    assert table["price"].notna().all()


def _table(noise: dict[str, float], n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = rng.gamma(2.0, 1.5, n)
    start = rng.uniform(0, 1, n)
    frames = []
    for variant, sd in noise.items():
        r = np.random.default_rng(seed + 1)  # same noise draw, scaled: variants differ only in sd
        frames.append(pd.DataFrame({
            "variant": variant, "gw": np.repeat([1, 2], n // 2), "Code": np.arange(n),
            "POS": np.tile(["DEF", "MID"], n // 2), "xp": truth + r.normal(0, sd, n),
            "points": np.round(truth), "minutes": 90, "started": (rng.uniform(0, 1, n) < start).astype(int) if variant == list(noise)[0] else 0,
            "start_pct": start, "exp_goals": 0.2, "exp_assists": 0.1, "goals": 0, "assists": 0,
        }))
    out = pd.concat(frames, ignore_index=True)
    out["started"] = out.groupby("Code")["started"].transform("max")
    return out


def test_evaluate_ranks_a_better_variant_and_refuses_to_rank_a_tie():
    ev = bt.evaluate(_table({"port": 3.0, "sharp": 0.5, "same": 3.0}), n_boot=60)
    assert ev.loc["sharp", "mae"] < ev.loc["port", "mae"] and ev.loc["sharp", "rank"] > ev.loc["port", "rank"]
    assert ev.loc["sharp", "mae_hi"] < 0 < ev.loc["sharp", "rank_lo"]
    assert ev.loc["same", "mae_lo"] <= 0 <= ev.loc["same", "mae_hi"]
    lines = {v.split(":")[0]: v for v in bt.verdicts(ev)}
    assert "better on" in lines["sharp"] and "mae" in lines["sharp"]
    assert "no metric distinguishable" in lines["same"]


def test_probability_metrics_use_everyone_not_just_those_who_played():
    t = _table({"port": 1.0})
    ghosts = t.iloc[:50].assign(Code=lambda d: d["Code"] + 10_000, xp=0.0, minutes=0, started=0, points=0, start_pct=0.0)
    t = pd.concat([t, ghosts], ignore_index=True)
    flagged = bt._relevant(t)
    assert not flagged.loc[flagged["Code"] >= 10_000, "relevant"].any()
    cal = bt.calibration(t, "port")
    assert cal.iloc[0]["n"] >= 50 and cal.iloc[0]["actual"] < 0.1   # the correct 0% calls are counted
    assert abs(cal.iloc[-2]["predicted"] - cal.iloc[-2]["actual"]) < 0.15
