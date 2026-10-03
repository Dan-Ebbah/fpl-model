"""Playing style: match log assembly, venue tilt, positional channels, scorer odds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_model import matchlog, style
from fpl_model.cli import build_pipeline, scorer_table
from fpl_model.config import DEFAULT_PARAMS

LA = 1.5
TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _strength() -> pd.DataFrame:
    return pd.DataFrame([{"ABVR": t, "LF xG/90": 1.5, "SF xG/90": 1.5, "LF xGC/90": 1.5, "SF xGC/90": 1.5} for t in TEAMS])


def _log(home_xg: dict[str, float], away_xg: dict[str, float], fwd_share: dict[str, float], rounds: int = 1) -> pd.DataFrame:
    """Double round robin; each side's xG split between one FWD and one DEF."""
    rows, fid = [], 0
    for _ in range(rounds):
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                fid += 1
                for team, opp, home, xg in [(h, a, True, home_xg[h]), (a, h, False, away_xg[a])]:
                    for pos, share in [("FWD", fwd_share[team]), ("DEF", 1 - fwd_share[team])]:
                        rows.append({"element": hash((team, pos)) % 10000, "code": 1, "pos": pos, "fixture": fid,
                                     "round": 1, "team": team, "opponent": opp, "was_home": home, "minutes": 90,
                                     "starts": 1, "goals": 0, "assists": 0, "xg": xg * share, "xa": xg * share, "xgc": 0.0})
    return pd.DataFrame(rows)


def test_typical_home_advantage_is_no_tilt():
    flat = dict.fromkeys(TEAMS, 0.5)
    log = _log(dict.fromkeys(TEAMS, 1.65), dict.fromkeys(TEAMS, 1.35), flat)
    v = style.venue_tilt(log, _strength(), LA, DEFAULT_PARAMS)
    assert v[["att_home", "att_away", "def_home", "def_away"]].to_numpy() == pytest.approx(1.0)


def test_home_side_tilt_is_shrunk_and_reciprocal():
    home = {**dict.fromkeys(TEAMS, 1.65), "AAA": 2.6}
    away = {**dict.fromkeys(TEAMS, 1.35), "AAA": 0.9}
    flat = dict.fromkeys(TEAMS, 0.5)
    v = style.venue_tilt(_log(home, away, flat), _strength(), LA, DEFAULT_PARAMS).loc["AAA"]
    assert 1.0 < v["att_home"] < v["att_tilt_raw"] ** 0.5      # right direction, shrunk
    assert v["att_home"] * v["att_away"] == pytest.approx(1.0)  # moves points between venues only
    more = style.venue_tilt(_log(home, away, flat, rounds=3), _strength(), LA, DEFAULT_PARAMS).loc["AAA"]
    assert more["att_home"] > v["att_home"]                     # more games, more belief


def test_channels_attack_and_concession():
    flat_h, flat_a = dict.fromkeys(TEAMS, 1.5), dict.fromkeys(TEAMS, 1.5)
    shares = {"AAA": 0.9, "BBB": 0.5, "CCC": 0.5, "DDD": 0.5}  # AAA plays through its striker
    st = style.team_style(_log(flat_h, flat_a, shares), _strength(), LA, DEFAULT_PARAMS)
    assert st.attack.loc[("AAA", "FWD"), "xg_share_raw"] == pytest.approx(0.9)
    assert 0.6 < st.attack.loc[("AAA", "FWD"), "xg_share"] < 0.9  # shrunk toward the league 0.6
    assert st.attack.loc[("AAA", "FWD"), "goal_rel"] > 1 > st.attack.loc[("AAA", "DEF"), "goal_rel"]
    # everyone who faced AAA conceded more to forwards than the league does
    assert st.concede.loc[("BBB", "FWD"), "goal_mult"] > 1
    lo, hi = style.DEFAULT_STYLE.channel_clip
    assert st.concede[["goal_mult", "assist_mult"]].to_numpy().min() >= lo
    assert st.attack[["goal_rel", "assist_rel"]].to_numpy().max() <= hi

    adj = style.fixture_adjustments(st, TEAMS)
    assert len(adj) == 4 * 3 * 2 * 4
    gk = adj[adj["POS"] == "GK"]
    assert gk["goal_mult"].to_numpy() == pytest.approx(1.0)  # no venue tilt here, no GK channel ever


def test_match_log_uses_fixture_team_and_caches(tmp_path):
    boot = {
        "teams": [{"id": 1, "short_name": "AAA"}, {"id": 2, "short_name": "BBB"}],
        "elements": [
            {"id": 10, "code": 100, "element_type": 4, "team": 2, "minutes": 90},   # now at BBB...
            {"id": 11, "code": 101, "element_type": 2, "team": 1, "minutes": 0},
        ],
    }
    fixtures = [{"id": 7, "team_h": 1, "team_a": 2}]
    hist = {"history": [{"fixture": 7, "round": 1, "opponent_team": 2, "was_home": True, "minutes": 90, "starts": 1,
                         "goals_scored": 1, "assists": 0, "expected_goals": "0.80", "expected_assists": "0.10",
                         "expected_goals_conceded": "1.20"}]}
    calls = []

    def fetch(pid: int) -> dict:
        calls.append(pid)
        return hist

    cache = tmp_path / "log.parquet"
    log = matchlog.load_match_log(boot, fixtures, cache_path=cache, fetch=fetch)
    assert calls == [10]                                  # zero-minute players are never fetched
    assert log.iloc[0]["team"] == "AAA"                   # ...but played this one for AAA (the home side)
    assert log.iloc[0]["xg"] == pytest.approx(0.8)
    matchlog.load_match_log(boot, fixtures, cache_path=cache, fetch=fetch)
    assert calls == [10]                                  # unchanged minutes: served from cache
    boot["elements"][0]["minutes"] = 180
    matchlog.load_match_log(boot, fixtures, cache_path=cache, fetch=fetch)
    assert calls == [10, 10]                              # minutes moved: refetched


def test_fixture_adj_scales_goals_and_clean_sheets_only_where_told():
    from fpl_model import reference
    from fpl_model.fixtures import build_fixture_slots, opponent_grid
    from fpl_model.ppts import compute_ppts

    teams = reference.load("TEAMS")
    teams["id"] = pd.to_numeric(teams["id"], errors="coerce").astype(int)
    fx = reference.load("FIXTURES")[["event", "team_h", "team_a"]].apply(pd.to_numeric, errors="coerce")
    la_xg, la_xgc = reference.league_average_xg()
    dcs, dca = reference.load_dcscore()
    kw = dict(model=reference.load_model(), team_strength=reference.load_team_strength(), dcscore=dcs,
              dc_league_avg=dca, fixture_grid=opponent_grid(build_fixture_slots(fx, teams), gws=range(1, 3)),
              league_avg_xg=la_xg, league_avg_xgc=la_xgc, params=DEFAULT_PARAMS)
    base = compute_ppts(**kw)
    row = base[(base["POS"] == "FWD") & (base["8"] > 0.5) & (base["VS"].str.len() >= 3)].iloc[0]
    adj = pd.DataFrame([{"TEAM": row["TEAM"].upper(), "vs_key": row["VS"].upper(), "is_home": row["VS"] == row["VS"].upper(),
                         "POS": "FWD", "goal_mult": 1.2, "assist_mult": 1.0, "conceded_mult": 1.0}])
    out = compute_ppts(**kw, fixture_adj=adj)
    hit = (out["Code"] == row["Code"]) & (out["gw"] == row["gw"]) & (out["slot"] == row["slot"])
    assert out.loc[hit, "8"].iloc[0] == pytest.approx(row["8"] * 1.2)
    assert out.loc[hit, "7"].iloc[0] == pytest.approx(row["7"])
    # keyed by team/position/fixture: his fellow forwards move too, nobody else does
    scoped = (out["TEAM"] == row["TEAM"]) & (out["POS"] == "FWD") & (out["VS"] == row["VS"])
    pd.testing.assert_series_equal(out.loc[~scoped, "TRUE TOTAL"], base.loc[~scoped, "TRUE TOTAL"])
    assert (out.loc[scoped, "TRUE TOTAL"] >= base.loc[scoped, "TRUE TOTAL"]).all()


def test_scorer_table_probabilities():
    ppts = build_pipeline().ppts
    t = scorer_table(ppts, gw=1)
    assert ((t["P(goal)"] >= 0) & (t["P(goal)"] < 1)).all()
    assert (t["P(either)"] >= t[["P(goal)", "P(assist)"]].max(axis=1) - 1e-12).all()
    assert np.allclose(t["P(goal)"], 1 - np.exp(-t["xG"]))
    top = t.sort_values("P(goal)", ascending=False).iloc[0]
    assert top["POS"] in ("FWD", "MID") and 0.25 < top["P(goal)"] < 0.75
    one = scorer_table(ppts, gw=1, team=top["TEAM"])
    assert set(one["TEAM"]) == {top["TEAM"]}
    assert 0.5 < one["xG"].sum() < 4.0   # a team's players add up to a plausible scoreline
