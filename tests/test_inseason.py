"""In-season layer: form multipliers, rate shrinkage, start-rate blend.

Synthetic inputs throughout — the properties under test are the blend's
behaviour (direction, shrinkage, schedule adjustment), not any real number.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers import LA, strength_table

from fpl_model import inseason
from fpl_model.config import DEFAULT_PARAMS


def _boot(team_xg: dict[str, float], team_xgc: dict[str, float]) -> dict:
    teams = [{"id": i + 1, "short_name": t} for i, t in enumerate(team_xg)]
    elements = []
    for t in teams:
        elements.append({"team": t["id"], "element_type": 4, "expected_goals": team_xg[t["short_name"]],
                         "expected_goals_conceded": 99.0})  # outfield xGC must be ignored
        elements.append({"team": t["id"], "element_type": 1, "expected_goals": 0.0,
                         "expected_goals_conceded": team_xgc[t["short_name"]]})
    return {"teams": teams, "elements": elements}


def _double_round_robin(n_teams: int, rounds: int = 1) -> list[dict]:
    """Every team hosts every other once per round: two meetings per pair."""
    fx = []
    for _ in range(rounds):
        for h in range(1, n_teams + 1):
            for a in range(1, n_teams + 1):
                if h != a:
                    fx.append({"team_h": h, "team_a": a, "finished": True})
    return fx


def test_team_on_expectation_gets_no_adjustment():
    ts = strength_table({"AAA": 1.5, "BBB": 1.5}, {"AAA": 1.5, "BBB": 1.5})
    fx = _double_round_robin(2)  # one home, one away each: home factor nets out
    form = inseason.team_form(_boot({"AAA": 3.0, "BBB": 3.0}, {"AAA": 3.0, "BBB": 3.0}), fx, ts, LA, DEFAULT_PARAMS)
    assert form["att_mult"].tolist() == pytest.approx([1.0, 1.0])
    assert form["def_mult"].tolist() == pytest.approx([1.0, 1.0])
    assert form["games"].tolist() == [2.0, 2.0]


def test_form_moves_toward_evidence_but_is_shrunk():
    teams = ["AAA", "BBB", "CCC", "DDD"]
    ts = strength_table(dict.fromkeys(teams, 1.5), dict.fromkeys(teams, 1.5))
    xg = {"AAA": 4.5, "BBB": 9.0, "CCC": 9.0, "DDD": 9.0}  # AAA creating half as much
    form = inseason.team_form(_boot(xg, dict.fromkeys(teams, 9.0)), _double_round_robin(4), ts, LA, DEFAULT_PARAMS)
    a = form.loc["AAA"]
    assert a["att_raw"] < a["att_mult"] < 1.0      # direction right, shrunk toward the prior
    assert form.loc["BBB", "att_mult"] > 1.0       # relative: the rest look better
    adj = inseason.adjust_team_strength(ts, form).set_index("ABVR")
    assert adj.loc["AAA", "LF xG/90"] == pytest.approx(1.5 * a["att_mult"])
    assert adj.loc["AAA", "LF xGC/90"] == pytest.approx(1.5 * a["def_mult"])


def test_form_is_schedule_adjusted():
    """Conceding a lot to an elite attack is not bad defensive form."""
    ts = strength_table({"TOP": 3.0, "MID": 1.5, "LOW": 1.5}, dict.fromkeys(["TOP", "MID", "LOW"], 1.5))
    # MID only ever plays TOP, LOW is idle filler to keep the league level sane
    fx = [{"team_h": 2, "team_a": 1, "finished": True}, {"team_h": 1, "team_a": 2, "finished": True}]
    expected_conceded = 3.0 * 1.5 / LA * 2  # what TOP's prior attack predicts over two games
    boot = _boot({"TOP": expected_conceded, "MID": 3.0, "LOW": 0.0},
                 {"TOP": 3.0, "MID": expected_conceded, "LOW": 0.0})
    form = inseason.team_form(boot, fx, ts, LA, DEFAULT_PARAMS)
    assert form.loc["MID", "def_mult"] == pytest.approx(1.0, abs=1e-6)
    assert form.loc["LOW", "games"] == 0 and form.loc["LOW", "att_mult"] == 1.0


def test_unfinished_fixtures_are_ignored():
    ts = strength_table({"AAA": 1.5, "BBB": 1.5}, {"AAA": 1.5, "BBB": 1.5})
    fx = [{"team_h": 1, "team_a": 2, "finished": False}]
    form = inseason.team_form(_boot({"AAA": 5.0, "BBB": 0.0}, {"AAA": 0.0, "BBB": 5.0}), fx, ts, LA, DEFAULT_PARAMS)
    assert form["att_mult"].tolist() == [1.0, 1.0]


def _model(rows: list[dict]) -> pd.DataFrame:
    base = {"TEAM": "AAA", "POS": "DEF", "POS2": "CB", "Goal": 6.0, "G+A PSxTS Mn/Start": 1.0,
            "7": 0.0, "PS Sts/SqdsMd": 0.9, "St/SqdsMd": 0.0, "PS Starts": 30, "PS Subs": 2, "PS UnSub": 2}
    return pd.DataFrame([{**base, **r} for r in rows])


def _api(rows: list[dict]) -> pd.DataFrame:
    base = {"minutes": 0, "starts": 0, "expected_goals": 0.0, "expected_assists": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_small_sample_rate_is_pulled_to_the_group_mean():
    # a big established group at 0.05 xG/90, one 100-minute player at 0.30
    regulars = [{"Code": i, "8": 0.05 * 6, "LF G+A Mins": 3000.0} for i in range(1, 6)]
    model = _model(regulars + [{"Code": 99, "8": 0.30 * 6, "LF G+A Mins": 100.0}])
    out = inseason.adjust_player_rates(model, _api([{"code": c} for c in model["Code"]]), DEFAULT_PARAMS.points)
    fluke, regular = out[out["Code"] == 99].iloc[0], out[out["Code"] == 1].iloc[0]
    assert fluke["xG90 prior"] == pytest.approx(0.30)
    assert fluke["xG90 used"] < 0.10                       # mostly the norm now
    assert regular["xG90 used"] == pytest.approx(0.05, abs=0.002)  # a full season barely moves
    assert fluke["8"] == pytest.approx(fluke["xG90 used"] * 6)


def test_current_season_evidence_moves_the_rate():
    regulars = [{"Code": i, "8": 0.10 * 6, "LF G+A Mins": 3000.0} for i in range(1, 4)]
    model = _model(regulars)
    api = _api([{"code": 1, "minutes": 900, "expected_goals": 0.0},    # gone cold
                {"code": 2, "minutes": 900, "expected_goals": 3.0},    # 0.30/90, on fire
                {"code": 3}])
    out = inseason.adjust_player_rates(model, api, DEFAULT_PARAMS.points).set_index("Code")
    assert out.loc[1, "xG90 used"] < out.loc[3, "xG90 used"] < out.loc[2, "xG90 used"]
    assert out.loc[1, "xG90 used"] > 0.05  # 10 games don't erase a full season
    assert out.loc[2, "Season Mins"] == 900


def test_player_with_no_minutes_anywhere_is_untouched():
    model = _model([{"Code": 1, "8": 0.10 * 6, "LF G+A Mins": 3000.0},
                    {"Code": 2, "8": 0.0, "LF G+A Mins": 0.0}])
    out = inseason.adjust_player_rates(model, _api([{"code": 1}, {"code": 2}]), DEFAULT_PARAMS.points)
    assert out.set_index("Code").loc[2, "8"] == 0.0


def test_group_norm_scales_with_team_attack():
    model = _model([{"Code": 1, "8": 0.10 * 6, "LF G+A Mins": 3000.0},
                    {"Code": 2, "TEAM": "WEAK", "8": 0.0, "LF G+A Mins": 0.0}])
    api = _api([{"code": 1}, {"code": 2, "minutes": 90, "expected_goals": 0.0}])
    rel = pd.Series({"AAA": 1.0, "WEAK": 0.5})
    out = inseason.adjust_player_rates(model, api, DEFAULT_PARAMS.points, team_attack_rel=rel).set_index("Code")
    assert out.loc[2, "xG90 group"] == pytest.approx(0.05)


def test_start_rate_blend():
    model = _model([
        {"Code": 1},                                                            # nailed both seasons
        {"Code": 2, "PS Sts/SqdsMd": 0.15, "PS Starts": 3, "PS Subs": 10, "PS UnSub": 7},  # backup turned starter
        {"Code": 3},                                                            # no minutes: injured or dropped, can't tell
    ])
    api = _api([{"code": 1, "minutes": 360, "starts": 4}, {"code": 2, "minutes": 360, "starts": 4}, {"code": 3}])
    form = pd.DataFrame({"games": {"AAA": 4.0}})
    out = inseason.adjust_start_rates(model, api, form).set_index("Code")
    assert out.loc[1, "Start rate used"] == pytest.approx(0.9, abs=0.03)
    assert 0.35 < out.loc[2, "Start rate used"] < 0.6        # up a lot, not to 100%
    assert out.loc[3, "Start rate used"] == 0.9
    assert out.loc[2, "PS Sts/SqdsMd"] == out.loc[2, "St/SqdsMd"] == out.loc[2, "Start rate used"]


def test_fit_player_with_no_minutes_loses_his_start_rate():
    model = _model([{"Code": 1}, {"Code": 2}, {"Code": 3}])
    api = _api([{"code": 1, "status": "a"},                  # fit, unused for 4 games: dropped
                {"code": 2, "status": "i"},                  # injured: the flag handles it
                {"code": 3, "status": "a", "minutes": 360, "starts": 4}])
    out = inseason.adjust_start_rates(model, api, pd.DataFrame({"games": {"AAA": 4.0}})).set_index("Code")
    assert out.loc[1, "Start rate used"] == pytest.approx(0.9 * 0.5 ** 4)
    assert out.loc[2, "Start rate used"] == 0.9
    assert out.loc[3, "Start rate used"] > 0.85
    # before a ball is kicked nobody has "missed" anything
    pre = inseason.adjust_start_rates(model, api, pd.DataFrame({"games": {"AAA": 0.0}})).set_index("Code")
    assert pre.loc[1, "Start rate used"] == 0.9
