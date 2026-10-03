"""Web UI payload builders, on the workbook snapshot (no network, no server)."""

from __future__ import annotations

import pytest

from fpl_model import web


def test_predictions_payload_shape():
    p = web.predictions_payload(1, 5, refresh=False, pos="MID", sort="pts", top=10)
    rows = p["rows"]
    assert 0 < len(rows) <= 10
    assert all(r["POS"] == "MID" for r in rows)
    pts = [r["true_pts"] for r in rows]
    assert pts == sorted(pts, reverse=True)
    assert {"FPL_Name", "TEAM", "price", "games", "value"} <= set(rows[0])


def test_player_payload_and_errors():
    p = web.player_payload("Bukayo", 1, 5, refresh=False)
    assert "Saka" in p["name"]
    assert 1 <= len(p["rows"]) <= 5
    assert {"gw", "vs", "goals", "true_total"} <= set(p["rows"][0])
    with pytest.raises(ValueError, match="no player"):
        web.player_payload("zzznotaplayer", 1, 5, refresh=False)
    with pytest.raises(ValueError, match="ambiguous"):
        web.player_payload("a", 1, 5, refresh=False)


def test_verdict_payload():
    v = web.verdict_payload("Bruno Borges", 1, 8, refresh=False)
    assert v["pos"] == "MID" and v["team"] == "MUN"
    assert v["rank"] >= 1 and v["pos_rank"] <= v["pos_count"]
    assert 0 <= v["start_pct"] <= 1
    assert len(v["fixtures"]) >= 6
    assert v["verdict"] and all(isinstance(s, str) for s in v["verdict"])
    # a top pick should read as one
    assert any("elite" in s or "strong" in s for s in v["verdict"])
    # priors-backed player carries the low-confidence caveat
    t = web.verdict_payload("Tzolis", 1, 8, refresh=False)
    assert any("priors.yaml" in s for s in t["verdict"])
    with pytest.raises(ValueError, match="ambiguous"):
        web.verdict_payload("Pedro", 1, 8, refresh=False)


def test_compare_payload():
    j = web.compare_payload("Calvert-Lewin", "Richarlison", 1, 8, refresh=False)
    assert j["a"]["name"].endswith("Calvert-Lewin")
    assert "Richarlison" in j["b"]["name"]
    assert j["same_pos"] is True
    assert len(j["rows"]) == 8
    assert j["xp_diff"] == pytest.approx(j["a"]["xp"] - j["b"]["xp"], abs=0.1)
    total_a = sum(r["a_xp"] for r in j["rows"])
    assert total_a == pytest.approx(j["a"]["xp"], abs=0.5)
    cross = web.compare_payload("Bukayo", "Gabriel dos", 1, 8, refresh=False)
    assert cross["same_pos"] is False


@pytest.mark.slow
def test_squad_payload_legal():
    s = web.squad_payload(1, 3, refresh=False, bench_weight=0.1, locks=[], excludes=[])
    assert s["status"] == "Optimal"
    assert len(s["squad"]) == 15
    assert len(s["xi"]) == 11
    assert len(s["bench"]) == 4
    assert s["cost"] <= 100.0
    assert sum(1 for r in s["xi"] if r["captain"]) == 1


def test_plan_to_json_shape():
    from test_team import _fake_ppts, _fake_team, toy_players
    from fpl_model.plan import plan_transfers

    players, xp = toy_players()
    p = plan_transfers(_fake_ppts(players, xp), players["price"], _fake_team(players), to_gw=2)
    j = web.plan_to_json(p)
    assert j["from_gw"] == 1 and j["to_gw"] == 2
    assert len(j["current"]) == 15 and len(j["xi"]) == 11 and len(j["bench"]) == 4
    assert sum(r["captain"] for r in j["xi"]) == 1 and sum(r["vice"] for r in j["xi"]) == 1
    assert j["gain"] == pytest.approx(j["proposal_xp"] - j["hit_points"] - j["baseline_xp"], abs=0.1)
    assert {"out", "in", "in_price", "out_sell", "xp_gain"} <= set(j["transfers"][0]) if j["transfers"] else True
    assert isinstance(j["chips_available"], list) and j["formation"].count("-") == 2
