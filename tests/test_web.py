"""Web UI payload builders, on the workbook snapshot (no network, no server)."""

from __future__ import annotations

import pytest

from fpl_model import payloads


def test_predictions_payload_shape():
    p = payloads.predictions_payload(1, 5, refresh=False, pos="MID", sort="pts", top=10)
    rows = p["rows"]
    assert 0 < len(rows) <= 10
    assert all(r["POS"] == "MID" for r in rows)
    pts = [r["true_pts"] for r in rows]
    assert pts == sorted(pts, reverse=True)
    assert {"FPL_Name", "TEAM", "price", "games", "value"} <= set(rows[0])


def test_player_payload_and_errors():
    p = payloads.player_payload("Bukayo", 1, 5, refresh=False)
    assert "Saka" in p["name"]
    assert 1 <= len(p["rows"]) <= 5
    assert {"gw", "vs", "goals", "true_total"} <= set(p["rows"][0])
    with pytest.raises(ValueError, match="no player"):
        payloads.player_payload("zzznotaplayer", 1, 5, refresh=False)
    with pytest.raises(ValueError, match="ambiguous"):
        payloads.player_payload("a", 1, 5, refresh=False)


def test_verdict_payload():
    v = payloads.verdict_payload("Bruno Borges", 1, 8, refresh=False)
    assert v["pos"] == "MID" and v["team"] == "MUN"
    assert v["rank"] >= 1 and v["pos_rank"] <= v["pos_count"]
    assert 0 <= v["start_pct"] <= 1
    assert len(v["fixtures"]) >= 6
    assert v["verdict"] and all(isinstance(s, str) for s in v["verdict"])
    # a top pick should read as one
    assert any("elite" in s or "strong" in s for s in v["verdict"])
    # priors-backed player carries the low-confidence caveat
    t = payloads.verdict_payload("Tzolis", 1, 8, refresh=False)
    assert any("priors.yaml" in s for s in t["verdict"])
    with pytest.raises(ValueError, match="ambiguous"):
        payloads.verdict_payload("Pedro", 1, 8, refresh=False)


def test_compare_payload():
    j = payloads.compare_payload("Calvert-Lewin", "Richarlison", 1, 8, refresh=False)
    assert j["a"]["name"].endswith("Calvert-Lewin")
    assert "Richarlison" in j["b"]["name"]
    assert j["same_pos"] is True
    assert len(j["rows"]) == 8
    assert j["xp_diff"] == pytest.approx(j["a"]["xp"] - j["b"]["xp"], abs=0.1)
    total_a = sum(r["a_xp"] for r in j["rows"])
    assert total_a == pytest.approx(j["a"]["xp"], abs=0.5)
    cross = payloads.compare_payload("Bukayo", "Gabriel dos", 1, 8, refresh=False)
    assert cross["same_pos"] is False


@pytest.mark.slow
def test_squad_payload_legal():
    s = payloads.squad_payload(1, 3, refresh=False, bench_weight=0.1, locks=[], excludes=[])
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
    j = payloads.plan_to_json(p)
    assert j["from_gw"] == 1 and j["to_gw"] == 2
    assert len(j["current"]) == 15 and len(j["xi"]) == 11 and len(j["bench"]) == 4
    assert sum(r["captain"] for r in j["xi"]) == 1 and sum(r["vice"] for r in j["xi"]) == 1
    assert j["gain"] == pytest.approx(j["proposal_xp"] - j["hit_points"] - j["baseline_xp"], abs=0.1)
    assert {"out", "in", "in_price", "out_sell", "xp_gain"} <= set(j["transfers"][0]) if j["transfers"] else True
    assert isinstance(j["chips_available"], list) and j["formation"].count("-") == 2


def test_sandbox_tables_shape():
    from fpl_model.pipeline import build_pipeline, player_prices

    pipe = build_pipeline(refresh=False)
    t = payloads.sandbox_tables(pipe.ppts, player_prices(False), 6, 10, boot=None, team_form=pipe.team_form)
    assert t["gws"] == [6, 7, 8, 9, 10]
    assert len(t["players"]) > 400 and len(t["teams"]) == 20
    p = next(v for v in t["players"].values() if v["name"] == "James Tarkowski")
    assert p["pos"] == "DEF" and p["team"] == "EVE" and p["price"] > 0
    # window xP is the sum of the per-GW entries
    assert p["xp"] == pytest.approx(sum(g["xp"] for g in p["gw"].values()), abs=0.05)
    assert set(p["gw"]) <= {"6", "7", "8", "9", "10"}
    assert {"goals", "assists", "cs", "conceded", "dc", "bonus"} == set(p["cats"])
    assert "actual" not in p  # no bootstrap given
    # team fixtures agree with the player's own
    assert all(set(g["vs"]) <= set(t["teams"]["EVE"]["fixtures"][gw]) for gw, g in p["gw"].items())
    if pipe.team_form is not None:
        assert "att_mult" in t["teams"]["EVE"]["form"]


def test_track_record_from_backtest_table(tmp_path):
    import pandas as pd

    rows = []
    for variant, xp in [("all", 4.5), ("port", 3.9)]:
        rows.append(dict(Code=17761, FPL_Name="James Tarkowski", TEAM="EVE", POS="DEF", xp=xp,
                         start_pct=1.0, exp_goals=0.08, exp_assists=0.1, points=6.0, minutes=90.0,
                         started=1.0, goals=0.0, assists=0.0, price=6.0, variant=variant, gw=1,
                         snapshot="2026-08-20"))
    pd.DataFrame(rows).to_parquet(tmp_path / "bt.parquet")
    tr = payloads.track_record(None, path=tmp_path / "bt.parquet")
    assert list(tr) == [17761]
    r = tr[17761][0]
    assert r["gw"] == 1 and r["xp"] == 4.5 and r["xp_port"] == 3.9 and r["pts"] == 6
    assert r["started"] is True and r["snapshot"] == "2026-08-20"
    assert payloads.track_record(None, path=tmp_path / "missing.parquet") == {}
