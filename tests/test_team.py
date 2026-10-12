"""Your-team reconstruction and transfer-mode optimizer, on synthetic data."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_model import plan, rules, team
from fpl_model.optimizer import optimize_squad
from test_optimizer import toy_players  # rootdir tests/ is on sys.path (rootdir conftest-less layout)


# --- pure helpers -------------------------------------------------------------


def test_selling_price_halves_profit_rounding_down():
    assert team.selling_price(5.0, 5.0) == 5.0
    assert team.selling_price(5.0, 5.1) == 5.0      # 0.1 profit -> floor(0.05) = 0
    assert team.selling_price(5.0, 5.2) == 5.1
    assert team.selling_price(5.0, 5.5) == 5.2
    assert team.selling_price(5.0, 4.7) == 4.7      # losses are yours in full


def _history(transfers_by_gw: dict[int, int], chips: dict[int, str] | None = None) -> dict:
    return {
        "current": [{"event": gw, "event_transfers": n} for gw, n in transfers_by_gw.items()],
        "chips": [{"name": c, "event": gw} for gw, c in (chips or {}).items()],
    }


def test_free_transfers_bank_and_cap():
    # GW2 you hold 1; unused ones accumulate, capped at 5
    assert team.free_transfers_after(_history({1: 0, 2: 0}), through_gw=2) == 2
    assert team.free_transfers_after(_history({g: 0 for g in range(1, 9)}), through_gw=8) == 5
    # spending one leaves you on 1 for next week (+1 - 1)
    assert team.free_transfers_after(_history({1: 0, 2: 1}), through_gw=2) == 1
    # hit: 3 transfers on 1 FT -> back to 1
    assert team.free_transfers_after(_history({1: 0, 2: 0, 3: 3}), through_gw=3) == 1


def test_free_transfers_chip_weeks_are_free():
    h = _history({1: 0, 2: 0, 3: 9}, chips={3: "wildcard"})
    assert team.free_transfers_after(h, through_gw=3) == 3  # 1 -> 2 -> 3, WC spent nothing
    h = _history({1: 0, 2: 0, 3: 9}, chips={3: "freehit"})
    assert team.free_transfers_after(h, through_gw=3) == 3
    # only gameweeks up to `through_gw` count
    assert team.free_transfers_after(h, through_gw=2) == 2


def test_chips_available_per_half():
    h = _history({1: 0}, chips={1: "bboost", 3: "3xc"})
    assert team.chips_available(h, next_gw=5) == ["wildcard", "freehit"]
    assert team.chips_available(h, next_gw=25) == list(team.CHIPS)  # second set
    h2 = _history({1: 0}, chips={22: "wildcard"})
    assert "wildcard" in team.chips_available(h2, next_gw=10)
    assert "wildcard" not in team.chips_available(h2, next_gw=30)


def _elements():
    return {
        1: {"id": 1, "code": 101, "web_name": "Keeper", "now_cost": 45, "cost_change_start": 0,
            "element_type": 1, "team": 1, "status": "a", "chance_of_playing_next_round": None},
        2: {"id": 2, "code": 102, "web_name": "Riser", "now_cost": 62, "cost_change_start": 4,
            "element_type": 3, "team": 2, "status": "a", "chance_of_playing_next_round": None},
        3: {"id": 3, "code": 103, "web_name": "Bought", "now_cost": 80, "cost_change_start": 2,
            "element_type": 4, "team": 3, "status": "d", "chance_of_playing_next_round": 75},
        4: {"id": 4, "code": 104, "web_name": "FHOnly", "now_cost": 50, "cost_change_start": 0,
            "element_type": 2, "team": 4, "status": "a", "chance_of_playing_next_round": None},
    }


def test_purchase_prices_opening_transfer_and_freehit():
    picks = [{"element": 1}, {"element": 2}, {"element": 3}, {"element": 4}]
    transfers = [
        {"element_in": 3, "element_in_cost": 77, "element_out": 9, "event": 4, "time": "t1"},
        # free-hit week buys are reverted: must not count
        {"element_in": 4, "element_in_cost": 99, "element_out": 9, "event": 5, "time": "t2"},
        # an earlier buy of 3 is superseded by the later one
        {"element_in": 3, "element_in_cost": 70, "element_out": 9, "event": 2, "time": "t0"},
    ]
    hist = _history({1: 0}, chips={5: "freehit"})
    paid = team.purchase_prices(picks, transfers, _elements(), hist)
    assert paid == {1: 4.5, 2: 5.8, 3: 7.7, 4: 5.0}


def test_build_team_end_to_end():
    boot = {
        "elements": list(_elements().values()),
        "teams": [{"id": i, "short_name": f"T{i}"} for i in range(1, 5)],
    }
    picks = {
        "picks": [
            {"element": 1, "position": 1, "is_captain": False, "is_vice_captain": False},
            {"element": 2, "position": 2, "is_captain": True, "is_vice_captain": False},
            {"element": 3, "position": 12, "is_captain": False, "is_vice_captain": True},
            {"element": 4, "position": 13, "is_captain": False, "is_vice_captain": False},
        ],
        "entry_history": {"bank": 13, "value": 1003},
    }
    hist = _history({1: 0, 2: 0, 3: 1}, chips={2: "bboost"})
    entry = {"player_first_name": "A", "player_last_name": "B", "name": "XI FC",
             "summary_overall_rank": 1234, "summary_overall_points": 200}
    t = team.build_team(7, entry, picks, hist, [], boot, gw=3, next_gw=4)
    assert t.codes == {101, 102, 103, 104}
    assert t.bank == 1.3
    assert t.free_transfers == 2          # 1 -> 2 -> (2-1)+1 = 2
    assert t.chips_available == ["wildcard", "freehit", "3xc"]
    by_name = {p.name: p for p in t.picks}
    assert by_name["Riser"].is_captain and by_name["Riser"].sell == 6.0  # bought 5.8, now 6.2
    assert by_name["Bought"].bench_slot == 1 and by_name["Bought"].chance == 0.75
    assert any("Bought (75%)" in n for n in t.notes)
    assert by_name["Bought"].sell == 7.9   # opening 7.8, now 8.0 -> half the profit
    assert t.squad_value == pytest.approx(4.5 + 6.0 + 7.9 + 5.0 + 1.3)


def test_resolve_and_remember_entry(tmp_path, monkeypatch):
    path = tmp_path / "me.json"
    monkeypatch.delenv("FPL_ENTRY", raising=False)
    with pytest.raises(SystemExit, match="--entry"):
        team.resolve_entry(None, path)
    team.remember_entry(42, path)
    assert json.loads(path.read_text()) == {"entry": 42}
    assert team.resolve_entry(None, path) == 42
    monkeypatch.setenv("FPL_ENTRY", "99")
    assert team.resolve_entry(None, path) == 99
    assert team.resolve_entry(7, path) == 7


# --- transfer-mode optimizer ---------------------------------------------------


def _owned(players: pd.DataFrame) -> dict[int, float]:
    """A legal but weak 15: the *last* players of each position, sold at price."""
    codes: list[int] = []
    for pos, n in rules.SQUAD_SHAPE.items():
        codes += list(players[players["POS"] == pos].index[-n:])
    return {c: float(players.loc[c, "price"]) for c in codes}


def test_no_transfers_keeps_squad_and_budget_ignored():
    players, xp = toy_players()
    owned = _owned(players)
    sol = optimize_squad(players, xp, current=owned, bank=0.0, max_transfers=0)
    assert set(sol.squad.index) == set(owned)
    assert sol.transfers_in == [] and sol.transfers_out == [] and sol.hit_points == 0


def test_free_transfer_used_and_hits_priced():
    players, xp = toy_players()
    owned = _owned(players)
    bank = 50.0  # money is no object; only the hit cost limits transfers
    # a prohibitive hit: exactly the one free transfer gets used
    one_ft = optimize_squad(players, xp, current=owned, bank=bank, free_transfers=1, hit_cost=100.0)
    assert one_ft.n_transfers == 1 and one_ft.hit_points == 0
    # the toy pool's upgrades are worth ~10 xP each over 2 GWs, so 4-pt hits get taken
    hits = optimize_squad(players, xp, current=owned, bank=bank, free_transfers=1, hit_cost=4.0, ft_value=0.0)
    assert hits.n_transfers > 1
    assert hits.hit_points == pytest.approx(4.0 * (hits.n_transfers - 1))
    assert hits.expected_points >= one_ft.expected_points
    # two free transfers: at least as good a squad, and 4 points cheaper net of hits
    # (ft_value=0, so xP - hits is exactly what the solver maximised)
    two_ft = optimize_squad(players, xp, current=owned, bank=bank, free_transfers=2, hit_cost=4.0, ft_value=0.0)
    assert two_ft.n_transfers >= hits.n_transfers
    assert two_ft.expected_points - two_ft.hit_points >= hits.expected_points - hits.hit_points + 4.0 - 1e-9


def test_free_transfer_value_stops_marginal_churn():
    players, xp = toy_players()
    owned = _owned(players)
    # make every upgrade tiny: flat projections apart from a +0.5 edge for the best of each position
    xp = xp.copy()
    xp[:] = 3.0
    for pos in rules.POSITIONS:
        xp.loc[players[players["POS"] == pos].index[0]] = 3.5
    churn = optimize_squad(players, xp, current=owned, bank=50.0, free_transfers=3, ft_value=0.0)
    assert churn.n_transfers >= 1                     # any gain is taken when FTs are free
    held = optimize_squad(players, xp, current=owned, bank=50.0, free_transfers=3, ft_value=5.0)
    assert held.n_transfers == 0                      # +1 xP over 2 GWs is not worth a 5-pt FT
    full = optimize_squad(players, xp, current=owned, bank=50.0, free_transfers=5, ft_value=5.0)
    assert full.n_transfers == 1                      # at the cap one FT expires anyway: use it


def test_transfer_budget_uses_bank_plus_sales():
    players, xp = toy_players()
    owned = _owned(players)
    # with an empty bank every swap must be self-funding: bought <= sold
    sol = optimize_squad(players, xp, current=owned, bank=0.0, free_transfers=15, hit_cost=0.0)
    sold = sum(owned[c] for c in sol.transfers_out)
    bought = sum(players.loc[c, "price"] for c in sol.transfers_in)
    assert bought <= sold + 1e-9
    assert len(sol.transfers_in) == len(sol.transfers_out)


def test_current_player_missing_from_pool_is_an_error():
    players, xp = toy_players()
    with pytest.raises(ValueError, match="current squad"):
        optimize_squad(players, xp, current={9999: 5.0})


# --- plan glue ----------------------------------------------------------------


def _fake_team(players: pd.DataFrame, next_gw: int = 1) -> team.MyTeam:
    owned = _owned(players)
    picks = [
        team.Pick(code=c, name=players.loc[c, "FPL_Name"], team=players.loc[c, "TEAM"],
                  pos=players.loc[c, "POS"], purchase=v, now=v, sell=v)
        for c, v in owned.items()
    ]
    return team.MyTeam(
        entry_id=1, manager="m", team_name="t", gw=next_gw - 1, next_gw=next_gw, picks=picks,
        bank=2.0, free_transfers=1, chips_used=[], chips_available=list(team.CHIPS),
        overall_rank=None, total_points=None,
    )


def _fake_ppts(players: pd.DataFrame, xp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in players.index:
        for gw in xp.columns:
            rows.append({"Code": c, "gw": gw, "TRUE TOTAL": xp.loc[c, gw],
                         **players.loc[c, ["FPL_Name", "TEAM", "POS"]].to_dict()})
    return pd.DataFrame(rows)


def test_plan_transfers_shapes_and_gain():
    players, xp = toy_players()
    ppts = _fake_ppts(players, xp)
    my = _fake_team(players)
    p = plan.plan_transfers(ppts, players["price"], my, to_gw=2)
    assert p.from_gw == 1 and p.to_gw == 2
    assert len(p.current) == 15 and {"window_xp", "sell", "purchase"} <= set(p.current.columns)
    assert len(p.xi) == 11 and len(p.bench) == 4
    assert p.xi["captain"].sum() == 1 and p.xi["vice"].sum() == 1
    assert p.bank == my.bank  # the bank the plan was priced against
    # a --bank override is what the plan (and the explainer's alternatives) price against
    assert plan.plan_transfers(ppts, players["price"], my, to_gw=2, bank=my.bank + 3.0).bank == my.bank + 3.0
    assert not (p.xi["captain"] & p.xi["vice"]).any()
    assert p.bench.iloc[0]["POS"] == "GK"  # bench GK sits in slot 1
    assert p.gain >= -1e-9
    for t in p.transfers:
        assert t["pos"] == players.loc[t["in_code"], "POS"]
    # money adds up
    assert p.bank_after == pytest.approx(
        my.bank + sum(t["out_sell"] for t in p.transfers) - sum(t["in_price"] for t in p.transfers), abs=0.05
    )


def test_plan_owned_player_outside_model_gets_zero():
    players, xp = toy_players()
    my = _fake_team(players)
    ppts = _fake_ppts(players, xp)
    ghost = my.picks[-1].code
    ppts = ppts[ppts["Code"] != ghost]
    p = plan.plan_transfers(ppts, players["price"], my, to_gw=2, max_transfers=0)
    assert p.current.loc[ghost, "window_xp"] == 0
    assert any("no projection" in w for w in p.warnings)


def test_excluding_an_owned_player_forces_a_sale():
    players, xp = toy_players()
    owned = _owned(players)
    victim = next(c for c in owned if players.loc[c, "POS"] == "DEF")  # the toy pool has spare DEFs
    sol = optimize_squad(players, xp, current=owned, bank=50.0, excludes={victim}, ft_value=0.0)
    assert victim in sol.transfers_out and victim not in sol.squad.index
