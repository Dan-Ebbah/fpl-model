"""Optimizer: known-answer checks on a toy pool, rulebook checks on real data."""

import pandas as pd
import pytest

from fpl_model import reference
from fpl_model.optimizer import SquadRules, optimize_squad, select_pool


def toy_players() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal pool: 2 GK, 6 DEF, 6 MID, 4 FWD across enough clubs."""
    rows, xp_rows = [], []
    spec = [("GK", 2), ("DEF", 6), ("MID", 6), ("FWD", 4)]
    code = 0
    for pos, n in spec:
        for i in range(n):
            code += 1
            rows.append(
                {
                    "Code": code,
                    "FPL_Name": f"{pos}{i}",
                    "TEAM": f"T{code % 7}",
                    "POS": pos,
                    # first player of each position is premium and best
                    "price": 10.0 if i == 0 else 4.0 + i * 0.5,
                }
            )
            xp_rows.append({"Code": code, 1: 8.0 - i, 2: 8.0 - i})
    players = pd.DataFrame(rows).set_index("Code")
    xp = pd.DataFrame(xp_rows).set_index("Code")
    return players, xp


def test_toy_squad_respects_all_rules():
    players, xp = toy_players()
    rules = SquadRules(budget=90.0)
    sol = optimize_squad(players, xp, rules=rules)

    assert sol.status == "Optimal"
    assert len(sol.squad) == 15
    assert sol.cost <= rules.budget + 1e-9
    assert sol.squad["POS"].value_counts().to_dict() == {"DEF": 5, "MID": 5, "FWD": 3, "GK": 2}
    assert sol.squad["TEAM"].value_counts().max() <= 3

    for gw, group in sol.lineups.groupby("gw"):
        xi = group[group["started"]]
        assert len(xi) == 11
        counts = xi["POS"].value_counts()
        assert counts.get("GK", 0) == 1
        assert 3 <= counts.get("DEF", 0) <= 5
        assert 1 <= counts.get("FWD", 0) <= 3
        captains = group[group["captain"]]
        assert len(captains) == 1
        assert captains["started"].all(), "captain must be a starter"


def test_lock_and_exclude():
    players, xp = toy_players()
    worst_mid = players[players["POS"] == "MID"].index[-1]
    best_fwd = players[players["POS"] == "FWD"].index[0]
    sol = optimize_squad(players, xp, locks={worst_mid}, excludes={best_fwd})
    assert worst_mid in sol.squad.index
    assert best_fwd not in sol.squad.index


def test_higher_budget_never_scores_worse():
    players, xp = toy_players()
    # the toy pool's cheapest legal squad costs 84.5, so 85 is tight but feasible
    tight = optimize_squad(players, xp, rules=SquadRules(budget=85.0))
    loose = optimize_squad(players, xp, rules=SquadRules(budget=100.0))
    assert loose.expected_points >= tight.expected_points - 1e-9


@pytest.mark.slow
def test_real_data_squad_is_legal():
    from fpl_model.cli import _prices, build_ppts

    ppts = build_ppts()
    window = ppts[(ppts["gw"] >= 1) & (ppts["gw"] <= 4)]
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")
    players = window.groupby("Code")[["FPL_Name", "TEAM", "POS"]].first().join(
        _prices(refresh=False).rename("price")
    )
    players = players[players["price"].notna()]
    pool = select_pool(players, xp.sum(axis=1))

    sol = optimize_squad(pool, xp)
    assert sol.cost <= 100.0 + 1e-9
    assert sol.squad["POS"].value_counts().to_dict() == {"DEF": 5, "MID": 5, "FWD": 3, "GK": 2}
    assert sol.squad["TEAM"].value_counts().max() <= 3
    assert sol.expected_points > 100  # 4 GWs of a legal XI should clear this easily
