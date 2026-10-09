"""Explainer: the reasoning attached to a plan is a faithful readout."""

from __future__ import annotations

import pytest

from fpl_model import explain, team
from fpl_model.pipeline import build_pipeline, player_prices
from fpl_model.optimizer import optimize_squad, select_pool
from fpl_model.plan import plan_transfers, squad_frame


@pytest.fixture(scope="module")
def explained():
    pipe = build_pipeline()  # frozen workbook snapshot: no network, no in-season layer
    prices = player_prices(refresh=False)
    players, xp = squad_frame(pipe.ppts, prices, 1, 3)
    best = optimize_squad(select_pool(players, xp.sum(axis=1)), xp)
    codes = [int(c) for c in best.squad["Code"]] if "Code" in best.squad else [int(c) for c in best.squad.index]
    picks = [
        team.Pick(code=c, element=c, name=players.loc[c, "FPL_Name"], team=players.loc[c, "TEAM"],
                  pos=players.loc[c, "POS"], purchase=players.loc[c, "price"],
                  now=players.loc[c, "price"], sell=players.loc[c, "price"])
        for c in codes
    ]
    my = team.MyTeam(
        entry_id=1, manager="m", team_name="t", gw=0, next_gw=1, picks=picks, bank=0.5,
        free_transfers=1, chips_used=[], chips_available=list(team.CHIPS),
        overall_rank=None, total_points=None,
    )
    # selling the best player forces exactly one transfer to explain
    sold = int(xp.loc[codes].sum(axis=1).idxmax())
    plan = plan_transfers(pipe.ppts, prices, my, to_gw=3, max_transfers=1, excludes={sold})
    explain.explain_plan(plan, pipe, prices)
    return plan, pipe


def test_every_transfer_gets_reasons(explained):
    plan, _ = explained
    assert len(plan.transfers) == 1
    why = plan.transfers[0]["why"]
    assert any(w.startswith("Where the") for w in why)
    assert any(w.startswith("XI effect") for w in why)
    assert any(w.startswith("Fixtures: GW1") for w in why)
    assert any(w.startswith("Minutes:") for w in why)
    # frozen pipeline has no in-season tables, so no form/rate claims are made
    assert not any("form, adjusted" in w or w.startswith("Rates:") for w in why)
    assert any(v.startswith("Captain GW1") for v in plan.verdict)


def test_category_breakdown_sums_to_total(explained):
    plan, pipe = explained
    rows = explain._window_rows(pipe.ppts, plan.transfers[0]["in_code"], 1, 3)
    parts = explain.category_breakdown(rows)
    assert sum(parts.values()) == pytest.approx(rows["TRUE TOTAL"].sum())
    assert parts["minutes"] > 0  # appearance points are never negative


def test_alternatives_exclude_owned_and_respect_budget(explained):
    plan, _ = explained
    t = plan.transfers[0]
    line = next(w for w in plan.transfers[0]["why"] if w.startswith("Next best"))
    assert t["in"] not in line
    assert f"£{t['out_sell'] + 0.5:.1f}m" in line
