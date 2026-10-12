"""The web UI's payload builders: plain functions from arguments to JSON-ready dicts.

Everything the browser renders is assembled here, with no HTTP in sight —
`web.Handler` only parses the query string and calls one of these. They are
testable without a server (see tests/test_web.py) and read their projections
through a per-process cache (`cached_pipeline`), so a workbook-snapshot build
and a live build each happen once for the life of the server.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from . import api, explain, team as team_mod
from .backtest import RESULTS_PATH
from .optimizer import captains_by_gw, formation, optimize_squad, select_pool
from .pipeline import Pipeline, build_pipeline, player_prices, resolve_codes
from .plan import plan_transfers
from .predict import aggregate_window, gameweek_window, has_fixture, with_fixture
from .priors import load_priors
from .rules import DEFAULT_FT_VALUE, DEFAULT_HORIZON, HIT_COST, LAST_GW, MAX_PER_CLUB

# --- the per-process projection cache ---------------------------------------
_cache: dict[bool, Pipeline] = {}  # refresh flag -> Pipeline
_cache_lock = threading.Lock()


def cached_pipeline(refresh: bool, force: bool = False) -> Pipeline:
    """One pipeline build per data source (workbook snapshot vs live pull), rebuilt on `force`."""
    with _cache_lock:
        if force or refresh not in _cache:
            _cache[refresh] = build_pipeline(refresh=refresh)
        return _cache[refresh]


def cached_ppts(refresh: bool, force: bool = False) -> pd.DataFrame:
    return cached_pipeline(refresh, force).ppts


# --- verdict thresholds ------------------------------------------------------
ELITE_RANK = 5  # overall rank at or above which a player is an "elite pick"
STRONG_RANK = 25  # ... a "strong pick"
SOLID_POS_RANK = 15  # positional rank at or above which a player is "solid"
CAPTAIN_RANK = 3  # overall rank that earns the captaincy sentence
NAILED_START = 0.85  # mean start probability thresholds for the minutes sentence
SECURE_START = 0.6
ROTATION_START = 0.35
GOOD_VALUE_RANK = 10  # points-per-£ rank within position that reads as good value...
GOOD_VALUE_MIN_POS_RANK = 5  # ...unless he is already a top-5 pick in his position
POOR_VALUE_SHARE = 0.7  # value rank beyond this share of the position pool is poor value
FRONT_LOAD_RATIO = 1.25  # early-half xP must beat late-half by this ratio (or vice versa)
MIN_FIXTURES_FOR_SHAPE = 4  # fewer fixtures than this and the window has no shape to speak of
MAX_AMBIGUOUS_NAMES = 8  # how many matches an "ambiguous" error lists


def predictions_payload(
    from_gw: int, to_gw: int, refresh: bool, pos: str | None = None,
    team: str | None = None, max_price: float | None = None,
    sort: str = "pts", top: int = 50, force: bool = False,
) -> dict:
    """Ranked window projections with price and value — the Predictions tab.

    Mirrors `cli.cmd_predict`: aggregate the window, attach prices, filter by
    position / team / max price, sort by `sort` (pts, value or price) and
    keep the top `top` rows.
    """

    ppts = cached_ppts(refresh, force)
    pred = aggregate_window(ppts, from_gw, to_gw)
    pred["price"] = pred["Code"].map(player_prices(refresh))
    pred = pred[pred["price"].notna()].copy()
    pred["value"] = (pred["true_pts"] / pred["price"] / pred["games"]).where(
        (pred["price"] > 0) & (pred["games"] > 0), 0.0
    )
    if pos:
        pred = pred[pred["POS"] == pos.upper()]
    if team:
        pred = pred[pred["TEAM"] == team.upper()]
    if max_price:
        pred = pred[pred["price"] <= max_price]
    key = {"pts": "true_pts", "value": "value", "price": "price"}[sort]
    pred = pred.sort_values(key, ascending=False).head(top)
    cols = ["FPL_Name", "TEAM", "POS", "price", "games", "true_pts", "value"]
    return {"rows": json.loads(pred[cols].round(2).to_json(orient="records"))}


def squad_payload(
    from_gw: int, to_gw: int, refresh: bool, bench_weight: float,
    locks: list[str], excludes: list[str], force: bool = False,
) -> dict:
    """Solve the optimal 15 over the window — the Squad tab.

    Same inputs as `cli.cmd_squad`: per-GW xP pivot, prices, lock/exclude
    names resolved to codes, pool trimmed, then the MILP. Returns the squad
    table, the first gameweek's XI/bench/formation and the captain per GW.
    """

    ppts = cached_ppts(refresh, force)
    window = gameweek_window(ppts, from_gw, to_gw)
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")
    players = (
        window.groupby("Code")[["FPL_Name", "TEAM", "POS"]]
        .first()
        .join(player_prices(refresh).rename("price"))
    )
    players = players[players["price"].notna()]

    lock_codes = resolve_codes(locks, players)
    exclude_codes = resolve_codes(excludes, players)
    pool = select_pool(players, xp.sum(axis=1), keep=lock_codes)
    sol = optimize_squad(
        players=pool, xp=xp, bench_weight=bench_weight,
        locks=lock_codes, excludes=exclude_codes,
    )

    first = sol.lineups[sol.lineups["gw"] == from_gw]
    xi = first[first["started"]]
    captains = captains_by_gw(sol.lineups)
    return {
        "cost": round(sol.cost, 1),
        "expected_points": round(sol.expected_points, 1),
        "squad": json.loads(sol.squad.round(2).to_json(orient="records")),
        "formation": formation(xi),
        "xi": json.loads(
            xi[["FPL_Name", "TEAM", "POS", "captain"]].to_json(orient="records")
        ),
        "bench": list(first.loc[~first["started"], "FPL_Name"]),
        "captains": json.loads(
            captains[["gw", "FPL_Name", "TEAM"]].to_json(orient="records")
        ),
    }


def team_payload(
    entry: int | None, to_gw: int | None, bench_weight: float,
    ft: int | None, bank: float | None, max_transfers: int | None,
    hit_cost: float, wildcard: bool, locks: list[str], excludes: list[str],
    force: bool = False, ft_value: float = DEFAULT_FT_VALUE,
) -> dict:
    """Your team: reconstructed state + the deadline plan (always live data)."""

    entry_id = team_mod.resolve_entry(entry)
    boot = api.fetch_bootstrap()
    my = team_mod.load_my_team(entry_id, boot)
    if entry:
        team_mod.remember_entry(entry_id)  # only once it resolved to a real team
    pipe = cached_pipeline(True, force)
    ppts, prices = pipe.ppts, player_prices(True, boot)
    names = ppts.groupby("Code")[["FPL_Name"]].first()
    plan = plan_transfers(
        ppts, prices, my, to_gw=to_gw, bench_weight=bench_weight,
        free_transfers=ft, bank=bank, max_transfers=max_transfers, hit_cost=hit_cost,
        ft_value=ft_value, wildcard=wildcard, locks=resolve_codes(locks, names),
        excludes=resolve_codes(excludes, names),
    )
    explain.explain_plan(plan, pipe, prices, ft_value=ft_value)
    return plan_to_json(plan)


def plan_to_json(plan) -> dict:
    """A `plan.TeamPlan` as the My Team tab's JSON.

    Team state (entry, bank, chips), the warnings and verdict lines, the
    current 15 with selling prices, the baseline-vs-proposal xP, each
    transfer (floats rounded to 2dp), the first-GW XI and bench, and the
    captain per gameweek.
    """
    my = plan.team
    caps = captains_by_gw(plan.proposal.lineups)
    return {
        "entry": my.entry_id, "manager": my.manager, "team_name": my.team_name,
        "overall_rank": my.overall_rank, "total_points": my.total_points,
        "gw": my.gw, "from_gw": plan.from_gw, "to_gw": plan.to_gw,
        "bank": my.bank, "squad_value": my.squad_value,
        "free_transfers": my.free_transfers,
        "chips_available": my.chips_available,
        "warnings": plan.warnings,
        "verdict": plan.verdict,
        "current": json.loads(
            plan.current.reset_index()[
                ["FPL_Name", "TEAM", "POS", "purchase", "sell", "chance", "window_xp"]
            ].round(2).to_json(orient="records")
        ),
        "baseline_xp": round(plan.baseline.expected_points, 1),
        "proposal_xp": round(plan.proposal.expected_points, 1),
        "gain": round(plan.gain, 1),
        "hit_points": plan.proposal.hit_points,
        "bank_after": plan.bank_after,
        "transfers": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()}
                      for t in plan.transfers],
        "formation": formation(plan.xi),
        "xi_xp": round(float(plan.xi["xp"].sum()), 1),
        "xi": json.loads(
            plan.xi[["FPL_Name", "TEAM", "POS", "xp", "captain", "vice"]].round(2).to_json(orient="records")
        ),
        "bench": json.loads(plan.bench[["FPL_Name", "POS", "xp"]].round(2).to_json(orient="records")),
        "captains": json.loads(caps[["gw", "FPL_Name", "TEAM"]].to_json(orient="records")),
    }


# ---------------------------------------------------------------------------
# Verdict: the model's opinion on one player, written out.
# ---------------------------------------------------------------------------


def _match_player(pred: pd.DataFrame, name: str) -> pd.Series:
    """The one aggregated row whose name contains `name` (case-insensitive).

    ValueError when nothing matches or more than one distinct player does;
    the ambiguous message lists at most `MAX_AMBIGUOUS_NAMES` of them.
    """
    hits = pred[pred["FPL_Name"].str.contains(name, case=False, na=False)]
    if hits.empty:
        raise ValueError(f"no player matching {name!r}")
    if hits["FPL_Name"].nunique() > 1:
        names = hits["FPL_Name"].unique()[:MAX_AMBIGUOUS_NAMES]
        raise ValueError(f"ambiguous — matches: {', '.join(names)}")
    return hits.iloc[0]


def _ranks(pred: pd.DataFrame, row: pd.Series) -> tuple[int, int, int, int]:
    """(overall rank, rank within position, points-per-£ rank within position, position pool size).

    Ranks are dense-from-the-top (`method="min"`) over the aggregated window
    table; value is window xP per £ of price, zero where the price is zero.
    """
    pred["rank"] = pred["true_pts"].rank(ascending=False, method="min")
    pos_pool = pred[pred["POS"] == row["POS"]].copy()
    pos_pool["pos_rank"] = pos_pool["true_pts"].rank(ascending=False, method="min")
    pos_pool["value"] = (pos_pool["true_pts"] / pos_pool["price"]).where(pos_pool["price"] > 0, 0)
    me = pos_pool[pos_pool["Code"] == row["Code"]].iloc[0]
    rank = int(pred.loc[pred["Code"] == row["Code"], "rank"].iloc[0])
    pos_rank = int(me["pos_rank"])
    value_rank = int(pos_pool["value"].rank(ascending=False, method="min")[me.name])
    return rank, pos_rank, value_rank, len(pos_pool)


def _fixture_info(
    ppts: pd.DataFrame, code: int, from_gw: int, to_gw: int,
) -> tuple[float, list[dict], list[int]]:
    """(mean start probability, per-fixture {gw, vs, xp} list, gameweeks with a double)."""
    mine = ppts[ppts["Code"] == code]
    mine = with_fixture(gameweek_window(mine, from_gw, to_gw)).sort_values(["gw", "slot"])
    start_pct = float(mine["Start %"].mean()) if len(mine) else 0.0
    fixtures = [
        {"gw": int(r["gw"]), "vs": r["VS"], "xp": round(float(r["TRUE TOTAL"]), 2)}
        for _, r in mine.iterrows()
    ]
    dgws = mine.groupby("gw").size()
    doubles = [int(g) for g, n in dgws.items() if n > 1]
    return start_pct, fixtures, doubles


def _verdict_sentences(
    row: pd.Series, rank: int, pos_rank: int, value_rank: int, n_pos: int,
    start_pct: float, fixtures: list[dict], doubles: list[int], prior: dict | None,
) -> list[str]:
    """The prose: standing, minutes, value, fixture shape, captaincy, confidence."""
    v: list[str] = []
    if rank <= ELITE_RANK:
        v.append(f"One of the model's elite picks: #{rank} overall for this window.")
    elif rank <= STRONG_RANK:
        v.append(f"A strong pick — #{rank} overall, #{pos_rank} of {n_pos} {row['POS']}s.")
    elif pos_rank <= SOLID_POS_RANK:
        v.append(f"Solid positional option: #{pos_rank} of {n_pos} {row['POS']}s (#{rank} overall).")
    else:
        v.append(f"Fringe by the numbers: #{pos_rank} of {n_pos} {row['POS']}s (#{rank} overall).")

    if start_pct >= NAILED_START:
        v.append(f"Minutes look nailed (avg start probability {start_pct:.0%}).")
    elif start_pct >= SECURE_START:
        v.append(f"Minutes fairly secure but not certain ({start_pct:.0%} start probability).")
    elif start_pct >= ROTATION_START:
        v.append(f"Real rotation risk — {start_pct:.0%} start probability; check team news before owning.")
    else:
        v.append(f"Serious minutes doubt ({start_pct:.0%} start probability) — avoid until that changes.")

    if value_rank <= GOOD_VALUE_RANK and pos_rank > GOOD_VALUE_MIN_POS_RANK:
        v.append(f"Good value at £{row['price']:.1f}m — #{value_rank} points-per-£ among {row['POS']}s.")
    elif value_rank > n_pos * POOR_VALUE_SHARE:
        v.append(f"Poor value at £{row['price']:.1f}m — you pay for the points you get.")

    if len(fixtures) >= MIN_FIXTURES_FOR_SHAPE:
        half = len(fixtures) // 2
        early = sum(f["xp"] for f in fixtures[:half]) / half
        late = sum(f["xp"] for f in fixtures[half:]) / (len(fixtures) - half)
        if early > late * FRONT_LOAD_RATIO:
            v.append("Fixtures are front-loaded — the good weeks come early in this window.")
        elif late > early * FRONT_LOAD_RATIO:
            v.append("Fixtures improve later in the window — patience may beat chasing.")
    if doubles:
        v.append(f"Double gameweek in GW{', GW'.join(map(str, doubles))} — extra fixture value.")

    if rank <= CAPTAIN_RANK:
        v.append("Captaincy material in the best weeks.")

    if prior:
        v.append(
            "Low confidence: no Premier League history — this projection rests on the "
            "hand-set prior in data/priors.yaml"
            + (f" ({prior.get('note', '').strip()})" if prior.get("note") else "")
            + ". Trust your read of team news over the number."
        )
    return v


def verdict_payload(name: str, from_gw: int, to_gw: int, refresh: bool, force: bool = False) -> dict:
    """The model's opinion on one player, written out.

    Deterministic prose assembled from the signals the pipeline already
    computes: window xP and ranks, minutes security, value, fixture shape,
    and confidence caveats (injury flags, hand-set priors).
    """

    ppts = cached_ppts(refresh, force)
    pred = aggregate_window(ppts, from_gw, to_gw)
    pred["price"] = pred["Code"].map(player_prices(refresh))
    pred = pred[pred["price"].notna()].copy()

    row = _match_player(pred, name)
    rank, pos_rank, value_rank, n_pos = _ranks(pred, row)
    start_pct, fixtures, doubles = _fixture_info(ppts, row["Code"], from_gw, to_gw)
    prior = next(
        (p for p in load_priors() if str(p["player"]).lower() in str(row["FPL_Name"]).lower()),
        None,
    )
    sentences = _verdict_sentences(
        row, rank, pos_rank, value_rank, n_pos, start_pct, fixtures, doubles, prior,
    )
    return {
        "name": row["FPL_Name"], "team": row["TEAM"], "pos": row["POS"],
        "price": round(float(row["price"]), 1),
        "window": [from_gw, to_gw],
        "xp": round(float(row["true_pts"]), 1),
        "rank": rank, "pos_rank": pos_rank, "pos_count": n_pos,
        "start_pct": round(start_pct, 2),
        "fixtures": fixtures,
        "verdict": sentences,
    }


def compare_payload(a: str, b: str, from_gw: int, to_gw: int, refresh: bool, force: bool = False) -> dict:
    """Side-by-side: the model's pick vs a user-suggested alternative."""
    va = verdict_payload(a, from_gw, to_gw, refresh, force)  # one rebuild on force, not two
    vb = verdict_payload(b, from_gw, to_gw, refresh)

    def by_gw(v: dict) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for f in v["fixtures"]:
            slot = out.setdefault(f["gw"], {"vs": [], "xp": 0.0})
            slot["vs"].append(f["vs"])
            slot["xp"] += f["xp"]
        return out

    ga, gb = by_gw(va), by_gw(vb)
    rows = []
    for gw in range(from_gw, to_gw + 1):
        fa, fb = ga.get(gw), gb.get(gw)
        rows.append({
            "gw": gw,
            "a_vs": "+".join(fa["vs"]) if fa else "-",
            "a_xp": round(fa["xp"], 2) if fa else 0.0,
            "b_vs": "+".join(fb["vs"]) if fb else "-",
            "b_xp": round(fb["xp"], 2) if fb else 0.0,
        })
    return {
        "a": va, "b": vb, "rows": rows,
        "xp_diff": round(va["xp"] - vb["xp"], 1),
        "price_diff": round(va["price"] - vb["price"], 1),
        "same_pos": va["pos"] == vb["pos"],
    }


def player_payload(name: str, from_gw: int, to_gw: int, refresh: bool, force: bool = False) -> dict:
    """One player's per-gameweek breakdown — the Player tab's table.

    Name matching is a case-insensitive substring over the PPts rows
    (ValueError when none or several distinct players match). Rows cover
    the window's real fixtures, columns renamed to the short JSON keys the
    browser reads, plus the backtester's track record for the player.
    """

    ppts = cached_ppts(refresh, force)
    rows = ppts[ppts["FPL_Name"].str.contains(name, case=False, na=False)]
    names = rows["FPL_Name"].unique()
    if len(names) == 0:
        raise ValueError(f"no player matching {name!r}")
    if len(names) > 1:
        raise ValueError(f"ambiguous — matches: {', '.join(names[:MAX_AMBIGUOUS_NAMES])}")
    code = int(rows["Code"].iloc[0])
    rows = _fixture_rows(rows, from_gw, to_gw)
    cols = {
        "gw": "gw", "VS": "vs", "8": "goals", "7": "assists", "9": "cs",
        "10": "gc2", "11": "dc", "6": "bonus", "Start %": "start_pct",
        "IF START TOTAL": "if_start", "TRUE TOTAL": "true_total",
    }
    out = rows[list(cols)].rename(columns=cols).round(2)
    past = track_record(ppts).get(code, [])
    return {"name": names[0], "rows": json.loads(out.to_json(orient="records")), "past": past}


# ---------------------------------------------------------------------------
# Sandbox: one payload the browser can recompute against without the solver.
# ---------------------------------------------------------------------------

_ACTUAL_COLS = {
    # bootstrap field -> short key in the payload
    "minutes": "mins", "starts": "starts", "total_points": "pts",
    "goals_scored": "g", "assists": "a", "expected_goals": "xg",
    "expected_assists": "xa", "expected_goal_involvements": "xgi",
    "expected_goals_conceded": "xgc", "clean_sheets": "cs",
    "defensive_contribution": "dc", "bonus": "bonus", "bps": "bps",
    "form": "form", "points_per_game": "ppg", "selected_by_percent": "own",
    "transfers_in_event": "tin", "transfers_out_event": "tout",
    "cost_change_start": "dprice", "ict_index": "ict",
}

# PPts category column -> payload key, in display order
_CATEGORY_COLS = [
    ("goals", "8"), ("assists", "7"), ("cs", "9"), ("conceded", "10"),
    ("dc", "11"), ("bonus", "6"),
]
_FORM_COLS = ["games", "xg", "exp_xg", "att_mult", "xgc", "exp_xgc", "def_mult"]
_TEAM_ACTUAL_COLS = ["played", "win", "draw", "loss", "points", "position"]


def _fixture_rows(ppts: pd.DataFrame, from_gw: int, to_gw: int) -> pd.DataFrame:
    """The PPts rows in the window that carry a real fixture (blank slots dropped)."""
    return with_fixture(gameweek_window(ppts, from_gw, to_gw))


def track_record(ppts: pd.DataFrame | None = None, path: Path | None = None) -> dict[int, list[dict]]:
    """What the model said before each finished deadline, next to what happened.

    Read from the backtester's table (`fpl backtest`), so every row was built
    from the last snapshot stored *before* that gameweek's deadline — no
    hindsight. `xp` is the live path (every in-season layer on), `xp_port` the
    bare pre-season port. Empty dict when no backtest has been run.
    """

    path = path or RESULTS_PATH
    if not path.exists():
        return {}
    t = pd.read_parquet(path)
    if t.empty:
        return {}
    live = t[t["variant"] == "all"].set_index(["Code", "gw"])
    port = t[t["variant"] == "port"].set_index(["Code", "gw"])["xp"]
    vs: dict[tuple[int, int], str] = {}
    if ppts is not None:
        past = ppts[(ppts["gw"] <= int(t["gw"].max())) & has_fixture(ppts["VS"])]
        vs = past.groupby(["Code", "gw"])["VS"].agg(" + ".join).to_dict()
    out: dict[int, list[dict]] = {}
    for (code, gw), r in live.sort_index().iterrows():
        out.setdefault(int(code), []).append({
            "gw": int(gw), "snapshot": r["snapshot"], "vs": vs.get((code, gw), ""),
            "xp": round(float(r["xp"]), 2), "xp_port": round(float(port.get((code, gw), float("nan"))), 2),
            "start": round(float(r["start_pct"]), 3),
            "xg": round(float(r["exp_goals"]), 3), "xa": round(float(r["exp_assists"]), 3),
            "pts": int(r["points"]), "mins": int(r["minutes"]), "started": bool(r["started"]),
            "g": int(r["goals"]), "a": int(r["assists"]),
        })
    return out


def _player_record(
    code: int, g: pd.DataFrame, price: float, past: list[dict] | None, element: dict | None,
) -> dict:
    """One sandbox player: the model view over the window, plus season actuals when live.

    `g` is the player's fixture rows in the window; `element` his bootstrap
    entry (None on the workbook snapshot, which has no live stats).
    """
    first = g.iloc[0]
    by_gw: dict[int, dict] = {}
    for gw, gg in g.groupby("gw"):
        by_gw[int(gw)] = {
            "xp": round(float(gg["TRUE TOTAL"].sum()), 2),
            "vs": list(gg["VS"]),
            "start": round(float(gg["Start %"].mean()), 3),
            "xg": round(float(gg["exp_goals"].sum()), 3),
            "xa": round(float(gg["exp_assists"].sum()), 3),
            "cs": round(float(gg["CS%"].mean()), 3),
        }
    cats = {k: round(float(g[c].sum()), 2) for k, c in _CATEGORY_COLS}
    rec = {
        "code": code, "name": first["FPL_Name"], "team": first["TEAM"],
        "pos": first["POS"], "price": round(float(price), 1),
        "xp": round(float(g["TRUE TOTAL"].sum()), 2),
        "xp_if_start": round(float(g["IF START TOTAL"].sum()), 2),
        "start": round(float(g["Start %"].mean()), 3),
        "gw": {str(k): v for k, v in sorted(by_gw.items())},
        "cats": cats,
    }
    if past:
        rec["past"] = past
    if element:
        e = element
        rec["web"] = e["web_name"]
        rec["status"] = e["status"]
        rec["chance"] = (e["chance_of_playing_next_round"] if e["chance_of_playing_next_round"] is not None else 100) / 100
        rec["news"] = e["news"] or ""
        rec["actual"] = {k: _num_or_str(e.get(f)) for f, k in _ACTUAL_COLS.items()}
        rec["setpiece"] = {
            "pen": e.get("penalties_order"), "fk": e.get("direct_freekicks_order"),
            "ck": e.get("corners_and_indirect_freekicks_order"),
        }
    return rec


def _team_records(
    w: pd.DataFrame, team_form: pd.DataFrame | None, boot: dict | None,
) -> dict[str, dict]:
    """Each club's fixtures per gameweek over the window, form multipliers and (live) league record."""
    teams: dict[str, dict] = {}
    for team, g in w.groupby("TEAM"):
        fx = {str(int(gw)): sorted(set(gg["VS"])) for gw, gg in g.groupby("gw")}
        teams[team] = {"fixtures": fx}
    if team_form is not None:
        for abvr, row in team_form.iterrows():
            t = teams.setdefault(str(abvr), {"fixtures": {}})
            t["form"] = {k: round(float(row[k]), 3) for k in _FORM_COLS if k in row.index}
    if boot:
        for t in boot["teams"]:
            tt = teams.get(t["short_name"])
            if tt is not None:
                tt["name"] = t["name"]
                tt["actual"] = {k: t.get(k) for k in _TEAM_ACTUAL_COLS}
    return teams


def sandbox_tables(
    ppts: pd.DataFrame, prices: pd.Series, from_gw: int, to_gw: int,
    boot: dict | None = None, team_form: pd.DataFrame | None = None,
    with_past: bool = True,
) -> dict:
    """Everything the sandbox needs, keyed by player code and team.

    Pure: no network, no solver. `players` carries the model view (per-GW xP,
    fixtures, start chance, expected goals/assists, category split) and, when
    a bootstrap is given, the season's actual stats. `teams` carries each
    club's fixture list over the window and the in-season form multipliers.
    """
    w = _fixture_rows(ppts, from_gw, to_gw)
    gws = list(range(from_gw, to_gw + 1))
    boot_by_code = {e["code"]: e for e in boot["elements"]} if boot else {}
    past = track_record(ppts) if with_past else {}

    players: dict[int, dict] = {}
    for code, g in w.groupby("Code", sort=False):
        price = prices.get(code)
        if price is None or pd.isna(price):
            continue
        code = int(code)
        players[code] = _player_record(code, g, price, past.get(code), boot_by_code.get(code))
    teams = _team_records(w, team_form, boot)
    return {"gws": gws, "players": players, "teams": teams}


def _num_or_str(v):
    """A bootstrap stat as a JSON number where it parses (ints stay ints, floats to 2dp), else as given."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() else round(f, 2)


def sandbox_payload(entry: int | None, to_gw: int | None, force: bool = False) -> dict:
    """Live sandbox: your squad + the full player/team tables for the window."""

    entry_id = team_mod.resolve_entry(entry)
    boot = api.fetch_bootstrap()
    my = team_mod.load_my_team(entry_id, boot)
    if entry:
        team_mod.remember_entry(entry_id)
    pipe = cached_pipeline(True, force)
    prices = player_prices(True, boot)
    from_gw = my.next_gw
    to_gw = min(to_gw or from_gw + DEFAULT_HORIZON, LAST_GW)  # the season ends at GW38
    tables = sandbox_tables(pipe.ppts, prices, from_gw, to_gw, boot=boot, team_form=pipe.team_form)
    squad = []
    for p in my.picks:
        rec = tables["players"].get(p.code)
        squad.append({
            "code": p.code, "name": p.name, "pos": p.pos, "sell": p.sell,
            "in_model": rec is not None,
        })
    return {
        "entry": my.entry_id, "manager": my.manager, "team_name": my.team_name,
        "overall_rank": my.overall_rank, "total_points": my.total_points,
        "from_gw": from_gw, "to_gw": to_gw,
        "bank": my.bank, "free_transfers": my.free_transfers,
        "hit_cost": HIT_COST, "max_per_club": MAX_PER_CLUB,
        "notes": my.notes,
        "squad": squad, **tables,
    }
