"""Local web UI: a thin browser layer over the prediction + optimizer pipeline.

`fpl web` serves a single-page app on localhost. Standard-library HTTP server
only — this is a single-user local tool, so no framework dependency is
warranted. The payload builders are plain functions (testable without a
server); the handler just maps URLs onto them.

Projections are cached per data source (workbook snapshot vs live pull) for
the life of the process; the Refresh button re-pulls the live API.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

HTML_PATH = Path(__file__).parent / "web.html"

_cache: dict = {}  # refresh flag -> cli.Pipeline
_cache_lock = threading.Lock()


def _pipeline(refresh: bool, force: bool = False):
    from .cli import build_pipeline

    with _cache_lock:
        if force or refresh not in _cache:
            _cache[refresh] = build_pipeline(refresh=refresh)
        return _cache[refresh]


def _ppts(refresh: bool, force: bool = False) -> pd.DataFrame:
    return _pipeline(refresh, force).ppts


def _prices(refresh: bool) -> pd.Series:
    from .cli import _prices as cli_prices

    return cli_prices(refresh)


def predictions_payload(
    from_gw: int, to_gw: int, refresh: bool, pos: str | None = None,
    team: str | None = None, max_price: float | None = None,
    sort: str = "pts", top: int = 50, force: bool = False,
) -> dict:
    from .predict import aggregate_window

    ppts = _ppts(refresh, force)
    pred = aggregate_window(ppts, from_gw, to_gw)
    pred["price"] = pred["Code"].map(_prices(refresh))
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
    from .cli import _resolve_codes
    from .optimizer import optimize_squad, select_pool

    ppts = _ppts(refresh, force)
    window = ppts[(ppts["gw"] >= from_gw) & (ppts["gw"] <= to_gw)]
    xp = window.pivot_table(index="Code", columns="gw", values="TRUE TOTAL", aggfunc="sum")
    players = (
        window.groupby("Code")[["FPL_Name", "TEAM", "POS"]]
        .first()
        .join(_prices(refresh).rename("price"))
    )
    players = players[players["price"].notna()]

    lock_codes = _resolve_codes(locks, players)
    exclude_codes = _resolve_codes(excludes, players)
    pool = select_pool(players, xp.sum(axis=1), keep=lock_codes)
    sol = optimize_squad(
        players=pool, xp=xp, bench_weight=bench_weight,
        locks=lock_codes, excludes=exclude_codes,
    )

    first = sol.lineups[sol.lineups["gw"] == from_gw]
    xi = first[first["started"]]
    formation = "-".join(str((xi["POS"] == p).sum()) for p in ["DEF", "MID", "FWD"])
    captains = sol.lineups[sol.lineups["captain"]].sort_values("gw")
    return {
        "status": sol.status,
        "cost": round(sol.cost, 1),
        "expected_points": round(sol.expected_points, 1),
        "squad": json.loads(sol.squad.round(2).to_json(orient="records")),
        "formation": formation,
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
    force: bool = False, ft_value: float = 1.5,
) -> dict:
    """Your team: reconstructed state + the deadline plan (always live data)."""
    from . import api, explain, team as team_mod
    from .cli import _prices as cli_prices, _resolve_codes
    from .plan import plan_transfers

    entry_id = team_mod.resolve_entry(entry)
    boot = api.fetch_bootstrap()
    my = team_mod.load_my_team(entry_id, boot)
    if entry:
        team_mod.remember_entry(entry_id)  # only once it resolved to a real team
    pipe = _pipeline(True, force)
    ppts, prices = pipe.ppts, cli_prices(True, boot)
    names = ppts.groupby("Code")[["FPL_Name"]].first()
    plan = plan_transfers(
        ppts, prices, my, to_gw=to_gw, bench_weight=bench_weight,
        free_transfers=ft, bank=bank, max_transfers=max_transfers, hit_cost=hit_cost,
        ft_value=ft_value, wildcard=wildcard, locks=_resolve_codes(locks, names),
        excludes=_resolve_codes(excludes, names),
    )
    explain.explain_plan(plan, pipe, prices, ft_value=ft_value)
    return plan_to_json(plan)


def plan_to_json(plan) -> dict:
    my = plan.team
    caps = plan.proposal.lineups[plan.proposal.lineups["captain"]].sort_values("gw")
    return {
        "entry": my.entry_id, "manager": my.manager, "team_name": my.team_name,
        "overall_rank": my.overall_rank, "total_points": my.total_points,
        "gw": my.gw, "from_gw": plan.from_gw, "to_gw": plan.to_gw,
        "bank": my.bank, "squad_value": my.squad_value,
        "free_transfers": my.free_transfers,
        "chips_available": my.chips_available, "chips_used": my.chips_used,
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
        "formation": "-".join(str((plan.xi["POS"] == p).sum()) for p in ["DEF", "MID", "FWD"]),
        "xi_xp": round(float(plan.xi["xp"].sum()), 1),
        "xi": json.loads(
            plan.xi[["FPL_Name", "TEAM", "POS", "xp", "captain", "vice"]].round(2).to_json(orient="records")
        ),
        "bench": json.loads(plan.bench[["FPL_Name", "POS", "xp"]].round(2).to_json(orient="records")),
        "captains": json.loads(caps[["gw", "FPL_Name", "TEAM"]].to_json(orient="records")),
    }


def verdict_payload(name: str, from_gw: int, to_gw: int, refresh: bool) -> dict:
    """The model's opinion on one player, written out.

    Deterministic prose assembled from the signals the pipeline already
    computes: window xP and ranks, minutes security, value, fixture shape,
    and confidence caveats (injury flags, hand-set priors).
    """
    from .predict import aggregate_window
    from .priors import load_priors

    ppts = _ppts(refresh)
    pred = aggregate_window(ppts, from_gw, to_gw)
    pred["price"] = pred["Code"].map(_prices(refresh))
    pred = pred[pred["price"].notna()].copy()

    hits = pred[pred["FPL_Name"].str.contains(name, case=False, na=False)]
    if hits.empty:
        raise ValueError(f"no player matching {name!r}")
    if hits["FPL_Name"].nunique() > 1:
        raise ValueError(f"ambiguous — matches: {', '.join(hits['FPL_Name'].unique()[:8])}")
    row = hits.iloc[0]

    pred["rank"] = pred["true_pts"].rank(ascending=False, method="min")
    pos_pool = pred[pred["POS"] == row["POS"]].copy()
    pos_pool["pos_rank"] = pos_pool["true_pts"].rank(ascending=False, method="min")
    pos_pool["value"] = (pos_pool["true_pts"] / pos_pool["price"]).where(pos_pool["price"] > 0, 0)
    me = pos_pool[pos_pool["Code"] == row["Code"]].iloc[0]
    rank = int(pred.loc[pred["Code"] == row["Code"], "rank"].iloc[0])
    pos_rank = int(me["pos_rank"])
    value_rank = int(pos_pool["value"].rank(ascending=False, method="min")[me.name])

    mine = ppts[(ppts["Code"] == row["Code"]) & (ppts["gw"] >= from_gw) & (ppts["gw"] <= to_gw)]
    mine = mine[mine["VS"].str.len() >= 3].sort_values(["gw", "slot"])
    start_pct = float(mine["Start %"].mean()) if len(mine) else 0.0
    fixtures = [
        {"gw": int(r["gw"]), "vs": r["VS"], "xp": round(float(r["TRUE TOTAL"]), 2)}
        for _, r in mine.iterrows()
    ]
    dgws = mine.groupby("gw").size()
    doubles = [int(g) for g, n in dgws.items() if n > 1]

    prior = next(
        (p for p in load_priors() if str(p["player"]).lower() in str(row["FPL_Name"]).lower()),
        None,
    )

    v: list[str] = []
    n_pos = len(pos_pool)
    if rank <= 5:
        v.append(f"One of the model's elite picks: #{rank} overall for this window.")
    elif rank <= 25:
        v.append(f"A strong pick — #{rank} overall, #{pos_rank} of {n_pos} {row['POS']}s.")
    elif pos_rank <= 15:
        v.append(f"Solid positional option: #{pos_rank} of {n_pos} {row['POS']}s (#{rank} overall).")
    else:
        v.append(f"Fringe by the numbers: #{pos_rank} of {n_pos} {row['POS']}s (#{rank} overall).")

    if start_pct >= 0.85:
        v.append(f"Minutes look nailed (avg start probability {start_pct:.0%}).")
    elif start_pct >= 0.6:
        v.append(f"Minutes fairly secure but not certain ({start_pct:.0%} start probability).")
    elif start_pct >= 0.35:
        v.append(f"Real rotation risk — {start_pct:.0%} start probability; check team news before owning.")
    else:
        v.append(f"Serious minutes doubt ({start_pct:.0%} start probability) — avoid until that changes.")

    if value_rank <= 10 and pos_rank > 5:
        v.append(f"Good value at £{row['price']:.1f}m — #{value_rank} points-per-£ among {row['POS']}s.")
    elif value_rank > n_pos * 0.7:
        v.append(f"Poor value at £{row['price']:.1f}m — you pay for the points you get.")

    if len(fixtures) >= 4:
        half = len(fixtures) // 2
        early = sum(f["xp"] for f in fixtures[:half]) / half
        late = sum(f["xp"] for f in fixtures[half:]) / (len(fixtures) - half)
        if early > late * 1.25:
            v.append("Fixtures are front-loaded — the good weeks come early in this window.")
        elif late > early * 1.25:
            v.append("Fixtures improve later in the window — patience may beat chasing.")
    if doubles:
        v.append(f"Double gameweek in GW{', GW'.join(map(str, doubles))} — extra fixture value.")

    if rank <= 3:
        v.append("Captaincy material in the best weeks.")

    if prior:
        v.append(
            "Low confidence: no Premier League history — this projection rests on the "
            "hand-set prior in data/priors.yaml"
            + (f" ({prior.get('note', '').strip()})" if prior.get("note") else "")
            + ". Trust your read of team news over the number."
        )

    return {
        "name": row["FPL_Name"], "team": row["TEAM"], "pos": row["POS"],
        "price": round(float(row["price"]), 1),
        "window": [from_gw, to_gw],
        "xp": round(float(row["true_pts"]), 1),
        "rank": rank, "pos_rank": pos_rank, "pos_count": n_pos,
        "value_rank": value_rank,
        "start_pct": round(start_pct, 2),
        "fixtures": fixtures,
        "verdict": v,
    }


def compare_payload(a: str, b: str, from_gw: int, to_gw: int, refresh: bool) -> dict:
    """Side-by-side: the model's pick vs a user-suggested alternative."""
    va = verdict_payload(a, from_gw, to_gw, refresh)
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


def player_payload(name: str, from_gw: int, to_gw: int, refresh: bool) -> dict:
    ppts = _ppts(refresh)
    rows = ppts[ppts["FPL_Name"].str.contains(name, case=False, na=False)]
    names = rows["FPL_Name"].unique()
    if len(names) == 0:
        raise ValueError(f"no player matching {name!r}")
    if len(names) > 1:
        raise ValueError(f"ambiguous — matches: {', '.join(names[:8])}")
    rows = rows[(rows["gw"] >= from_gw) & (rows["gw"] <= to_gw)]
    rows = rows[rows["VS"].str.len() >= 3]
    cols = {
        "gw": "gw", "VS": "vs", "8": "goals", "7": "assists", "9": "cs",
        "10": "gc2", "11": "dc", "6": "bonus", "Start %": "start_pct",
        "IF START TOTAL": "if_start", "TRUE TOTAL": "true_total",
    }
    out = rows[list(cols)].rename(columns=cols).round(2)
    return {"name": names[0], "rows": json.loads(out.to_json(orient="records"))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/predictions":
                self._json(200, predictions_payload(
                    int(q.get("from", 1)), int(q.get("to", 8)),
                    q.get("refresh") == "1", q.get("pos") or None,
                    q.get("team") or None,
                    float(q["max_price"]) if q.get("max_price") else None,
                    q.get("sort", "pts"), int(q.get("top", 50)),
                    q.get("force") == "1",
                ))
            elif url.path == "/api/squad":
                names = lambda k: [s for s in q.get(k, "").split(",") if s.strip()]  # noqa: E731
                self._json(200, squad_payload(
                    int(q.get("from", 1)), int(q.get("to", 8)),
                    q.get("refresh") == "1", float(q.get("bench_weight", 0.1)),
                    names("lock"), names("exclude"), q.get("force") == "1",
                ))
            elif url.path == "/api/team":
                names = lambda k: [s for s in q.get(k, "").split(",") if s.strip()]  # noqa: E731
                opt = lambda k, f: f(q[k]) if q.get(k) not in (None, "") else None  # noqa: E731
                self._json(200, team_payload(
                    opt("entry", int), opt("to", int), float(q.get("bench_weight", 0.1)),
                    opt("ft", int), opt("bank", float), opt("max_transfers", int),
                    float(q.get("hit_cost", 4.0)), q.get("wildcard") == "1",
                    names("lock"), names("exclude"), q.get("force") == "1",
                    float(q.get("ft_value", 1.5)),
                ))
            elif url.path == "/api/compare":
                self._json(200, compare_payload(
                    q.get("a", ""), q.get("b", ""), int(q.get("from", 1)),
                    int(q.get("to", 8)), q.get("refresh") == "1",
                ))
            elif url.path == "/api/verdict":
                self._json(200, verdict_payload(
                    q.get("name", ""), int(q.get("from", 1)),
                    int(q.get("to", 8)), q.get("refresh") == "1",
                ))
            elif url.path == "/api/player":
                self._json(200, player_payload(
                    q.get("name", ""), int(q.get("from", 1)),
                    int(q.get("to", 38)), q.get("refresh") == "1",
                ))
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, SystemExit) as e:  # bad input / ambiguous name
            self._json(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover - surfaced in the UI
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"fpl web UI at {url} (ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
