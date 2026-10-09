"""Backtester: replay past deadlines, score what the model would have said.

For each finished gameweek the replay takes the last snapshot stored *before*
that deadline, runs the pipeline on it, and joins the projection to what
happened. Variants (the bare port, each in-season layer added in turn, all of
them, any custom constants) run on the same snapshot, so every comparison is
paired: same players, same information, one thing changed.

Leakage is the failure mode that makes a backtest worse than none, so the
rules are structural rather than careful:
- a snapshot qualifies for gameweek g only if its *own* bootstrap marks g as
  the next deadline — no clocks, no file dates;
- the match log is fetched once, current, and cut to rounds < g before the
  style layer sees it;
- data/priors.yaml is off: it is today's judgment, not the model's;
- a gameweek without a qualifying snapshot is reported as skipped, never
  filled from a later one.

What it cannot do: last season (the frozen inputs are end-of-season
aggregates with no point-in-time view), or recover a week no snapshot was
taken. Small-sample honesty is part of the output — every comparison carries
a bootstrap interval, and with a handful of gameweeks most of them straddle
zero. That is the correct answer, not a defect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import api, matchlog, reference
from .inseason import PORT_ONLY, Layers
from .pipeline import base_model_from_bootstrap, build_pipeline
from .rules import GAMEWEEKS

RESULTS_PATH = api.SNAPSHOT_DIR / "backtest.parquet"
UNAVAILABLE = "u"  # bootstrap status of a player who has left the game
DEFAULT_N_BOOT = 300  # bootstrap resamples behind each interval
INTERVAL_QUANTILES = (0.05, 0.95)  # a 90% interval
TOP_N = 10  # the "top10" metric: mean actual points of the N highest projections per (gw, POS)
CALIBRATION_BINS = (0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001)  # right-open; the last catches p == 1

_BASE = replace(PORT_ONLY, priors=False)
VARIANTS: dict[str, Layers] = {
    "port": _BASE,
    "+form": replace(_BASE, form=True),
    "+rates": replace(_BASE, rates=True),
    "+starts": replace(_BASE, starts=True),
    "+style": replace(_BASE, style=True),
    "all": replace(_BASE, form=True, rates=True, starts=True, style=True),
}
# a player counts as "relevant" to a gameweek if any variant projected at
# least this, or he played: keeps the 300 never-play squad fillers from
# flattering every metric with easy zeros
RELEVANT_XP = 1.5


@dataclass
class Snapshot:
    date: str
    boot: dict
    fixtures: list[dict]

    @property
    def next_gw(self) -> int | None:
        return next((int(e["id"]) for e in self.boot["events"] if e.get("is_next")), None)


def load_snapshots(snap_dir: Path | None = None) -> list[Snapshot]:
    """Every stored snapshot, oldest first (folder names are ISO dates).

    A folder without both bootstrap.json and fixtures.parquet — a failed
    pull — is skipped; the fixtures' nested `stats` column is dropped.
    """
    out = []
    for d in sorted(p for p in (snap_dir or api.SNAPSHOT_DIR).iterdir() if p.is_dir()):
        boot, fx = d / "bootstrap.json", d / "fixtures.parquet"
        if not (boot.exists() and fx.exists()):
            continue  # a failed pull's empty folder
        fixtures = pd.read_parquet(fx).drop(columns=["stats"], errors="ignore")
        out.append(Snapshot(d.name, json.loads(boot.read_text()), fixtures.to_dict("records")))
    return out


def snapshot_for(snapshots: list[Snapshot], gw: int) -> Snapshot | None:
    """Latest snapshot taken while `gw` was still the upcoming deadline."""
    ok = [s for s in snapshots if s.next_gw == gw]
    return ok[-1] if ok else None


def actuals(log: pd.DataFrame, gw: int) -> pd.DataFrame:
    """What each player did in `gw`, by code. Absent = did not play = zeros."""
    return log[log["round"] == gw].groupby("code").agg(
        points=("points", "sum"), minutes=("minutes", "sum"), started=("starts", "max"),
        goals=("goals", "sum"), assists=("assists", "sum"),
    )


def _score_variant(
    snap: Snapshot, gw: int, layers: Layers, past: pd.DataFrame, act: pd.DataFrame,
    live: pd.DataFrame, base_model: pd.DataFrame,
) -> pd.DataFrame:
    """One variant's projection for `gw` from `snap`, joined to the actuals:
    one row per player still in the game, with his price at the snapshot."""
    pipe = build_pipeline(
        refresh=True, boot=snap.boot, fixtures_raw=snap.fixtures, layers=layers,
        persist=False, log=past, base_model=base_model, gws=range(gw, gw + 1),
    )
    p = pipe.ppts[(pipe.ppts["gw"] == gw) & (pipe.ppts["VS"].astype(str).str.len() >= 3)]
    pred = p.groupby("Code").agg(
        FPL_Name=("FPL_Name", "first"), TEAM=("TEAM", "first"), POS=("POS", "first"),
        xp=("TRUE TOTAL", "sum"), start_pct=("Start %", "mean"),
        exp_goals=("exp_goals", "sum"), exp_assists=("exp_assists", "sum"),
    )
    pred = pred[pred.index.map(live["status"]).fillna(UNAVAILABLE) != UNAVAILABLE]  # left the game
    pred = pred.join(act, how="left").fillna(
        {"points": 0, "minutes": 0, "started": 0, "goals": 0, "assists": 0}
    )
    pred["price"] = pred.index.map(live["now_cost"]) / 10
    return pred


def replay(
    snapshots: list[Snapshot], log: pd.DataFrame, variants: dict[str, Layers] | None = None,
    gws: list[int] | None = None, finished: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Long table (variant, gw, Code, ...) plus {gw: why it was skipped}.

    `finished` is the set of completed gameweeks (a round in progress can't be
    scored); defaults to the newest snapshot's view.
    """
    variants = variants or VARIANTS
    rounds = sorted({int(r) for r in log["round"].unique()})
    done = finished if finished is not None else (
        {int(e["id"]) for e in snapshots[-1].boot["events"] if e.get("finished")} if snapshots else set()
    )
    rows, skipped = [], {}
    for gw in gws or rounds:
        if gw not in done:
            skipped[gw] = "gameweek not finished"
            continue
        snap = snapshot_for(snapshots, gw)
        if snap is None:
            skipped[gw] = "no snapshot taken before the deadline"
            continue
        past = log[log["round"] < gw]
        act = actuals(log, gw)
        live = pd.DataFrame(snap.boot["elements"]).set_index("code")
        # build_model doesn't depend on the layers: once per snapshot, not per variant
        base_model = base_model_from_bootstrap(snap.boot)
        for name, layers in variants.items():
            pred = _score_variant(snap, gw, layers, past, act, live, base_model)
            rows.append(pred.reset_index().assign(variant=name, gw=gw, snapshot=snap.date))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return table, skipped


def add_naive_baseline(table: pd.DataFrame) -> pd.DataFrame:
    """'naive' variant: last season's points per game, nothing else. A model
    that can't out-rank this isn't earning its complexity."""
    if table.empty:
        return table
    ps = reference.load("PSAPI")
    ppg = pd.Series(
        pd.to_numeric(ps["total_points"], errors="coerce").values / GAMEWEEKS,
        index=pd.to_numeric(ps["code"], errors="coerce"),
    ).groupby(level=0).first()
    base = table[table["variant"] == table["variant"].iloc[0]].copy()
    base["variant"] = "naive"
    base["xp"] = base["Code"].map(ppg).fillna(0.0)
    base[["start_pct", "exp_goals", "exp_assists"]] = np.nan
    return pd.concat([table, base], ignore_index=True)


def _relevant(table: pd.DataFrame) -> pd.DataFrame:
    """Flag the pool the points metrics are scored on. Selecting on "he
    played" is fine for comparing projections of points, but it would rig any
    probability metric (a 0% start chance only enters the pool when it was
    wrong), so those are scored on everyone — see score()."""
    model_rows = table[table["variant"] != "naive"]
    best = model_rows.groupby(["gw", "Code"])["xp"].max()
    played = model_rows.groupby(["gw", "Code"])["minutes"].max() > 0
    keep = (best >= RELEVANT_XP) | played
    idx = pd.MultiIndex.from_frame(table[["gw", "Code"]])
    return table.assign(relevant=keep.reindex(idx).fillna(False).values)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation; NaN when there are too few rows or either side is constant."""
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def _brier(p: pd.Series, y: pd.Series) -> float:
    """Mean squared error of probabilities `p` against 0/1 outcomes `y`
    (lower is better); NaN when the variant carries no probabilities."""
    return float(((p - y) ** 2).mean()) if p.notna().any() else np.nan


def score(d: pd.DataFrame) -> dict[str, float]:
    """All metrics for one variant's rows: points metrics on the relevant
    pool, probability metrics on every player in the game."""
    full, d = d, d[d["relevant"]]
    by_gw_pos = [g for _, g in d.groupby(["gw", "POS"])]
    top = [g.nlargest(TOP_N, "xp")["points"].mean() for g in by_gw_pos if len(g) >= TOP_N]
    p_goal, p_assist = 1 - np.exp(-full["exp_goals"]), 1 - np.exp(-full["exp_assists"])
    return {
        "mae": float((d["xp"] - d["points"]).abs().mean()),
        "bias": float((d["xp"] - d["points"]).mean()),
        "rank": float(np.nanmean([_spearman(g["xp"], g["points"]) for g in by_gw_pos])),
        "top10": float(np.mean(top)) if top else np.nan,
        "start_brier": _brier(full["start_pct"], (full["started"] > 0).astype(float)),
        "goal_brier": _brier(p_goal, (full["goals"] > 0).astype(float)),
        "assist_brier": _brier(p_assist, (full["assists"] > 0).astype(float)),
    }


# lower is better for these; the rest are higher-is-better
LOWER_BETTER = {"mae", "start_brier", "goal_brier", "assist_brier"}


def evaluate(
    table: pd.DataFrame, reference_variant: str = "port", n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> pd.DataFrame:
    """Metrics per variant plus a paired bootstrap interval on the difference
    from `reference_variant`. Resampling is over players within each gameweek,
    the same draw for every variant — the pairing is what gives a handful of
    gameweeks any power at all. (It does not capture week-to-week variation:
    with few gameweeks the intervals are, if anything, too narrow.)
    """
    d = _relevant(table)
    wide = {v: g.set_index(["gw", "Code"]).sort_index() for v, g in d.groupby("variant")}
    ref = wide[reference_variant]
    rng = np.random.default_rng(seed)
    pos_by_gw = {gw: np.flatnonzero(ref.index.get_level_values("gw") == gw) for gw in ref.index.unique("gw")}
    draws = [np.concatenate([rng.choice(ix, len(ix)) for ix in pos_by_gw.values()]) for _ in range(n_boot)]

    out = []
    for name, w in wide.items():
        w = w.reindex(ref.index)
        point = score(w.reset_index())
        row = {"variant": name, "n": int(w["relevant"].sum()), "gws": ref.index.get_level_values("gw").nunique(), **point}
        if name != reference_variant:
            boots = pd.DataFrame([
                {k: a - b for (k, a), b in zip(score(w.iloc[ix].reset_index()).items(), score(ref.iloc[ix].reset_index()).values())}
                for ix in draws
            ])
            for k in point:
                lo, hi = boots[k].quantile(list(INTERVAL_QUANTILES))
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out).set_index("variant")


def calibration(table: pd.DataFrame, variant: str, col: str = "start_pct", outcome: str = "started",
                bins: tuple = CALIBRATION_BINS) -> pd.DataFrame:
    """Predicted probability bucket vs how often it happened (all players)."""
    d = table[table["variant"] == variant]
    p = d[col] if col == "start_pct" else 1 - np.exp(-d[col])
    g = pd.DataFrame({"p": p, "y": (d[outcome] > 0).astype(float)}).dropna()
    g["bucket"] = pd.cut(g["p"], list(bins), right=False)
    return g.groupby("bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))


def verdicts(ev: pd.DataFrame, reference_variant: str = "port") -> list[str]:
    """Plain-language read of the intervals: better / worse / can't tell."""
    out = []
    metrics = [c for c in ev.columns if f"{c}_lo" in ev.columns]
    for name, r in ev.iterrows():
        if name == reference_variant or pd.isna(r.get("mae_lo")):
            continue
        better, worse = [], []
        for k in metrics:
            lo, hi = r[f"{k}_lo"], r[f"{k}_hi"]
            if pd.isna(lo):
                continue
            if k in LOWER_BETTER:
                lo, hi = -hi, -lo
            (better if lo > 0 else worse if hi < 0 else []).append(k)
        if not better and not worse:
            out.append(f"{name}: no metric distinguishable from {reference_variant} at this sample size.")
        else:
            out.append(f"{name}: " + "; ".join(
                s for s in [f"better on {', '.join(better)}" if better else "", f"worse on {', '.join(worse)}" if worse else ""] if s
            ) + " (90% interval excludes zero); the rest can't be told apart.")
    return out


def run(
    variants: dict[str, Layers] | None = None, gws: list[int] | None = None, n_boot: int = DEFAULT_N_BOOT,
) -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame]:
    """The weekly backtest: replay every finished gameweek from its stored
    snapshot against the live match log, add the naive baseline, store the
    long table at RESULTS_PATH and evaluate it.

    Returns (table, skipped, evaluation); the evaluation is empty when no
    gameweek could be scored, and nothing is stored in that case.
    """
    snaps = load_snapshots()
    if not snaps:
        raise SystemExit("no snapshots to replay — run `fpl squad --refresh` or the weekly job first")
    boot = api.fetch_bootstrap()
    log = matchlog.load_match_log(boot, api.fetch_fixtures())
    finished = {int(e["id"]) for e in boot["events"] if e.get("finished")}
    table, skipped = replay(snaps, log, variants, gws, finished)
    if table.empty:
        return table, skipped, pd.DataFrame()
    table = add_naive_baseline(table)
    table.to_parquet(RESULTS_PATH)
    return table, skipped, evaluate(table, n_boot=n_boot)
