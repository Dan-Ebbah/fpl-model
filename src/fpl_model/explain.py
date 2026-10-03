"""The model's reasoning for a deadline plan, written out.

`fpl team` used to print a transfer and a number. This assembles the *why*
from tables the pipeline already has: where the points gap comes from by
scoring category, what the move does to the XI, the fixture run, the team's
schedule-adjusted form, the evidence behind the player's rates (last season
vs this season vs what the model used), minutes risks, the alternatives that
lost, and a verdict on whether the edge clears the noise.

Deterministic prose, no model of its own: every sentence is a readout of a
number, so a wrong sentence means a wrong number upstream.
"""

from __future__ import annotations

import pandas as pd

from .plan import TeamPlan, squad_frame

# if-start category columns -> the label a manager would use
CATEGORIES = {
    "goals": ["8"], "assists": ["7"], "clean sheets": ["9", "10"],
    "defensive contribution": ["11"], "bonus": ["6"], "saves": ["3"], "cards": ["4", "5"],
}
# FBRef role -> the shirts that actually compete with each other
ROLE_FAMILY = {
    "CB": "back", "LB": "back", "RB": "back", "DM": "mid", "CM": "mid",
    "AM": "att", "LW": "att", "RW": "att", "LM": "att", "RM": "att", "CF": "cf",
}
# net gain below this is inside the model's error bars
MARGINAL_GAIN, STRONG_GAIN = 3.0, 6.0


def _window_rows(ppts: pd.DataFrame, code: int, from_gw: int, to_gw: int) -> pd.DataFrame:
    rows = ppts[(ppts["Code"] == code) & (ppts["gw"] >= from_gw) & (ppts["gw"] <= to_gw)]
    return rows[rows["VS"].astype(str).str.len() >= 3].sort_values(["gw", "slot"])


def category_breakdown(rows: pd.DataFrame) -> dict[str, float]:
    """Window xP split by scoring category; sums to TRUE TOTAL.

    Category columns are if-start values, so each is weighted by start chance
    and the home/away multiplier; what is left of TRUE TOTAL is appearance
    points plus sub cameos, reported as "minutes".
    """
    weight = rows["Start %"] * rows["H/A"]
    out = {label: float((rows[cols].sum(axis=1) * weight).sum()) for label, cols in CATEGORIES.items()}
    out["minutes"] = float(rows["TRUE TOTAL"].sum()) - sum(out.values())
    return out


def _form_line(team: str, form: pd.DataFrame | None) -> str | None:
    if form is None or team.upper() not in form.index:
        return None
    f = form.loc[team.upper()]
    n = int(f["games"])
    if n == 0:
        return None

    def word(raw: float, good_low: bool) -> str:
        better = raw < 1 if good_low else raw > 1
        if abs(raw - 1) < 0.10:
            return "in line with expectation"
        return f"{abs(raw - 1):.0%} {'better' if better else 'worse'} than expected"

    return (
        f"{team} form, adjusted for the {n} opponents faced: attack {word(f['att_raw'], False)} "
        f"({f['xg']:.1f} xG vs {f['exp_xg']:.1f}), defence {word(f['def_raw'], True)} "
        f"({f['xgc']:.1f} xGC vs {f['exp_xgc']:.1f}). {n} games is thin, so the model moves "
        f"only part-way: attack x{f['att_mult']:.2f}, goals conceded x{f['def_mult']:.2f}."
    )


def _style_line(team: str, pos: str, style) -> str | None:
    """Where the team's chances go and how venue-dependent it is, if notable."""
    team = team.upper()
    if style is None or (team, pos) not in style.attack.index or team not in style.venue.index:
        return None
    a, lg, v = style.attack.loc[(team, pos)], style.league.loc[pos], style.venue.loc[team]
    bits = []
    for col, raw, label in [("xg_share", "xg_share_raw", "xG"), ("xa_share", "xa_share_raw", "xA")]:
        if abs(a[raw] - lg[col]) >= 0.05:
            bits.append(f"its {pos}s take {a[raw]:.0%} of team {label} (league {lg[col]:.0%})")
    if abs(v["att_home"] - 1) >= 0.02:
        side = "home" if v["att_home"] > 1 else "away"
        bits.append(f"attack has been stronger {side} (x{max(v['att_home'], v['att_away']):.2f} there after shrinkage)")
    if abs(v["def_home"] - 1) >= 0.02:
        side = "away" if v["def_home"] > 1 else "home"
        bits.append(f"defence has been tighter {side}")
    if not bits:
        return None
    return f"{team} style over {int(a['games'])} games: " + "; ".join(bits) + ". Applied at a fraction of face value until the sample grows."


def _rate_line(m: pd.Series) -> str | None:
    if "xG90 used" not in m.index:
        return None
    bits = []
    for tag, label in [("xG90", "xG"), ("xA90", "xA")]:
        if max(m[f"{tag} prior"], m[f"{tag} season"], m[f"{tag} used"]) < 0.03:
            continue
        bits.append(
            f"{label}/90 {m[f'{tag} prior']:.2f} last season ({m['LF G+A Mins']:.0f} mins), "
            f"{m[f'{tag} season']:.2f} this season ({m['Season Mins']:.0f} mins), "
            f"position norm {m[f'{tag} group']:.2f} -> model uses {m[f'{tag} used']:.2f}"
        )
    if not bits:
        return None
    line = "Rates: " + "; ".join(bits) + "."
    if m["LF G+A Mins"] + m["Season Mins"] < 900:
        line += " Small sample — this projection leans on the position norm."
    return line


def _minutes_lines(m: pd.Series, start_pct: float, model: pd.DataFrame, boot: dict | None) -> list[str]:
    out = []
    started = f", started {m['Season Starts']:.0f} this season ({m['Season Mins']:.0f} mins)" if "Season Starts" in m.index else ""
    basis = (
        f" (selection rate {m['Start rate prior']:.0%} last season -> {m['Start rate used']:.0%} blended with this one)"
        if "Start rate used" in m.index and abs(m["Start rate used"] - m["Start rate prior"]) >= 0.05 else ""
    )
    out.append(f"Minutes: {start_pct:.0%} start chance{basis}{started}.")
    if not boot:
        return out
    el = pd.DataFrame(boot["elements"])
    me = el[el["code"] == m["Code"]]
    if me.empty:
        return out
    me = me.iloc[0]
    # a rival for the shirt coming back is the rotation risk the rates can't see
    rivals = el[(el["team"] == me["team"]) & (el["element_type"] == me["element_type"])
                & el["status"].isin(["i", "d", "s"]) & (el["code"] != me["code"])]
    ps_starts = model.set_index("Code")["PS Starts"]
    pos2 = model.set_index("Code")["POS2"]
    for r in rivals.itertuples():
        if ps_starts.get(r.code, 0) < 10:
            continue
        mine, his = ROLE_FAMILY.get(m["POS2"]), ROLE_FAMILY.get(pos2.get(r.code))
        if mine and his and mine != his:
            continue
        out.append(f"Rotation risk: {r.web_name} ({ps_starts[r.code]:.0f} starts last season) is out — \"{r.news}\". His return competes for this spot.")
    duties = [
        label for col, label in [
            ("penalties_order", "penalties"), ("direct_freekicks_order", "direct free kicks"),
            ("corners_and_indirect_freekicks_order", "corners"),
        ] if me.get(col) == 1
    ]
    if duties:
        out.append(f"Set pieces: first choice on {', '.join(duties)}.")
    return out


def explain_plan(plan: TeamPlan, pipe, prices: pd.Series, ft_value: float = 1.5) -> None:
    """Attach `why` (list[str]) to each transfer and fill `plan.verdict`."""
    ppts, model = pipe.ppts, pipe.model
    by_code = model.drop_duplicates("Code").set_index("Code", drop=False)
    players, xp = squad_frame(ppts, prices, plan.from_gw, plan.to_gw)
    wxp = xp.sum(axis=1)
    n_gw = plan.to_gw - plan.from_gw + 1

    def starts(sol, code: int) -> int:
        lu = sol.lineups
        return int(lu[(lu["Code"] == code) & lu["started"]].shape[0])

    budget = plan.team.bank if plan.team.bank is not None else 0.0
    for t in plan.transfers:
        why: list[str] = []
        rin = _window_rows(ppts, t["in_code"], plan.from_gw, plan.to_gw)
        rout = _window_rows(ppts, t["out_code"], plan.from_gw, plan.to_gw)
        if rin.empty:
            t["why"] = why
            continue
        bin_, bout = category_breakdown(rin), category_breakdown(rout) if len(rout) else {}
        deltas = sorted(((k, bin_[k] - bout.get(k, 0.0)) for k in bin_), key=lambda kv: -abs(kv[1]))
        top = [f"{k} {d:+.1f}" for k, d in deltas if abs(d) >= 0.3][:4]
        why.append(f"Where the {t['xp_gain']:+.1f} comes from: {', '.join(top) or 'small edges across the board'}.")

        out_pick = next((p for p in plan.team.picks if p.code == t["out_code"]), None)
        if out_pick is not None and out_pick.chance < 1:
            why.append(f"{t['out']} is flagged at {out_pick.chance:.0%} to play, which discounts every gameweek he is projected for.")

        s_in, s_out = starts(plan.proposal, t["in_code"]), starts(plan.baseline, t["out_code"])
        if s_in != s_out:
            why.append(
                f"XI effect: {t['in']} starts {s_in} of {n_gw} gameweeks for you; {t['out']} was starting {s_out}. "
                "Only started points count in full, which is why the net gain differs from the raw gap."
            )
        else:
            why.append(f"XI effect: like-for-like — both start {s_in} of {n_gw} gameweeks.")

        per_gw = rin.groupby("gw").agg(vs=("VS", "+".join), xp=("TRUE TOTAL", "sum"))
        best = per_gw["xp"].idxmax()
        why.append(
            "Fixtures: " + ", ".join(f"GW{g} {r.vs} {r.xp:.1f}" for g, r in per_gw.iterrows())
            + f". GW{best} alone is {per_gw['xp'].max() / per_gw['xp'].sum():.0%} of the window."
        )

        if (line := _form_line(t["in_team"], pipe.team_form)):
            why.append(line)
        if (line := _style_line(t["in_team"], t["pos"], getattr(pipe, "style", None))):
            why.append(line)
        if t["in_code"] in by_code.index:
            m = by_code.loc[t["in_code"]]
            if (line := _rate_line(m)):
                why.append(line)
            why.extend(_minutes_lines(m, float(rin["Start %"].mean()), model, pipe.boot))

        cap = t["out_sell"] + budget
        alts = players[(players["POS"] == t["pos"]) & (players["price"] <= cap + 1e-9)
                       & ~players.index.isin(plan.team.codes | {t["in_code"]})]
        alts = alts.assign(xp=wxp.reindex(alts.index)).sort_values("xp", ascending=False).head(3)
        if len(alts):
            why.append(
                f"Next best {t['pos']}s within £{cap:.1f}m: "
                + ", ".join(f"{r.FPL_Name} ({r.TEAM}, £{r.price:.1f}m, {r.xp:.1f} xP)" for r in alts.itertuples())
                + "."
            )
        dearer = players[(players["POS"] == t["pos"]) & (players["price"] > cap) & (players["price"] <= cap + 0.5)
                         & ~players.index.isin(plan.team.codes)]
        dearer = dearer.assign(xp=wxp.reindex(dearer.index)).sort_values("xp", ascending=False).head(1)
        if len(dearer) and dearer["xp"].iloc[0] > t["in_xp"] + 1:
            r = dearer.iloc[0]
            why.append(f"Priced out: {r['FPL_Name']} ({r['TEAM']}, £{r['price']:.1f}m) projects {r['xp']:.1f} xP but is £{r['price'] - cap:.1f}m over budget.")
        t["why"] = why

    v: list[str] = []
    if plan.transfers and not plan.proposal.hit_points:
        g = plan.gain
        if g >= STRONG_GAIN:
            v.append(f"Verdict: make it. +{g:.1f} xP is a clear edge over the ~{ft_value:.1f} xP a banked transfer is worth.")
        elif g >= MARGINAL_GAIN:
            v.append(f"Verdict: worthwhile. +{g:.1f} xP beats the ~{ft_value:.1f} xP value of rolling, but not by a lot.")
        else:
            v.append(
                f"Verdict: marginal. +{g:.1f} xP over {n_gw} gameweeks is inside the model's error bars and close to the "
                f"~{ft_value:.1f} xP a banked transfer is worth — rolling is just as defensible."
            )
    caps = plan.xi.sort_values("xp", ascending=False).head(3)
    if len(caps) >= 2:
        lead = caps["xp"].iloc[0] - caps["xp"].iloc[1]
        v.append(
            f"Captain GW{plan.from_gw}: " + ", ".join(f"{r.FPL_Name} {r.xp:.1f}" for r in caps.itertuples())
            + (f" — a {lead:.1f} xP lead, so a clear call." if lead >= 1 else f" — only {lead:.1f} xP apart, so a coin flip; go with fixture and form.")
        )
    plan.verdict = v
