"""The model's reasoning for a deadline plan, written out.

`explain_plan` attaches a `why` list to every transfer in a `TeamPlan` and
fills its `verdict`. The *why* is assembled from tables the pipeline already
has: where the points gap comes from by scoring category, what the move does
to the XI, the fixture run, the team's schedule-adjusted form and style, the
evidence behind the player's rates (last season vs this season vs what the
model used), minutes and rotation risks, the alternatives that lost, and a
verdict on whether the edge clears the noise plus the captain call.

Deterministic prose, no model of its own: every sentence is a readout of a
number, so a wrong sentence means a wrong number upstream.
"""

from __future__ import annotations

import pandas as pd

from .optimizer import SquadSolution
from .plan import TeamPlan, squad_frame
from .predict import gameweek_window
from .rules import DEFAULT_FT_VALUE

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
# --- what is worth a sentence ------------------------------------------------
CATEGORY_DELTA_MIN = 0.3      # xP: a scoring-category swing smaller than this is not named
CATEGORY_DELTAS_SHOWN = 4     # the largest swings listed in "Where the gain comes from"
FORM_IN_LINE = 0.10           # |xG ratio - 1| below this is "in line with expectation"
STYLE_SHARE_NOTABLE = 0.05    # a position's xG/xA share this far from the league norm
STYLE_VENUE_NOTABLE = 0.02    # a home/away multiplier this far from 1
RATE_FLOOR = 0.03             # xG90/xA90 below this in every source: skip the rate
SMALL_SAMPLE_MINS = 900       # last season + this season minutes under this is a thin sample
START_RATE_SHIFT = 0.05       # selection rate moved this much from last season: say so
RIVAL_MIN_STARTS = 10         # last-season starts before a sidelined teammate counts as a rival
ALTERNATIVES_SHOWN = 3        # next-best affordable players listed
PRICED_OUT_REACH = 0.5        # £m above budget to look for a better player
PRICED_OUT_MIN_EDGE = 1       # xP the dearer player must add to be worth mentioning
CAPTAIN_CANDIDATES = 3        # top starters named in the captain line
CAPTAIN_CLEAR_LEAD = 1        # xP gap to the second choice that makes it "a clear call"


def _player_window(ppts: pd.DataFrame, code: int, from_gw: int, to_gw: int) -> pd.DataFrame:
    """One player's fixture rows over the window, blank slots dropped, in (gw, slot) order."""
    window = gameweek_window(ppts, from_gw, to_gw)
    rows = window[window["Code"] == code]
    # the sheet's LEN(VS) >= 3 with NaN kept as "nan" (3 chars) — not predict.has_fixture
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
    """The team's attack and defence vs expectation given the opponents faced, if known."""
    if form is None or team.upper() not in form.index:
        return None
    row = form.loc[team.upper()]
    n = int(row["games"])
    if n == 0:
        return None

    def word(raw: float, good_low: bool) -> str:
        better = raw < 1 if good_low else raw > 1
        if abs(raw - 1) < FORM_IN_LINE:
            return "in line with expectation"
        return f"{abs(raw - 1):.0%} {'better' if better else 'worse'} than expected"

    return (
        f"{team} form, adjusted for the {n} opponents faced: attack {word(row['att_raw'], False)} "
        f"({row['xg']:.1f} xG vs {row['exp_xg']:.1f}), defence {word(row['def_raw'], True)} "
        f"({row['xgc']:.1f} xGC vs {row['exp_xgc']:.1f}). {n} games is thin, so the model moves "
        f"only part-way: attack x{row['att_mult']:.2f}, goals conceded x{row['def_mult']:.2f}."
    )


def _style_line(team: str, pos: str, style) -> str | None:
    """Where the team's chances go and how venue-dependent it is, if notable."""
    team = team.upper()
    if style is None or (team, pos) not in style.attack.index or team not in style.venue.index:
        return None
    attack, league, venue = style.attack.loc[(team, pos)], style.league.loc[pos], style.venue.loc[team]
    bits = []
    for col, raw, label in [("xg_share", "xg_share_raw", "xG"), ("xa_share", "xa_share_raw", "xA")]:
        if abs(attack[raw] - league[col]) >= STYLE_SHARE_NOTABLE:
            bits.append(f"its {pos}s take {attack[raw]:.0%} of team {label} (league {league[col]:.0%})")
    if abs(venue["att_home"] - 1) >= STYLE_VENUE_NOTABLE:
        side = "home" if venue["att_home"] > 1 else "away"
        bits.append(f"attack has been stronger {side} (x{max(venue['att_home'], venue['att_away']):.2f} there after shrinkage)")
    if abs(venue["def_home"] - 1) >= STYLE_VENUE_NOTABLE:
        side = "away" if venue["def_home"] > 1 else "home"
        bits.append(f"defence has been tighter {side}")
    if not bits:
        return None
    return f"{team} style over {int(attack['games'])} games: " + "; ".join(bits) + ". Applied at a fraction of face value until the sample grows."


def _rate_line(row: pd.Series) -> str | None:
    """The player's xG/xA per 90 — last season, this season, position norm, what the model used."""
    if "xG90 used" not in row.index:
        return None
    bits = []
    for tag, label in [("xG90", "xG"), ("xA90", "xA")]:
        if max(row[f"{tag} prior"], row[f"{tag} season"], row[f"{tag} used"]) < RATE_FLOOR:
            continue
        bits.append(
            f"{label}/90 {row[f'{tag} prior']:.2f} last season ({row['LF G+A Mins']:.0f} mins), "
            f"{row[f'{tag} season']:.2f} this season ({row['Season Mins']:.0f} mins), "
            f"position norm {row[f'{tag} group']:.2f} -> model uses {row[f'{tag} used']:.2f}"
        )
    if not bits:
        return None
    line = "Rates: " + "; ".join(bits) + "."
    if row["LF G+A Mins"] + row["Season Mins"] < SMALL_SAMPLE_MINS:
        line += " Small sample — this projection leans on the position norm."
    return line


def _minutes_lines(row: pd.Series, start_pct: float, model: pd.DataFrame, boot: dict | None) -> list[str]:
    """Start chance and its basis, then the rotation risks and set-piece duties the API shows."""
    out = []
    started = f", started {row['Season Starts']:.0f} this season ({row['Season Mins']:.0f} mins)" if "Season Starts" in row.index else ""
    basis = (
        f" (selection rate {row['Start rate prior']:.0%} last season -> {row['Start rate used']:.0%} blended with this one)"
        if "Start rate used" in row.index and abs(row["Start rate used"] - row["Start rate prior"]) >= START_RATE_SHIFT else ""
    )
    out.append(f"Minutes: {start_pct:.0%} start chance{basis}{started}.")
    if not boot:
        return out
    elements = pd.DataFrame(boot["elements"])
    player = elements[elements["code"] == row["Code"]]
    if player.empty:
        return out
    player = player.iloc[0]
    # a rival for the shirt coming back is the rotation risk the rates can't see
    rivals = elements[(elements["team"] == player["team"]) & (elements["element_type"] == player["element_type"])
                      & elements["status"].isin(["i", "d", "s"]) & (elements["code"] != player["code"])]
    ps_starts = model.set_index("Code")["PS Starts"]
    pos2 = model.set_index("Code")["POS2"]
    for rival in rivals.itertuples():
        if ps_starts.get(rival.code, 0) < RIVAL_MIN_STARTS:
            continue
        my_role, his_role = ROLE_FAMILY.get(row["POS2"]), ROLE_FAMILY.get(pos2.get(rival.code))
        if my_role and his_role and my_role != his_role:
            continue
        out.append(f"Rotation risk: {rival.web_name} ({ps_starts[rival.code]:.0f} starts last season) is out — \"{rival.news}\". His return competes for this spot.")
    duties = [
        label for col, label in [
            ("penalties_order", "penalties"), ("direct_freekicks_order", "direct free kicks"),
            ("corners_and_indirect_freekicks_order", "corners"),
        ] if player.get(col) == 1
    ]
    if duties:
        out.append(f"Set pieces: first choice on {', '.join(duties)}.")
    return out


def _starts(sol: SquadSolution, code: int) -> int:
    """How many gameweeks of the window `code` starts in this solution."""
    lineups = sol.lineups
    return int(lineups[(lineups["Code"] == code) & lineups["started"]].shape[0])


def _alternative_lines(
    transfer: dict, players: pd.DataFrame, wxp: pd.Series, owned: set[int], budget: float
) -> list[str]:
    """The next-best affordable players in the slot, and the one just out of reach if clearly better."""
    out: list[str] = []
    pos = transfer["pos"]
    cap = transfer["out_sell"] + budget
    alts = players[(players["POS"] == pos) & (players["price"] <= cap + 1e-9)
                   & ~players.index.isin(owned | {transfer["in_code"]})]
    alts = alts.assign(xp=wxp.reindex(alts.index)).sort_values("xp", ascending=False).head(ALTERNATIVES_SHOWN)
    if len(alts):
        out.append(
            f"Next best {pos}s within £{cap:.1f}m: "
            + ", ".join(f"{r.FPL_Name} ({r.TEAM}, £{r.price:.1f}m, {r.xp:.1f} xP)" for r in alts.itertuples())
            + "."
        )
    dearer = players[(players["POS"] == pos) & (players["price"] > cap) & (players["price"] <= cap + PRICED_OUT_REACH)
                     & ~players.index.isin(owned)]
    dearer = dearer.assign(xp=wxp.reindex(dearer.index)).sort_values("xp", ascending=False).head(1)
    if len(dearer) and dearer["xp"].iloc[0] > transfer["in_xp"] + PRICED_OUT_MIN_EDGE:
        r = dearer.iloc[0]
        out.append(f"Priced out: {r['FPL_Name']} ({r['TEAM']}, £{r['price']:.1f}m) projects {r['xp']:.1f} xP but is £{r['price'] - cap:.1f}m over budget.")
    return out


def _transfer_reasons(
    transfer: dict,
    plan: TeamPlan,
    pipe,
    by_code: pd.DataFrame,
    players: pd.DataFrame,
    wxp: pd.Series,
    budget: float,
) -> list[str]:
    """Every sentence behind one transfer; empty when the incoming player has no fixtures."""
    ppts, model = pipe.ppts, pipe.model
    n_gw = plan.to_gw - plan.from_gw + 1
    why: list[str] = []
    rows_in = _player_window(ppts, transfer["in_code"], plan.from_gw, plan.to_gw)
    rows_out = _player_window(ppts, transfer["out_code"], plan.from_gw, plan.to_gw)
    if rows_in.empty:
        return why
    in_parts, out_parts = category_breakdown(rows_in), category_breakdown(rows_out) if len(rows_out) else {}
    deltas = sorted(((k, in_parts[k] - out_parts.get(k, 0.0)) for k in in_parts), key=lambda kv: -abs(kv[1]))
    top = [f"{k} {d:+.1f}" for k, d in deltas if abs(d) >= CATEGORY_DELTA_MIN][:CATEGORY_DELTAS_SHOWN]
    why.append(f"Where the {transfer['xp_gain']:+.1f} comes from: {', '.join(top) or 'small edges across the board'}.")

    out_pick = next((p for p in plan.team.picks if p.code == transfer["out_code"]), None)
    if out_pick is not None and out_pick.chance < 1:
        why.append(f"{transfer['out']} is flagged at {out_pick.chance:.0%} to play, which discounts every gameweek he is projected for.")

    s_in, s_out = _starts(plan.proposal, transfer["in_code"]), _starts(plan.baseline, transfer["out_code"])
    if s_in != s_out:
        why.append(
            f"XI effect: {transfer['in']} starts {s_in} of {n_gw} gameweeks for you; {transfer['out']} was starting {s_out}. "
            "Only started points count in full, which is why the net gain differs from the raw gap."
        )
    else:
        why.append(f"XI effect: like-for-like — both start {s_in} of {n_gw} gameweeks.")

    per_gw = rows_in.groupby("gw").agg(vs=("VS", "+".join), xp=("TRUE TOTAL", "sum"))
    best = per_gw["xp"].idxmax()
    why.append(
        "Fixtures: " + ", ".join(f"GW{g} {r.vs} {r.xp:.1f}" for g, r in per_gw.iterrows())
        + f". GW{best} alone is {per_gw['xp'].max() / per_gw['xp'].sum():.0%} of the window."
    )

    if (line := _form_line(transfer["in_team"], pipe.team_form)):
        why.append(line)
    if (line := _style_line(transfer["in_team"], transfer["pos"], getattr(pipe, "style", None))):
        why.append(line)
    if transfer["in_code"] in by_code.index:
        row = by_code.loc[transfer["in_code"]]
        if (line := _rate_line(row)):
            why.append(line)
        why.extend(_minutes_lines(row, float(rows_in["Start %"].mean()), model, pipe.boot))

    why.extend(_alternative_lines(transfer, players, wxp, plan.team.codes, budget))
    return why


def _verdict_lines(plan: TeamPlan, ft_value: float) -> list[str]:
    """Make it / worthwhile / marginal, by net gain against the value of a banked transfer."""
    if not plan.transfers or plan.proposal.hit_points:
        return []
    n_gw = plan.to_gw - plan.from_gw + 1
    g = plan.gain
    if g >= STRONG_GAIN:
        return [f"Verdict: make it. +{g:.1f} xP is a clear edge over the ~{ft_value:.1f} xP a banked transfer is worth."]
    if g >= MARGINAL_GAIN:
        return [f"Verdict: worthwhile. +{g:.1f} xP beats the ~{ft_value:.1f} xP value of rolling, but not by a lot."]
    return [
        f"Verdict: marginal. +{g:.1f} xP over {n_gw} gameweeks is inside the model's error bars and close to the "
        f"~{ft_value:.1f} xP a banked transfer is worth — rolling is just as defensible."
    ]


def _captain_line(plan: TeamPlan) -> str | None:
    """The top captain candidates for the next gameweek and whether the call is clear."""
    caps = plan.xi.sort_values("xp", ascending=False).head(CAPTAIN_CANDIDATES)
    if len(caps) < 2:
        return None
    lead = caps["xp"].iloc[0] - caps["xp"].iloc[1]
    return (
        f"Captain GW{plan.from_gw}: " + ", ".join(f"{r.FPL_Name} {r.xp:.1f}" for r in caps.itertuples())
        + (f" — a {lead:.1f} xP lead, so a clear call." if lead >= CAPTAIN_CLEAR_LEAD else f" — only {lead:.1f} xP apart, so a coin flip; go with fixture and form.")
    )


def explain_plan(plan: TeamPlan, pipe, prices: pd.Series, ft_value: float = DEFAULT_FT_VALUE) -> None:
    """Attach `why` (list[str]) to each transfer and fill `plan.verdict`."""
    ppts, model = pipe.ppts, pipe.model
    by_code = model.drop_duplicates("Code").set_index("Code", drop=False)
    players, xp = squad_frame(ppts, prices, plan.from_gw, plan.to_gw)
    wxp = xp.sum(axis=1)

    budget = plan.bank  # the same bank the plan itself was priced against (honours a --bank override)
    for transfer in plan.transfers:
        transfer["why"] = _transfer_reasons(transfer, plan, pipe, by_code, players, wxp, budget)

    verdict = _verdict_lines(plan, ft_value)
    if (line := _captain_line(plan)):
        verdict.append(line)
    plan.verdict = verdict
