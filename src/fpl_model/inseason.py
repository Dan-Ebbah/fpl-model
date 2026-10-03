"""In-season adjustment layer: current-season evidence on top of the port.

The workbook port is a pre-season model — every rate is last season's, and
the CONTROL weights on current-season data are zero. Three consequences this
layer fixes, each as a Bayesian blend so early-season noise can't run away:

1. Team form. A team's attack/defence strength is last season's xG/xGC. Here
   each team's actual xG for/against this season is compared with what the
   prior strengths predicted *for the fixtures it actually played* (so four
   hard games don't read as bad form), and the prior is scaled by the shrunk
   ratio: mult = (K + n * ratio) / (K + n), K pseudo-games of prior.

2. Player xG/xA rates. The sheet takes last season's per-90 at face value —
   100 minutes and one big chance projects as an elite scorer — and ignores
   this season entirely. Here the rate is a minutes-weighted blend of the
   position-group mean (K pseudo-minutes, scaled by the team's attack), last
   season, and this season (up-weighted for recency).

3. Start rates. Selection rate is last season's, so this season's regular
   who was injured or a backup last year stays near zero. Blended with this
   season's starts over team games.

This is a deliberate deviation from the faithful-port policy, so it lives
outside build_model/compute_ppts: parity tests run the untouched port, and
the live pipeline applies this between the two. The constants are reasoned
defaults, not fitted — they are the first thing the backtester should tune.

Data caveats: team xG is the sum of its current players' season xG and team
xGC is its goalkeepers' season xGC, so a mid-season transfer between PL clubs
carries his totals to the new club. Negligible early on; the snapshot-delta
data layer removes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ControlParams, PositionPoints

XG_COLS = ["LF xG/90", "SF xG/90"]
XGC_COLS = ["LF xGC/90", "SF xGC/90", "LFxSF US xGC FIX", "LF US xGC FIX", "SF US xGC FIX"]

# a position group needs this many prior minutes behind its mean, else fall
# back to the broader FPL position (RM/LM have a handful of players)
MIN_GROUP_MINS = 5000.0


@dataclass(frozen=True)
class InSeasonParams:
    # pseudo-games of pre-season strength: after n games the current season
    # carries n / (n + K) of the weight (4 games -> 29%, 19 games -> 66%)
    team_prior_games: float = 10.0
    # pseudo-minutes at the position-group mean rate (10 full games)
    rate_prior_mins: float = 900.0
    # recency: a current-season minute counts this many last-season minutes
    current_season_weight: float = 1.5
    # start-rate blend: last season counts for at most this many games, and
    # everyone carries a few pseudo-games at a coin-flip so 4 starts in 4
    # isn't read as 100% nailed
    start_prior_max_games: float = 19.0
    start_pseudo_games: float = 3.0
    # a fit player with no minutes while his team has played: start rate is
    # multiplied by this per game missed. Measured on GW2-4 of 2026/27, fit
    # 80%+ starters with zero minutes started 47% / 17% / 0% of the time
    # after 1 / 2 / 3 team games — roughly a halving per game
    absent_decay: float = 0.5
    # home sides take ~55% of xG; used only to schedule-adjust expectations
    home_xg_factor: float = 1.10
    ratio_clip: tuple[float, float] = (0.4, 2.5)


DEFAULT_INSEASON = InSeasonParams()


@dataclass(frozen=True)
class Layers:
    """Which live-path adjustments run, and with what constants.

    The backtester's unit of comparison: every switch and every constant the
    live pipeline uses is here, so a variant is just another Layers value.
    `style_params` is typed loosely to keep this module free of a style import.
    """

    form: bool = True
    rates: bool = True
    starts: bool = True
    style: bool = True
    priors: bool = True          # data/priors.yaml — today's judgment; off when replaying the past
    params: InSeasonParams = DEFAULT_INSEASON
    style_params: object | None = None

    @property
    def any_inseason(self) -> bool:
        return self.form or self.rates or self.starts or self.style


ALL_LAYERS = Layers()
PORT_ONLY = Layers(form=False, rates=False, starts=False, style=False)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def team_form(
    boot: dict,
    fixtures_raw: list[dict],
    team_strength: pd.DataFrame,
    league_avg_xgc: float,
    control: ControlParams,
    params: InSeasonParams = DEFAULT_INSEASON,
) -> pd.DataFrame:
    """Schedule-adjusted current-season form per team, indexed by ABVR.

    att_mult > 1: creating more than the prior predicted against these
    opponents. def_mult > 1: conceding more (worse defence).
    """
    ts = team_strength.set_index(team_strength["ABVR"].str.upper())
    w_lf, w_sf = control.fdr_long_form_weight, control.fdr_short_form_weight
    prior_att = w_lf * ts["LF xG/90"] + w_sf * ts["SF xG/90"]
    prior_def = w_lf * ts["LF xGC/90"] + w_sf * ts["SF xGC/90"]

    short = {int(t["id"]): str(t["short_name"]).upper() for t in boot["teams"]}
    el = pd.DataFrame(boot["elements"])
    el["T"] = _num(el["team"]).astype(int).map(short)
    xg = _num(el["expected_goals"]).groupby(el["T"]).sum()
    gk = el[_num(el["element_type"]) == 1]
    xgc = _num(gk["expected_goals_conceded"]).groupby(gk["T"]).sum()

    played = [
        (short[int(f["team_h"])], short[int(f["team_a"])])
        for f in fixtures_raw
        if f.get("finished") and f.get("team_h") in short and f.get("team_a") in short
    ]
    teams = [t for t in ts.index if t in set(short.values())]
    games = pd.Series({t: sum(t in g for g in played) for t in teams}, dtype=float)

    att = pd.Series(1.0, index=teams)
    dfn = pd.Series(1.0, index=teams)
    lo, hi = params.ratio_clip
    k = params.team_prior_games
    exp_xg = exp_xgc = pd.Series(0.0, index=teams)
    att_target, def_target = att, dfn
    # opponents' strengths are themselves being re-estimated, so iterate the
    # expectation a few times; it settles almost immediately
    for _ in range(3):
        exp_xg = pd.Series(0.0, index=teams)
        exp_xgc = pd.Series(0.0, index=teams)
        for h, a in played:
            if h not in att.index or a not in att.index:
                continue
            h_xg = prior_att[h] * att[h] * prior_def[a] * dfn[a] / league_avg_xgc * params.home_xg_factor
            a_xg = prior_att[a] * att[a] * prior_def[h] * dfn[h] / league_avg_xgc * (2 - params.home_xg_factor)
            exp_xg[h] += h_xg
            exp_xgc[a] += h_xg
            exp_xg[a] += a_xg
            exp_xgc[h] += a_xg
        act_xg = xg.reindex(teams).fillna(0.0)
        act_xgc = xgc.reindex(teams).fillna(0.0)
        # league-level drift (scoring environment, xG provider) is not form:
        # normalise so the multipliers are relative to the league this season
        level_att = act_xg.sum() / exp_xg.sum() if exp_xg.sum() > 0 else 1.0
        level_def = act_xgc.sum() / exp_xgc.sum() if exp_xgc.sum() > 0 else 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            att_ratio = (act_xg / exp_xg / level_att).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            def_ratio = (act_xgc / exp_xgc / level_def).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # the ratios are measured against the *current* estimate; fold them in
        att_target = (att * att_ratio).clip(lo, hi)
        def_target = (dfn * def_ratio).clip(lo, hi)
        att = (k + games * att_target) / (k + games)
        dfn = (k + games * def_target) / (k + games)

    return pd.DataFrame({
        "games": games, "xg": xg.reindex(teams).fillna(0.0), "xgc": xgc.reindex(teams).fillna(0.0),
        "exp_xg": exp_xg, "exp_xgc": exp_xgc,
        "att_raw": (att_target).round(4), "def_raw": (def_target).round(4),
        "att_mult": att, "def_mult": dfn,
    })


def adjust_team_strength(team_strength: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Team strength table with the form multipliers folded into every xG/xGC column."""
    ts = team_strength.copy()
    key = ts["ABVR"].str.upper()
    att = key.map(form["att_mult"]).fillna(1.0)
    dfn = key.map(form["def_mult"]).fillna(1.0)
    for c in XG_COLS:
        ts[c] = ts[c] * att.values
    for c in XGC_COLS:
        ts[c] = ts[c] * dfn.values
    return ts


def _group_means(rate90: pd.Series, mins: pd.Series, pos: pd.Series, pos2: pd.Series) -> pd.Series:
    """Minutes-weighted mean per-90 for each player's position group."""
    def wmean(by: pd.Series) -> tuple[pd.Series, pd.Series]:
        tot = mins.groupby(by).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = ((rate90 * mins).groupby(by).sum() / tot).fillna(0.0)
        return mean, tot

    mean2, tot2 = wmean(pos2)
    mean1, _ = wmean(pos)
    use_fine = pos2.map(tot2).fillna(0.0) >= MIN_GROUP_MINS
    return pos2.map(mean2).where(use_fine, pos.map(mean1)).fillna(0.0)


def adjust_player_rates(
    model: pd.DataFrame,
    api: pd.DataFrame,
    points: PositionPoints,
    params: InSeasonParams = DEFAULT_INSEASON,
    team_attack_rel: pd.Series | None = None,
    channel_rel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replace the goal ("8") and assist ("7") columns with the blended rates.

    `team_attack_rel` (ABVR -> team xG/90 over league average) scales the
    position-group mean: the norm for a striker is a share of *his team's*
    chances, so a promoted side's forward isn't handed the PL-average rate.
    `channel_rel` (style.TeamStyle.attack: goal_rel/assist_rel per (team, pos))
    refines that by where the team's chances go: a defender in a side whose
    back line takes 5% of its xG gets a lower goal norm than the league's.

    Keeps the evidence next to the result (`xG90 prior/season/used`, same for
    xA, plus `Season Mins`/`Season Starts`) so the explainer can show its work.
    Players with no minutes in either season are left alone: there is nothing
    to blend, and a mean rate would hand phantom points to non-players.
    """
    m = model.copy()
    by_code = api.assign(_code=_num(api["code"]).astype(int)).drop_duplicates("_code").set_index("_code")

    def cur(col: str) -> pd.Series:
        return m["Code"].map(_num(by_code[col])).fillna(0.0) if col in by_code else pd.Series(0.0, index=m.index)

    cur_mins = cur("minutes")
    m["Season Mins"], m["Season Starts"] = cur_mins, cur("starts")
    blend = _num(m["G+A PSxTS Mn/Start"])
    prior_mins = _num(m["LF G+A Mins"])
    w, k = params.current_season_weight, params.rate_prior_mins
    has_data = (prior_mins + cur_mins) > 0

    team_pos = pd.MultiIndex.from_arrays([m["TEAM"].str.upper(), m["POS"]])
    for col, tag, season_col, pts, rel_col in [
        ("8", "xG90", "expected_goals", _num(m["Goal"]), "goal_rel"),
        ("7", "xA90", "expected_assists", pd.Series(points.assist, index=m.index), "assist_rel"),
    ]:
        with np.errstate(divide="ignore", invalid="ignore"):
            prior90 = (_num(m[col]) / pts / blend).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            season90 = (cur(season_col) / cur_mins * 90).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mu = _group_means(prior90, prior_mins, m["POS"], m["POS2"])
        if team_attack_rel is not None:
            mu = mu * m["TEAM"].str.upper().map(team_attack_rel).fillna(1.0)
        if channel_rel is not None:
            mu = mu * pd.Series(channel_rel[rel_col].reindex(team_pos).values, index=m.index).fillna(1.0)
        num = mu * k + prior90 * prior_mins + w * season90 * cur_mins
        used90 = num / (k + prior_mins + w * cur_mins)
        m[f"{tag} prior"], m[f"{tag} season"], m[f"{tag} group"] = prior90, season90, mu
        m[f"{tag} used"] = used90.where(has_data, prior90)
        m[col] = (used90 * blend * pts).where(has_data, _num(m[col]))
    return m


def team_attack_relative(team_strength: pd.DataFrame, league_avg_xg: float, control: ControlParams) -> pd.Series:
    ts = team_strength.set_index(team_strength["ABVR"].str.upper())
    blend = control.fdr_long_form_weight * ts["LF xG/90"] + control.fdr_short_form_weight * ts["SF xG/90"]
    return blend / league_avg_xg


def adjust_start_rates(
    model: pd.DataFrame,
    api: pd.DataFrame,
    form: pd.DataFrame,
    params: InSeasonParams = DEFAULT_INSEASON,
) -> pd.DataFrame:
    """Blend this season's starts into the selection rate behind Start %.

    The port's rate is last season's starts per squad made, so a player who
    was injured or a backup last year stays at ~20% after starting every game
    this year. Blend: last season (capped pseudo-games), a few coin-flip
    pseudo-games, and this season's starts over team games, recency-weighted.

    Zero minutes splits on the availability flag. Flagged (injured,
    suspended, doubtful): left at last season's rate — the flag already
    discounts him and he may walk back into the side. Fit and still unused:
    he has lost his place, and the blend is far too slow to say so (it would
    hold a 90% starter near 80% after a game on the bench), so the rate is
    decayed per team game missed instead. The result is written to both rate
    columns so compute_ppts' season weights can't undo it.
    """
    m = model.copy()
    by_code = api.assign(_code=_num(api["code"]).astype(int)).drop_duplicates("_code").set_index("_code")
    starts = m["Code"].map(_num(by_code["starts"])).fillna(0.0)
    mins = m["Code"].map(_num(by_code["minutes"])).fillna(0.0)
    games = m["TEAM"].str.upper().map(form["games"]).fillna(0.0)

    ps_rate = _num(m["PS Sts/SqdsMd"])
    ps_games = (_num(m["PS Starts"]) + _num(m["PS Subs"]) + _num(m["PS UnSub"])).clip(upper=params.start_prior_max_games)
    w, k = params.current_season_weight, params.start_pseudo_games
    blended = (ps_rate * ps_games + 0.5 * k + w * starts.clip(upper=games)) / (ps_games + k + w * games)
    touch = (mins > 0) & (games > 0)
    # no status column (synthetic inputs): can't tell fit from injured, so leave alone
    status = m["Code"].map(by_code["status"]) if "status" in by_code else pd.Series("", index=m.index)
    absent = (mins == 0) & (games > 0) & status.eq("a")
    m["Start rate prior"] = ps_rate
    m["Start rate used"] = blended.where(touch, ps_rate).where(~absent, ps_rate * params.absent_decay ** games)
    m["PS Sts/SqdsMd"] = m["St/SqdsMd"] = m["Start rate used"]
    return m
