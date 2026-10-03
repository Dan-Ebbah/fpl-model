"""Team playing style from the match log: venue tilt and positional channels.

Three questions the season totals can't answer, each turned into a multiplier
and each shrunk hard toward "no effect" because a handful of games is noise:

- Venue tilt: is this team more of a home side (or less) than the league? The
  port already applies a flat home/away multiplier to everyone, so only the
  team's tilt *relative to the league* is modelled — 1.0 means "typical home
  advantage", not "none".
- Attacking channels: what share of the team's xG and xA falls to each
  position. This sets the norm a player's own rate is shrunk toward: a
  wing-back in a side that creates through its defenders gets a higher
  assist prior than the league's average defender.
- Concession channels: what share of the xG/xA a team *allows* goes to each
  position. A striker's goal expectation against a side that leaks to
  forwards is scaled up, beyond that side's overall defensive strength.

Everything is this-season-only (the frozen last-season inputs have no
per-match venue or opponent), so after four games these factors sit within a
few percent of 1.0 by construction and open up as the season goes on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ControlParams

OUTFIELD = ["DEF", "MID", "FWD"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]


@dataclass(frozen=True)
class StyleParams:
    # pseudo-games at the league-typical value; venue splits halve the sample
    # (home games only vs away games only), so they get a heavier prior
    venue_prior_games: float = 19.0
    channel_prior_games: float = 15.0
    # positional channels are the noisiest thing here (one set-piece chance
    # moves a defender share a lot): cap what they can do to a projection
    channel_clip: tuple[float, float] = (0.8, 1.25)
    home_xg_factor: float = 1.10
    tilt_clip: tuple[float, float] = (0.6, 1.6)


DEFAULT_STYLE = StyleParams()


@dataclass
class TeamStyle:
    venue: pd.DataFrame     # per team: att_home/att_away/def_home/def_away multipliers (+ raw tilts, games)
    attack: pd.DataFrame    # per (team, pos): xg_share, xa_share, slots, goal_rel, assist_rel
    concede: pd.DataFrame   # per (team, pos): goal_mult, assist_mult (vs league channel)
    league: pd.DataFrame    # per pos: league xg_share, xa_share, slots


def _team_matches(log: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture): xG for, xG against, venue."""
    own = log.groupby(["team", "fixture"]).agg(
        xg=("xg", "sum"), opponent=("opponent", "first"), home=("was_home", "first")
    ).reset_index()
    own["home"] = own["home"].astype(bool)  # object dtype after a cache round-trip: ~ would flip bits, not truth
    against = own.rename(columns={"team": "opponent", "opponent": "team", "xg": "xga"})[["team", "fixture", "xga"]]
    return own.merge(against, on=["team", "fixture"], how="left").fillna({"xga": 0.0})


def venue_tilt(
    log: pd.DataFrame, team_strength: pd.DataFrame, league_avg_xgc: float,
    control: ControlParams, params: StyleParams = DEFAULT_STYLE,
) -> pd.DataFrame:
    ts = team_strength.set_index(team_strength["ABVR"].str.upper())
    w_lf, w_sf = control.fdr_long_form_weight, control.fdr_short_form_weight
    att = w_lf * ts["LF xG/90"] + w_sf * ts["SF xG/90"]
    dfn = w_lf * ts["LF xGC/90"] + w_sf * ts["SF xGC/90"]

    tm = _team_matches(log)
    tm = tm[tm["team"].isin(ts.index) & tm["opponent"].isin(ts.index)].copy()
    hf = np.where(tm["home"], params.home_xg_factor, 2 - params.home_xg_factor)
    # expectation given who they played and where, at the league-typical venue effect
    tm["exp_xg"] = tm["team"].map(att).values * tm["opponent"].map(dfn).values / league_avg_xgc * hf
    tm["exp_xga"] = tm["opponent"].map(att).values * tm["team"].map(dfn).values / league_avg_xgc * (2 - hf)

    rows = {}
    league = tm.groupby("home")[["xg", "exp_xg"]].sum()
    # if the league as a whole is more/less home-tilted than the assumed
    # factor, that is not any one team's style
    league_tilt = 1.0
    if {True, False} <= set(league.index) and (league["exp_xg"] > 0).all():
        league_tilt = (league.loc[True, "xg"] / league.loc[True, "exp_xg"]) / (league.loc[False, "xg"] / league.loc[False, "exp_xg"])
    lo, hi = params.tilt_clip
    for team, g in tm.groupby("team"):
        h, a = g[g["home"]], g[~g["home"]]
        n = 2 * min(len(h), len(a))  # a split needs both halves
        w = n / (n + params.venue_prior_games)

        def tilt(act: str, exp: str, invert: bool) -> float:
            if not len(h) or not len(a) or min(h[exp].sum(), a[exp].sum(), h[act].sum(), a[act].sum()) <= 0:
                return 1.0
            t = (h[act].sum() / h[exp].sum()) / (a[act].sum() / a[exp].sum())
            return float(np.clip(t * (league_tilt if invert else 1 / league_tilt), lo, hi))

        att_raw, def_raw = tilt("xg", "exp_xg", False), tilt("xga", "exp_xga", True)
        # split the shrunk tilt evenly so a team's home and away factors
        # multiply to 1: style moves points between venues, form owns the level
        a_t, d_t = att_raw ** w, def_raw ** w
        rows[team] = {
            "home_games": len(h), "away_games": len(a), "att_tilt_raw": att_raw, "def_tilt_raw": def_raw,
            "att_home": a_t ** 0.5, "att_away": a_t ** -0.5, "def_home": d_t ** 0.5, "def_away": d_t ** -0.5,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.reindex(ts.index).fillna({c: 1.0 for c in ["att_tilt_raw", "def_tilt_raw", "att_home", "att_away", "def_home", "def_away"]}).fillna(0)


def _channels(log: pd.DataFrame, by: str, k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positional shares of xG/xA grouped by `by` (the creating or the conceding
    team), shrunk toward the league shares with k pseudo-games."""
    # keepers take no meaningful share of chances; a 0.2% league share turns
    # one long ball into a 5x multiplier, so channels are outfield-only
    lg = log[log["pos"].isin(OUTFIELD)]
    league = lg.groupby("pos")[["xg", "xa", "minutes"]].sum().reindex(OUTFIELD).fillna(0.0)
    league_share = league[["xg", "xa"]] / league[["xg", "xa"]].sum().replace(0, np.nan)
    games = lg.groupby(by)["fixture"].nunique()
    tot = lg.groupby([by, "pos"])[["xg", "xa", "minutes"]].sum()
    idx = pd.MultiIndex.from_product([games.index, OUTFIELD], names=["team", "pos"])
    tot = tot.reindex(idx).fillna(0.0)
    n = pd.Series(idx.get_level_values(0).map(games).values, index=idx, dtype=float)
    team_tot = tot.groupby(level=0)[["xg", "xa"]].transform("sum").replace(0, np.nan)
    raw = (tot[["xg", "xa"]] / team_tot)
    lshare = league_share.reindex(idx.get_level_values(1)).set_axis(idx)
    raw = raw.fillna(lshare)
    shrunk = raw.mul(n, axis=0).add(lshare * k).div(n + k, axis=0)
    shrunk = shrunk / shrunk.groupby(level=0).transform("sum")
    # players of this position on the pitch at once
    league_slots = league["minutes"] / (90 * games.sum())
    lslots = pd.Series(idx.get_level_values(1).map(league_slots).values, index=idx)
    slots = (tot["minutes"] / 90 + lslots * k) / (n + k)
    out = pd.DataFrame({
        "games": n, "xg_share_raw": raw["xg"], "xa_share_raw": raw["xa"],
        "xg_share": shrunk["xg"], "xa_share": shrunk["xa"], "slots": slots,
    })
    league_out = league_share.rename(columns={"xg": "xg_share", "xa": "xa_share"})
    league_out["slots"] = league_slots
    return out, league_out


def team_style(
    log: pd.DataFrame, team_strength: pd.DataFrame, league_avg_xgc: float,
    control: ControlParams, params: StyleParams = DEFAULT_STYLE,
) -> TeamStyle:
    attack, league = _channels(log, "team", params.channel_prior_games)
    concede, _ = _channels(log, "opponent", params.channel_prior_games)
    lg = league.reindex(attack.index.get_level_values("pos")).set_axis(attack.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        # per-player norm relative to the league's: (share / slots) over the same for the league
        attack["goal_rel"] = ((attack["xg_share"] / attack["slots"]) / (lg["xg_share"] / lg["slots"])).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        attack["assist_rel"] = ((attack["xa_share"] / attack["slots"]) / (lg["xa_share"] / lg["slots"])).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        lc = league.reindex(concede.index.get_level_values("pos")).set_axis(concede.index)
        concede["goal_mult"] = (concede["xg_share"] / lc["xg_share"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        concede["assist_mult"] = (concede["xa_share"] / lc["xa_share"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for frame, cols in [(attack, ["goal_rel", "assist_rel"]), (concede, ["goal_mult", "assist_mult"])]:
        frame[cols] = frame[cols].clip(*params.channel_clip)
    return TeamStyle(
        venue=venue_tilt(log, team_strength, league_avg_xgc, control, params),
        attack=attack, concede=concede, league=league,
    )


def fixture_adjustments(style: TeamStyle, teams: list[str]) -> pd.DataFrame:
    """Per (TEAM, vs, home, POS) multipliers for compute_ppts.

    goal/assist: own attack at this venue x opponent defence at theirs x how
    much of what the opponent concedes goes to this position.
    conceded: the mirror, for the clean-sheet lambda (position-free).
    """
    v = style.venue
    rows = []
    for t in teams:
        for o in teams:
            if t == o or t not in v.index or o not in v.index:
                continue
            for home in (True, False):
                att = v.loc[t, "att_home" if home else "att_away"] * v.loc[o, "def_away" if home else "def_home"]
                conceded = v.loc[t, "def_home" if home else "def_away"] * v.loc[o, "att_away" if home else "att_home"]
                for pos in POSITIONS:
                    c = style.concede.loc[(o, pos)] if (o, pos) in style.concede.index else None
                    rows.append({
                        "TEAM": t, "vs_key": o, "is_home": home, "POS": pos,
                        "goal_mult": att * (c["goal_mult"] if c is not None else 1.0),
                        "assist_mult": att * (c["assist_mult"] if c is not None else 1.0),
                        "conceded_mult": conceded,
                    })
    return pd.DataFrame(rows)
