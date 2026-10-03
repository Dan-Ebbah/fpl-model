"""Build the MODEL-sheet player aggregates from raw inputs.

This ports the MODEL sheet's formulas (docs/formulas/MODEL.json) so the player
table can be rebuilt from a live FPL API pull instead of consumed as a frozen
extract. Live inputs: the current bootstrap (players + teams). Frozen inputs
(last-season data that no longer changes): PSAPI, the FBRef playing-time
tables PT/PSPT, the per-GW pivot delta matrices (MINS/xG/xA/DC), team
strength (TABLES), and the CONTROL cell values.

Only the dependency closure of the columns PPts consumes is ported (about 90
of the sheet's 190 columns). Faithful-port policy applies: Excel's quirks are
reproduced, e.g. MATCH is case-insensitive and returns the first hit, the
current-season YC rate skips the minutes-share factor the previous-season one
applies, and a player whose Mn/Start is exactly 50 falls off the end of the
TREND interpolation table and gets 0.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import PositionPoints

ELEMENT_TYPE_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# CONTROL rows defining the pivot windows (C=offset, D=span, E=end offset)
LF_WINDOW, SF_WINDOW = 56, 57
LF_DC_WINDOW, SF_DC_WINDOW = 83, 84


def control_value(control: pd.Series, cell: str) -> float:
    return float(control[cell])


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first_match_map(keys: pd.Series, values: pd.Series) -> dict[str, float]:
    """Excel MATCH semantics: case-insensitive, first occurrence wins."""
    out: dict[str, float] = {}
    vals = _num(values)
    for k, v in zip(keys, vals):
        if k is None or pd.isna(k):
            continue
        out.setdefault(str(k).lower(), v)
    return out


def _fbref_lookup(table: pd.DataFrame, column: str, names: list[pd.Series]) -> list[pd.Series]:
    """IFERROR(INDEX(T[column], MATCH(name, T[Player], 0)), 0) per name variant."""
    lookup = _first_match_map(table["Player"], table[column])
    return [
        n.map(lambda x: lookup.get(str(x).lower(), np.nan)).fillna(0.0) for n in names
    ]


def _dedup4(a: pd.Series, b: pd.Series, c: pd.Series, d: pd.Series) -> pd.Series:
    """The sheet's double-count guard: if the 4 name variants matched more than
    one distinct row, the sum is halved (bug-for-bug — halving is only correct
    when exactly two variants matched)."""
    s = a + b + c + d
    single = (s == a) | (s == b) | (s == c) | (s == d)
    return s.where(single, s / 2)


def _pivot_window_sum(
    pivot: pd.DataFrame, codes: pd.Series, lo: int, hi: int
) -> pd.Series:
    """SUM(INDEX(sheet!B1001:CZ1999, row, lo):INDEX(..., row, hi)) keyed by code.

    lo/hi are 1-based column indices within the 103-wide block; out-of-range
    indices or an unmatched code error out in Excel and land on IFERROR -> 0.
    """
    width = pivot.shape[1] - 1
    if not (1 <= lo <= width and 1 <= hi <= width):
        return pd.Series(0.0, index=codes.index)
    lo, hi = min(lo, hi), max(lo, hi)
    block = pivot.iloc[:, lo : hi + 1].apply(pd.to_numeric, errors="coerce")
    sums = block.sum(axis=1, skipna=True)
    lookup = {}
    for key, val in zip(pivot["key"], sums):
        if key is None or pd.isna(key):
            continue
        try:  # keys may have been stored as '17761' or '17761.0'
            key = str(int(float(key)))
        except ValueError:
            key = str(key)
        lookup.setdefault(key, float(val))
    return codes.map(lambda c: lookup.get(str(int(c)), 0.0)).astype(float)


def _window(control: pd.Series, row: int) -> tuple[int, int]:
    e2 = int(control_value(control, "E2"))
    c = int(control_value(control, f"C{row}"))
    d = int(control_value(control, f"D{row}"))
    e = int(control_value(control, f"E{row}"))
    return e2 - d + 1 - c, e2 - e


def _trend_interp(x: pd.Series, bx: list[float], by: list[float], base: float) -> pd.Series:
    """base - IF(x < bx[-1], 0.5, TREND(segment)) with IFERROR -> 0.

    bx is descending. MATCH(x, bx, -1) picks the last breakpoint >= x; TREND
    over that breakpoint and the next is plain linear interpolation. x exactly
    at the final breakpoint indexes past the table -> error -> 0 (preserved).
    """
    def one(v: float) -> float:
        if pd.isna(v):
            v = 0.0
        if v < bx[-1]:
            return base - 0.5
        pos = None
        for i, b in enumerate(bx):
            if b >= v:
                pos = i
        if pos is None or pos + 1 >= len(bx):
            # MATCH fails (x above all) or segment runs off the table: error -> 0
            if pos is None:
                return 0.0
            return 0.0
        x0, x1, y0, y1 = bx[pos], bx[pos + 1], by[pos], by[pos + 1]
        return base - (y0 + (y1 - y0) * (v - x0) / (x1 - x0))

    return x.map(one)


def _poisson_cdf(k: pd.Series, lam: pd.Series) -> pd.Series:
    def one(kk: float, ll: float) -> float:
        if pd.isna(ll) or ll < 0:
            return np.nan
        return math.exp(-ll) * sum(ll**i / math.factorial(i) for i in range(int(kk) + 1))

    return pd.Series([one(kk, ll) for kk, ll in zip(k, lam)], index=k.index)


def build_model(
    api: pd.DataFrame,
    teams: pd.DataFrame,
    psapi: pd.DataFrame,
    pt: pd.DataFrame,
    pspt: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    team_strength: pd.DataFrame,
    control: pd.Series,
    points: PositionPoints,
    name_map: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the MODEL sheet rows (one per API player, API order).

    `name_map` carries the sheet's manual columns keyed by Code: FBRef_Name,
    "Alt Name 2", POS2 — name spellings used to match the FBRef tables. New
    players absent from it fall back to "-" (no extra FBRef match), POS2=POS.
    """
    cv = lambda cell: control_value(control, cell)  # noqa: E731

    m = pd.DataFrame(index=api.index)
    m["Code"] = _num(api["code"]).astype(int)
    m["FPL_Name"] = api["first_name"].astype(str) + " " + api["second_name"].astype(str)
    m["web_name"] = api["web_name"].astype(str)
    nm = name_map.reindex(m["Code"].values)
    m["FBRef_Name"] = nm["FBRef_Name"].fillna("-").values
    m["Alt Name 2"] = nm["Alt Name 2"].fillna("-").values
    m["POS"] = _num(api["element_type"]).map(ELEMENT_TYPE_POS).fillna("-")
    m["POS2"] = pd.Series(nm["POS2"].values, index=m.index).fillna(m["POS"])

    team_ids = _num(teams["id"]).astype(int)
    short = dict(zip(team_ids, teams["short_name"]))
    m["TEAM"] = _num(api["team"]).map(short).fillna("-")

    names = [m["FPL_Name"], m["web_name"], m["FBRef_Name"], m["Alt Name 2"]]

    # --- FBRef playing-time aggregates (PS = previous season via PSPT) ---
    for prefix, table in [("PS ", pspt), ("", pt)]:
        for out_col, src in [
            ("Starts", "Starts"), ("Subs", "Subs"), ("UnSub", "unSub"),
            ("Mn/St", "Mn/Start"), ("Mn/Sub", "Mn/Sub"),
        ]:
            m[prefix + out_col] = _dedup4(*_fbref_lookup(table, src, names))

    m["PS Mn/St %"] = np.where(
        m["POS"] == "GK", 1.0, np.where(m["PS Starts"] == 0, 0.66, m["PS Mn/St"] / 90)
    )
    m["PS Mn/Sub %"] = np.where(m["PS Mn/Sub"] == 0, 0.2, m["PS Mn/Sub"] / 90)
    m["Mn/St %"] = np.where(
        m["POS"] == "GK", 1.0, np.where(m["Starts"] == 0, 0.66, m["Mn/St"] / 90)
    )
    m["Mn/Sub %"] = np.where(m["Mn/Sub"] == 0, 0.2, m["Mn/Sub"] / 90)

    for prefix in ["PS ", ""]:
        squads = m[prefix + "Starts"] + m[prefix + "Subs"] + m[prefix + "UnSub"]
        with np.errstate(invalid="ignore"):
            m[prefix + ("Sts/SqdsMd" if prefix else "St/SqdsMd")] = (
                (m[prefix + "Starts"] / squads).replace([np.inf, -np.inf], 0).fillna(0.0)
            )
            m[prefix + "UnSub/SqdsMd"] = (
                (m[prefix + "UnSub"] / squads).replace([np.inf, -np.inf], 0).fillna(0.0)
            )
    # --- season-total rates from the API pulls (PS = frozen PSAPI) ---
    ps_by_code = psapi.set_index(_num(psapi["code"]).astype(int))

    def ps_col(col: str) -> pd.Series:
        return m["Code"].map(_num(ps_by_code[col])).fillna(0.0)

    def api_col(col: str) -> pd.Series:
        return _num(api[col]).fillna(0.0)

    def per90(count: pd.Series, mins: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (count / mins) * 90
        return r.replace([np.inf, -np.inf], 0).fillna(0.0)

    m["PS Mins"], m["Mins"] = ps_col("minutes"), api_col("minutes")
    m["PS Saves/90"] = per90(ps_col("saves"), m["PS Mins"])
    m["Saves/90"] = per90(api_col("saves"), m["Mins"])
    m["PS YC/St"] = per90(ps_col("yellow_cards"), m["PS Mins"]) * m["PS Mn/St %"]
    m["YC/90"] = per90(api_col("yellow_cards"), m["Mins"])  # sheet skips the Mn/St % factor here
    m["PS RC/90"] = per90(ps_col("red_cards"), m["PS Mins"])
    m["RC/90"] = per90(api_col("red_cards"), m["Mins"])
    m["PS A/St"] = per90(ps_col("assists"), m["PS Mins"]) * m["PS Mn/St %"]
    m["A/St"] = per90(api_col("assists"), m["Mins"]) * m["Mn/St %"]
    m["PS xA/St"] = per90(ps_col("expected_assists"), m["PS Mins"]) * m["PS Mn/St %"]
    m["xA/St"] = per90(api_col("expected_assists"), m["Mins"]) * m["Mn/St %"]

    # bonus / BPS per start
    ps_bps, bps = ps_col("bps"), api_col("bps")
    ps_bonus, bonus = ps_col("bonus"), api_col("bonus")
    with np.errstate(divide="ignore", invalid="ignore"):
        ps_start_bps = (ps_bps * ((m["PS Starts"] * m["PS Mn/St"]) / m["PS Mins"])).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        start_bps = (bps * ((m["Starts"] * m["Mn/St"]) / m["Mins"])).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        # PS side treats a 0 ratio as "no data" and falls back; current side doesn't
        ps_bps_per_start = (ps_start_bps / m["PS Starts"]).replace([np.inf, -np.inf], np.nan)
        ps_bps_per_start = ps_bps_per_start.where(ps_bps_per_start != 0, np.nan).fillna(cv("B43"))
        ps_bonus_per_start = (ps_bonus / m["PS Starts"]).replace([np.inf, -np.inf], np.nan)
        ps_bonus_per_start = ps_bonus_per_start.where(ps_bonus_per_start != 0, np.nan).fillna(cv("B44"))
        bps_per_start = (start_bps / m["Starts"]).replace([np.inf, -np.inf], np.nan).fillna(cv("B43"))
        bonus_per_start = (bonus / m["Starts"]).replace([np.inf, -np.inf], np.nan).fillna(cv("B44"))
    m["PS BPS/Start"], m["PS Bonus/Start"] = ps_bps_per_start, ps_bonus_per_start
    m["BPS/Start"], m["Bonus/Start"] = bps_per_start, bonus_per_start

    chance = _num(api["chance_of_playing_next_round"])
    m["Chance of Playing"] = np.where(chance.isna(), 1.0, chance / 100)

    # minutes-share blends (PS weight B62, current weight B63)
    m["G+A PSxTS Mn/Start"] = m["PS Mn/St %"] * cv("B62") + cv("B63") * m["Mn/St %"]
    m["G+A PSxTS Mn/Sub"] = m["PS Mn/Sub %"] * cv("B62") + cv("B63") * m["Mn/Sub %"]

    # --- pivot windows: long-form / short-form xG, xA, minutes, DC ---
    codes = m["Code"]
    for tag, wrow in [("LF", LF_WINDOW), ("SF", SF_WINDOW)]:
        lo, hi = _window(control, wrow)
        mins_w = _pivot_window_sum(pivots["MINS"], codes, lo, hi)
        xa_w = _pivot_window_sum(pivots["xA"], codes, lo, hi)
        xg_w = _pivot_window_sum(pivots["xG"], codes, lo, hi)
        m[f"{tag} G+A Mins" if tag == "LF" else "SF Mins"] = mins_w
        m[f"{tag} xA"], m[f"{tag} xG"] = xa_w, xg_w
        m[f"{tag} xA/St"] = per90(xa_w, mins_w) * m["G+A PSxTS Mn/Start"]
        m[f"{tag} xG/St"] = per90(xg_w, mins_w) * m["G+A PSxTS Mn/Start"]

    dc_blend = m["PS Mn/St %"] * cv("B89") + cv("B90") * m["Mn/St %"]
    dc_threshold = np.where(m["POS"] == "DEF", 9, 11)
    for tag, wrow in [("LF", LF_DC_WINDOW), ("SF", SF_DC_WINDOW)]:
        lo, hi = _window(control, wrow)
        dc_w = _pivot_window_sum(pivots["DC"], codes, lo, hi)
        dc_mins = _pivot_window_sum(pivots["MINS"], codes, lo, hi)
        dc_per_start = per90(dc_w, dc_mins) * dc_blend
        m[f"{tag} DC/St"] = dc_per_start
        m[f"{tag} DC %"] = 1 - _poisson_cdf(pd.Series(dc_threshold, index=m.index), dc_per_start)

    # --- scoring category columns "1".."11" ---
    m["PS %1"] = np.where(
        (m["PS Starts"] == 0) & (m["PS Subs"] == 0), 0.01, (1 - m["PS UnSub/SqdsMd"]) * 0.99
    )
    m["%1"] = np.where(
        (m["Starts"] == 0) & (m["Subs"] == 0), 0.01, (1 - m["UnSub/SqdsMd"]) * 0.99
    )
    m["1"] = (m["PS %1"] * cv("B8") + m["%1"] * cv("B9")) * 1.0 * m["Chance of Playing"]

    bx = [cv(f"B{r}") for r in range(13, 18)]
    by = [cv(f"C{r}") for r in range(13, 18)]
    m["PS %2"] = _trend_interp(m["PS Mn/St"], bx, by, 0.98)
    m["%2"] = _trend_interp(m["Mn/St"], bx, by, 0.99)
    m["2"] = (m["PS %2"] * cv("B19") + cv("B20") * m["%2"]) * 1.0

    gk = (m["POS"] == "GK").astype(float)
    m["3"] = (m["PS Saves/90"] / 3 * cv("B25") + cv("B26") * m["Saves/90"] / 3) * gk
    m["4"] = (m["PS YC/St"] * cv("B31") + cv("B32") * m["YC/90"]) * -1
    m["5"] = (m["PS RC/90"] * cv("B37") + cv("B38") * m["RC/90"]) * -3

    ps_pct6 = np.where(
        m["PS Starts"] > cv("B46"),
        m["PS Bonus/Start"] * cv("B48") + cv("B49") * m["PS BPS/Start"] * cv("B45"),
        m["PS BPS/Start"] * cv("B45"),
    )
    pct6 = np.where(
        m["Starts"] > cv("B46"),
        m["Bonus/Start"] * cv("B48") + cv("B49") * m["BPS/Start"] * cv("B45"),
        m["BPS/Start"] * cv("B45"),
    )
    m["6"] = ps_pct6 * cv("B51") + cv("B52") * pct6 * 1.0

    m["7"] = (m["LF xA/St"] * cv("B59") + m["SF xA/St"] * cv("B60")) * points.assist
    m["Goal"] = m["POS"].map(points.goal).fillna(0.0)
    m["8"] = (m["LF xG/St"] * cv("B59") + m["SF xG/St"] * cv("B60")) * m["Goal"]

    ts = team_strength.set_index(team_strength["ABVR"].str.upper())
    lf_xgc90 = m["TEAM"].str.upper().map(_num(ts["LF US xGC FIX"])).fillna(0.0)
    sf_xgc90 = m["TEAM"].str.upper().map(_num(ts["SF US xGC FIX"])).fillna(0.0)
    m["LF xGC/90"], m["SF xGC/90"] = lf_xgc90, sf_xgc90
    m["LF % CS/St"] = np.exp(-lf_xgc90 * m["G+A PSxTS Mn/Start"]) * m["2"]
    m["SF % CS/90"] = np.exp(-sf_xgc90 * m["G+A PSxTS Mn/Start"]) * m["2"]
    m["LF % 2+ GC"] = 1 - m["LF % CS/St"] * (1 + lf_xgc90)
    m["SF % 2+ GC"] = 1 - m["SF % CS/90"] * (1 + sf_xgc90)

    m["9"] = (m["LF % CS/St"] * cv("B72") + m["SF % CS/90"] * cv("B73")) * m["POS"].map(
        points.clean_sheet
    ).fillna(0.0)
    m["10"] = (m["LF % 2+ GC"] * cv("B78") + m["SF % 2+ GC"] * cv("B79")) * m["POS"].map(
        points.two_plus_conceded
    ).fillna(0.0)
    m["11"] = (m["LF DC %"] * cv("B86") + m["SF DC %"] * cv("B87")) * m["POS"].map(
        points.defensive_contribution
    ).fillna(0.0)

    m["TOTAL"] = m[[str(i) for i in range(1, 12)]].sum(axis=1)
    return m
