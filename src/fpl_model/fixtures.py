"""Fixture slots: which opponent each team faces in each gameweek slot.

Port of the workbook's FIXHELPER table. Each gameweek has up to two "slots"
per team (slot 2 exists only in double gameweeks). Opponent codes carry the
venue in their case, matching the spreadsheet's convention: uppercase =
playing at home, lowercase = away. NO_FIXTURE ("-") marks a blank gameweek.
"""

from __future__ import annotations

import pandas as pd

SLOTS_PER_GW = 2  # fixture slots per team per gameweek: a double gameweek fills both
NO_FIXTURE = "-"


def build_fixture_slots(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (team_abrv, gw, slot) with the opponent and venue.

    `fixtures` is the raw FPL fixtures list (event, team_h, team_a) in API
    order; `teams` maps team id -> short_name. Slot numbering follows the
    workbook: a team's fixtures within a gameweek are numbered 1, 2 in list
    order, counting home and away appearances together.

    Columns: team, gw, slot, vs (opponent code, cased by venue: uppercase at
    home, lowercase away — the PPts port reads the venue from the case).
    """
    id_to_abrv = dict(zip(teams["id"].astype(int), teams["short_name"]))

    rows = []
    slot_counter: dict[tuple[str, int], int] = {}
    for fx in fixtures.itertuples(index=False):
        if pd.isna(fx.event):
            continue  # unscheduled (postponed) fixtures have no gameweek yet
        gw = int(fx.event)
        home = id_to_abrv[int(fx.team_h)]
        away = id_to_abrv[int(fx.team_a)]
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            slot = slot_counter.get((team, gw), 0) + 1
            slot_counter[(team, gw)] = slot
            rows.append(
                {
                    "team": team,
                    "gw": gw,
                    "slot": slot,
                    "vs": opponent if is_home else opponent.lower(),
                }
            )
    return pd.DataFrame(rows)


def opponent_grid(fixture_slots: pd.DataFrame, gws: range) -> pd.DataFrame:
    """Dense (team, gw, slot) grid over `gws`, NO_FIXTURE where no game exists.

    Every team gets `SLOTS_PER_GW` rows per gameweek (two covers double
    gameweeks), so blank slots exist as rows — the PPts
    sheet carries them too and zeroes their totals. Teams are sorted, so the
    grid order is (team, gw, slot) ascending.
    """
    teams = sorted(fixture_slots["team"].unique())
    grid = pd.MultiIndex.from_product(
        [teams, list(gws), range(1, SLOTS_PER_GW + 1)], names=["team", "gw", "slot"]
    ).to_frame(index=False)
    grid = grid.merge(fixture_slots, on=["team", "gw", "slot"], how="left")
    grid["vs"] = grid["vs"].fillna(NO_FIXTURE)
    return grid
