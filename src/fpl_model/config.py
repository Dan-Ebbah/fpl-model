"""Model parameters, ported from the workbook's CONTROL sheet.

Every weight here is a tunable the spreadsheet exposed as a cell. Defaults are
the workbook's pre-season (2026-27 GW0) values: previous-season data weighted
1.0 everywhere because no current-season data exists yet. As the season
progresses these shift toward current-season data — and once the backtester
exists they should be fit, not hand-set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rules

POSITIONS = rules.POSITIONS  # kept as a name here; the tuple now lives in rules


@dataclass(frozen=True)
class PositionPoints:
    """FPL scoring by position (CONTROL element_type table)."""

    goal: dict[str, float] = field(
        default_factory=lambda: {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
    )
    clean_sheet: dict[str, float] = field(
        default_factory=lambda: {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    )
    two_plus_conceded: dict[str, float] = field(
        default_factory=lambda: {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0}
    )
    defensive_contribution: dict[str, float] = field(
        default_factory=lambda: {"GK": 0, "DEF": 2, "MID": 2, "FWD": 2}
    )
    assist: float = 3.0


@dataclass(frozen=True)
class ControlParams:
    """Blend weights and multipliers (CONTROL sheet cells noted per field).

    These are the cells PPts reads. `build_model` reads the rest of the CONTROL
    sheet directly from the `CONTROL_values` extract — including B8/B9, which
    it uses for the appearance blend ("1") independently of
    `starts_rate_prev_season_weight` here, so the two can disagree.
    """

    # FDR: long-form (38 GW) vs short-form (6 GW) opponent strength blend (B97/B98)
    fdr_long_form_weight: float = 0.8

    # League-average xG/xGC: previous vs current season blend (B110/B111)
    league_avg_prev_season_weight: float = 1.0

    # FPL assists run ~40% above xA over recent seasons (B65)
    fantasy_assist_boost: float = 0.4

    # Starts/squads-made rate: previous vs current season blend (B8/B9)
    starts_rate_prev_season_weight: float = 1.0

    # Home/away multipliers on the fixture total (PPts H/A column)
    home_multiplier: float = 1.05
    away_multiplier: float = 0.95

    points: PositionPoints = field(default_factory=PositionPoints)

    @property
    def fdr_short_form_weight(self) -> float:
        return 1.0 - self.fdr_long_form_weight

    @property
    def league_avg_this_season_weight(self) -> float:
        return 1.0 - self.league_avg_prev_season_weight

    @property
    def starts_rate_this_season_weight(self) -> float:
        return 1.0 - self.starts_rate_prev_season_weight


DEFAULT_PARAMS = ControlParams()
