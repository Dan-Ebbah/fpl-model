"""The rules of the game, named once.

Everything here is a fact about Fantasy Premier League (squad shape, the
transfer market, the calendar) or a default the tools agree on. Model
parameters — anything that came from the workbook's CONTROL sheet or the
in-season layer — live in `config`, `inseason` and `style`, not here.
"""

from __future__ import annotations

# --- positions ---------------------------------------------------------------
POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
OUTFIELD: tuple[str, ...] = ("DEF", "MID", "FWD")
POSITION_ORDER: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
ELEMENT_TYPE_POS: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}  # API element_type

# --- squad -------------------------------------------------------------------
SQUAD_SIZE = 15
SQUAD_SHAPE: dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
MAX_PER_CLUB = 3
BUDGET = 100.0  # £m, a fresh squad
PRICE_UNIT = 10  # the API quotes prices in £0.1m

# --- transfers ---------------------------------------------------------------
MAX_FREE_TRANSFERS = 5  # banked transfers cap (2024-25 rules onward)
HIT_COST = 4.0  # points per transfer beyond the free ones
DEFAULT_FT_VALUE = 1.5  # xP a free transfer must beat to be worth spending
DEFAULT_BENCH_WEIGHT = 0.1  # share of bench xP the optimizer counts

# --- calendar ----------------------------------------------------------------
GAMEWEEKS = 38
FIRST_GW = 1
LAST_GW = GAMEWEEKS
FIRST_HALF_END = 19  # chips reset after this gameweek
DEFAULT_HORIZON = 4  # planning window: next gameweek plus this many
MINUTES_PER_MATCH = 90
