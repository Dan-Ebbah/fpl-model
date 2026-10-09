"""Your own FPL team, reconstructed from the public API.

The manager endpoints are unauthenticated but incomplete: the picks for the
last deadline are public, the pre-deadline team and purchase prices are not.
This module rebuilds what the optimizer needs from what is public:

- **squad**: the 15 picked at the last deadline (`is_current` gameweek)
- **bank**: from that gameweek's entry history (bank only moves on transfers)
- **selling prices**: purchase price = the price paid in the latest transfer
  in for that player, else the start-of-season price (`now_cost -
  cost_change_start`); FPL pays back half of any profit, rounded down
- **free transfers**: replayed from the per-gameweek history with the
  2024-25+ rules (1 FT per week, unused ones bank up to 5, wildcard and
  free-hit weeks neither spend nor lose them)
- **chips**: one of each per half-season (GW1-19, GW20-38), minus those played

Anything already done for the *next* deadline (transfers made this week,
a chip activated) is invisible until that deadline passes — the report
says so. Use `--ft`/`--bank` overrides when the reconstruction is off.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import api
from .paths import DATA_DIR
from .rules import ELEMENT_TYPE_POS, FIRST_HALF_END, MAX_FREE_TRANSFERS, PRICE_UNIT, XI_SIZE

ME_PATH = DATA_DIR / "me.json"
CHIPS = ("wildcard", "freehit", "bboost", "3xc")


@dataclass
class Pick:
    code: int
    element: int
    name: str
    team: str
    pos: str
    purchase: float   # £m paid
    now: float        # £m current price
    sell: float       # £m you would get back
    is_captain: bool = False
    is_vice: bool = False
    bench_slot: int = 0  # 0 = started last deadline, 1-4 = bench order
    status: str = "a"
    chance: float = 1.0


@dataclass
class MyTeam:
    entry_id: int
    manager: str
    team_name: str
    gw: int             # gameweek the picks were locked for
    next_gw: int        # the deadline you are planning for
    picks: list[Pick]
    bank: float
    free_transfers: int
    chips_used: list[dict]
    chips_available: list[str]
    overall_rank: int | None
    total_points: int | None
    notes: list[str] = field(default_factory=list)

    @property
    def codes(self) -> set[int]:
        return {p.code for p in self.picks}

    @property
    def selling_prices(self) -> dict[int, float]:
        return {p.code: p.sell for p in self.picks}

    @property
    def squad_value(self) -> float:
        return round(sum(p.sell for p in self.picks) + self.bank, 1)


# --- entry id persistence ----------------------------------------------------


def remember_entry(entry_id: int, path: Path | None = None) -> None:
    """Save the entry id to `path` (default data/me.json) so later runs need no `--entry`."""
    path = path or ME_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entry": int(entry_id)}, indent=1))


def resolve_entry(entry_id: int | None = None, path: Path | None = None) -> int:
    """`--entry` flag > FPL_ENTRY env var > data/me.json (saved by a prior run)."""
    if entry_id:
        return int(entry_id)
    if os.environ.get("FPL_ENTRY"):
        return int(os.environ["FPL_ENTRY"])
    path = path or ME_PATH
    if path.exists():
        return int(json.loads(path.read_text())["entry"])
    raise SystemExit(
        "no FPL entry id: pass --entry <id> once (it is remembered in data/me.json). "
        "Find it in the URL of your Points page: fantasy.premierleague.com/entry/<id>/event/…"
    )


# --- pure reconstruction helpers (unit-tested on synthetic histories) ---------


def selling_price(purchase: float, now: float) -> float:
    """FPL sell rule: you get purchase + floor(profit / 2), in £0.1m steps."""
    # work in whole tenths so the floor is exact
    p, n = round(purchase * PRICE_UNIT), round(now * PRICE_UNIT)
    if n <= p:
        return n / PRICE_UNIT
    return (p + (n - p) // 2) / PRICE_UNIT


def chip_events(history: dict) -> dict[int, str]:
    """{gameweek: chip name} for every chip played this season."""
    return {int(c["event"]): c["name"] for c in history.get("chips", [])}


def free_transfers_after(history: dict, through_gw: int) -> int:
    """Free transfers available for the deadline after `through_gw`.

    2024-25+ rules: 1 FT from GW2, +1 each week, banked to a max of 5. A
    wildcard or free hit week spends nothing and still earns next week's +1.
    Transfers beyond the FT count are hits and leave you on 1 next week.
    """
    chips = chip_events(history)
    ft = 1  # what you hold going into GW2
    rows = sorted(history.get("current", []), key=lambda r: r["event"])
    for row in rows:
        gw = int(row["event"])
        if gw < 2 or gw > through_gw:
            continue
        used = 0 if chips.get(gw) in ("wildcard", "freehit") else int(row["event_transfers"])
        ft = min(MAX_FREE_TRANSFERS, max(ft - used, 0) + 1)
    return ft


def chips_available(history: dict, next_gw: int) -> list[str]:
    """Chips still playable at `next_gw`: one of each per half-season."""
    first_half = next_gw <= FIRST_HALF_END
    used = {
        name for gw, name in chip_events(history).items()
        if (gw <= FIRST_HALF_END) == first_half
    }
    return [c for c in CHIPS if c not in used]


def purchase_prices(
    picks: list[dict], transfers: list[dict], elements: dict[int, dict], history: dict
) -> dict[int, float]:
    """£m paid for each current pick.

    Latest transfer-in wins; free-hit-week transfers are reverted by the game
    so they are ignored; anything never transferred in was bought at the
    season's opening price.
    """
    chips = chip_events(history)
    bought: dict[int, int] = {}  # element -> cost in 0.1m
    for t in sorted(transfers, key=lambda t: (t["event"], t["time"])):
        if chips.get(int(t["event"])) == "freehit":
            continue
        bought[int(t["element_in"])] = int(t["element_in_cost"])
    out: dict[int, float] = {}
    for p in picks:
        el = elements[int(p["element"])]
        opening = int(el["now_cost"]) - int(el["cost_change_start"])
        out[int(p["element"])] = bought.get(int(p["element"]), opening) / PRICE_UNIT
    return out


def _chance_of_playing(element: dict) -> float:
    """The API's percentage flag as a fraction; an unflagged player is 1.0."""
    chance = element.get("chance_of_playing_next_round")
    return 1.0 if chance is None else float(chance) / 100


def build_team(
    entry_id: int, entry: dict, picks_payload: dict, history: dict,
    transfers: list[dict], bootstrap: dict, gw: int, next_gw: int,
) -> MyTeam:
    """Assemble a MyTeam from already-fetched payloads (no network)."""
    elements = {int(e["id"]): e for e in bootstrap["elements"]}
    teams = {int(t["id"]): t["short_name"] for t in bootstrap["teams"]}
    paid = purchase_prices(picks_payload["picks"], transfers, elements, history)

    picks: list[Pick] = []
    for p in sorted(picks_payload["picks"], key=lambda p: p["position"]):
        el = elements[int(p["element"])]
        now = int(el["now_cost"]) / PRICE_UNIT
        purchase = paid[int(p["element"])]
        picks.append(Pick(
            code=int(el["code"]), element=int(el["id"]), name=el["web_name"],
            team=teams[int(el["team"])], pos=ELEMENT_TYPE_POS[int(el["element_type"])],
            purchase=purchase, now=now, sell=selling_price(purchase, now),
            is_captain=bool(p["is_captain"]), is_vice=bool(p["is_vice_captain"]),
            bench_slot=max(0, int(p["position"]) - XI_SIZE),
            status=el.get("status", "a"),
            chance=_chance_of_playing(el),
        ))

    entry_history = picks_payload.get("entry_history", {})
    notes = [
        f"Squad as locked for GW{gw}; transfers or chips already set for GW{next_gw} "
        "are not public until the deadline passes."
    ]
    flagged = [f"{p.name} ({p.chance:.0%})" for p in picks if p.chance < 1]
    if flagged:
        notes.append("Availability flags: " + ", ".join(flagged) + ".")

    return MyTeam(
        entry_id=entry_id,
        manager=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        team_name=entry.get("name", ""),
        gw=gw, next_gw=next_gw, picks=picks,
        bank=int(entry_history.get("bank", entry.get("last_deadline_bank", 0) or 0)) / PRICE_UNIT,
        free_transfers=free_transfers_after(history, gw),
        chips_used=[{"name": n, "gw": g} for g, n in sorted(chip_events(history).items())],
        chips_available=chips_available(history, next_gw),
        overall_rank=entry.get("summary_overall_rank"),
        total_points=entry.get("summary_overall_points"),
        notes=notes,
    )


def current_and_next_gw(bootstrap: dict) -> tuple[int | None, int]:
    """(gameweek whose deadline last passed, next deadline gameweek)."""
    events = bootstrap["events"]
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    if nxt is None:  # season over: plan for the last gameweek
        nxt = events[-1]["id"]
    return current, int(nxt)


def load_my_team(entry_id: int, bootstrap: dict | None = None) -> MyTeam:
    """Fetch everything about one entry and reconstruct its current state."""
    bootstrap = bootstrap or api.fetch_bootstrap()
    current, next_gw = current_and_next_gw(bootstrap)
    if current is None:
        raise SystemExit(
            "the season has not started — picks are only public after the GW1 deadline. "
            "Use `fpl squad` to build a squad from scratch."
        )
    picks = api.fetch_entry_picks(entry_id, current)
    if picks is None:
        raise SystemExit(f"entry {entry_id} has no picks for GW{current} — wrong id?")
    entry = api.fetch_entry(entry_id)
    history = api.fetch_entry_history(entry_id)
    transfers = api.fetch_entry_transfers(entry_id)
    return build_team(entry_id, entry, picks, history, transfers, bootstrap, current, next_gw)
