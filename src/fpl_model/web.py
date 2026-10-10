"""Local web UI: a thin browser layer over the prediction + optimizer pipeline.

`fpl web` serves a single-page app on localhost. Standard-library HTTP server
only — this is a single-user local tool, so no framework dependency is
warranted. The payload builders live in `payloads` (plain functions, testable
without a server); this module is the handler that maps URLs onto them.

Projections are cached per data source (workbook snapshot vs live pull) for
the life of the process; the Refresh button re-pulls the live API.
"""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


from . import payloads
from .rules import DEFAULT_BENCH_WEIGHT, DEFAULT_FT_VALUE, FIRST_GW, HIT_COST, LAST_GW

HTML_PATH = Path(__file__).parent / "web.html"

DEFAULT_PORT = 8765
DEFAULT_WINDOW_END = 8  # `to` when a query leaves it out (the Player tab uses LAST_GW)

# ---------------------------------------------------------------------------
# Routes: query-string dict -> payload. Each parses its own parameters so the
# defaults stay next to the call they feed.
# ---------------------------------------------------------------------------

Query = dict[str, str]


def _names(q: Query, key: str) -> list[str]:
    """A comma-separated list parameter, blanks dropped."""
    return [s for s in q.get(key, "").split(",") if s.strip()]


def _opt(q: Query, key: str, parse: Callable):
    """An optional parameter: None when absent or empty, else parsed."""
    return parse(q[key]) if q.get(key) not in (None, "") else None


def _flag(q: Query, key: str) -> bool:
    return q.get(key) == "1"


def _predictions(q: Query) -> dict:
    return payloads.predictions_payload(
        int(q.get("from", FIRST_GW)), int(q.get("to", DEFAULT_WINDOW_END)),
        _flag(q, "refresh"), q.get("pos") or None,
        q.get("team") or None,
        float(q["max_price"]) if q.get("max_price") else None,
        q.get("sort", "pts"), force=_flag(q, "force"),
    )


def _squad(q: Query) -> dict:
    return payloads.squad_payload(
        int(q.get("from", FIRST_GW)), int(q.get("to", DEFAULT_WINDOW_END)),
        _flag(q, "refresh"), float(q.get("bench_weight", DEFAULT_BENCH_WEIGHT)),
        _names(q, "lock"), _names(q, "exclude"), _flag(q, "force"),
    )


def _team(q: Query) -> dict:
    return payloads.team_payload(
        _opt(q, "entry", int), _opt(q, "to", int), float(q.get("bench_weight", DEFAULT_BENCH_WEIGHT)),
        _opt(q, "ft", int), _opt(q, "bank", float), _opt(q, "max_transfers", int),
        float(q.get("hit_cost", HIT_COST)), _flag(q, "wildcard"),
        _names(q, "lock"), _names(q, "exclude"), _flag(q, "force"),
        float(q.get("ft_value", DEFAULT_FT_VALUE)),
    )


def _compare(q: Query) -> dict:
    return payloads.compare_payload(
        q.get("a", ""), q.get("b", ""), int(q.get("from", FIRST_GW)),
        int(q.get("to", DEFAULT_WINDOW_END)), _flag(q, "refresh"),
    )


def _verdict(q: Query) -> dict:
    return payloads.verdict_payload(
        q.get("name", ""), int(q.get("from", FIRST_GW)),
        int(q.get("to", DEFAULT_WINDOW_END)), _flag(q, "refresh"),
    )


def _sandbox(q: Query) -> dict:
    return payloads.sandbox_payload(_opt(q, "entry", int), _opt(q, "to", int), _flag(q, "force"))


def _player(q: Query) -> dict:
    return payloads.player_payload(
        q.get("name", ""), int(q.get("from", FIRST_GW)),
        int(q.get("to", LAST_GW)), _flag(q, "refresh"),
    )


_ROUTES: dict[str, Callable[[Query], dict]] = {
    "/api/predictions": _predictions,
    "/api/squad": _squad,
    "/api/team": _team,
    "/api/compare": _compare,
    "/api/verdict": _verdict,
    "/api/sandbox": _sandbox,
    "/api/player": _player,
}


class Handler(BaseHTTPRequestHandler):
    """GET-only: `/` serves the page, `/api/*` a JSON payload from `_ROUTES`.

    Bad input (a name that matches nothing or several players, an unparsable
    number) comes back as 400 with the message; anything else as 500 so the
    UI can show it.
    """

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif url.path in _ROUTES:
                self._json(200, _ROUTES[url.path](q))
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, SystemExit) as e:  # bad input / ambiguous name
            self._json(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover - surfaced in the UI
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Serve the UI on 127.0.0.1:`port` until ctrl-C, opening a browser tab unless told not to."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"fpl web UI at {url} (ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
