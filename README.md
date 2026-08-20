# fpl-model

Python port of the Excel FPL expected-points model (MODEL.xlsx from the video),
on its way to a full predict-then-optimize pipeline: per-gameweek expected
points → planning-window aggregation → squad optimization.

## Status

- **Ported and parity-verified against the workbook** (pre-season 2026-27 snapshot):
  - `fpl_model.ppts` — the PPts engine: expected points per player per gameweek
    per fixture slot (double gameweeks supported), all 11 FPL scoring categories,
    opponent-strength and home/away adjusted, marginalized over start probability.
    Matches all 76,000 workbook rows to 1e-9.
  - `fpl_model.predict` — planning-window aggregation (the PREDICT sheet).
  - `fpl_model.fixtures` — fixture slot builder (the FIXHELPER table).
  - `fpl_model.config` — the CONTROL sheet as typed parameters.
  - `fpl_model.api` — live FPL API client + dated snapshot store.
  - `fpl_model.optimizer` — PuLP/CBC squad MILP (replaces the Excel Solver
    workbook): optimal 15 under budget/squad/club rules, with per-gameweek
    XI, formation, and captain choice; lock/exclude support; live price and
    availability refresh via `fpl squad --refresh`.
- **Not yet ported**: the MODEL-sheet player aggregates (currently consumed as
  extracted values), team strength tables from raw per-match data, the
  backtester.
- `fpl_model.archive` — weekly elite-manager archiver. The FPL API deletes all
  manager picks/transfers/chips at the July season rollover, so `fpl archive`
  captures the overall-league leaders (cohort = union of everyone ever in the
  top N) while the season runs; each pass backfills missing gameweeks, so a
  missed week self-heals. `data/archive/` is gitignored here but is its own
  git repo, backed up to the private `Dan-Ebbah/fpl-archive` GitHub repo;
  `scripts/weekly_archive.sh` (run by the `com.fpl-model.archive` launchd
  agent, Mondays 10:00) does the archive pass then commits and pushes.

## Layout

- `src/fpl_model/` — the package
- `data/reference/` — frozen values extracted from the workbook; parity fixtures
  for the tests and interim model inputs
- `docs/formulas/` — the workbook's formulas (one JSON per sheet); the port spec
- `scripts/extract_*.py` — the extractors that produced both of the above
- `tests/` — parity tests against the workbook's own computed numbers

## Usage

```bash
uv sync
uv run pytest                      # parity suite
uv run python -m fpl_model.api     # snapshot today's FPL API data
uv run fpl archive                 # weekly: archive elite managers' seasons
```

## Faithful-port policy

The port reproduces the spreadsheet bug-for-bug (see notes in `ppts.py` —
e.g. the home/away multiplier is applied twice on the sub-appearance path).
Improvements go in only once the backtester exists to measure them, and each
deliberate deviation must update the parity tests explicitly.
