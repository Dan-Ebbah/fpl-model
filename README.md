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
  - `fpl_model.model_build` — the MODEL sheet itself: rebuilds the player
    aggregate table from a live bootstrap pull (current clubs, positions,
    prices, injury flags, new registrations) plus the frozen last-season
    inputs (PSAPI, FBRef playing-time tables, per-GW pivot deltas, CONTROL
    values). `--refresh` now runs this instead of patching the extract, and
    pulls live fixtures too. Every `--refresh` pull is persisted to
    `data/snapshots/<date>/` (as is a weekly pull by the Tuesday launchd job) —
    that dated series is the raw material for the in-season data layer.
- `fpl_model.team` + `fpl_model.plan` — **your own team**. `fpl team --entry
  <id>` reconstructs your squad from the public manager endpoints (picks at
  the last deadline, bank, selling prices from the transfer history and
  opening prices, free transfers replayed under the 2024-25+ banking rules,
  chips left per half-season) and runs the optimizer in transfer mode:
  budget = bank + sales, hits at 4 points, a small penalty per free transfer
  used (`--ft-value`, banked transfers have option value), `--max-transfers`,
  `--wildcard`. Output: recommended transfers with the net xP gain over
  standing pat, next-GW XI with captain, vice and bench order, captain by
  gameweek. The entry id is remembered in `data/me.json` (gitignored). The
  web UI's **My Team** tab is the same thing with a form. Caveat: anything
  you have already done for the *next* deadline is not public until it
  passes, so run it before you make your moves, or override `--ft`/`--bank`.
- `fpl_model.inseason` — **current-season evidence on the live path.** The
  port is a pre-season model (every CONTROL weight on this season is 0), so
  on `--refresh`/`fpl team` three Bayesian blends run between `build_model`
  and `compute_ppts`: schedule-adjusted **team form** (this season's xG/xGC vs
  what prior strength predicted for the fixtures actually played, shrunk with
  10 pseudo-games and folded into the team strength table), **xG/xA rates**
  (position-group norm scaled by team attack + last season + this season at
  1.5x, so 100-minute flukes and cold/hot starts are both handled), and
  **start rates** (this season's starts over team games blended into last
  season's selection rate). `fpl form` prints the team multipliers in use;
  `fpl team --no-inseason` gives the pre-season model for comparison. The
  constants are reasoned defaults, not fitted — first job for the backtester.
- `fpl_model.matchlog` + `fpl_model.style` — **team playing style.** A cached
  per-match player log (one element-summary call per player, refetched only
  when his minutes move; ~20s after a gameweek, instant otherwise) feeds
  three shrunk factors: **venue tilt** (is this side more or less of a home
  team than the league — relative, because the port already applies a flat
  home/away multiplier), **attacking channels** (share of team xG/xA by
  position; sets the norm a player's rate is shrunk toward) and **concession
  channels** (which positions a team leaks chances to; scales the opponent's
  goal/assist expectation by position, capped at 0.8-1.25). They enter
  `compute_ppts` through its one extension point, `fixture_adj` (None = the
  workbook's numbers exactly). This-season-only data, so after four games
  everything sits within a few percent of 1.0 and opens up with the sample.
  `fpl style [--team NEW]` prints the profile; `fpl scorers [--gw N] [--team
  NEW]` is the **anytime scorer / assist predictor** built on it (Poisson on
  each player's fixture xG / FPL-assist expectation, team minutes normalised
  to ten outfielders so the table reconciles with a plausible scoreline).
- `fpl_model.explain` — **the reasoning, written out.** Every recommended
  transfer in `fpl team` (and the web My Team tab) lists where the xP gap
  comes from by scoring category, the effect on your XI, the fixture run,
  the team's form, the rate evidence (last season / this season / norm /
  used), minutes and rotation risks (flagged rivals returning, set-piece
  duty), the alternatives that lost, then a verdict on whether the gain
  clears the value of rolling, and how close the captain call is.
- `fpl_model.backtest` — **the backtester.** `fpl backtest` replays every
  finished gameweek from the last snapshot stored *before* its deadline (a
  snapshot qualifies only if its own bootstrap marks that gameweek as next —
  no clocks), with the match log cut off before the gameweek and
  `priors.yaml` off, then scores the bare port, each in-season layer added
  alone, all of them, and a naive last-season-PPG baseline on the same
  players: MAE/bias, within-position rank correlation, actual points of the
  ten best-projected per position, and Brier scores for start / goal / assist
  probabilities (scored on every player, so correct 0% calls count), plus a
  start-chance calibration table. Every difference from the port carries a
  paired-bootstrap 90% interval and a plain verdict — "can't be told apart"
  is a valid and common one. A week with no snapshot is skipped for good,
  which is why the weekly job matters. Variants are `inseason.Layers` values,
  so sweeping a constant is one more dict entry in `backtest.VARIANTS`. The
  per-player table lands in `data/snapshots/backtest.parquet`.
- **Not yet ported**: team strength tables from raw per-match data, the
  snapshot-delta data layer (per-GW columns from weekly snapshots — the
  in-season layer above uses season totals from the bootstrap instead).
  Backtester still to come: constant sweeps as a CLI flag, week-level
  (cluster) intervals once there are enough weeks, and the decision test
  (replaying `fpl team` against holding and against the elite archive).
- `fpl_model.archive` — weekly elite-manager archiver. The FPL API deletes all
  manager picks/transfers/chips at the July season rollover, so `fpl archive`
  captures the overall-league leaders (cohort = union of everyone ever in the
  top N) while the season runs; each pass backfills missing gameweeks, so a
  missed week self-heals. `data/archive/` is gitignored here but is its own
  git repo, backed up to the private `Dan-Ebbah/fpl-archive` GitHub repo;
  `scripts/weekly.sh` (run by the `com.fpl-model.archive` launchd agent,
  Tuesdays 10:00, log in `data/weekly.log`) is the whole weekly job: API
  snapshot, archive pass, backtest replay, then commit and push the archive
  repo — each step on its own, so one failing does not stop the rest, with a
  summary and non-zero exit if any did. `scripts/weekly.sh backtest` runs
  one step by hand.

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
uv run fpl web                     # browser UI: solve squads, rank players
uv run fpl squad --to-gw 8 --refresh   # same from the terminal
uv run fpl team --entry 1234567    # YOUR team: transfers, XI, captain (id remembered)
uv run fpl web                     # then open http://127.0.0.1:8765/#sandbox — what-if squad sandbox
uv run fpl team --max-transfers 1 --to-gw 12
uv run fpl form                    # team form multipliers the model is applying
uv run fpl style --team NEW        # home/away tilt, which positions the chances go through
uv run fpl scorers --gw 5          # anytime goalscorer / assist probabilities
uv run fpl backtest                # replay past deadlines, score each model layer (~1 min)
uv run python -m fpl_model.api     # snapshot today's FPL API data
uv run fpl archive                 # weekly: archive elite managers' seasons
scripts/weekly.sh [step...]        # the Tuesday job by hand: snapshot, archive, backtest, backup
```

`fpl web` (src/fpl_model/web.py + web.html) serves a local single-page UI on
127.0.0.1:8765 — squad solver with lock/exclude and bench weight, ranked
predictions with filters, and per-player gameweek breakdowns. Every player
name is clickable and opens the verdict view: the model's written opinion
(rank, minutes security, value, fixture shape, and low-confidence caveats for
priors-backed players). The Compare tab (or a squad row's "vs…" link) puts
any two players head-to-head: verdict cards side by side, per-gameweek
fixtures and xP, and the window/price deltas. "Live data" uses the same refresh path as the CLI
(and stores a snapshot per pull); untick it to run off the frozen workbook
snapshot.

The **Sandbox** tab is the what-if view: your squad as cards, each with its
window xP and a per-gameweek sparkline, the best XI / formation / captain for
every gameweek recomputed in the browser from the model's per-GW projections
(same objective as the optimizer's lineup step, so an untouched squad matches
`fpl team`'s no-transfer baseline exactly). Click a player for his xP split by
scoring category, gameweek-by-gameweek fixtures with start chance and xG/xA,
this season's actual numbers from the FPL API (minutes, starts, xGI, defensive
contribution, bonus, ownership, price moves, set-piece duty), his club's
schedule-adjusted form, and a swap list of every player in his position
ranked by xP with the delta, filtered to what you can afford. Swaps update the
window xP, net gain, transfers used, hit cost and bank instantly; `Reset
squad` goes back. `/#sandbox/Tarkowski` deep-links to a player. Every player
detail (Sandbox and Player tabs) also carries a **track record**: for each
finished gameweek, the xP the model gave him from the last snapshot stored
before that deadline (the backtester's replay, so no hindsight — the live
model and the bare port side by side), next to his actual points, minutes
and goal involvement. Run `fpl backtest` after each gameweek to extend it.

## Faithful-port policy

The port reproduces the spreadsheet bug-for-bug (see notes in `ppts.py` —
e.g. the home/away multiplier is applied twice on the sub-appearance path).
Improvements go in only once the backtester exists to measure them, and each
deliberate deviation must update the parity tests explicitly.

One deliberate exception: `inseason.py` (and `style.py`, same rules). A pre-season model four gameweeks in
is wrong in known directions, so the in-season layer ships ahead of the
backtester — but outside the ported functions. `build_model` and
`compute_ppts` are untouched and still parity-tested; the layer only edits
their inputs, only on live data, and `--no-inseason` switches it off.
