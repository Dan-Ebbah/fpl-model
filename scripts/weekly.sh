#!/bin/zsh
# The weekly in-season job, as one script that runs each step on its own.
#
# Run by launchd (com.fpl-model.archive) every Tuesday 10:00 — after the
# weekend's gameweek has finished and before the next deadline — and safe to
# run by hand any time (`scripts/weekly.sh`, or `scripts/weekly.sh backtest`
# for one step). Absolute paths because launchd provides a bare environment.
#
# Steps, in order. Each runs in its own subshell: one failing does not stop
# the rest, and the summary at the end says which did; the exit status is
# non-zero if any did.
#
#   snapshot  pull today's FPL API data into data/snapshots/<date>/ — the
#             dated series the backtester replays from (a gameweek with no
#             snapshot before its deadline can never be scored) and the raw
#             material for the in-season data layer
#   archive   elite-manager picks/transfers/chips for the week (the API
#             deletes them at the season rollover)
#   backtest  replay every finished gameweek from its pre-deadline snapshot
#             and score each model layer; refreshes data/snapshots/backtest.parquet,
#             which feeds the track-record tables in the web UI
#   backup    commit and push the archive repo (data/archive) to GitHub
set -uo pipefail

ROOT="$(cd "$(dirname "$0:A")/.." && pwd)"
UV=/Users/danieldan-ebbah/.local/bin/uv
# Apple's /usr/bin/git refuses to run until the Xcode licence is accepted;
# prefer a Homebrew git when there is one.
GIT=/usr/bin/git
for g in /opt/homebrew/bin/git /usr/local/bin/git; do
  [[ -x "$g" ]] && { GIT="$g"; break; }
done

cd "$ROOT"
export PYTHONUNBUFFERED=1

step_snapshot() { "$UV" run python -m fpl_model.api; }
step_archive()  { "$UV" run fpl archive; }
step_backtest() { "$UV" run fpl backtest; }
step_backup() {
  cd "$ROOT/data/archive"
  "$GIT" add -A
  if "$GIT" diff --cached --quiet; then
    echo "archive repo: nothing new to commit"
  else
    "$GIT" commit -m "Archive pass $(date +%F)"
    "$GIT" push
  fi
}

ALL=(snapshot archive backtest backup)
STEPS=("$@")
(( ${#STEPS[@]} )) || STEPS=("${ALL[@]}")

typeset -A result
failed=0
echo "=== fpl-model weekly job, $(date '+%F %T') ==="
for s in "${STEPS[@]}"; do
  if ! typeset -f "step_$s" >/dev/null; then
    echo "unknown step: $s (choose from: ${ALL[*]})" >&2
    result[$s]=unknown; failed=1; continue
  fi
  echo; echo "--- $s ($(date '+%T'))"
  if ( "step_$s" ); then
    result[$s]=ok
  else
    result[$s]="FAILED (exit $?)"; failed=1
  fi
done

echo; echo "=== summary ==="
for s in "${STEPS[@]}"; do
  printf '  %-9s %s\n' "$s" "${result[$s]}"
done
exit $failed
