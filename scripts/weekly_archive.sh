#!/bin/zsh
# Weekly elite-manager archive pass + off-machine backup.
# Run by launchd (com.fpl-model.archive) every Tuesday 10:00; absolute paths
# because launchd provides a bare environment.
set -euo pipefail

cd "$(dirname "$0:A")/.."
# weekly API snapshot: the dated series in data/snapshots/ is what the
# in-season data layer diffs into per-GW values, so every week needs one
# even if no `fpl squad --refresh` ran
/Users/danieldan-ebbah/.local/bin/uv run python -m fpl_model.api
/Users/danieldan-ebbah/.local/bin/uv run fpl archive

cd data/archive
/usr/bin/git add -A
if ! /usr/bin/git diff --cached --quiet; then
  /usr/bin/git commit -m "Archive pass $(date +%F)"
  /usr/bin/git push
fi
