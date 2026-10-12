"""Where the repository keeps its data.

One place to derive the `data/` tree from, so the modules that read or write
it (reference extracts, snapshots, the match-log cache, the elite archive,
the remembered entry id) agree on the layout.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"  # frozen workbook extracts (committed)
SNAPSHOT_DIR = DATA_DIR / "snapshots"  # dated FPL API pulls (gitignored)
ARCHIVE_DIR = DATA_DIR / "archive"  # elite-manager archive, its own git repo
