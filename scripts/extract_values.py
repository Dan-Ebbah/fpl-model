"""Extract cached (computed) values from MODEL.xlsx into data/reference/*.parquet.

These are the parity fixtures: each layer of the Python port is validated
against the spreadsheet's own computed numbers. The workbook is a pre-season
(2026-27 GW0) snapshot, so inputs are frozen — parity is exact, not "close".
"""

import sys
from pathlib import Path

import openpyxl
import pandas as pd

WORKBOOK = Path.home() / "Downloads" / "MODEL.xlsx"
OUT = Path(__file__).resolve().parents[1] / "data" / "reference"

# sheet -> (header_row, max_col or None for all)
SHEETS = {
    "API": (1, None),
    "PSAPI": (1, None),
    "TEAMS": (1, 25),
    "TABLE": (1, 30),
    "PSTABLE": (1, 30),
    "TABLES": (1, 80),
    "FIXTURES": (1, 80),
    "MODEL": (1, 250),
    "PREDICT": (2, 60),
    "PPts": (1, 70),
}


def extract(wb, name, header_row, max_col):
    ws = wb[name]
    rows = ws.iter_rows(min_row=header_row, max_col=max_col, values_only=True)
    headers = next(rows)
    # de-duplicate / fill unnamed headers so pandas columns stay addressable
    cols, seen = [], {}
    for i, h in enumerate(headers):
        h = str(h) if h is not None else f"_col{i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}.{seen[h]}"
        else:
            seen[h] = 0
        cols.append(h)
    data = [r for r in rows if any(v is not None for v in r)]
    df = pd.DataFrame(data, columns=cols)
    df = df.dropna(axis=1, how="all")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    for name, (hrow, maxcol) in SHEETS.items():
        df = extract(wb, name, hrow, maxcol)
        # parquet needs uniform types; the sheets mix numbers and strings freely
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).replace({"None": None})
        df.to_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols", flush=True)
    wb.close()


if __name__ == "__main__":
    sys.exit(main())
