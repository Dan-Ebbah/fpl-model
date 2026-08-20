"""Extract the DCSCORE table (TABLES!C30:N50) and its league-average row
(TABLES!E53:N53) into data/reference/. These sit below the TABLES table
proper, so the sheet-level extraction misaligns their columns.
"""

import sys
from pathlib import Path

import openpyxl
import pandas as pd

WORKBOOK = Path.home() / "Downloads" / "MODEL.xlsx"
OUT = Path(__file__).resolve().parents[1] / "data" / "reference"


def main():
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    ws = wb["TABLES"]
    rows = list(ws.iter_rows(min_row=30, max_row=53, min_col=3, max_col=14, values_only=True))
    wb.close()

    headers = [str(h) for h in rows[0]]  # Team, ABVR, CB..CF
    body = [r for r in rows[1:21] if r[0] is not None]
    df = pd.DataFrame(body, columns=headers)
    df.to_parquet(OUT / "DCSCORE.parquet")

    avg_row = rows[23]  # sheet row 53: "25/26 Averages" starting at col C
    avg = pd.DataFrame([avg_row[2:]], columns=headers[2:])
    avg.to_parquet(OUT / "DCSCORE_league_avg.parquet")
    print(f"DCSCORE: {df.shape}, league avg: {avg.shape}")


if __name__ == "__main__":
    sys.exit(main())
