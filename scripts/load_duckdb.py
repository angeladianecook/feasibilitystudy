#!/usr/bin/env python
"""Load the four feasibility CSVs into a local DuckDB database.

Reads data/{studies,sites,enrollment,patients}.csv (produced by
scripts/generate_data.py) and writes data/icon.duckdb, replacing any
existing tables of the same name.

Usage:
    python scripts/load_duckdb.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "icon.duckdb"

TABLES = ["studies", "sites", "enrollment", "patients"]


def main():
    missing = [t for t in TABLES if not (DATA_DIR / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSVs {missing} in {DATA_DIR} - run scripts/generate_data.py first"
        )

    con = duckdb.connect(str(DB_PATH))
    for table in TABLES:
        csv_path = DATA_DIR / f"{table}.csv"
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS "
            f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,} rows loaded")
    con.close()

    print(f"\nDuckDB database written to: {DB_PATH}")


if __name__ == "__main__":
    main()
