#!/usr/bin/env python
"""Run sql/feasibility_queries.sql against data/icon.duckdb and print
each query's result as a formatted table.

Queries are parsed out of the .sql file by their "-- N. Title" header
lines, so adding a new numbered query to the file picks it up automatically.

Usage:
    python scripts/run_queries.py [--limit 20]
"""
import argparse
import re
from pathlib import Path

import duckdb
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "icon.duckdb"
SQL_PATH = ROOT / "sql" / "feasibility_queries.sql"

HEADER_RE = re.compile(r"^-- (\d+)\.\s*(.+)$", re.MULTILINE)


def parse_queries(sql_text: str):
    headers = list(HEADER_RE.finditer(sql_text))
    if not headers:
        raise ValueError(f"No '-- N. Title' headers found in {SQL_PATH}")

    queries = []
    for i, match in enumerate(headers):
        number, title = match.group(1), match.group(2).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(sql_text)
        body = sql_text[start:end].strip()
        if body.endswith(";"):
            body = body[:-1].strip()
        queries.append((number, title, body))
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15, help="max rows to print per query")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found - run scripts/load_duckdb.py first")

    queries = parse_queries(SQL_PATH.read_text())
    con = duckdb.connect(str(DB_PATH), read_only=True)

    for number, title, body in queries:
        print(f"\n{'=' * 80}\nQuery {number}: {title}\n{'=' * 80}")
        df = con.execute(body).df()
        print(f"({len(df)} rows total, showing up to {args.limit})\n")
        print(tabulate(df.head(args.limit), headers="keys", tablefmt="psql", showindex=False))

    con.close()


if __name__ == "__main__":
    main()
