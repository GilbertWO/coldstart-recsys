"""Refresh the serving fallback list (7-day popularity) for a chosen as_of date.

The pure functions here are covered by test_refresh_fallback.py. The command-line entry point
reads the transactions, computes the list, checks it, and writes one row to
fallback_lists through psql in the container.
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

SOURCE = "fallback_popularity_7d"
K = 12


def recent_top(tx, as_of: dt.date, days: int = 7, k: int = 12) -> list:
    """Top-k article ids by purchase count in the `days` days ending at as_of (inclusive).

    The window is anchored to as_of, not to the last sale in the data: a purchase counts when
    as_of - days < t_dat <= as_of. Ties are broken by the lower article id.
    """
    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(days=days)
    in_window = (tx["t_dat"] > start) & (tx["t_dat"] < end + pd.Timedelta(days=1))
    counts = tx.loc[in_window, "article_id"].value_counts()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [int(article) for article, _ in ranked[:k]]


def validate_items(items) -> list:
    """Exactly 12 distinct integer ids, otherwise ValueError."""
    ids = [int(x) for x in items]
    if len(ids) != K or len(set(ids)) != K:
        raise ValueError(f"expected {K} distinct article ids, got {len(ids)} ({len(set(ids))} distinct)")
    return ids


def check_as_of(as_of: dt.date, data_min: dt.date, data_max: dt.date,
                current_latest: dt.date, days: int = 7) -> None:
    """Refuse an as_of the API would ignore or that the data cannot support (ValueError)."""
    if as_of > data_max:
        raise ValueError(f"as_of {as_of} is after the last day in the data ({data_max})")
    if as_of - dt.timedelta(days=days) < data_min:
        raise ValueError(f"the {days}-day window ending {as_of} starts before the data does ({data_min})")
    if as_of < current_latest:
        raise ValueError(
            f"as_of {as_of} is older than the list the API serves ({current_latest}); "
            "it would be written but never served"
        )


def build_insert_sql(as_of: dt.date, window_days: int, items) -> str:
    """One idempotent INSERT that carries model_version over from the latest existing row."""
    ids = validate_items(items)
    array = ",".join(str(i) for i in ids)
    return (
        "INSERT INTO fallback_lists (source, as_of, window_days, items, model_version) "
        f"SELECT source, DATE '{as_of.isoformat()}', {int(window_days)}, ARRAY[{array}], model_version "
        f"FROM fallback_lists WHERE source = '{SOURCE}' ORDER BY as_of DESC LIMIT 1 "
        "ON CONFLICT (source, as_of) DO UPDATE "
        "SET window_days = EXCLUDED.window_days, items = EXCLUDED.items"
    )


ROOT = Path(__file__).parent
PARTS = ("transactions_train_split.parquet", "transactions_val.parquet", "transactions_test.parquet")


def psql(sql: str, compose_file: Path) -> str:
    """Run one statement in the postgres container and return its trimmed output."""
    cmd = ["docker", "compose", "-f", str(compose_file), "exec", "-T", "postgres",
           "psql", "-U", "coldstart", "-d", "coldstart", "-v", "ON_ERROR_STOP=1", "-t", "-A", "-c", sql]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "psql failed")
    return result.stdout.strip()


def current_row(compose_file: Path):
    """(as_of, items) of the row the API serves, or None."""
    out = psql("SELECT as_of || ' ' || array_to_string(items, ',') FROM fallback_lists "
               f"WHERE source = '{SOURCE}' ORDER BY as_of DESC LIMIT 1", compose_file)
    if not out:
        return None
    day, items = out.split(" ", 1)
    return dt.date.fromisoformat(day), [int(x) for x in items.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--as-of", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="compute and check, write nothing")
    ap.add_argument("--compose-file", type=Path, default=ROOT / "serving" / "docker-compose.yml")
    args = ap.parse_args(argv)

    tx = pd.concat([pd.read_parquet(ROOT / "data" / "processed" / n, columns=["t_dat", "article_id"])
                    for n in PARTS], ignore_index=True)
    data_min, data_max = tx["t_dat"].min().date(), tx["t_dat"].max().date()

    current = current_row(args.compose_file)
    if current is None:
        print(f"no existing {SOURCE} row to carry the model version from; load the export first")
        return 1
    latest, old_items = current

    try:
        check_as_of(args.as_of, data_min, data_max, latest, args.days)
        items = validate_items(recent_top(tx, args.as_of, args.days, K))
    except ValueError as e:
        print(f"refusing: {e}")
        return 1

    window_start = args.as_of - dt.timedelta(days=args.days - 1)
    print(f"as_of          : {args.as_of} (window {window_start}..{args.as_of}, {args.days} days)")
    print(f"data range     : {data_min}..{data_max}")
    print(f"current latest : {latest} (the list the API serves)")
    print(f"current list   : {old_items}")
    print(f"new list       : {items}")
    print(f"articles kept  : {len(set(items) & set(old_items))} of {K}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    psql(build_insert_sql(args.as_of, args.days, items), args.compose_file)
    row = current_row(args.compose_file)
    if row != (args.as_of, items):
        print(f"WRITE CHECK FAILED: the latest row is {row}")
        return 1
    print("written and read back: the API serves the new list within about 60 seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
