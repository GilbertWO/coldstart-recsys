#!/usr/bin/env python3
"""Write the local Postman environment that holds real ids from the Kaggle data.

The Postman collection leaves knownId, knownRepurchase and knownDiscover empty because they
are Competition Data, which the H&M competition rules (section 7.B) do not allow to be
redistributed. This script fills them from the local export instead.

Run from the repo root:

    python serving/postman/make_local_env.py

Rule: knownId is the customer on the first data row of recommendations.csv; the two lists
are that customer's repurchase and discover rows, read from the same file. The output,
serving/postman/local.postman_environment.json, is gitignored. In Postman, import it and
select it as the active environment before running the collection. Re-run after a new export.
Standard library only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_CSV = Path("models/serving_export/recommendations.csv")
DEFAULT_OUT = Path("serving/postman/local.postman_environment.json")


def parse_items(literal):
    """'{1,2,3}' (PostgreSQL array literal) -> [1, 2, 3]."""
    return [int(x) for x in literal.strip("{}").split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.csv.exists():
        sys.exit(f"missing {args.csv}. Run from the repo root after export_recommendations.py.")

    known_id, lists = None, {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if known_id is None:
                known_id = row["customer_id"]
            if row["customer_id"] == known_id:
                lists[row["mode"]] = parse_items(row["items"])
                if "repurchase" in lists and "discover" in lists:
                    break
    missing = {"repurchase", "discover"} - lists.keys()
    if known_id is None or missing:
        sys.exit(f"could not find both modes for the first customer (missing: {sorted(missing)})")
    for mode, items in lists.items():
        if len(items) != 12 or len(set(items)) != 12:
            sys.exit(f"{mode} list is not 12 distinct ids: {len(items)} items")

    env = {
        "name": "coldstart-recsys local (real ids, do not commit)",
        "values": [
            {"key": "knownId", "value": known_id, "type": "default", "enabled": True},
            {"key": "knownRepurchase", "value": json.dumps(lists["repurchase"]), "type": "default", "enabled": True},
            {"key": "knownDiscover", "value": json.dumps(lists["discover"]), "type": "default", "enabled": True},
        ],
        "_postman_variable_scope": "environment",
    }
    args.out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: knownId = first customer in {args.csv.name}, 12 + 12 items")


if __name__ == "__main__":
    main()
