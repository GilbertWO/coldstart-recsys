#!/usr/bin/env python3
"""Parity check: does the live endpoint return exactly what the offline export wrote?

Run from the repo root with the API running (default http://localhost:8081):

    python serving/parity_check_api.py
    python serving/parity_check_api.py --samples 2000 --seed 7

What it checks:
  1. A random sample of (customer, mode) rows from recommendations.csv: the endpoint must
     return HTTP 200, source "als", the same mode, the same model version and exactly the
     same 12 article ids in the same order.
  2. Random valid-format ids that are not in the table: HTTP 200, source
     fallback_popularity_7d, the requested mode echoed, and the list from fallback.csv.
  3. Four malformed requests: each must return HTTP 400.

It reads the export CSV once (streaming, reservoir sampling), so memory stays small.
Standard library only. Exit code 0 means every check passed, 1 means at least one failed.
"""
import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_EXPORT_DIR = Path("models/serving_export")
FALLBACK_SOURCE = "fallback_popularity_7d"


def parse_items(text):
    """Postgres array literal '{1,2,3}' -> [1, 2, 3]."""
    return [int(x) for x in text.strip().strip("{}").split(",")]


def reservoir_sample(path, n, rng):
    sample, total = [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            total += 1
            if i < n:
                sample.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = row
    return sample, total


def read_fallback(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one row in {path}, found {len(rows)}")
    row = rows[0]
    if row["source"] != FALLBACK_SOURCE:
        raise SystemExit(f"unexpected fallback source {row['source']!r}")
    return parse_items(row["items"]), row["model_version"]


def call(base, path):
    """Return (status, parsed_json_or_None)."""
    try:
        with urllib.request.urlopen(base + path, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = None
        return e.code, body


def check_server(base):
    try:
        status, _ = call(base, "/actuator/health")
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise SystemExit(f"cannot reach {base} ({reason}). Is the app running, with POSTGRES_PASSWORD set?")
    if status != 200:
        raise SystemExit(f"{base}/actuator/health returned {status}; fix the app before checking parity")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8081")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--samples", type=int, default=600, help="rows sampled from recommendations.csv")
    ap.add_argument("--unknown", type=int, default=20, help="random not-in-table ids to check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rec_path = args.export_dir / "recommendations.csv"
    fb_path = args.export_dir / "fallback.csv"
    for p in (rec_path, fb_path):
        if not p.exists():
            raise SystemExit(f"missing {p}. Run from the repo root, or pass --export-dir.")

    base = args.base_url.rstrip("/")
    check_server(base)
    rng = random.Random(args.seed)

    fb_items, fb_version = read_fallback(fb_path)
    print(f"reading {rec_path} (one pass)...", flush=True)
    t0 = time.time()
    sample, total = reservoir_sample(rec_path, args.samples, rng)
    print(f"  {total:,} rows read, {len(sample)} sampled, seed {args.seed}, {time.time() - t0:.0f}s", flush=True)

    failures = []

    def fail(kind, ident, detail):
        failures.append((kind, ident, detail))

    # 1. Known customers
    per_mode = {"repurchase": 0, "discover": 0}
    for row in sample:
        cid, mode = row["customer_id"], row["mode"]
        per_mode[mode] = per_mode.get(mode, 0) + 1
        expected = parse_items(row["items"])
        status, body = call(base, f"/v1/recommendations/{cid}?mode={mode}")
        label = f"{cid[:12]}../{mode}"
        if status != 200 or body is None:
            fail("known", label, f"status {status}")
            continue
        if body.get("source") != "als":
            fail("known", label, f"source {body.get('source')!r}, expected 'als'")
        if body.get("mode") != mode:
            fail("known", label, f"mode {body.get('mode')!r}")
        if body.get("customerId") != cid:
            fail("known", label, "customerId not echoed")
        if body.get("modelVersion") != row["model_version"]:
            fail("known", label, f"modelVersion {body.get('modelVersion')!r}")
        if body.get("items") != expected:
            fail("known", label, f"items differ: api {body.get('items')} vs export {expected}")

    # 2. Unknown but well-formed ids -> fallback, in both modes
    for i in range(args.unknown):
        cid = f"{rng.getrandbits(256):064x}"
        mode = "repurchase" if i % 2 == 0 else "discover"
        status, body = call(base, f"/v1/recommendations/{cid}?mode={mode}")
        label = f"unknown{i}/{mode}"
        if status != 200 or body is None:
            fail("fallback", label, f"status {status}")
            continue
        if body.get("source") != FALLBACK_SOURCE:
            fail("fallback", label, f"source {body.get('source')!r} (a random id should not be a customer)")
        if body.get("mode") != mode:
            fail("fallback", label, f"mode {body.get('mode')!r}")
        if body.get("items") != fb_items:
            fail("fallback", label, "items differ from fallback.csv")
        if body.get("modelVersion") != fb_version:
            fail("fallback", label, f"modelVersion {body.get('modelVersion')!r}")

    # 3. Malformed requests -> 400
    good = sample[0]["customer_id"] if sample else "a" * 64
    bad_requests = {
        "not-a-valid-id": "/v1/recommendations/not-a-valid-id",
        "leading space": f"/v1/recommendations/%20{good}",
        "uppercase id": f"/v1/recommendations/{good.upper()}",
        "invalid mode": f"/v1/recommendations/{good}?mode=everything",
    }
    for name, path in bad_requests.items():
        status, _ = call(base, path)
        if status != 400:
            fail("contract", name, f"status {status}, expected 400")

    n_checks = len(sample) + args.unknown + len(bad_requests)
    print()
    print(f"known-customer rows checked : {len(sample)} (repurchase {per_mode.get('repurchase', 0)}, discover {per_mode.get('discover', 0)})")
    print(f"unknown-id fallback checks  : {args.unknown}")
    print(f"malformed-request checks    : {len(bad_requests)}")
    print(f"total checks                : {n_checks}")
    print(f"failures                    : {len(failures)}")
    for kind, ident, detail in failures[:10]:
        print(f"  FAIL [{kind}] {ident}: {detail}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    print("RESULT:", "ALL CHECKS PASSED" if not failures else "FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
