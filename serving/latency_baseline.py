#!/usr/bin/env python3
"""Latency baseline for GET /v1/recommendations/{customerId}.

Run from the repo root with the API running (Postgres-only, before any Redis code):

    python serving/latency_baseline.py
    python serving/latency_baseline.py --concurrency 8 --label postgres-only-c8

Method (the same method must be used for every later run, e.g. with Redis):
  * Customer ids come from recommendations.csv (one streaming pass, reservoir sample of
    --pool distinct customers). Requests mix both modes 50/50, and --unknown-share of them
    use random well-formed ids that are not customers (the fallback path).
  * --workload uniform picks ids evenly from the pool; --workload zipf skews traffic toward
    a few hot ids (a cache only helps when ids repeat, so report both before claiming a gain).
  * --warmup requests are sent first and discarded (JVM, connection pool, page cache).
  * Each measured run sends --requests requests; --repeat runs are executed back to back
    so the run-to-run spread is visible.
  * Connections are kept alive and TCP connect time is excluded: only request -> full
    response is timed. Percentiles use the nearest-rank method.
  * Raw per-request latencies are saved to --out-dir as JSON.

Limits to state whenever you quote these numbers: client, app, PostgreSQL and Redis all run
on one machine, and the timing includes the Python client's own overhead.
Standard library only. Exit code 1 if any request returned a non-200 status.
"""
import argparse
import csv
import http.client
import itertools
import json
import math
import platform
import random
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_EXPORT_DIR = Path("models/serving_export")
DEFAULT_OUT_DIR = Path("serving/latency_results")
MODES = ("repurchase", "discover")


def sample_customer_ids(path, n, rng):
    """Reservoir-sample n distinct customer ids (one row per customer: the repurchase row)."""
    sample, seen = [], 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] != "repurchase":
                continue
            seen += 1
            if len(sample) < n:
                sample.append(row["customer_id"])
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    sample[j] = row["customer_id"]
    return sample, seen


def build_requests(pool, count, unknown_share, workload, zipf_s, rng):
    """Return a list of (path, kind). kind is 'known' or 'unknown'."""
    if workload == "zipf":
        cum = list(itertools.accumulate(1.0 / (r + 1) ** zipf_s for r in range(len(pool))))
    out = []
    for _ in range(count):
        mode = rng.choice(MODES)
        if rng.random() < unknown_share:
            cid, kind = f"{rng.getrandbits(256):064x}", "unknown"
        else:
            if workload == "zipf":
                cid = pool[rng.choices(range(len(pool)), cum_weights=cum, k=1)[0]]
            else:
                cid = pool[rng.randrange(len(pool))]
            kind = "known"
        out.append((f"/v1/recommendations/{cid}?mode={mode}", kind))
    return out


class Client:
    """One keep-alive connection. TCP connect is done before the timer starts."""

    def __init__(self, host, port):
        self.host, self.port, self.conn = host, port, None

    def _ensure(self):
        if self.conn is None or self.conn.sock is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=15)
            self.conn.connect()
            self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def get(self, path):
        for attempt in (1, 2):
            try:
                self._ensure()
                t0 = time.perf_counter()
                self.conn.request("GET", path)
                resp = self.conn.getresponse()
                resp.read()
                return resp.status, (time.perf_counter() - t0) * 1000.0
            except (http.client.HTTPException, OSError):
                self.conn = None
                if attempt == 2:
                    return 0, float("nan")


def pct(sorted_vals, p):
    """Nearest-rank percentile of an ascending list."""
    k = max(1, math.ceil(p / 100.0 * len(sorted_vals)))
    return sorted_vals[k - 1]


def summarize(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 3),
        "p50": round(pct(s, 50), 3),
        "p90": round(pct(s, 90), 3),
        "p95": round(pct(s, 95), 3),
        "p99": round(pct(s, 99), 3),
        "max": round(s[-1], 3),
    }


def run_phase(host, port, requests, concurrency):
    local = threading.local()

    def work(item):
        client = getattr(local, "client", None)
        if client is None:
            client = local.client = Client(host, port)
        return client.get(item[0])

    t0 = time.perf_counter()
    if concurrency == 1:
        results = [work(it) for it in requests]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(work, requests))
    wall = time.perf_counter() - t0
    return results, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--label", default="postgres-only", help="names this configuration in the output file")
    ap.add_argument("--pool", type=int, default=20000, help="distinct customers to draw ids from")
    ap.add_argument("--workload", choices=("uniform", "zipf"), default="uniform")
    ap.add_argument("--zipf-s", type=float, default=1.1)
    ap.add_argument("--unknown-share", type=float, default=0.10)
    ap.add_argument("--requests", type=int, default=5000, help="requests per measured run")
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3, help="number of measured runs")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    u = urlparse(args.base_url)
    host, port = u.hostname, u.port or 80
    rec_path = args.export_dir / "recommendations.csv"
    if not rec_path.exists():
        raise SystemExit(f"missing {rec_path}. Run from the repo root, or pass --export-dir.")

    probe = Client(host, port)
    status, _ = probe.get("/actuator/health")
    if status != 200:
        raise SystemExit(f"cannot use {args.base_url}: /actuator/health returned {status or 'no response'}. "
                         "Is the app running, with POSTGRES_PASSWORD set?")

    rng = random.Random(args.seed)
    print(f"reading {rec_path} (one pass)...", flush=True)
    pool, n_customers = sample_customer_ids(rec_path, args.pool, rng)
    rng.shuffle(pool)  # so zipf rank is not tied to file order
    print(f"  {n_customers:,} customers in the export, pool of {len(pool):,} sampled", flush=True)

    warm = build_requests(pool, args.warmup, args.unknown_share, args.workload, args.zipf_s, rng)
    print(f"warm-up: {len(warm)} requests (discarded)...", flush=True)
    run_phase(host, port, warm, args.concurrency)

    runs, any_errors = [], False
    for r in range(1, args.repeat + 1):
        reqs = build_requests(pool, args.requests, args.unknown_share, args.workload, args.zipf_s, rng)
        results, wall = run_phase(host, port, reqs, args.concurrency)
        ok = [(k, dt) for (_, k), (st, dt) in zip(reqs, results) if st == 200]
        n_err = len(results) - len(ok)
        any_errors |= n_err > 0
        stats = summarize([dt for _, dt in ok])
        by_kind = {k: summarize([dt for kk, dt in ok if kk == k]) for k in ("known", "unknown")}
        runs.append({
            "run": r,
            "wall_seconds": round(wall, 3),
            "throughput_rps": round(len(results) / wall, 1),
            "n_requests": len(results),
            "n_errors": n_err,
            "stats_ms": stats,
            "by_kind_ms": by_kind,
            "latencies_ms": [round(dt, 3) for _, dt in ok],
        })
        print(f"run {r}/{args.repeat}: {len(results)} requests in {wall:.1f}s "
              f"({runs[-1]['throughput_rps']} req/s), errors {n_err} | "
              f"p50 {stats.get('p50')} ms, p95 {stats.get('p95')} ms, p99 {stats.get('p99')} ms", flush=True)

    def med(key):
        vals = sorted(r["stats_ms"][key] for r in runs if r["stats_ms"].get("n"))
        return round(vals[len(vals) // 2], 3) if vals else None

    summary = {k: med(k) for k in ("p50", "p95", "p99")}
    out = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "median_of_runs_ms": summary,
        "runs": runs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out_dir / f"{args.label}-{args.workload}-c{args.concurrency}-{stamp}.json"
    path.write_text(json.dumps(out), encoding="utf-8")

    print()
    print(f"label                : {args.label}")
    print(f"workload             : {args.workload} (pool {len(pool):,}, unknown share {args.unknown_share:.0%})")
    print(f"concurrency          : {args.concurrency}")
    print(f"requests per run     : {args.requests} x {args.repeat} runs (warm-up {args.warmup} discarded)")
    print(f"median across runs   : p50 {summary['p50']} ms | p95 {summary['p95']} ms | p99 {summary['p99']} ms")
    p50s = [r["stats_ms"].get("p50") for r in runs]
    print(f"per-run p50 spread   : {min(p50s)} to {max(p50s)} ms")
    k = [r["by_kind_ms"]["known"].get("p50") for r in runs]
    f = [r["by_kind_ms"]["unknown"].get("p50") for r in runs]
    print(f"p50 known / fallback : {sorted(k)[len(k)//2]} ms / {sorted(f)[len(f)//2]} ms")
    print(f"saved                : {path}")
    print("note: one machine runs client, app and databases; timing includes Python client overhead; "
          "TCP connect excluded (keep-alive).")
    if any_errors:
        print("WARNING: some requests returned a non-200 status; see n_errors in the saved file.")
    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
