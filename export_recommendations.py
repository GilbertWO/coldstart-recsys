"""
export_recommendations.py -- Week 7-8 prerequisite (written Oct 8, pulled forward from Oct 14).

The serving design (serving-design-draft.md) precomputes every customer's top-12 offline and
serves lookups. This script produces those lists from the SAVED ALS artifact in models/ -- no
refit -- for both modes:

  repurchase : top-12 by ALS score, already-bought articles allowed   (default mode)
  discover   : top-12 by ALS score, already-bought articles removed

and the 7-day popularity fallback list saved in the artifact's metadata.

Outputs (default models/serving_export/, which is covered by .gitignore via `models/`):
  recommendations.csv  header: customer_id,mode,items,model_version. One row per customer
                       per mode. `items` is a PostgreSQL array literal such as
                       "{108775015,111565001,...}" (12 integer article ids, best first), so
                       the file can be loaded with COPY ... CSV HEADER into an integer[] column.
  fallback.csv         header: source,as_of,window_days,items,model_version
  export_manifest.json counts, model_version, history_end, file sizes, checks that passed.

Design decisions (same style as the other scripts):
  * One matrix product per block of customers serves BOTH modes: scores are computed once,
    the repurchase top-12 is taken, then already-bought columns are set to -inf and the
    discover top-12 is taken. Scoring logic is the same plain "factors dot product, drop
    already-bought, take 12" that smoke_test_artifact.py proved reproduces the reported test
    MAP, so no new scoring semantics are introduced here.
  * "Already bought" = the customer's purchases in train+val, the same history the artifact
    was fitted on (artifact history_end is asserted). Purchases from the held-out test week
    (2020-09-16 to 2020-09-22) are NOT in the exclusion, because the model has not seen them
    either; model_version carries the history end date so this is visible in every response.
  * Every check runs BEFORE the long loop where possible (id formats, fallback list, history
    end date) so a bad input fails in a minute, not after 25 minutes. Output files only get
    their final names after all post-loop checks pass; otherwise a .tmp file is left behind.
  * --limit exports only the first N customers (artifact row order) for a quick trial run. Use a
    different --out-dir for trial runs so they never overwrite a full export.
  * Refuses to overwrite existing output unless --overwrite is given.

Checks (any failure stops the script):
  1. every customer id is 64 lowercase hex characters and unique; every article id is an
     integer and unique; fallback list is 12 distinct integer ids that exist in the artifact
  2. history end date in the data equals the artifact's history_end
  3. every exported list has 12 distinct articles; discover lists never select a removed
     article (scores of removed articles are -inf, so a finite score is required)
  4. row count = 2 x customers exported
  5. independent parity check: for a random sample, the exported lists equal what
     smoke_test_artifact.topk_scores returns (>= 99.5% identical; float rounding can reorder
     near-ties when the block shape differs)

Does not: refit, touch models/als_*, use the test-week transactions, load anything into
PostgreSQL (that is Thursday Oct 15), or compute a refreshed fallback list (the as_of refresh
job is Week B).

Usage (close Docker Desktop first; the factors are about 800 MB in memory):
    python export_recommendations.py --limit 2000 --out-dir models/serving_export_trial
    python export_recommendations.py
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import smoke_test_artifact as st

log = logging.getLogger(__name__)

K = 12
CHUNK = 500
OUT_DIR = Path("models") / "serving_export"
MODES = ("repurchase", "discover")
FALLBACK_SOURCE = "fallback_popularity_7d"
FALLBACK_WINDOW_DAYS = 7          # model.COLD_START_WINDOW_DAYS; kept as a constant so this
                                  # script does not need to import the implicit library
CUSTOMER_ID_RE = re.compile(r"[0-9a-f]{64}")
PARITY_SAMPLE = 300
PARITY_MIN = 0.995


def model_version(meta):
    """String stored with every list and returned by the API as modelVersion."""
    c = meta["config"]
    try:
        return "als-f{}-r{}-a{:g}-i{}-h{}".format(
            c["factors"], c["regularization"], float(c["alpha"]), c["iterations"], meta["history_end"])
    except KeyError as e:
        raise KeyError(f"als_meta.json config is missing {e}; keys present: {sorted(c)}") from None


def validate_ids(a):
    """Check 1 (customers and articles). Returns the article ids as an int64 array."""
    cids = a["customer_ids"].tolist()
    bad = [c for c in cids if not CUSTOMER_ID_RE.fullmatch(c)]
    if bad:
        raise ValueError(f"{len(bad)} customer ids are not 64 lowercase hex characters, e.g. {bad[:3]}")
    if len(set(cids)) != len(cids):
        raise ValueError("duplicate customer ids in the artifact")
    try:
        item_ints = np.array([int(s) for s in a["item_ids"].tolist()], dtype=np.int64)
    except ValueError as e:
        raise ValueError(f"article ids in the artifact are not all integers: {e}") from None
    if len(np.unique(item_ints)) != len(item_ints):
        raise ValueError("duplicate article ids in the artifact")
    log.info("PASS: ids -- %d customers (64 lowercase hex, unique), %d articles (integer, unique)",
             len(cids), len(item_ints))
    return item_ints


def validate_fallback(meta, item_ints):
    """Check 1 (fallback). Returns the fallback list as ints."""
    try:
        fb = [int(x) for x in meta["fallback_top12"]]
    except (KeyError, ValueError) as e:
        raise ValueError(f"fallback_top12 missing or not integer ids in als_meta.json: {e}") from None
    if len(fb) != K or len(set(fb)) != K:
        raise ValueError(f"fallback list is not {K} distinct ids: {fb}")
    missing = set(fb) - set(item_ints.tolist())
    if missing:
        raise ValueError(f"fallback article ids not present in the artifact: {sorted(missing)}")
    log.info("PASS: fallback list -- %d distinct ids, all present in the artifact", K)
    return fb


def _customer_rows(s, cust_index):
    """Map a customer_id Series to artifact row numbers (-1 when absent). Uses category codes
    when the column is categorical, which avoids converting ~30M strings."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat_rows = cust_index.get_indexer(s.cat.categories.astype(str))
        codes = s.cat.codes.to_numpy()
        return np.where(codes >= 0, cat_rows[np.maximum(codes, 0)], -1)
    return cust_index.get_indexer(s.astype(str).to_numpy())


def build_exclusions(a, item_ints):
    """Per-customer already-bought article columns from train+val, as CSR-style arrays
    (indptr, indices). Check 2: the data's last day must equal the artifact's history_end."""
    n_users, n_items = a["U"].shape[0], a["V"].shape[0]
    cust_index = pd.Index(a["customer_ids"])
    order = np.argsort(item_ints)
    sorted_vals = item_ints[order]

    train, val = base.load_train(), base.load_val()
    last = max(train["t_dat"].max(), val["t_dat"].max())
    if str(last.date()) != a["meta"]["history_end"]:
        raise ValueError(f"history in the data ends {last.date()} but the artifact says "
                         f"{a['meta']['history_end']} -- wrong history for this artifact")
    log.info("PASS: history end -- data ends %s, artifact history_end %s", last.date(), a["meta"]["history_end"])

    parts, kept, dropped = [], 0, 0
    for df in (train, val):
        rows = _customer_rows(df["customer_id"], cust_index)
        art = df["article_id"].astype(np.int64).to_numpy()
        pos = np.minimum(np.searchsorted(sorted_vals, art), len(sorted_vals) - 1)
        cols = np.where(sorted_vals[pos] == art, order[pos], -1)
        keep = (rows >= 0) & (cols >= 0)
        kept += int(keep.sum())
        dropped += int((~keep).sum())
        parts.append(np.unique(rows[keep].astype(np.int64) * n_items + cols[keep]))
    del train, val
    keys = np.unique(np.concatenate(parts))
    r = keys // n_items
    indices = (keys % n_items).astype(np.int32)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(r, minlength=n_users), out=indptr[1:])
    per_user = np.diff(indptr)
    log.info("exclusions: %d purchase rows used (%d unmatched rows ignored), %d distinct customer-article pairs; "
             "customers with no history items: %d; median items per customer: %d",
             kept, dropped, len(keys), int((per_user == 0).sum()), int(np.median(per_user)))
    return indptr, indices


def topk_from_scores(scores, k=K):
    """Top-k columns per row, best first, plus their scores."""
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _all_distinct(ids):
    s = np.sort(ids, axis=1)
    return bool((s[:, 1:] != s[:, :-1]).all())


def array_literal(row):
    return "{" + ",".join(map(str, row)) + "}"


def export(a, item_ints, indptr, indices, out_dir, limit, chunk, version, seed=0):
    U, V = a["U"], a["V"]
    cust_ids = a["customer_ids"]
    n = U.shape[0] if limit is None else min(limit, U.shape[0])
    rng = np.random.default_rng(seed)
    sample_rows = set(rng.choice(n, size=min(PARITY_SAMPLE, n), replace=False).tolist())
    sampled = {}                                   # row -> {mode: top columns}

    tmp = out_dir / "recommendations.csv.tmp"
    n_rows, t0 = 0, time.time()
    n_chunks = (n + chunk - 1) // chunk
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["customer_id", "mode", "items", "model_version"])
        for ci, start in enumerate(range(0, n, chunk)):
            stop = min(start + chunk, n)
            scores = U[start:stop] @ V.T
            top_rep, _ = topk_from_scores(scores)
            for i in range(stop - start):
                s, e = indptr[start + i], indptr[start + i + 1]
                scores[i, indices[s:e]] = -np.inf
            top_dis, vals_dis = topk_from_scores(scores)
            if not np.isfinite(vals_dis).all():
                raise RuntimeError(f"discover list selected an already-bought article in rows {start}-{stop}")
            for i in range(stop - start):
                if start + i in sample_rows:
                    sampled[start + i] = {"repurchase": top_rep[i].copy(), "discover": top_dis[i].copy()}
            for mode, top in (("repurchase", top_rep), ("discover", top_dis)):
                ids = item_ints[top]
                if ids.shape[1] != K or not _all_distinct(ids):
                    raise RuntimeError(f"{mode} list without {K} distinct articles in rows {start}-{stop}")
                for cid, row in zip(cust_ids[start:stop].tolist(), ids.tolist()):
                    w.writerow([cid, mode, array_literal(row), version])
                    n_rows += 1
            if (ci + 1) % 100 == 0 or ci + 1 == n_chunks:
                el = time.time() - t0
                log.info("  %d / %d customers (%.1f%%), %.0f s elapsed, about %.0f s remaining",
                         stop, n, 100 * stop / n, el, el / stop * (n - stop))
    elapsed = time.time() - t0

    if n_rows != len(MODES) * n:
        raise RuntimeError(f"row count {n_rows} != {len(MODES)} x {n}")
    log.info("PASS: row count -- %d rows = %d modes x %d customers; every list has %d distinct articles; "
             "no discover list contains an already-bought article", n_rows, len(MODES), n, K)

    # check 5: independent recomputation through smoke_test_artifact.topk_scores
    rows = np.array(sorted(sampled))
    mine = {
        "repurchase": st.topk_scores(U[rows], V),
        "discover": st.topk_scores(U[rows], V, [indices[indptr[r]:indptr[r + 1]] for r in rows]),
    }
    ok_all = True
    for mode in MODES:
        same = float(np.mean([list(sampled[r][mode]) == list(mine[mode][j]) for j, r in enumerate(rows)]))
        ok = same >= PARITY_MIN
        ok_all &= ok
        log.info("%s: parity -- exported %s lists equal smoke_test_artifact.topk_scores for %.1f%% of %d sampled customers",
                 "PASS" if ok else "FAIL", mode, 100 * same, len(rows))
    if not ok_all:
        raise RuntimeError(f"parity check failed; leaving {tmp} in place for inspection, nothing finalised")
    return tmp, n, n_rows, elapsed


def write_fallback(out_dir, meta, fb, version):
    tmp = out_dir / "fallback.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["source", "as_of", "window_days", "items", "model_version"])
        w.writerow([FALLBACK_SOURCE, meta["history_end"], FALLBACK_WINDOW_DAYS, array_literal(fb), version])
    return tmp


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Export precomputed top-12 lists from the saved ALS artifact.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="export only the first N customers (trial run)")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    final_recs = args.out_dir / "recommendations.csv"
    final_fb = args.out_dir / "fallback.csv"
    final_manifest = args.out_dir / "export_manifest.json"
    if not args.overwrite and any(f.exists() for f in (final_recs, final_fb, final_manifest)):
        sys.exit(f"{args.out_dir} already contains an export; pass --overwrite or use another --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a = st.load_artifact()
    meta = a["meta"]
    version = model_version(meta)
    log.info("artifact: %s customers, %s articles, %s; model_version %s",
             f"{meta['n_customers']:,}", f"{meta['n_items']:,}", {k: meta["config"][k] for k in meta["config"]}, version)
    if not st.check_structure(a):
        sys.exit("artifact structure check failed; run smoke_test_artifact.py")
    item_ints = validate_ids(a)
    fb = validate_fallback(meta, item_ints)
    indptr, indices = build_exclusions(a, item_ints)

    tmp_recs, n, n_rows, elapsed = export(a, item_ints, indptr, indices, args.out_dir, args.limit, args.chunk, version)
    tmp_fb = write_fallback(args.out_dir, meta, fb, version)
    os.replace(tmp_recs, final_recs)
    os.replace(tmp_fb, final_fb)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": version,
        "history_end": meta["history_end"],
        "k": K,
        "modes": list(MODES),
        "customers_exported": n,
        "customers_in_artifact": int(meta["n_customers"]),
        "partial_export": n < int(meta["n_customers"]),
        "rows": n_rows,
        "fallback_source": FALLBACK_SOURCE,
        "fallback_items": fb,
        "export_seconds": round(elapsed, 1),
        "files": {f.name: f.stat().st_size for f in (final_recs, final_fb)},
        "checks_passed": ["ids", "fallback", "history_end", "row_count_and_distinct", "no_bought_in_discover", "parity"],
    }
    final_manifest.write_text(json.dumps(manifest, indent=2))
    log.info("done: %d customers, %d rows in %.0f s -> %s", n, n_rows, elapsed, args.out_dir)
    if manifest["partial_export"]:
        log.warning("PARTIAL export (--limit): do not load this into PostgreSQL as the real table")
    log.info("OVERALL: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
