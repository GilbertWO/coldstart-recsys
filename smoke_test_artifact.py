"""
smoke_test_artifact.py -- Week 5-6 wrap-up / Week 7-8 prerequisite (Oct 6).

Nothing has ever RELOADED the saved model in models/ and used it. The serving stage will
depend on it, so before any Java is written this checks that the artifact on disk really is
the model that produced the reported test-week numbers.

Checks (each prints PASS / FAIL):
  1. STRUCTURE: factor matrices, id arrays and metadata agree on sizes; no NaN/inf; reports how
     many customers have an all-zero embedding.
  2. IMPLICIT PARITY (best effort): raw top-12 from the factors in plain numpy equals the saved
     `implicit` model's own recommend() on a random sample. Skipped with a warning if the saved
     implicit file can't be loaded by this version of the library.
  3. REPRODUCTION: using ONLY the saved factors + id mappings + saved fallback list (no refit),
     re-scores the test week from the train+val history under both exclusion settings and
     compares aggregate MAP@12 with what test_evaluation.py reported
     (exclusion on 0.004996, off 0.008802). This is the check that matters: it proves the
     artifact reproduces the reported result, and that the plain "factors dot product, drop
     already-bought items, take 12" logic a serving layer would use is equivalent to the
     library's.
  It also times the scoring and prints an estimate of how long precomputing top-12 lists for
  EVERY customer in the artifact would take, which decides how the serving stage is designed.

This is read-only: it does not refit, overwrite models/, or touch the test-evaluation results.
The test week is used here only to verify a number that was already produced and reported.

Usage:
    python smoke_test_artifact.py
    python smoke_test_artifact.py --skip-reproduce        # structure + parity only (fast)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import baseline as base

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
# aggregate MAP@12 on the test week from test_evaluation.py (frozen config, train+val history)
EXPECTED_TEST_MAP = {True: 0.004996, False: 0.008802}
TOL = 1.5e-6
K = 12


def load_artifact(model_dir=MODEL_DIR):
    f = np.load(model_dir / "als_factors.npz")
    meta = json.loads((model_dir / "als_meta.json").read_text())
    return {"U": f["user_factors"], "V": f["item_factors"], "customer_ids": f["customer_ids"],
            "item_ids": f["item_ids"], "meta": meta}


def check_structure(a):
    U, V, meta = a["U"], a["V"], a["meta"]
    ok = True
    problems = []
    if U.shape[0] != len(a["customer_ids"]) or U.shape[0] != meta["n_customers"]:
        problems.append(f"user rows {U.shape[0]} vs ids {len(a['customer_ids'])} vs meta {meta['n_customers']}")
    if V.shape[0] != len(a["item_ids"]) or V.shape[0] != meta["n_items"]:
        problems.append(f"item rows {V.shape[0]} vs ids {len(a['item_ids'])} vs meta {meta['n_items']}")
    if U.shape[1] != V.shape[1]:
        problems.append(f"factor dims differ: {U.shape[1]} vs {V.shape[1]}")
    if not np.isfinite(U).all() or not np.isfinite(V).all():
        problems.append("NaN or inf in the factors")
    if len(set(a["customer_ids"].tolist())) != len(a["customer_ids"]):
        problems.append("duplicate customer ids")
    zero_users = int((np.abs(U).sum(axis=1) == 0).sum())
    ok = not problems
    log.info("%s: structure -- users %s, items %s, dims %d; zero-embedding customers: %d; factors dtype %s",
             "PASS" if ok else "FAIL", U.shape[0], V.shape[0], U.shape[1], zero_users, U.dtype)
    for p in problems:
        log.error("  %s", p)
    return ok


def topk_scores(U_rows, V, exclude_cols=None, k=K):
    """Plain-numpy top-k for a block of users. exclude_cols: list (one array of item columns per
    user) to remove, or None. Returns (n, k) item columns ordered best-first."""
    scores = U_rows @ V.T
    if exclude_cols is not None:
        for i, cols in enumerate(exclude_cols):
            if len(cols):
                scores[i, cols] = -np.inf
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def check_implicit_parity(a, model_dir=MODEL_DIR, n=200, seed=0):
    path = model_dir / "als_implicit.npz"
    try:
        from implicit.cpu.als import AlternatingLeastSquares
        model = AlternatingLeastSquares.load(str(path))
    except Exception as e:
        log.warning("SKIPPED: implicit parity -- could not load %s (%s). The reproduction check below "
                    "does not depend on it.", path, e)
        return True
    rng = np.random.default_rng(seed)
    rows = rng.choice(a["U"].shape[0], size=min(n, a["U"].shape[0]), replace=False)
    mine = topk_scores(a["U"][rows], a["V"])
    empty = sp.csr_matrix((len(rows), a["V"].shape[0]), dtype=np.float32)
    theirs, _ = model.recommend(rows, empty, N=K, filter_already_liked_items=False)
    same = float(np.mean([list(x) == list(y) for x, y in zip(mine, theirs)]))
    ok = same >= 0.99
    log.info("%s: implicit parity -- numpy top-%d equals implicit's recommend() for %.1f%% of %d sampled customers",
             "PASS" if ok else "FAIL", K, 100 * same, len(rows))
    return ok


def reproduce_test_map(a, history, test, chunk=500):
    """Re-score `test` using only the artifact + history. Returns {exclude: MAP@12} and timing."""
    meta = a["meta"]
    c2r = {c: i for i, c in enumerate(a["customer_ids"].tolist())}
    i2c = {s: i for i, s in enumerate(a["item_ids"].tolist())}
    orig_item = {str(x): x for x in history["article_id"].unique()}
    items_by_col = np.array([orig_item.get(s, s) for s in a["item_ids"].tolist()], dtype=object)
    fallback = [orig_item.get(s, s) for s in meta["fallback_top12"]]

    test_customers = list(test["customer_id"].unique())
    rows_of = {c: c2r.get(str(c)) for c in test_customers}
    with_row = [c for c in test_customers if rows_of[c] is not None]
    log.info("test customers: %d, with an embedding: %d, fallback: %d",
             len(test_customers), len(with_row), len(test_customers) - len(with_row))

    # history purchases of the test customers only (for the already-bought exclusion)
    sub = history[history["customer_id"].isin(with_row)]
    hr = sub["customer_id"].astype(str).map(c2r)
    hc = sub["article_id"].astype(str).map(i2c)
    keep = hr.notna() & hc.notna()
    B = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                       (hr[keep].astype(int).to_numpy(), hc[keep].astype(int).to_numpy())),
                      shape=(a["U"].shape[0], a["V"].shape[0]))

    out, timing = {}, {}
    for exclude in (True, False):
        t0 = time.time()
        recs = {c: fallback for c in test_customers if rows_of[c] is None}
        for start in range(0, len(with_row), chunk):
            part = with_row[start:start + chunk]
            ridx = np.array([rows_of[c] for c in part])
            excl = [B.indices[B.indptr[r]:B.indptr[r + 1]] for r in ridx] if exclude else None
            top = topk_scores(a["U"][ridx], a["V"], excl)
            for c, cols in zip(part, top):
                recs[c] = items_by_col[cols].tolist()
        timing[exclude] = (time.time() - t0, len(with_row))
        df = pd.DataFrame({"customer_id": list(recs.keys()), "recommendations": list(recs.values())})
        out[exclude] = (base.map_at_k(df, test), base.hit_rate_at_k(df, test))
    return out, timing


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Smoke-test the saved ALS model artifact.")
    parser.add_argument("--skip-reproduce", action="store_true")
    args = parser.parse_args()

    a = load_artifact()
    log.info("loaded artifact: %s", {k: a["meta"][k] for k in ("config", "history_end", "n_customers", "n_items")})
    results = [check_structure(a), check_implicit_parity(a)]

    if not args.skip_reproduce:
        train, val = base.load_train(), base.load_val()
        history = pd.concat([train, val], ignore_index=True)
        test = pd.read_parquet(Path("data/processed") / "transactions_test.parquet")
        assert str(history["t_dat"].max().date()) == a["meta"]["history_end"], \
            "artifact history_end differs from train+val's last day -- wrong history for this artifact"
        out, timing = reproduce_test_map(a, history, test)
        for exclude, (m, h) in out.items():
            exp = EXPECTED_TEST_MAP[exclude]
            ok = abs(m - exp) <= TOL
            results.append(ok)
            log.info("%s: reproduction, exclusion=%s -- MAP@12 %.6f (expected %.6f), hit_rate@12 %.4f",
                     "PASS" if ok else "FAIL", exclude, m, exp, h)
        secs, n = timing[True]
        per_cust = secs / max(n, 1)
        total = per_cust * a["meta"]["n_customers"]
        log.info("timing: %.1f s for %d customers (exclusion on) = %.2f ms/customer; precomputing top-%d for all %d "
                 "customers in the artifact would take about %.0f min on this machine (single process, plain numpy)",
                 secs, n, 1000 * per_cust, K, a["meta"]["n_customers"], total / 60)

    log.info("OVERALL: %s", "ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED -- do not build serving on this artifact yet")


if __name__ == "__main__":
    main()
