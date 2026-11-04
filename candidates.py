#!/usr/bin/env python3
"""Stage 1 of the re-ranker: candidate generation and candidate recall.

For one target week, fits the frozen ALS config on history strictly before that week
(never the saved artifact in models/, which was trained through the validation week),
then builds each target customer's candidate set:

    ALS top-N (repurchases allowed)  UNION  7-day popularity top-12

Customers with no ALS row get the 7-day popularity top-N instead (the cold-start path).

Candidate recall is the ceiling for any ranker built on these candidates: a purchase that
is not in the candidate set cannot be ranked into the top 12.

    python candidates.py --week val     # target = validation week (history = train split)
    python candidates.py --week train   # target = the week before validation (ranker training labels)

Prints recall at each N, overall and per segment; appends aggregate rows to
candidate_recall_results.csv; saves the candidates at the largest N to
data/candidates/candidates_<week>.parquet (gitignored: contains customer and article ids).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl

import baseline as base
import cold_start_policy as csp
import model as als_mod

log = logging.getLogger("candidates")

FROZEN_ALS_CONFIG = {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 40}
RANDOM_STATE = 42
FALLBACK_DAYS = 7
POP_UNION_K = 12
RESULTS_CSV = Path("candidate_recall_results.csv")
OUT_DIR = Path("data/candidates")


def split_week(week):
    """Returns (history, target, week_start) with the same leak guards as rolling_backtest.run_folds."""
    train, val = base.load_train(), base.load_val()
    all_hist = pd.concat([train, val], ignore_index=True)
    val_start = val["t_dat"].min().normalize()
    week_start = val_start if week == "val" else val_start - pd.Timedelta(days=7)
    week_end = week_start + pd.Timedelta(days=7)
    target = all_hist[(all_hist["t_dat"] >= week_start) & (all_hist["t_dat"] < week_end)]
    history = all_hist[all_hist["t_dat"] < week_start]
    assert len(target) > 0 and len(history) > 0, "empty target or history"
    assert history["t_dat"].max() < target["t_dat"].min(), "history overlaps the target week"
    test = pd.read_parquet(Path("data/processed") / "transactions_test.parquet", columns=["t_dat"])
    assert target["t_dat"].max() < test["t_dat"].min(), "target week reaches the test split"
    log.info("target %s -> %s (%d rows, %d customers) | history ends %s (%d rows)",
             target["t_dat"].min().date(), target["t_dat"].max().date(), len(target),
             target["customer_id"].nunique(), history["t_dat"].max().date(), len(history))
    return history, target, week_start


def als_topn(history, customers, n):
    """ALS top-n per customer (repurchases allowed). Returns {customer: (items, scores)} for customers with a row."""
    matrix, c2r, i2c = als_mod.build_confidence_weighted_matrix(history)
    log.info("fitting ALS %s on %s", FROZEN_ALS_CONFIG, matrix.shape)
    model = als_mod.fit_als_model(matrix, random_state=RANDOM_STATE, **FROZEN_ALS_CONFIG)
    col_to_item = np.empty(len(i2c), dtype=object)
    for item, col in i2c.items():
        col_to_item[col] = item
    known = [c for c in customers if c in c2r]
    rows = np.array([c2r[c] for c in known], dtype=np.int64)
    out = {}
    batch = 20000
    with threadpoolctl.threadpool_limits(1, "blas"):
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            ids, scores = model.recommend(r, matrix[r], N=n, filter_already_liked_items=False)
            for c, i_row, s_row in zip(known[s:s + batch], ids, scores):
                out[c] = (col_to_item[i_row], s_row)
    log.info("ALS candidates for %d of %d target customers", len(out), len(customers))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", choices=("val", "train"), required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[12, 50, 100])
    args = ap.parse_args()
    ns = sorted(args.n)
    n_max = ns[-1]

    history, target, week_start = split_week(args.week)
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg = base.segment_customers(history, customers).astype(str)
    pop = csp.compute_recent_popularity(history, FALLBACK_DAYS).head(max(n_max, POP_UNION_K)).index.tolist()
    pop12 = set(pop[:POP_UNION_K])

    als = als_topn(history, customers, n_max)

    rows = []
    for n in ns:
        per = {"als_only": [], "als_plus_pop12": []}
        for c in customers:
            truth = actual[c]
            if c in als:
                base_set = set(als[c][0][:n])
            else:
                base_set = set(pop[:n])
            union = base_set | pop12
            per["als_only"].append((len(truth & base_set), len(truth), bool(truth & base_set), len(base_set)))
            per["als_plus_pop12"].append((len(truth & union), len(truth), bool(truth & union), len(union)))
        for name, vals in per.items():
            v = np.array(vals, dtype=float)
            for scope in ["aggregate", "0 purchases", "1-2 purchases", "3+ purchases"]:
                mask = np.ones(len(customers), bool) if scope == "aggregate" else (seg.to_numpy() == scope)
                if mask.sum() == 0:
                    continue
                vv = v[mask]
                rows.append({
                    "week": args.week, "week_start": str(week_start.date()), "candidates": name, "n": n,
                    "scope": scope, "n_customers": int(mask.sum()),
                    "pair_recall": vv[:, 0].sum() / vv[:, 1].sum(),
                    "customer_coverage": vv[:, 2].mean(),
                    "mean_candidates": vv[:, 3].mean(),
                })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res[res["scope"] == "aggregate"][["candidates", "n", "pair_recall", "customer_coverage", "mean_candidates"]]
          .to_string(index=False))
    print("\nper segment, ALS + popularity-12:")
    print(res[(res["candidates"] == "als_plus_pop12") & (res["scope"] != "aggregate")]
          [["n", "scope", "n_customers", "pair_recall", "customer_coverage"]].to_string(index=False))

    agg = res[res["scope"] == "aggregate"]
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        old = old[old["week"] != args.week]
        agg = pd.concat([old, agg], ignore_index=True)
    agg.to_csv(RESULTS_CSV, index=False)
    log.info("aggregate rows written to %s", RESULTS_CSV)

    cols = {"customer_id": [], "article_id": [], "source": [], "als_score": [], "source_rank": [], "label": []}
    for c in customers:
        if c in als:
            items, scores = als[c]
            src = {a: ["als", float(sc), r + 1] for r, (a, sc) in enumerate(zip(items, scores))}
        else:
            src = {a: ["pop", np.nan, r + 1] for r, a in enumerate(pop[:n_max])}
        for a in pop12:
            if a in src:
                src[a][0] += "+pop12"
            else:
                src[a] = ["pop12", np.nan, np.nan]
        truth = actual[c]
        for a, (s_, sc, rank) in src.items():
            cols["customer_id"].append(c); cols["article_id"].append(a); cols["source"].append(s_)
            cols["als_score"].append(sc); cols["source_rank"].append(rank); cols["label"].append(int(a in truth))
    cand = pd.DataFrame(cols)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"candidates_{args.week}.parquet"
    cand.to_parquet(out, index=False)
    log.info("saved %d candidate rows (%d customers, positive rate %.4f) to %s",
             len(cand), cand["customer_id"].nunique(), cand["label"].mean(), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
