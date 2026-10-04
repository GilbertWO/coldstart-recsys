"""
exclusion_comparison.py -- Week 5-6, Oct 4: the full model comparison in ONE
place, under BOTH exclusion settings, with matched cold-start fallbacks.

Models compared (every one scored aggregate + per segment, MAP@12 and friends):
  - ALS (confidence-weighted, factors=50 / alpha=15 / reg=0.01 / iter=15, the
    config model.py --best-only uses), last-7d popularity fallback
  - item-item CF, last-7d popularity fallback
  - 14-day recency popularity (no personalization)
under exclusion=True (own train purchases removed) and exclusion=False.

Why: (1) your README's no-exclusion item-item numbers come from ablation code
that isn't in baseline.py -- this script reproduces that row, so the repo can
regenerate it; (2) the ALS-vs-item-item ordering flips with the exclusion
setting, and the Monday write-up needs both tables from one consistent run.

Efficiency notes: ALS is fit ONCE -- exclusion is a recommend-time flag, the
model is identical under both settings. Item-item CF is scored once per
exclusion setting; its 7-day-fallback variant is built by swapping the fallback
list in for customers with no row in its interaction matrix (their personalized
lists don't depend on the fallback).

Does not modify baseline.py, model.py or any other file.

Usage:
    python exclusion_comparison.py
"""

import logging

import numpy as np
import pandas as pd

import baseline as base
import cold_start_policy as csp
import item_item_matched as iim
import model as als_mod

log = logging.getLogger(__name__)

ALS_CONFIG = {"factors": 50, "regularization": 0.01, "alpha": 15.0, "iterations": 15}
RECENCY_BASELINE_DAYS = 14
FALLBACK_DAYS = als_mod.COLD_START_WINDOW_DAYS

# README.md's reported item-item CF row WITHOUT exclusion (all-time fallback).
README_ITEM_ITEM_NO_EXCL = {"precision@12": 0.00441, "recall@12": 0.02095, "hit_rate@12": 0.0465, "map@12": 0.00839}


def item_item_recommendations(val_customers, matrix, similarity, customer_to_row, item_to_col,
                              fallback_top12, exclude, k=12):
    """Same as baseline.get_item_item_recommendations, but the already-purchased
    exclusion is a flag. (baseline.py's version always excludes.)"""
    col_to_item = {v: item for item, v in item_to_col.items()}
    recommendations = {}
    for customer_id in val_customers:
        row_idx = customer_to_row.get(customer_id)
        if row_idx is None:
            recommendations[customer_id] = fallback_top12
            continue
        customer_row = matrix[row_idx]
        scores = np.asarray((customer_row @ similarity).todense()).flatten()
        if exclude:
            scores[customer_row.indices] = -np.inf
        top_k_cols = np.argpartition(scores, -k)[-k:]
        top_k_cols = top_k_cols[np.argsort(-scores[top_k_cols])]
        recommendations[customer_id] = [col_to_item[c] for c in top_k_cols]
    return recommendations


def als_recommendations(val_customers, model, matrix, customer_to_row, item_to_col, fallback_top12, exclude, k=12):
    col_to_item = {v: item for item, v in item_to_col.items()}
    recommendations = {}
    for customer_id in val_customers:
        row_idx = customer_to_row.get(customer_id)
        if row_idx is None:
            recommendations[customer_id] = fallback_top12
            continue
        item_ids, _ = model.recommend(
            userid=row_idx, user_items=matrix[row_idx], N=k, filter_already_liked_items=exclude
        )
        recommendations[customer_id] = [col_to_item[c] for c in item_ids]
    return recommendations


def run(train, val, n_candidates=None, als_config=None):
    als_config = als_config or ALS_CONFIG
    n_candidates = n_candidates or base.N_CANDIDATE_ITEMS
    val_customers = val["customer_id"].unique()
    segments = base.segment_customers(train, val_customers)

    popularity = base.compute_popularity(train)
    alltime_top12 = base.get_popularity_recommendations(popularity)
    fallback_top12 = csp.compute_recent_popularity(train, FALLBACK_DAYS).head(12).index.tolist()
    recency_pop = csp.compute_recent_popularity(train, RECENCY_BASELINE_DAYS)
    purchased = (
        train[train["customer_id"].isin(val_customers)]
        .groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    # ---- item-item CF: matrix + similarity once, scored once per exclusion setting
    matrix, c2r, i2c = base.build_interaction_matrix(train, popularity, n_candidates)
    similarity = base.compute_item_similarity(matrix)

    # ---- ALS: fit once
    cw_matrix, cw_c2r, cw_i2c = als_mod.build_confidence_weighted_matrix(train)
    log.info("Fitting ALS once: %s", als_config)
    model = als_mod.fit_als_model(cw_matrix, random_state=42, **als_config)

    rows = []
    readme_check_row = None
    for exclude in (True, False):
        tag = "exclusion=True" if exclude else "exclusion=False"

        log.info("[%s] item-item CF...", tag)
        ii_alltime = item_item_recommendations(val_customers, matrix, similarity, c2r, i2c, alltime_top12, exclude)
        fb = {c for c in val_customers if c not in c2r}
        ii_recent = {c: (fallback_top12 if c in fb else r) for c, r in ii_alltime.items()}

        if not exclude:
            # reproduce the README's no-exclusion item-item row (all-time fallback)
            readme_check_row = iim.score_variant("check", ii_alltime, val, segments)[0]

        log.info("[%s] ALS...", tag)
        als_recs = als_recommendations(val_customers, model, cw_matrix, cw_c2r, cw_i2c, fallback_top12, exclude)

        log.info("[%s] recency baseline...", tag)
        rec_recs = csp.get_recency_recommendations(val_customers, recency_pop, purchased if exclude else None)

        for name, recs in (
            (f"ALS (7d fallback)", als_recs),
            (f"item-item CF (7d fallback)", ii_recent),
            (f"recency {RECENCY_BASELINE_DAYS}d", rec_recs),
        ):
            for r in iim.score_variant(name, recs, val, segments):
                rows.append({"setting": tag, **r})

    return pd.DataFrame(rows), readme_check_row


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="ALS vs item-item CF vs recency, both exclusion settings.")
    parser.add_argument("--factors", type=int, default=ALS_CONFIG["factors"])
    parser.add_argument("--regularization", type=float, default=ALS_CONFIG["regularization"])
    parser.add_argument("--alpha", type=float, default=ALS_CONFIG["alpha"])
    parser.add_argument("--iterations", type=int, default=ALS_CONFIG["iterations"])
    args = parser.parse_args()
    als_config = {"factors": args.factors, "regularization": args.regularization,
                  "alpha": args.alpha, "iterations": args.iterations}

    train = base.load_train()
    val = base.load_val()
    results, check = run(train, val, als_config=als_config)
    log.info("ALS config used for every ALS row below: %s", als_config)

    ok = all(abs(check[m] - README_ITEM_ITEM_NO_EXCL[m]) < 6e-5 for m in ("precision@12", "recall@12", "map@12")) \
        and abs(check["hit_rate@12"] - README_ITEM_ITEM_NO_EXCL["hit_rate@12"]) < 6e-4
    log.info("%s: no-exclusion item-item (all-time fallback) vs README row: got P=%.5f R=%.5f HR=%.4f MAP=%.5f, README %s",
             "CHECK PASSED" if ok else "CHECK FAILED",
             check["precision@12"], check["recall@12"], check["hit_rate@12"], check["map@12"], README_ITEM_ITEM_NO_EXCL)

    pd.set_option("display.width", 220)
    order = ["aggregate", "0 purchases", "1-2 purchases", "3+ purchases"]
    for metric in ("map@12", "hit_rate@12"):
        print(f"\n=== {metric} ===")
        pivot = results.pivot_table(index=["setting", "variant"], columns="segment", values=metric)
        print(pivot[order].round(6).to_string())

    results.to_csv("exclusion_comparison_results.csv", index=False)
    log.info("wrote exclusion_comparison_results.csv")