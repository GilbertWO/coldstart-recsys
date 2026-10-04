"""
item_item_matched.py -- Week 5-6, Oct 4: item-item CF re-scored with the SAME
cold-start fallback ALS now uses (last-7-day popularity), so the ALS vs
item-item CF comparison is no longer confounded by two different fallbacks.

Why this exists: model.py's ALS now falls back to last-7-day popularity for
customers with no train history; baseline.py's item-item CF (and the README
numbers) still fall back to all-time popularity. ALS's aggregate MAP@12 gain
from the new fallback (+0.000185) is exactly segment 0's share of customers
times segment 0's gain, i.e. it came from the fallback list, not the model.
This script gives item-item CF the same list and re-scores it.

Does not modify baseline.py. Reuses its functions as-is. The expensive step
(per-customer item-item scoring) runs ONCE: a customer's personalized list
doesn't depend on the fallback list, so the 7-day variant is built by swapping
the fallback list in for customers with no row in the interaction matrix.

NOTE on scope of the swap: item-item CF falls back for every val customer with
no row in its top-5,000-item interaction matrix. That includes the true
cold-start customers (segment 0) AND any customer whose train purchases were
all outside the top-5,000 candidate articles. Both groups get the new list
here; the log prints how many customers that is.

Usage:
    python item_item_matched.py
"""

import logging

import pandas as pd

import baseline as base
import cold_start_policy as csp

log = logging.getLogger(__name__)

COLD_START_WINDOW_DAYS = 7  # keep equal to model.py's COLD_START_WINDOW_DAYS

# Reference rows pasted from earlier runs, for side-by-side printing only
# (ALS = factors=50, alpha=15, reg=0.01, iter=15, exclusion on, 7-day fallback).
ALS_REFERENCE = {
    "aggregate": {"precision@12": 0.003083, "recall@12": 0.013190, "hit_rate@12": 0.03432, "map@12": 0.004864},
    "0 purchases": {"map@12": 0.006393},
    "1-2 purchases": {"map@12": 0.009197},
    "3+ purchases": {"map@12": 0.004598},
}
# README's reported item-item CF row (all-time fallback, exclusion on).
README_ITEM_ITEM = {"precision@12": 0.00314, "recall@12": 0.01285, "hit_rate@12": 0.0344, "map@12": 0.00478}


def score_variant(name, recs_dict, val, segments):
    recs = pd.DataFrame({
        "customer_id": list(recs_dict.keys()),
        "recommendations": list(recs_dict.values()),
    })
    precision, recall = base.precision_recall_at_k(recs, val)
    row = {
        "variant": name, "segment": "aggregate",
        "precision@12": precision, "recall@12": recall,
        "hit_rate@12": base.hit_rate_at_k(recs, val), "map@12": base.map_at_k(recs, val),
    }
    rows = [row]
    for segment_label in ["0 purchases", "1-2 purchases", "3+ purchases"]:
        seg = base.evaluate_segment(recs, val, segments, segment_label)
        rows.append({"variant": name, **{k: v for k, v in seg.items()}})
    return rows


def run(train, val, n_candidates=None):
    n_candidates = n_candidates or base.N_CANDIDATE_ITEMS
    val_customers = val["customer_id"].unique()
    segments = base.segment_customers(train, val_customers)

    popularity = base.compute_popularity(train)
    alltime_top12 = base.get_popularity_recommendations(popularity)
    recent_top12 = csp.compute_recent_popularity(train, COLD_START_WINDOW_DAYS).head(12).index.tolist()

    matrix, customer_to_row, item_to_col = base.build_interaction_matrix(train, popularity, n_candidates)
    similarity = base.compute_item_similarity(matrix)

    log.info("Scoring item-item CF once (all-time fallback)...")
    recs_alltime = base.get_item_item_recommendations(
        val_customers, matrix, similarity, customer_to_row, item_to_col, alltime_top12
    )

    fallback_customers = [c for c in val_customers if c not in customer_to_row]
    log.info("%d / %d val customers have no row in the item-item matrix and use the fallback list",
             len(fallback_customers), len(val_customers))

    fallback_set = set(fallback_customers)
    recs_recent = {c: (recent_top12 if c in fallback_set else r) for c, r in recs_alltime.items()}

    rows = score_variant("item-item, all-time fallback", recs_alltime, val, segments)
    rows += score_variant(f"item-item, last-{COLD_START_WINDOW_DAYS}d fallback", recs_recent, val, segments)
    return pd.DataFrame(rows), len(fallback_customers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train = base.load_train()
    val = base.load_val()
    results, n_fallback = run(train, val)

    pd.set_option("display.width", 200)
    cols = ["variant", "segment", "n_customers", "precision@12", "recall@12", "hit_rate@12", "map@12"]
    print(results.reindex(columns=cols).to_string(index=False))

    # Sanity check: the all-time variant must reproduce README.md's reported
    # item-item CF row. If it doesn't, this script's pipeline differs from
    # baseline.py's and nothing below it should be trusted.
    agg = results[(results["variant"] == "item-item, all-time fallback") & (results["segment"] == "aggregate")].iloc[0]
    ok = all(abs(agg[m] - README_ITEM_ITEM[m]) < 6e-5 for m in ("precision@12", "recall@12", "map@12")) \
        and abs(agg["hit_rate@12"] - README_ITEM_ITEM["hit_rate@12"]) < 6e-4
    log.info("%s: all-time variant aggregate vs README item-item row (%s)",
             "CHECK PASSED" if ok else "CHECK FAILED", README_ITEM_ITEM)

    print("\n=== matched comparison, MAP@12 (exclusion on, both models use the 7-day fallback) ===")
    matched = results[results["variant"].str.contains("last-")].set_index("segment")["map@12"]
    for seg in ["aggregate", "0 purchases", "1-2 purchases", "3+ purchases"]:
        als = ALS_REFERENCE[seg]["map@12"]
        ii = matched[seg]
        print(f"{seg:>14}:  ALS {als:.6f}   item-item {ii:.6f}   ALS vs item-item {100 * (als / ii - 1):+.1f}%")

    results.to_csv("item_item_matched_results.csv", index=False)
    log.info("wrote item_item_matched_results.csv")