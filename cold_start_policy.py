"""
cold_start_policy.py -- Week 5-6, Sun Oct 4: explicit cold-start policy for
segment "0 purchases" (customers with no train history, who get no learned
ALS embedding and currently fall back to all-time popularity).

Candidate policy tested here: popularity computed over only the most recent N
days of train, instead of all of train. README.md already showed a 7-day
recency list scoring MAP@12 0.00677 vs 0.00344 for all-time popularity
(overall, no exclusion) -- this script asks whether that gap holds up on the
segment that actually needs the policy, and at which window length.

Does NOT retrain ALS: it only needs train, val and baseline.py, so it runs in
minutes, not the ~25 minutes of model.py's sweep.

LEAK WARNING (same as baseline.py): recency windows are anchored to the max
date in TRAIN, and counted from train rows only. Val is only ever scored
against.

DESIGN DECISION -- every window is scored per segment, with and without
exclusion of the customer's own train purchases. Segment 0 has no purchases to
exclude, so its two rows are identical by construction (a built-in sanity
check). Segments 1-2 and 3+ are scored too, with exclusion on, because that is
the setting the ALS and item-item CF numbers in README.md use -- it gives the
equal-terms recency-vs-personalization comparison README.md flagged as missing.

Usage:
    python cold_start_policy.py
"""

import logging

import pandas as pd

import baseline as base

log = logging.getLogger(__name__)

WINDOWS_DAYS = [7, 14, 28, None]  # None = all-time popularity, today's fallback
N_CANDIDATES = 200  # candidate pool so exclusion can still fill 12 slots


def compute_recent_popularity(train: pd.DataFrame, days) -> pd.Series:
    """Article purchase counts over the last `days` days of train (anchored to
    train's own max t_dat). days=None means all of train."""
    if days is None:
        return base.compute_popularity(train)
    cutoff = train["t_dat"].max() - pd.Timedelta(days=days)
    return train.loc[train["t_dat"] > cutoff, "article_id"].value_counts()


def get_recency_recommendations(customer_ids, popularity: pd.Series, purchased=None, k: int = 12) -> dict:
    """Same ranked popularity list for everyone; if `purchased` (a dict of
    customer_id -> set of train article_ids) is given, each customer's own
    train purchases are skipped and the next-most-popular items fill the list.
    """
    candidates = popularity.head(N_CANDIDATES).index.tolist()
    top_k = candidates[:k]

    recommendations = {}
    for customer_id in customer_ids:
        owned = purchased.get(customer_id) if purchased is not None else None
        if not owned:
            recommendations[customer_id] = top_k
        else:
            recommendations[customer_id] = [a for a in candidates if a not in owned][:k]
    return recommendations


def run_policy_comparison(train: pd.DataFrame, val: pd.DataFrame, windows=WINDOWS_DAYS) -> pd.DataFrame:
    val_customers = val["customer_id"].unique()
    segments = base.segment_customers(train, val_customers)
    purchased = (
        train[train["customer_id"].isin(val_customers)]
        .groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    rows = []
    for days in windows:
        label = "all-time" if days is None else f"last {days}d"
        popularity = compute_recent_popularity(train, days)

        for exclusion in (False, True):
            recs_dict = get_recency_recommendations(
                val_customers, popularity, purchased if exclusion else None
            )
            recs = pd.DataFrame({
                "customer_id": list(recs_dict.keys()),
                "recommendations": list(recs_dict.values()),
            })
            for segment_label in ["0 purchases", "1-2 purchases", "3+ purchases"]:
                result = base.evaluate_segment(recs, val, segments, segment_label)
                rows.append({"policy": label, "exclusion": exclusion, **result})
            log.info("done: %s, exclusion=%s", label, exclusion)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train = base.load_train()
    val = base.load_val()
    log.info("train date range: %s -> %s", train["t_dat"].min(), train["t_dat"].max())

    results = run_policy_comparison(train, val)

    pd.set_option("display.width", 200)
    cols = ["policy", "exclusion", "n_customers", "precision@12", "recall@12", "hit_rate@12", "map@12"]
    for segment_label in ["0 purchases", "1-2 purchases", "3+ purchases"]:
        print(f"\n=== segment: {segment_label} ===")
        print(results.loc[results["segment"] == segment_label, cols].to_string(index=False))

    results.to_csv("cold_start_policy_results.csv", index=False)
    log.info("wrote cold_start_policy_results.csv")