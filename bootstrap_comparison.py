"""
bootstrap_comparison.py -- Week 5-6, Oct 4: are the model-comparison gaps real?

exclusion_comparison.py reports point estimates (e.g. ALS +8.2% MAP@12 over
item-item CF with exclusion off, -1.9% with exclusion on). With ~72k customers
a gap of that size may or may not exceed sampling noise. This script puts a
95% confidence interval on every pairwise difference using a PAIRED bootstrap
over customers: the same resampled customers are scored under both models, so
customer-level difficulty cancels out and the interval reflects the model
difference itself.

Pairs compared (each under exclusion on AND off, for MAP@12 and hit_rate@12,
aggregate and per segment):
    ALS - item-item CF
    ALS - recency 14d
    item-item CF - recency 14d

Setup is identical to exclusion_comparison.py: ALS on the purchase-count matrix
(default factors=150 / reg=0.1 / alpha=15 / 20 iters, the config that won on
MAP@12), ALS and item-item CF both on the last-7-day fallback, recency list =
last-14-day popularity. Per-customer scores are cached in
bootstrap_per_customer_scores.npz so the (slow) recommendation step runs once.

Self-checks: (1) per-customer scores are verified against baseline.py's own
map_at_k / hit_rate_at_k on a random subsample; (2) with the default ALS config
the aggregate point estimates must match the exclusion_comparison.py table
(CHECK PASSED / CHECK FAILED in the log).

CAVEATS, stated up front:
  - This quantifies sampling noise over CUSTOMERS in one validation week. It
    says nothing about week-to-week variation, which only a second evaluation
    week (the test split) can show.
  - ~48 intervals are reported; at 95% about 1 in 20 of them would exclude 0
    by chance alone even if every true difference were 0. Read the pattern,
    not any single borderline row.

Usage:
    python bootstrap_comparison.py                  # compute + bootstrap (~15 min)
    python bootstrap_comparison.py --from-cache     # re-bootstrap from the saved scores (seconds)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import cold_start_policy as csp
import exclusion_comparison as ex
import model as als_mod

log = logging.getLogger(__name__)

CACHE_PATH = Path("bootstrap_per_customer_scores.npz")
DEFAULT_ALS_CONFIG = {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 20}
MODELS = ["ALS", "item-item CF", "recency 14d"]
SETTINGS = ["exclusion=True", "exclusion=False"]
SCOPES = ["aggregate", "0 purchases", "1-2 purchases", "3+ purchases"]
PAIRS = [("ALS", "item-item CF"), ("ALS", "recency 14d"), ("item-item CF", "recency 14d")]

# Aggregate MAP@12 from exclusion_comparison.py run with the default config.
KNOWN_AGGREGATE_MAP = {
    ("exclusion=False", "ALS"): 0.009422, ("exclusion=False", "item-item CF"): 0.008709,
    ("exclusion=False", "recency 14d"): 0.006995,
    ("exclusion=True", "ALS"): 0.004995, ("exclusion=True", "item-item CF"): 0.005094,
    ("exclusion=True", "recency 14d"): 0.006435,
}


def per_customer_scores(recs_dict, customers, actual, k=12):
    """Per-customer AP@k and hit indicator, same definitions as baseline.map_at_k
    and baseline.hit_rate_at_k (verified against them in validate_against_baseline)."""
    ap = np.zeros(len(customers))
    hit = np.zeros(len(customers))
    for i, customer_id in enumerate(customers):
        true_items = actual[customer_id]
        hits = 0
        precision_sum = 0.0
        seen = set()
        for rank, item in enumerate(recs_dict[customer_id][:k], start=1):
            if item in true_items and item not in seen:
                hits += 1
                precision_sum += hits / rank
            seen.add(item)
        ap[i] = precision_sum / min(len(true_items), k)
        hit[i] = 1.0 if hits > 0 else 0.0
    return ap, hit


def validate_against_baseline(name, recs_dict, customers, actual, val, ap, hit, n_sample=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(customers), size=min(n_sample, len(customers)), replace=False)
    sample_ids = [customers[i] for i in idx]
    recs_df = pd.DataFrame({
        "customer_id": sample_ids,
        "recommendations": [recs_dict[c] for c in sample_ids],
    })
    base_map = base.map_at_k(recs_df, val)
    base_hit = base.hit_rate_at_k(recs_df, val)
    assert abs(base_map - ap[idx].mean()) < 1e-9, (name, "MAP mismatch", base_map, ap[idx].mean())
    assert abs(base_hit - hit[idx].mean()) < 1e-9, (name, "hit_rate mismatch", base_hit, hit[idx].mean())


def compute_scores(train, val, als_config, n_candidates=None):
    """Returns (scores, seg_labels): scores[(setting, model)] = (ap, hit) arrays
    aligned to val['customer_id'].unique(); seg_labels = per-customer segment."""
    n_candidates = n_candidates or base.N_CANDIDATE_ITEMS
    customers = list(val["customer_id"].unique())
    actual = val.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg_labels = base.segment_customers(train, customers).to_numpy().astype(str)

    popularity = base.compute_popularity(train)
    alltime_top12 = base.get_popularity_recommendations(popularity)
    fallback_top12 = csp.compute_recent_popularity(train, ex.FALLBACK_DAYS).head(12).index.tolist()
    recency_pop = csp.compute_recent_popularity(train, ex.RECENCY_BASELINE_DAYS)
    purchased = (
        train[train["customer_id"].isin(customers)]
        .groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    matrix, c2r, i2c = base.build_interaction_matrix(train, popularity, n_candidates)
    similarity = base.compute_item_similarity(matrix)

    cw_matrix, cw_c2r, cw_i2c = als_mod.build_confidence_weighted_matrix(train)
    log.info("Fitting ALS once: %s", als_config)
    model = als_mod.fit_als_model(cw_matrix, random_state=42, **als_config)

    scores = {}
    for exclude in (True, False):
        setting = f"exclusion={exclude}"

        log.info("[%s] item-item CF...", setting)
        ii_alltime = ex.item_item_recommendations(customers, matrix, similarity, c2r, i2c, alltime_top12, exclude)
        no_row = {c for c in customers if c not in c2r}
        ii_recent = {c: (fallback_top12 if c in no_row else r) for c, r in ii_alltime.items()}

        log.info("[%s] ALS...", setting)
        als_recs = ex.als_recommendations(customers, model, cw_matrix, cw_c2r, cw_i2c, fallback_top12, exclude)

        log.info("[%s] recency baseline...", setting)
        rec_recs = csp.get_recency_recommendations(customers, recency_pop, purchased if exclude else None)

        for name, recs in (("ALS", als_recs), ("item-item CF", ii_recent), ("recency 14d", rec_recs)):
            ap, hit = per_customer_scores(recs, customers, actual)
            validate_against_baseline(f"{setting}/{name}", recs, customers, actual, val, ap, hit)
            scores[(setting, name)] = (ap, hit)
            log.info("[%s] %s: MAP@12 %.6f, hit_rate@12 %.6f (validated vs baseline.py)",
                     setting, name, ap.mean(), hit.mean())

    return scores, seg_labels


def save_cache(scores, seg_labels, path=CACHE_PATH):
    arrays = {"seg_labels": seg_labels}
    for (setting, name), (ap, hit) in scores.items():
        arrays[f"{setting}|{name}|ap"] = ap
        arrays[f"{setting}|{name}|hit"] = hit
    np.savez(path, **arrays)


def load_cache(path=CACHE_PATH):
    data = np.load(path, allow_pickle=False)
    scores = {}
    for setting in SETTINGS:
        for name in MODELS:
            scores[(setting, name)] = (data[f"{setting}|{name}|ap"], data[f"{setting}|{name}|hit"])
    return scores, data["seg_labels"]


def bootstrap_mean_diff(d, n_boot, rng, chunk=100):
    """Bootstrap distribution of mean(d) over customers (resampling with replacement)."""
    n = len(d)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        out[start:start + m] = d[idx].mean(axis=1)
    return out


def run_bootstrap(scores, seg_labels, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for setting in SETTINGS:
        for metric_name, which in (("MAP@12", 0), ("hit_rate@12", 1)):
            for scope in SCOPES:
                mask = np.ones(len(seg_labels), dtype=bool) if scope == "aggregate" else (seg_labels == scope)
                for a_name, b_name in PAIRS:
                    a = scores[(setting, a_name)][which][mask]
                    b = scores[(setting, b_name)][which][mask]
                    d = a - b
                    point = d.mean()
                    if not np.any(d):
                        lo = hi = 0.0
                        verdict = "identical"
                    else:
                        boot = bootstrap_mean_diff(d, n_boot, rng)
                        lo, hi = np.percentile(boot, [2.5, 97.5])
                        verdict = f"{a_name} > {b_name}" if lo > 0 else (f"{b_name} > {a_name}" if hi < 0 else "no clear difference")
                    rows.append({
                        "setting": setting, "metric": metric_name, "scope": scope, "n": int(mask.sum()),
                        "A": a_name, "B": b_name, "mean_A": a.mean(), "mean_B": b.mean(),
                        "diff": point, "rel_diff_pct": 100 * point / b.mean() if b.mean() else np.nan,
                        "ci_low": lo, "ci_high": hi, "verdict": verdict,
                    })
        log.info("bootstrapped %s", setting)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for the model comparison.")
    parser.add_argument("--from-cache", action="store_true", help="skip recommendation step, load saved per-customer scores")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--factors", type=int, default=DEFAULT_ALS_CONFIG["factors"])
    parser.add_argument("--regularization", type=float, default=DEFAULT_ALS_CONFIG["regularization"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALS_CONFIG["alpha"])
    parser.add_argument("--iterations", type=int, default=DEFAULT_ALS_CONFIG["iterations"])
    args = parser.parse_args()
    als_config = {"factors": args.factors, "regularization": args.regularization,
                  "alpha": args.alpha, "iterations": args.iterations}

    if args.from_cache:
        log.info("Loading cached per-customer scores from %s", CACHE_PATH)
        scores, seg_labels = load_cache()
    else:
        train = base.load_train()
        val = base.load_val()
        scores, seg_labels = compute_scores(train, val, als_config)
        save_cache(scores, seg_labels)
        log.info("Saved per-customer scores to %s (re-run with --from-cache to re-bootstrap quickly)", CACHE_PATH)
        log.info("ALS config used: %s", als_config)

    # Check the point estimates reproduce exclusion_comparison.py's table (default config only).
    if not args.from_cache and als_config == DEFAULT_ALS_CONFIG:
        bad = {k: (scores[k][0].mean(), v) for k, v in KNOWN_AGGREGATE_MAP.items() if abs(scores[k][0].mean() - v) > 1.5e-6}
        log.info("%s: aggregate MAP@12 point estimates vs exclusion_comparison.py table%s",
                 "CHECK FAILED" if bad else "CHECK PASSED", f" -- mismatches: {bad}" if bad else "")

    results = run_bootstrap(scores, seg_labels, n_boot=args.n_boot)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["scope", "A", "B", "mean_A", "mean_B", "rel_diff_pct", "ci_low", "ci_high", "verdict"]
    for setting in SETTINGS:
        for metric in ("MAP@12", "hit_rate@12"):
            print(f"\n=== {setting} | {metric} | diff = mean(A) - mean(B), 95% paired-bootstrap CI ===")
            sub = results[(results["setting"] == setting) & (results["metric"] == metric)]
            print(sub[show].round(6).to_string(index=False))

    results.to_csv("bootstrap_comparison_results.csv", index=False)
    log.info("wrote bootstrap_comparison_results.csv")