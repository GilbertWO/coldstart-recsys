"""
recency_window_recheck.py -- Week 5-6, Oct 5: is the "recency list collapses" finding
partly an artifact of the 14-day window?

fallback_diagnostic.py showed that in the week starting 2020-08-19 the 14-day popularity
list overlapped only 1 of the week's 12 best-selling articles (and scored all-customer
MAP@12 0.0028), while the 7-day list overlapped 4 of 12 and scored 0.0079 on the SAME
customers. The rolling backtest compared ALS and item-item against the 14-day list only,
so part of the "recency swings by week" result may be window choice, not recency in general.

This script re-scores the recency baseline with a 7-day window (any window via --window)
for the same weeks, reuses the cached per-customer ALS / item-item scores from
rolling_backtest.py and test_evaluation.py (no model refits), and reruns the paired
comparison models-vs-recency using the 7-day list.

Self-checks, so the cached scores are provably aligned with the customers recomputed here:
  1. recomputed segment labels must equal the cached ones;
  2. recomputing the 14-day recency scores must reproduce the cached "recency 14d" arrays.
If either fails the script stops: nothing below it can be trusted.

Which window counts as "the" recency baseline was NOT fixed before seeing results; the
14-day choice (made earlier) and the 7-day one (the cold-start fallback window) are both
reported. Do not select the window by looking at the test column.

Usage (a few minutes; needs backtest_fold_1..4.npz and test_per_customer_scores.npz):
    python recency_window_recheck.py
    python recency_window_recheck.py --window 7 --folds 1 2 3 4 --no-test
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import bootstrap_comparison as bs
import cold_start_policy as csp
import rolling_backtest as rb

log = logging.getLogger(__name__)

TEST_CACHE = Path("test_per_customer_scores.npz")


def recency_scores(history, customers, actual, window):
    """Per-customer (ap, hit) for the last-`window`-day list under both exclusion settings."""
    pop = csp.compute_recent_popularity(history, window)
    purchased = (
        history[history["customer_id"].isin(customers)]
        .groupby("customer_id")["article_id"].apply(set).to_dict()
    )
    out = {}
    for exclude in (True, False):
        recs = csp.get_recency_recommendations(customers, pop, purchased if exclude else None)
        out[f"exclusion={exclude}"] = bs.per_customer_scores(recs, customers, actual)
    return out


def augment_week(label, history, target, cache_path, window):
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    scores, seg_cached = bs.load_cache(cache_path)

    seg = base.segment_customers(history, customers).to_numpy().astype(str)
    assert len(seg) == len(seg_cached) and np.array_equal(seg, seg_cached), \
        f"{label}: recomputed customers/segments do not match the cache -- alignment failed"

    rec14 = recency_scores(history, customers, actual, 14)
    for setting, (ap, hit) in rec14.items():
        c_ap, c_hit = scores[(setting, "recency 14d")]
        assert np.allclose(ap, c_ap) and np.allclose(hit, c_hit), \
            f"{label}/{setting}: recomputed 14d recency does not reproduce the cached scores"

    new = recency_scores(history, customers, actual, window)
    name = f"recency {window}d"
    for setting, arrays in new.items():
        scores[(setting, name)] = arrays
    log.info("%s: alignment checks passed (segments + 14d recency reproduced); added '%s'", label, name)
    return scores, seg


def build_weeks(folds, include_test, window):
    train, val = base.load_train(), base.load_val()
    all_hist = pd.concat([train, val], ignore_index=True)
    val_start = val["t_dat"].min().normalize()
    weeks = {}
    for k in sorted(folds, reverse=True):
        if not rb.fold_path(k).exists():
            log.warning("fold %d: %s missing, skipped", k, rb.fold_path(k))
            continue
        start = val_start - pd.Timedelta(days=7 * k)
        end = start + pd.Timedelta(days=7)
        target = all_hist[(all_hist["t_dat"] >= start) & (all_hist["t_dat"] < end)]
        history = all_hist[all_hist["t_dat"] < start]
        weeks[f"wk {start.date()}"] = augment_week(f"fold {k}", history, target, rb.fold_path(k), window)
    if include_test and TEST_CACHE.exists():
        test = pd.read_parquet(Path("data/processed") / "transactions_test.parquet")
        weeks["TEST"] = augment_week("TEST", all_hist, test, TEST_CACHE, window)
    return weeks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Re-run models-vs-recency with a different recency window.")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    name = f"recency {args.window}d"
    weeks = build_weeks(args.folds, not args.no_test, args.window)
    if not weeks:
        raise SystemExit("No cached weeks found; run rolling_backtest.py / test_evaluation.py first.")
    labels = list(weeks.keys())

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print(f"\n=== absolute recency scores: {args.window}d window vs 14d window (all customers) ===")
    rows = []
    for lab, (scores, _) in weeks.items():
        for setting in bs.SETTINGS:
            for w in (name, "recency 14d"):
                ap, hit = scores[(setting, w)]
                rows.append({"week": lab, "setting": setting, "list": w, "MAP@12": ap.mean(), "hit_rate@12": hit.mean()})
    print(pd.DataFrame(rows).pivot_table(index=["setting", "list"], columns="week",
                                          values="MAP@12").reindex(columns=labels).round(6).to_string())

    # models vs the NEW recency list (patch the pair list; weekly_comparison reads bs.PAIRS at call time)
    bs.PAIRS = [("ALS", name), ("item-item CF", name)]
    long, wide = rb.summarize(weeks, n_boot=args.n_boot)

    print(f"\nRelative gap, A vs B (% of B), vs the {args.window}-day recency list. "
          f"'*' = that week's 95% paired-bootstrap CI excludes 0. Weeks, oldest first: {labels}")
    for setting in bs.SETTINGS:
        for metric in ("MAP@12", "hit_rate@12"):
            sub = wide[(wide["setting"] == setting) & (wide["metric"] == metric)]
            print(f"\n=== {setting} | {metric} ===")
            cols = ["scope", "pair"] + labels + ["mean", "sd", "weeks A ahead (sig)", "weeks B ahead (sig)"]
            print(sub[cols].round(2).to_string(index=False))

    long.to_csv(f"recency_{args.window}d_recheck_weekly.csv", index=False)
    wide.to_csv(f"recency_{args.window}d_recheck_summary.csv", index=False)
    log.info("wrote recency_%dd_recheck_weekly.csv and recency_%dd_recheck_summary.csv", args.window, args.window)
