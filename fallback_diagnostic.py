"""
fallback_diagnostic.py -- Week 5-6, Oct 5: do the popularity-window choices hold up
outside the week they were chosen on?

Why: the 7-day cold-start fallback (chosen from segment 0 on the validation week)
and the 14-day recency baseline were both picked on a week from the period where
recency lists work well. The rolling backtest then showed the 14-day recency list
collapsing in the weeks starting 08-12 and 08-19 (MAP@12 about 0.0026-0.0029 vs
about 0.006 later). Two questions:
  1. Cold-start: for segment 0 (no history), how do popularity windows of 7, 14,
     28 days and all-time compare in EACH week -- is 7 days still sensible, or was
     it a favorable-regime pick?
  2. Mechanism: when the recency list collapses, is it because the list stopped
     overlapping with what customers actually bought that week? Per window and
     week this prints (a) how many of the 12 listed articles were among the
     week's 12 most-purchased articles and (b) the share of that week's purchases
     covered by the 12 listed articles.

Weeks: the val week (fold 0; configs were tuned on it) and folds 1-4 from
rolling_backtest.py (target week starts 7*k days before the val week; history =
everything before that week). The test week is NOT included by default so nothing
here can leak into a choice made on it; --include-test adds it for completeness.
DO NOT pick a new window by looking at the test column.

For segment 0 the comparison is paired (same customers, same week), with a
bootstrap CI on the MAP difference vs the 7-day list. Segment 0 has only about
4,400-5,900 customers per week, so single-week differences are noisy; read the
pattern across weeks.

No model fitting; takes a few minutes (mostly loading the data).

Usage:
    python fallback_diagnostic.py
    python fallback_diagnostic.py --folds 0 1 2 3 4 --include-test
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import bootstrap_comparison as bs
import cold_start_policy as csp

log = logging.getLogger(__name__)

WINDOWS = [7, 14, 28, None]
SEG0 = "0 purchases"


def win_label(w):
    return "all-time" if w is None else f"{w}d"


def diagnose_week(history, target, week_label, n_boot=2000, seed=42):
    """Rows (one per window) for one target week. Everyone gets the same top-12 list
    from the window (no per-customer exclusion), so this isolates the list itself."""
    rng = np.random.default_rng(seed)
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg = base.segment_customers(history, customers).to_numpy().astype(str)
    seg0_mask = seg == SEG0

    week_counts = target["article_id"].value_counts()
    week_top12 = set(week_counts.head(12).index)
    total_rows = int(week_counts.sum())

    scored = {}
    rows = []
    for w in WINDOWS:
        top12 = csp.compute_recent_popularity(history, w).head(12).index.tolist()
        recs = {c: top12 for c in customers}
        ap, hit = bs.per_customer_scores(recs, customers, actual)
        scored[w] = ap
        rows.append({
            "week": week_label, "window": win_label(w), "n_seg0": int(seg0_mask.sum()),
            "seg0_MAP": ap[seg0_mask].mean() if seg0_mask.any() else np.nan,
            "seg0_hit": hit[seg0_mask].mean() if seg0_mask.any() else np.nan,
            "all_MAP": ap.mean(), "all_hit": hit.mean(),
            "overlap_with_week_top12": len(set(top12) & week_top12),
            "purchase_coverage_pct": 100 * float(week_counts.reindex(top12).fillna(0).sum()) / total_rows,
        })

    # paired seg-0 MAP difference of each window vs the 7-day list
    if seg0_mask.any():
        for r in rows:
            wlabel = r["window"]
            if wlabel == "7d":
                r["seg0_vs_7d"] = ""
                continue
            w = None if wlabel == "all-time" else int(wlabel[:-1])
            d = (scored[w] - scored[7])[seg0_mask]
            if not np.any(d):
                r["seg0_vs_7d"] = "identical"
                continue
            lo, hi = np.percentile(bs.bootstrap_mean_diff(d, n_boot, rng), [2.5, 97.5])
            r["seg0_vs_7d"] = f"{100 * d.mean() / scored[7][seg0_mask].mean():+.1f}%" + ("*" if (lo > 0 or hi < 0) else "")
    return rows


def build_weeks(folds, include_test):
    train, val = base.load_train(), base.load_val()
    all_hist = pd.concat([train, val], ignore_index=True)
    val_start = val["t_dat"].min().normalize()
    weeks = []
    for k in sorted(folds, reverse=True):  # oldest first
        start = val_start - pd.Timedelta(days=7 * k)
        end = start + pd.Timedelta(days=7)
        target = all_hist[(all_hist["t_dat"] >= start) & (all_hist["t_dat"] < end)]
        history = all_hist[all_hist["t_dat"] < start]
        assert len(target) and len(history) and history["t_dat"].max() < target["t_dat"].min()
        weeks.append((f"{start.date()}{' (val)' if k == 0 else ''}", history, target))
    if include_test:
        test = pd.read_parquet(Path("data/processed") / "transactions_test.parquet")
        assert all_hist["t_dat"].max() < test["t_dat"].min()
        weeks.append(("TEST (spent; do not select on it)", all_hist, test))
    return weeks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Do the popularity-window choices hold outside the val week?")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    rows = []
    for label, history, target in build_weeks(args.folds, args.include_test):
        log.info("week %s: %d target rows, history ends %s", label, len(target), history["t_dat"].max().date())
        rows += diagnose_week(history, target, label, n_boot=args.n_boot)
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["week", "window", "n_seg0", "seg0_MAP", "seg0_hit", "seg0_vs_7d", "all_MAP", "all_hit",
            "overlap_with_week_top12", "purchase_coverage_pct"]
    print("\n=== popularity windows by week (same top-12 list for everyone; no exclusion) ===")
    print("seg0_vs_7d = relative seg-0 MAP difference vs the 7-day list ('*' = paired-bootstrap 95% CI excludes 0)")
    print(df[cols].round(5).to_string(index=False))

    best = df.loc[df.groupby("week")["seg0_MAP"].idxmax(), ["week", "window"]]
    print("\nbest window on segment-0 MAP@12, by week:")
    print(best.to_string(index=False))
    print("\nweeks won per window:", best["window"].value_counts().to_dict())

    df.to_csv("fallback_diagnostic_results.csv", index=False)
    log.info("wrote fallback_diagnostic_results.csv")
