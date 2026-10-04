"""
rolling_backtest.py -- Week 5-6, Oct 4: how much do the model gaps move from week to week?

Why: the bootstrap in bootstrap_comparison.py / test_evaluation.py only measures
sampling noise over CUSTOMERS inside one week. The two weeks seen so far already
disagree on one comparison (ALS vs item-item MAP with exclusion on: -1.9% on val,
+6.2% on test). This script scores the same three models over several EARLIER
weekly folds so the gaps become a spread across weeks instead of two points.

Folds: fold k has target week [val_start - 7k days, +7 days) and history = every
row of train+val BEFORE that week (so each fold has the same "history ends the day
before the target week" structure every config was tuned under). Default folds
k = 1..4. The test week is never loaded or scored here, and no fold touches it.
Fold 0 (= the val week, which the configs WERE tuned on) is available with
`--folds 0 ...` but is not independent evidence, so it is off by default.

Config is FROZEN to test_evaluation.py's (factors=150, reg=0.1, alpha=15, 40
iterations, 7-day fallback for ALS and item-item, 14-day recency baseline). Do
not tune on these folds; they exist to measure variation, not to select a model.

Per fold: ALS fit once on the fold's history, item-item built once, all three
models scored under exclusion on AND off; per-customer AP / hit arrays are cached
to backtest_fold_<k>.npz so a crashed or interrupted run resumes where it stopped.
It also logs, per fold and segment, how many customers item-item cannot
personalize (no row in its top-5,000-article matrix) -- the confound flagged in
the session notes.

Summary: for each comparison (aggregate and 3+ purchases; MAP@12 and hit_rate@12;
both exclusion settings) the relative gap per week with a paired-bootstrap
significance mark, plus the mean, spread, and number of weeks each side wins. The
test week's cached scores (test_per_customer_scores.npz) are included as a column
if the file exists (use --no-test to leave it out).

CAVEATS, stated up front:
  - Earlier weeks differ in season and assortment; a gap that moves across weeks
    may be moving with the calendar, not noise. A handful of weeks cannot separate
    those. Treat the spread as descriptive.
  - Many intervals are computed with no multiple-comparison correction. Read the
    pattern across weeks, not any single marked cell.
  - Folds share customers, so they are not independent samples.

Usage (each fold takes roughly 12-15 min on your machine; 4 folds ~ 1 hour):
    python rolling_backtest.py                      # folds 1-4, then the summary
    python rolling_backtest.py --folds 1 2          # a subset (cached folds are skipped)
    python rolling_backtest.py --summarize-only     # re-summarize from cached folds
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import bootstrap_comparison as bs
import cold_start_policy as csp
import exclusion_comparison as ex
import model as als_mod

log = logging.getLogger(__name__)

# ---- FROZEN CONFIG (identical to test_evaluation.py) ------------------------
FROZEN_ALS_CONFIG = {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 40}
FALLBACK_DAYS = 7
RECENCY_BASELINE_DAYS = 14
RANDOM_STATE = 42
# -----------------------------------------------------------------------------

TEST_CACHE = Path("test_per_customer_scores.npz")
SUMMARY_SCOPES = ["aggregate", "3+ purchases"]
SIG_MARK = "*"  # appended to a cell whose 95% paired-bootstrap CI excludes 0


def fold_path(k):
    return Path(f"backtest_fold_{k}.npz")


def fold_meta_path(k):
    return Path(f"backtest_fold_{k}.json")


def score_fold(history, target, week_start):
    """Scores ALS / item-item CF / recency on one fold. Returns (scores, seg_labels, meta)."""
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg_labels = base.segment_customers(history, customers).to_numpy().astype(str)

    popularity = base.compute_popularity(history)
    alltime_top12 = base.get_popularity_recommendations(popularity)
    fallback_top12 = csp.compute_recent_popularity(history, FALLBACK_DAYS).head(12).index.tolist()
    recency_pop = csp.compute_recent_popularity(history, RECENCY_BASELINE_DAYS)
    purchased = (
        history[history["customer_id"].isin(customers)]
        .groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    matrix, c2r, i2c = base.build_interaction_matrix(history, popularity, base.N_CANDIDATE_ITEMS)
    similarity = base.compute_item_similarity(matrix)
    no_row_mask = np.array([c not in c2r for c in customers])
    no_row = {c for c, m in zip(customers, no_row_mask) if m}

    meta = {"week_start": str(week_start.date()), "history_end": str(history["t_dat"].max().date()),
            "n_customers": len(customers),
            "segment_sizes": {s: int((seg_labels == s).sum()) for s in bs.SCOPES[1:]},
            "item_item_no_row": {s: int(((seg_labels == s) & no_row_mask).sum()) for s in bs.SCOPES[1:]}}
    log.info("fold %s: %d customers | segments %s | item-item has NO row for %s",
             week_start.date(), len(customers), meta["segment_sizes"], meta["item_item_no_row"])

    cw_matrix, cw_c2r, cw_i2c = als_mod.build_confidence_weighted_matrix(history)
    log.info("fold %s: fitting ALS %s", week_start.date(), FROZEN_ALS_CONFIG)
    model = als_mod.fit_als_model(cw_matrix, random_state=RANDOM_STATE, **FROZEN_ALS_CONFIG)

    scores = {}
    for exclude in (True, False):
        setting = f"exclusion={exclude}"
        ii_alltime = ex.item_item_recommendations(customers, matrix, similarity, c2r, i2c, alltime_top12, exclude)
        ii_recent = {c: (fallback_top12 if c in no_row else r) for c, r in ii_alltime.items()}
        als_recs = ex.als_recommendations(customers, model, cw_matrix, cw_c2r, cw_i2c, fallback_top12, exclude)
        rec_recs = csp.get_recency_recommendations(customers, recency_pop, purchased if exclude else None)

        for name, recs in (("ALS", als_recs), ("item-item CF", ii_recent), ("recency 14d", rec_recs)):
            ap, hit = bs.per_customer_scores(recs, customers, actual)
            bs.validate_against_baseline(f"fold {week_start.date()}/{setting}/{name}",
                                         recs, customers, actual, target, ap, hit)
            scores[(setting, name)] = (ap, hit)
            log.info("fold %s [%s] %s: MAP@12 %.6f, hit_rate@12 %.6f",
                     week_start.date(), setting, name, ap.mean(), hit.mean())
    return scores, seg_labels, meta


def run_folds(folds):
    train, val = base.load_train(), base.load_val()
    all_hist = pd.concat([train, val], ignore_index=True)
    val_start = val["t_dat"].min().normalize()
    test_start = None
    try:
        test_start = pd.read_parquet(Path("data/processed") / "transactions_test.parquet", columns=["t_dat"])["t_dat"].min()
    except Exception:  # only used for the leak guard below
        log.warning("could not read the test split for the leak guard; relying on val_start only")

    for k in folds:
        if fold_path(k).exists() and fold_meta_path(k).exists():
            log.info("fold %d: cached (%s), skipping", k, fold_path(k))
            continue
        week_start = val_start - pd.Timedelta(days=7 * k)
        week_end = week_start + pd.Timedelta(days=7)
        target = all_hist[(all_hist["t_dat"] >= week_start) & (all_hist["t_dat"] < week_end)]
        history = all_hist[all_hist["t_dat"] < week_start]
        assert len(target) > 0 and len(history) > 0, f"fold {k}: empty target or history"
        assert history["t_dat"].max() < target["t_dat"].min(), f"fold {k}: history overlaps target week"
        if test_start is not None:
            assert target["t_dat"].max() < test_start, f"fold {k}: target week reaches the test split"
        log.info("fold %d: target %s -> %s (%d rows), history ends %s (%d rows)",
                 k, target["t_dat"].min().date(), target["t_dat"].max().date(), len(target),
                 history["t_dat"].max().date(), len(history))
        scores, seg_labels, meta = score_fold(history, target, week_start)
        bs.save_cache(scores, seg_labels, fold_path(k))
        fold_meta_path(k).write_text(json.dumps(meta, indent=2))
        log.info("fold %d saved to %s", k, fold_path(k))


# ---------------------------------------------------------------- summary --

def weekly_comparison(scores, seg_labels, label, n_boot, rng):
    """One row per (setting, metric, scope, pair) for one week."""
    rows = []
    for setting in bs.SETTINGS:
        for metric_name, which in (("MAP@12", 0), ("hit_rate@12", 1)):
            for scope in SUMMARY_SCOPES:
                mask = np.ones(len(seg_labels), dtype=bool) if scope == "aggregate" else (seg_labels == scope)
                for a_name, b_name in bs.PAIRS:
                    a = scores[(setting, a_name)][which][mask]
                    b = scores[(setting, b_name)][which][mask]
                    d = a - b
                    mean_b = b.mean()
                    if not mask.any() or not np.any(d):
                        lo = hi = 0.0
                    else:
                        lo, hi = np.percentile(bs.bootstrap_mean_diff(d, n_boot, rng), [2.5, 97.5])
                    rows.append({
                        "week": label, "setting": setting, "metric": metric_name, "scope": scope,
                        "pair": f"{a_name} vs {b_name}", "n": int(mask.sum()),
                        "rel_diff_pct": 100 * d.mean() / mean_b if mean_b else np.nan,
                        "sig": 1 if lo > 0 else (-1 if hi < 0 else 0),
                    })
    return rows


def summarize(week_scores, n_boot=2000, seed=42):
    """week_scores: ordered dict label -> (scores, seg_labels). Returns (long df, wide df)."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, (scores, seg_labels) in week_scores.items():
        rows += weekly_comparison(scores, seg_labels, label, n_boot, rng)
        log.info("bootstrapped week %s", label)
    long = pd.DataFrame(rows)

    keys = ["setting", "metric", "scope", "pair"]
    labels = list(week_scores.keys())
    out = []
    for key, g in long.groupby(keys, sort=False):
        g = g.set_index("week").reindex(labels)
        rel = g["rel_diff_pct"].astype(float)
        row = dict(zip(keys, key))
        for lab in labels:
            row[lab] = f"{rel[lab]:+.1f}{SIG_MARK if g.loc[lab, 'sig'] != 0 else ''}"
        row["mean"] = rel.mean()
        row["sd"] = rel.std(ddof=1) if len(rel) > 1 else np.nan
        row["weeks A ahead (sig)"] = int((g["sig"] == 1).sum())
        row["weeks B ahead (sig)"] = int((g["sig"] == -1).sum())
        row["weeks A ahead (any)"] = int((rel > 0).sum())
        out.append(row)
    return long, pd.DataFrame(out)


def load_week_scores(folds, include_test):
    week_scores = {}
    for k in sorted(folds, reverse=True):  # oldest week first
        if not fold_path(k).exists():
            log.warning("fold %d has no cached scores (%s); left out of the summary", k, fold_path(k))
            continue
        meta = json.loads(fold_meta_path(k).read_text()) if fold_meta_path(k).exists() else {}
        label = f"wk {meta.get('week_start', f'-{k}')}"
        week_scores[label] = bs.load_cache(fold_path(k))
    if include_test and TEST_CACHE.exists():
        week_scores["TEST"] = bs.load_cache(TEST_CACHE)
    elif include_test:
        log.info("%s not found; summary will not include the test week", TEST_CACHE)
    return week_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Rolling backtest of ALS vs item-item CF vs recency.")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="fold indices (k weeks before the val week); 0 = the val week itself (tuned on, not independent)")
    parser.add_argument("--summarize-only", action="store_true", help="skip scoring, summarize cached folds")
    parser.add_argument("--no-test", action="store_true", help="leave the test week out of the summary")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if not args.summarize_only:
        run_folds(args.folds)

    week_scores = load_week_scores(args.folds, include_test=not args.no_test)
    if not week_scores:
        raise SystemExit("No cached folds found; run without --summarize-only first.")

    long, wide = summarize(week_scores, n_boot=args.n_boot)
    labels = list(week_scores.keys())

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)
    print(f"\nRelative gap, A vs B, in % of B's score. '{SIG_MARK}' = that week's 95% paired-bootstrap CI excludes 0. "
          f"Weeks (oldest first): {labels}")
    for setting in bs.SETTINGS:
        for metric in ("MAP@12", "hit_rate@12"):
            sub = wide[(wide["setting"] == setting) & (wide["metric"] == metric)]
            print(f"\n=== {setting} | {metric} ===")
            cols = ["scope", "pair"] + labels + ["mean", "sd", "weeks A ahead (sig)", "weeks B ahead (sig)", "weeks A ahead (any)"]
            print(sub[cols].round(2).to_string(index=False))

    long.to_csv("backtest_weekly_comparisons.csv", index=False)
    wide.to_csv("backtest_summary.csv", index=False)
    log.info("wrote backtest_weekly_comparisons.csv and backtest_summary.csv")