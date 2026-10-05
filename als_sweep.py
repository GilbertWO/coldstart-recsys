"""
als_sweep.py -- Week 5-6, Oct 5: how sensitive are the ALS results to its hyperparameters,
measured over SEVERAL weeks instead of the one validation week the original sweep used?

Background: the frozen ALS config (factors=150, reg=0.1, alpha=15, 40 iterations) came from a
coarse five-config pass on ONE week, where several knobs changed at once. This script varies
one knob at a time around the frozen config, and scores every variant on several earlier
weeks (the same weekly folds as rolling_backtest.py) under both exclusion settings.

PROTOCOL (fixed before any result was seen):
  * Selection weeks: folds 1, 2, 3 (targets starting 2020-08-26, 08-19, 08-12... see the
    printed dates). Confirmation week: fold 4 (targets 2020-08-12), never used to choose.
    (Fold k's target week starts 7*k days before the validation week.)
  * The test week and the validation week are NOT used here.
  * A variant is only a "candidate" if, on the selection weeks, its mean MAP@12 beats the
    frozen config by at least 3% in BOTH exclusion settings AND it is not significantly worse
    (paired bootstrap over customers) in any selection week. A candidate must then also be
    confirmed on the confirmation week.
  * Regardless of the outcome, the frozen config remains the one reported on the test week:
    the test number was produced before this sweep, so this is a sensitivity analysis, not a
    re-tuning of the reported result. If a variant clearly wins, that is a finding to report,
    and its test-week number would not be a clean out-of-sample figure.
  * Item-item CF is NOT tuned here. Tuning only ALS widens the existing asymmetry (ALS got a
    sweep, item-item none), so any ALS gain must be read with that in mind.

Efficiency: one fit per (week, config) -- fits are the cost (about 3-7 min each depending on
factors) -- and recommendations use implicit's batched recommend, verified against the
per-customer loop used elsewhere (it must agree on at least 99% of a sample, else the script
stops). Every (week, config) result is cached to sweep_cache/ so an interrupted run resumes.
The frozen config's per-customer scores come from the rolling_backtest.py cache (no refit);
alignment is verified via the segment labels.

Usage:
    python als_sweep.py                         # selection weeks 1 2 3, all six variants (~2 h)
    python als_sweep.py --weeks 4 --configs alpha25   # confirm one variant on week 4
    python als_sweep.py --summarize-only        # print the comparison from the caches
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as base
import bootstrap_comparison as bs
import cold_start_policy as csp
import exclusion_comparison as ex
import model as als_mod
import rolling_backtest as rb

log = logging.getLogger(__name__)

FROZEN = {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 40}
GRID = {  # one knob at a time around the frozen config
    "alpha8": {**FROZEN, "alpha": 8.0},
    "alpha25": {**FROZEN, "alpha": 25.0},
    "reg0.03": {**FROZEN, "regularization": 0.03},
    "reg0.3": {**FROZEN, "regularization": 0.3},
    "f100": {**FROZEN, "factors": 100},
    "f200": {**FROZEN, "factors": 200},
}
SELECTION_WEEKS = [1, 2, 3]
CONFIRM_WEEKS = [4]
CACHE_DIR = Path("sweep_cache")
SETTINGS = ["exclusion=True", "exclusion=False"]
MIN_GAIN_PCT = 3.0


def cache_path(k, key):
    return CACHE_DIR / f"wk{k}_{key}.npz"


def batch_recommendations(model, matrix, c2r, i2c, customers, fallback_top12, exclude, k=12, chunk=4000):
    """Same output shape as exclusion_comparison.als_recommendations, but batched."""
    items_by_col = np.array([item for item, _ in sorted(i2c.items(), key=lambda kv: kv[1])])
    rows = [(c, c2r[c]) for c in customers if c in c2r]
    recs = {c: fallback_top12 for c in customers if c not in c2r}
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        idx = np.array([r for _, r in part])
        ids, _ = model.recommend(idx, matrix[idx], N=k, filter_already_liked_items=exclude)
        for (c, _), row_ids in zip(part, ids):
            recs[c] = items_by_col[row_ids].tolist()
    return recs


def check_batch_matches_loop(model, matrix, c2r, i2c, customers, fallback_top12, n=400, seed=0):
    rng = np.random.default_rng(seed)
    with_row = [c for c in customers if c in c2r]
    sample = [with_row[i] for i in rng.choice(len(with_row), size=min(n, len(with_row)), replace=False)]
    for exclude in (True, False):
        a = batch_recommendations(model, matrix, c2r, i2c, sample, fallback_top12, exclude)
        b = ex.als_recommendations(sample, model, matrix, c2r, i2c, fallback_top12, exclude)
        same = np.mean([list(a[c]) == list(b[c]) for c in sample])
        log.info("batched vs per-customer recommend (exclude=%s): %.1f%% identical lists", exclude, 100 * same)
        if same < 0.99:
            raise RuntimeError(f"batched recommend disagrees with the loop version (exclude={exclude}, {same:.3f})")


def prepare_week(k, all_hist, val_start):
    start = val_start - pd.Timedelta(days=7 * k)
    end = start + pd.Timedelta(days=7)
    target = all_hist[(all_hist["t_dat"] >= start) & (all_hist["t_dat"] < end)]
    history = all_hist[all_hist["t_dat"] < start]
    assert len(target) and len(history) and history["t_dat"].max() < target["t_dat"].min()
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg = base.segment_customers(history, customers).to_numpy().astype(str)
    if rb.fold_path(k).exists():
        _, seg_cached = bs.load_cache(rb.fold_path(k))
        assert np.array_equal(seg, seg_cached), f"fold {k}: customers/segments differ from the backtest cache"
    fallback_top12 = csp.compute_recent_popularity(history, rb.FALLBACK_DAYS).head(12).index.tolist()
    matrix, c2r, i2c = als_mod.build_confidence_weighted_matrix(history)
    log.info("week %d (target %s): %d customers, history ends %s", k, start.date(), len(customers), history["t_dat"].max().date())
    return {"start": start, "customers": customers, "actual": actual, "matrix": matrix, "c2r": c2r,
            "i2c": i2c, "fallback": fallback_top12}


def run_sweep(weeks, config_keys):
    train, val = base.load_train(), base.load_val()
    all_hist = pd.concat([train, val], ignore_index=True)
    val_start = val["t_dat"].min().normalize()
    CACHE_DIR.mkdir(exist_ok=True)
    checked = False
    for k in weeks:
        todo = [key for key in config_keys if not cache_path(k, key).exists()]
        if not todo:
            log.info("week %d: all requested configs cached", k)
            continue
        wk = prepare_week(k, all_hist, val_start)
        for key in todo:
            cfg = GRID[key]
            t0 = time.time()
            log.info("week %d: fitting %s %s", k, key, cfg)
            model = als_mod.fit_als_model(wk["matrix"], random_state=42, **cfg)
            if not checked:
                check_batch_matches_loop(model, wk["matrix"], wk["c2r"], wk["i2c"], wk["customers"], wk["fallback"])
                checked = True
            arrays = {}
            for exclude in (True, False):
                recs = batch_recommendations(model, wk["matrix"], wk["c2r"], wk["i2c"], wk["customers"], wk["fallback"], exclude)
                ap, hit = bs.per_customer_scores(recs, wk["customers"], wk["actual"])
                tag = "on" if exclude else "off"
                arrays[f"ap_{tag}"], arrays[f"hit_{tag}"] = ap, hit
                log.info("week %d %s [exclusion=%s]: MAP@12 %.6f, hit_rate@12 %.6f", k, key, exclude, ap.mean(), hit.mean())
            np.savez(cache_path(k, key), **arrays)
            log.info("week %d %s done in %.1f min, cached", k, key, (time.time() - t0) / 60)


def load_week(k, key):
    """(ap_on, hit_on, ap_off, hit_off) for a config on week k, or None."""
    if key == "frozen":
        if not rb.fold_path(k).exists():
            return None
        scores, _ = bs.load_cache(rb.fold_path(k))
        return (scores[("exclusion=True", "ALS")][0], scores[("exclusion=True", "ALS")][1],
                scores[("exclusion=False", "ALS")][0], scores[("exclusion=False", "ALS")][1])
    if not cache_path(k, key).exists():
        return None
    d = np.load(cache_path(k, key))
    return d["ap_on"], d["hit_on"], d["ap_off"], d["hit_off"]


def summarize(weeks, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for k in weeks:
        frozen = load_week(k, "frozen")
        if frozen is None:
            log.warning("week %d: no frozen-config scores (backtest cache missing); skipped", k)
            continue
        for key in ["frozen"] + list(GRID):
            cur = load_week(k, key)
            if cur is None:
                continue
            for si, setting in enumerate(SETTINGS):
                ap, f_ap = cur[2 * si], frozen[2 * si]
                hit, f_hit = cur[2 * si + 1], frozen[2 * si + 1]
                d = ap - f_ap
                if key == "frozen" or not np.any(d):
                    lo = hi = 0.0
                else:
                    lo, hi = np.percentile(bs.bootstrap_mean_diff(d, n_boot, rng), [2.5, 97.5])
                rows.append({"week": k, "config": key, "setting": setting, "MAP": ap.mean(), "hit": hit.mean(),
                             "rel_MAP_vs_frozen_pct": 100 * d.mean() / f_ap.mean(),
                             "rel_hit_vs_frozen_pct": 100 * (hit.mean() - f_hit.mean()) / f_hit.mean(),
                             "sig": 1 if lo > 0 else (-1 if hi < 0 else 0)})
    return pd.DataFrame(rows)


def print_summary(df, select_weeks, confirm_weeks):
    pd.set_option("display.width", 250)
    for setting in SETTINGS:
        sub = df[df["setting"] == setting]
        print(f"\n=== {setting} | MAP@12 relative to the frozen config (% ; '*' = paired CI excludes 0) ===")
        out = []
        for key in ["frozen"] + list(GRID):
            g = sub[sub["config"] == key].set_index("week")
            if g.empty:
                continue
            row = {"config": key}
            for k in sorted(g.index):
                tag = "sel" if k in select_weeks else ("conf" if k in confirm_weeks else "other")
                row[f"wk{k} ({tag})"] = f"{g.loc[k, 'rel_MAP_vs_frozen_pct']:+.1f}{'*' if g.loc[k, 'sig'] else ''}"
            sel = g.loc[[k for k in g.index if k in select_weeks]]
            row["sel mean %"] = sel["rel_MAP_vs_frozen_pct"].mean() if len(sel) else np.nan
            row["sel mean MAP"] = sel["MAP"].mean() if len(sel) else np.nan
            row["sel mean hit %"] = 100 * sel["hit"].mean() if len(sel) else np.nan
            out.append(row)
        print(pd.DataFrame(out).round(4).to_string(index=False))

    print("\n=== candidates under the pre-specified rule (>= +%.0f%% mean MAP on selection weeks in BOTH settings, "
          "and not significantly worse in any selection week) ===" % MIN_GAIN_PCT)
    found = False
    for key in GRID:
        ok = True
        details = []
        for setting in SETTINGS:
            g = df[(df["config"] == key) & (df["setting"] == setting) & (df["week"].isin(select_weeks))]
            if g["week"].nunique() < len(select_weeks):
                ok = False
                details.append(f"{setting}: incomplete ({g['week'].nunique()}/{len(select_weeks)} weeks)")
                continue
            mean_rel = g["rel_MAP_vs_frozen_pct"].mean()
            worse = int((g["sig"] == -1).sum())
            details.append(f"{setting}: mean {mean_rel:+.1f}%, sig-worse weeks {worse}")
            ok = ok and mean_rel >= MIN_GAIN_PCT and worse == 0
        print(f"  {key:8s} {'CANDIDATE' if ok else 'no'}   ({'; '.join(details)})")
        found = found or ok
    if not found:
        print("  no variant meets the rule: the frozen config stands; ALS results are not sensitive to these knobs "
              "within the ranges tried.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Multi-week one-knob-at-a-time ALS sensitivity sweep.")
    parser.add_argument("--weeks", type=int, nargs="+", default=SELECTION_WEEKS)
    parser.add_argument("--configs", nargs="+", default=list(GRID), choices=list(GRID))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if not args.summarize_only:
        run_sweep(args.weeks, args.configs)

    all_weeks = sorted(set(SELECTION_WEEKS + CONFIRM_WEEKS) | set(args.weeks))
    df = summarize(all_weeks, n_boot=args.n_boot)
    if df.empty:
        raise SystemExit("Nothing cached yet.")
    print_summary(df, SELECTION_WEEKS, CONFIRM_WEEKS)
    df.to_csv("als_sweep_results.csv", index=False)
    log.info("wrote als_sweep_results.csv")
