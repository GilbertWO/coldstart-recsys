"""
test_evaluation.py -- Week 5-6, Oct 4: the ONE final evaluation on the TEST split.

DESIGN DECISIONS (fixed before this script was ever run):

  * History = train + val. The target is the test week, which starts the day
    after val ends. Tuning (7-day fallback, 14-day recency baseline, ALS config)
    was done on "history ends the day before the target week"; train+val -> test
    reproduces that structure. Train-only -> test would leave a one-week gap the
    configs were never tuned for, and would also be unlike deployment.
  * Everything is refit on train+val and segments are recomputed from
    train+val purchase counts, so segment sizes differ from the val tables.
    Do NOT read these numbers as "val vs test"; they are a separate table.
  * Config is FROZEN below. It was chosen on validation MAP@12 (factors=150,
    reg=0.1, alpha=15) and the convergence check (40 iterations). Nothing here
    may be changed after seeing test numbers.
  * Run once. The script refuses to run again if its results file exists,
    unless --rerun is passed. A rerun after seeing test numbers is a second
    look at test; if you do it, say so in the write-up.

What it does: builds history, fits ALS + item-item CF, scores ALS / item-item /
recency-14d under BOTH exclusion settings (aggregate + per segment), runs the
same paired bootstrap as bootstrap_comparison.py, and saves the trained ALS
model artifact (factors + id mappings + fallback list + config) to models/.

Self-checks: temporal ordering (no history row is on or after the test week);
per-customer scores verified against baseline.py's own metrics on a subsample.

Usage:
    python test_evaluation.py              # ~25 min
    python test_evaluation.py --rerun      # only if you accept a second look at test
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
import item_item_matched as iim
import model as als_mod

log = logging.getLogger(__name__)

# ---- FROZEN CONFIG -- do not edit after seeing test results -----------------
FROZEN_ALS_CONFIG = {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 40}
FALLBACK_DAYS = 7
RECENCY_BASELINE_DAYS = 14
RANDOM_STATE = 42
# -----------------------------------------------------------------------------

RESULTS_PATH = Path("test_evaluation_results.csv")
MODEL_DIR = Path("models")


def load_test(out_dir: Path = Path("data/processed")) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "transactions_test.parquet")


def save_model_artifact(model, c2r, i2c, fallback_top12, config, history_end, model_dir=MODEL_DIR):
    """Saves what a serving layer needs: factors, row/column id order, the cold-start
    fallback list, and the config. Plain npz + json, readable without `implicit`."""
    model_dir.mkdir(parents=True, exist_ok=True)
    customer_ids = [c for c, _ in sorted(c2r.items(), key=lambda kv: kv[1])]
    item_ids = [i for i, _ in sorted(i2c.items(), key=lambda kv: kv[1])]
    np.savez_compressed(
        model_dir / "als_factors.npz",
        user_factors=np.asarray(model.user_factors),
        item_factors=np.asarray(model.item_factors),
        customer_ids=np.asarray(customer_ids).astype(str),
        item_ids=np.asarray(item_ids).astype(str),
    )
    meta = {
        "config": config, "random_state": RANDOM_STATE,
        "fallback_days": FALLBACK_DAYS, "fallback_top12": [str(a) for a in fallback_top12],
        "history_end": str(history_end.date()),
        "n_customers": len(customer_ids), "n_items": len(item_ids),
        "note": "Trained on train+val history. exclusion of already-bought items is applied at recommend time.",
    }
    (model_dir / "als_meta.json").write_text(json.dumps(meta, indent=2))
    try:
        model.save(str(model_dir / "als_implicit.npz"))
    except Exception as e:  # implicit's own format is a convenience, not required
        log.warning("implicit model.save() failed (%s); factors npz + meta json were still saved.", e)
    log.info("Saved model artifact to %s/", model_dir)


def run(history, test):
    customers = list(test["customer_id"].unique())
    actual = test.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg_labels = base.segment_customers(history, customers).to_numpy().astype(str)

    log.info("Test customers: %d | segments (by train+val purchases): %s",
             len(customers), pd.Series(seg_labels).value_counts().to_dict())

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
    no_row = {c for c in customers if c not in c2r}
    log.info("item-item CF: %d / %d test customers have no matrix row and use the fallback", len(no_row), len(customers))

    cw_matrix, cw_c2r, cw_i2c = als_mod.build_confidence_weighted_matrix(history)
    log.info("Fitting ALS once on train+val: %s", FROZEN_ALS_CONFIG)
    model = als_mod.fit_als_model(cw_matrix, random_state=RANDOM_STATE, **FROZEN_ALS_CONFIG)
    save_model_artifact(model, cw_c2r, cw_i2c, fallback_top12, FROZEN_ALS_CONFIG, history["t_dat"].max())

    scores = {}
    for exclude in (True, False):
        setting = f"exclusion={exclude}"

        log.info("[%s] item-item CF...", setting)
        ii_alltime = ex.item_item_recommendations(customers, matrix, similarity, c2r, i2c, alltime_top12, exclude)
        ii_recent = {c: (fallback_top12 if c in no_row else r) for c, r in ii_alltime.items()}

        log.info("[%s] ALS...", setting)
        als_recs = ex.als_recommendations(customers, model, cw_matrix, cw_c2r, cw_i2c, fallback_top12, exclude)

        log.info("[%s] recency baseline...", setting)
        rec_recs = csp.get_recency_recommendations(customers, recency_pop, purchased if exclude else None)

        for name, recs in (("ALS", als_recs), ("item-item CF", ii_recent), ("recency 14d", rec_recs)):
            ap, hit = bs.per_customer_scores(recs, customers, actual)
            bs.validate_against_baseline(f"{setting}/{name}", recs, customers, actual, test, ap, hit)
            scores[(setting, name)] = (ap, hit)
            log.info("[%s] %s: MAP@12 %.6f, hit_rate@12 %.6f (validated vs baseline.py)",
                     setting, name, ap.mean(), hit.mean())

    return scores, seg_labels


def point_table(scores, seg_labels):
    rows = []
    for (setting, name), (ap, hit) in scores.items():
        for scope in bs.SCOPES:
            mask = np.ones(len(seg_labels), dtype=bool) if scope == "aggregate" else (seg_labels == scope)
            rows.append({"setting": setting, "model": name, "scope": scope, "n": int(mask.sum()),
                         "map@12": ap[mask].mean() if mask.any() else np.nan,
                         "hit_rate@12": hit[mask].mean() if mask.any() else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Final TEST-split evaluation (run once).")
    parser.add_argument("--rerun", action="store_true", help="allow a second run (a second look at test; disclose it)")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if RESULTS_PATH.exists() and not args.rerun:
        raise SystemExit(f"{RESULTS_PATH} already exists: the test split has been evaluated. "
                         "Pass --rerun only if you accept (and will disclose) a second look at test.")

    train, val, test = base.load_train(), base.load_val(), load_test()
    history = pd.concat([train, val], ignore_index=True)

    # Leak guard: every history row must predate every test row, and val must follow train.
    assert history["t_dat"].max() < test["t_dat"].min(), "history overlaps the test period"
    assert train["t_dat"].max() < val["t_dat"].min(), "val overlaps train"
    log.info("history: %s -> %s | test: %s -> %s",
             history["t_dat"].min().date(), history["t_dat"].max().date(),
             test["t_dat"].min().date(), test["t_dat"].max().date())

    scores, seg_labels = run(history, test)
    bs.save_cache(scores, seg_labels, Path("test_per_customer_scores.npz"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    points = point_table(scores, seg_labels)
    for metric in ("map@12", "hit_rate@12"):
        print(f"\n=== TEST {metric} (history = train+val; ALS {FROZEN_ALS_CONFIG}) ===")
        pivot = points.pivot_table(index=["setting", "model"], columns="scope", values=metric)
        print(pivot[bs.SCOPES].round(6).to_string())

    boot = bs.run_bootstrap(scores, seg_labels, n_boot=args.n_boot)
    show = ["scope", "A", "B", "mean_A", "mean_B", "rel_diff_pct", "ci_low", "ci_high", "verdict"]
    for setting in bs.SETTINGS:
        for metric in ("MAP@12", "hit_rate@12"):
            print(f"\n=== TEST | {setting} | {metric} | diff = mean(A) - mean(B), 95% paired-bootstrap CI ===")
            sub = boot[(boot["setting"] == setting) & (boot["metric"] == metric)]
            print(sub[show].round(6).to_string(index=False))

    points.to_csv(RESULTS_PATH, index=False)
    boot.to_csv("test_bootstrap_results.csv", index=False)
    log.info("wrote %s and test_bootstrap_results.csv", RESULTS_PATH)