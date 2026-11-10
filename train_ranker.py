#!/usr/bin/env python3
"""Stage 3 of the re-ranker: train LightGBM LambdaRank on the training week, score the validation week.

    python train_ranker.py                 # defaults, repurchase mode
    python train_ranker.py --label first-default

Training uses only customers with at least one purchase among their candidates (a group with no
positive teaches a ranker nothing). Evaluation uses EVERY validation-week customer, as all earlier
tables do: a customer whose candidates miss all their purchases scores 0.

Compared on the same customers, repurchases allowed (exclusion=False):
  ranker    LightGBM top-12 from the candidates
  ALS       ALS top-12 (popularity fallback for customers without an ALS row), the serving default
  pop7d     7-day popularity top-12, the strongest simple baseline so far
  repurchase+pop  the customer's own purchases from the last 28 days (most recent first), then
            7-day popularity: a heuristic built from the ranker's two strongest signals, no model
Paired-bootstrap 95% intervals on the per-customer AP@12 difference (2,000 resamples).
Appends aggregate and per-segment rows to reranker_validation_results.csv. The test week is not used.
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import baseline as base
import bootstrap_comparison as bs
import candidates as cands
import cold_start_policy as csp
import features as ft

log = logging.getLogger("train_ranker")
FEAT_DIR = Path("data/features")
RESULTS_CSV = Path("reranker_validation_results.csv")
K = 12


def top_k(df, score_col, ascending=False):
    """{customer: [article, ...]} top-K by score_col within each customer."""
    df = df.sort_values(["customer_id", score_col], ascending=[True, ascending], kind="stable")
    return df.groupby("customer_id", sort=False)["article_id"].apply(lambda s: list(s.head(K))).to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="default", help="names this run in the results file")
    ap.add_argument("--mode", choices=("repurchase", "discover"), default="repurchase",
                    help="discover: items the customer bought before are removed from training, candidates "
                         "and every baseline (exclusion=True)")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    discover = args.mode == "discover"
    train = pd.read_parquet(FEAT_DIR / "features_train.parquet")
    if discover:
        train = train[train["ci_count"] == 0]
    has_pos = train.groupby("customer_id")["label"].transform("max") > 0
    train = train[has_pos].sort_values("customer_id", kind="stable")
    groups = train.groupby("customer_id", sort=True).size().to_numpy()
    log.info("training rows %d, customers %d, positive rate %.4f", len(train), len(groups), train["label"].mean())

    params = {"objective": "lambdarank", "metric": "map", "eval_at": [K],
              "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
              "min_data_in_leaf": 50, "seed": 42, "deterministic": True, "verbose": -1}
    dtrain = lgb.Dataset(train[ft.FEATURES], label=train["label"], group=groups)
    model = lgb.train(params, dtrain, num_boost_round=args.n_estimators)

    val = pd.read_parquet(FEAT_DIR / "features_val.parquet")
    if discover:
        val = val[val["ci_count"] == 0].copy()
    val["score"] = model.predict(val[ft.FEATURES])

    history, target, week_start = cands.split_week("val")
    customers = list(target["customer_id"].unique())
    actual = target.groupby("customer_id")["article_id"].apply(set).to_dict()
    seg = base.segment_customers(history, customers).astype(str).to_numpy()
    assert set(val["customer_id"].unique()) <= set(customers), "candidates for customers outside the validation week"
    if not discover:
        assert set(val["customer_id"].unique()) == set(customers), "candidates do not cover the validation customers"

    recs = {"ranker": top_k(val, "score")}
    als_rows = val[val["src_als"] == 1]
    cold_rows = val[(val["src_als"] == 0) & val["source"].astype(str).str.startswith("pop") & val["source_rank"].notna()]
    als = top_k(als_rows, "als_rank", ascending=True)
    als.update({c: r for c, r in top_k(cold_rows, "source_rank", ascending=True).items() if c not in als})
    recs["ALS"] = als
    pop7 = csp.compute_recent_popularity(history, 7)
    if discover:
        purchased = (history[history["customer_id"].isin(set(customers))]
                     .groupby("customer_id")["article_id"].apply(set).to_dict())
        recs["pop7d"] = csp.get_recency_recommendations(customers, pop7, purchased)
    else:
        pop12 = pop7.head(K).index.tolist()
        recs["pop7d"] = {c: pop12 for c in customers}
    # Strong heuristic: the customer's own purchases from the last 28 days (most recent first),
    # then the remaining candidates by 7-day popularity. No model.
    recent = val["ci_days_since"].notna() & (val["ci_days_since"] <= 28)
    val["heuristic"] = np.where(recent, 1e9 - val["ci_days_since"].fillna(0), val["item_cnt_7d"])
    if not discover:  # in discover mode there are no repurchases to put first
        recs["repurchase+pop"] = top_k(val, "heuristic")
    for r in recs.values():  # a customer with no candidates left gets an empty list (scores 0)
        for c in customers:
            r.setdefault(c, [])
    # Cold-start routing at the ranking stage: customers with no purchase history get the 7-day
    # popularity list (the project's cold-start policy), everyone else gets the ranker.
    cold = {c for c, s_ in zip(customers, seg) if s_ == "0 purchases"}
    recs["ranker+cold"] = {c: (recs["pop7d"][c] if c in cold else recs["ranker"][c]) for c in customers}

    scores = {name: bs.per_customer_scores(r, customers, actual) for name, r in recs.items()}
    rng = np.random.default_rng(42)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for scope in bs.SCOPES:
        mask = np.ones(len(customers), bool) if scope == "aggregate" else (seg == scope)
        row = {"run": args.label, "run_utc": run_at, "mode": "exclusion=True" if discover else "exclusion=False", "scope": scope,
               "n": int(mask.sum()), "n_estimators": args.n_estimators,
               "learning_rate": args.learning_rate, "num_leaves": args.num_leaves}
        for name, (ap_, hit) in scores.items():
            row[f"{name}_MAP@12"] = ap_[mask].mean()
            row[f"{name}_hit@12"] = hit[mask].mean()
        for mine, other in [("ranker", o) for o in ("ALS", "pop7d", "repurchase+pop") if o in scores] + \
                [("ranker+cold", o) for o in ("pop7d", "repurchase+pop") if o in scores]:
            d = scores[mine][0][mask] - scores[other][0][mask]
            lo, hi = np.percentile(bs.bootstrap_mean_diff(d, args.n_boot, rng), [2.5, 97.5])
            row[f"{mine}_vs_{other}_rel_pct"] = 100 * d.mean() / scores[other][0][mask].mean()
            row[f"{mine}_vs_{other}_ci"] = f"[{lo:+.6f}, {hi:+.6f}]" + (" *" if lo > 0 or hi < 0 else "")
        rows.append(row)
    res = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in ["scope", "n", "ranker_MAP@12", "ALS_MAP@12", "pop7d_MAP@12", "repurchase+pop_MAP@12",
            "ranker_vs_repurchase+pop_rel_pct", "ranker_vs_repurchase+pop_ci"] if c in res]
    print(f"\nvalidation week, mode={args.mode} (* = 95% interval excludes 0):")
    print(res[cols].to_string(index=False))
    print("\nhit rate@12:")
    print(res[[c for c in ["scope", "ranker_hit@12", "ALS_hit@12", "pop7d_hit@12", "repurchase+pop_hit@12"] if c in res]].to_string(index=False))
    rc = [c for c in res if c.startswith("ranker+cold_vs_")]
    print("\nranker+cold (no-history customers get the 7-day list), MAP@12 and comparisons:")
    print(res[["scope", "ranker+cold_MAP@12"] + rc].to_string(index=False))
    print("\nranker vs ALS and pop7d:")
    print(res[["scope", "ranker_vs_ALS_rel_pct", "ranker_vs_ALS_ci", "ranker_vs_pop7d_rel_pct", "ranker_vs_pop7d_ci"]].to_string(index=False))
    imp = pd.Series(model.feature_importance("gain"), index=ft.FEATURES)
    print("\nfeature importance (share of total gain):")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

    if RESULTS_CSV.exists():
        res = pd.concat([pd.read_csv(RESULTS_CSV), res], ignore_index=True)
    res.to_csv(RESULTS_CSV, index=False)
    log.info("appended to %s", RESULTS_CSV)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
