#!/usr/bin/env python3
"""Stage 2 of the re-ranker: one feature row per (customer, candidate article).

Every feature is computed only from history strictly before the target week's first day
(`cutoff`). build_features() refuses a history frame that contains any row on or after the
cutoff, and test_features.py checks that this guard holds and that the values are right.

    python features.py --week train     # reads data/candidates/candidates_train_<config>.parquet
    python features.py --week val

Writes data/features/features_<week>.parquet (gitignored: customer and article ids).

Features (first set):
  als_score, als_rank          ALS score and rank; NaN when the article did not come from ALS
  src_als, src_pop, src_recent 1 if the article came from that candidate source
  item_cnt_7d, item_cnt_28d    purchases of the article in the last 7 / 28 days of history
  item_price_mean_28d          mean price of the article in the last 28 days (NaN if unsold)
  ci_count                     times this customer bought this article before
  ci_days_since                days since the customer last bought it (NaN if never)
  cust_n_purchases             the customer's purchases in history
  cust_days_since_last         days since the customer's last purchase (NaN if none)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import candidates as cands

log = logging.getLogger("features")

CONFIG = "als+pop50+recent28d"
CAND_DIR = Path("data/candidates")
OUT_DIR = Path("data/features")
FEATURES = ["als_score", "als_rank", "src_als", "src_pop", "src_recent",
            "item_cnt_7d", "item_cnt_28d", "item_price_mean_28d",
            "ci_count", "ci_days_since", "cust_n_purchases", "cust_days_since_last"]


def build_features(history: pd.DataFrame, cand: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """history: transactions (customer_id, article_id, t_dat, price), all strictly before `cutoff`.
    cand: candidates (customer_id, article_id, source, als_score, source_rank, label).
    Returns cand with the FEATURES columns added, in the same row order."""
    cutoff = pd.Timestamp(cutoff).normalize()
    if len(history) and history["t_dat"].max() >= cutoff:
        raise ValueError(f"history has rows on or after the cutoff {cutoff.date()}: "
                         f"latest is {history['t_dat'].max().date()}")
    out = cand.copy()
    src = out["source"].astype(str)
    out["src_als"] = src.str.contains("als").astype(np.int8)
    out["src_pop"] = src.str.contains("pop").astype(np.int8)
    out["src_recent"] = src.str.contains("recent").astype(np.int8)
    out["als_rank"] = np.where(out["src_als"] == 1, out["source_rank"], np.nan)
    out["als_score"] = np.where(out["src_als"] == 1, out["als_score"], np.nan)

    def last_days(days):
        return history[history["t_dat"] >= cutoff - pd.Timedelta(days=days)]

    h7, h28 = last_days(7), last_days(28)
    item7 = h7.groupby("article_id").size().rename("item_cnt_7d")
    item28 = h28.groupby("article_id").size().rename("item_cnt_28d")
    price28 = h28.groupby("article_id")["price"].mean().rename("item_price_mean_28d")
    out = out.join(item7, on="article_id").join(item28, on="article_id").join(price28, on="article_id")
    out[["item_cnt_7d", "item_cnt_28d"]] = out[["item_cnt_7d", "item_cnt_28d"]].fillna(0)

    hc = history[history["customer_id"].isin(set(out["customer_id"]))]
    ci = hc.groupby(["customer_id", "article_id"]).agg(ci_count=("t_dat", "size"), ci_last=("t_dat", "max"))
    out = out.join(ci, on=["customer_id", "article_id"])
    out["ci_count"] = out["ci_count"].fillna(0)
    out["ci_days_since"] = (cutoff - out.pop("ci_last")).dt.days

    cu = hc.groupby("customer_id").agg(cust_n_purchases=("t_dat", "size"), cust_last=("t_dat", "max"))
    out = out.join(cu, on="customer_id")
    out["cust_n_purchases"] = out["cust_n_purchases"].fillna(0)
    out["cust_days_since_last"] = (cutoff - out.pop("cust_last")).dt.days
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", choices=("val", "train", "test"), required=True)
    args = ap.parse_args()

    history, target, week_start = cands.split_week(args.week)
    cand_path = CAND_DIR / f"candidates_{args.week}_{CONFIG}.parquet"
    cand = pd.read_parquet(cand_path)
    log.info("%s: %d candidate rows for %d customers", cand_path, len(cand), cand["customer_id"].nunique())
    feats = build_features(history[["customer_id", "article_id", "t_dat", "price"]], cand, week_start)

    print(feats[FEATURES + ["label"]].describe().T[["count", "mean", "min", "max"]].to_string())
    print("\nmean of each feature, positives vs negatives:")
    print(feats.groupby("label")[FEATURES].mean().T.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"features_{args.week}.parquet"
    feats.to_parquet(out, index=False)
    log.info("saved %d rows x %d features to %s", len(feats), len(FEATURES), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
