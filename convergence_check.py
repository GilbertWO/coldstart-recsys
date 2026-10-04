"""
convergence_check.py -- Week 5-6, Oct 4: was 20 ALS iterations enough?

model.py's docstring has promised since Oct 1 to check convergence via
calculate_training_loss=True instead of assuming the library default of 15
iterations is enough. Every ALS number so far used 15 or 20 iterations.

Two checks, one fit:
  1. Training-loss curve over MANY iterations (default 40) of the chosen config
     (factors=150 / reg=0.1 / alpha=15). Shows whether the loss was still
     falling meaningfully at iteration 20.
  2. Validation MAP@12 of the 40-iteration model (exclusion on, 7-day fallback,
     aggregate + per segment) against the recorded 20-iteration result from
     exclusion_comparison.py. Training loss can keep falling while validation
     quality stops improving (or gets worse), so loss alone is not the answer.

Because random_state=42 and ALS here is deterministic, the first 20 iterations
of this 40-iteration run are the same computation as the earlier 20-iteration
run -- so any validation difference is due to the extra iterations only.

Usage:
    python convergence_check.py                 # loss curve + validation (~10 min)
    python convergence_check.py --skip-eval     # loss curve only (~5 min)
"""

import argparse
import logging

import numpy as np
import pandas as pd
import threadpoolctl
from implicit.als import AlternatingLeastSquares

import baseline as base
import cold_start_policy as csp
import exclusion_comparison as ex
import item_item_matched as iim
import model as als_mod

log = logging.getLogger(__name__)

# From exclusion_comparison.py, ALS factors=150 / reg=0.1 / alpha=15 / 20 iters,
# exclusion on, 7-day fallback. MAP@12.
REFERENCE_20_ITER_MAP = {"aggregate": 0.004995, "0 purchases": 0.006393,
                         "1-2 purchases": 0.010363, "3+ purchases": 0.004706}


def fit_with_loss(matrix, factors, regularization, alpha, iterations, random_state=42):
    """Fit ALS recording the training loss after every iteration.
    Mirrors model.fit_als_model's settings (num_threads=0, BLAS limited to 1)."""
    records = []

    def callback(*args):
        # implicit calls callback(iteration, elapsed_seconds, loss)
        loss = args[2] if len(args) > 2 else None
        records.append({"iteration": args[0] + 1, "seconds": args[1], "loss": loss})

    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=alpha, iterations=iterations,
        num_threads=0, random_state=random_state, calculate_training_loss=True,
    )
    with threadpoolctl.threadpool_limits(1, "blas"):
        try:
            model.fit(matrix, callback=callback)
        except TypeError as e:
            raise RuntimeError(
                "This version of `implicit` doesn't accept a fit() callback, so per-iteration "
                "loss can't be recorded. Upgrade with: pip install -U implicit"
            ) from e
    return model, pd.DataFrame(records)


def summarize_loss(loss_df, reference_iter=20):
    df = loss_df.copy()
    if df["loss"].isna().all():
        log.warning("implicit returned no loss values; cannot assess training-loss convergence.")
        return df
    df["rel_change_pct"] = 100 * df["loss"].pct_change()
    first, last = df["loss"].iloc[0], df["loss"].iloc[-1]
    at_ref = df.loc[df["iteration"] == reference_iter, "loss"]
    if len(at_ref) and first != last:
        share = (first - at_ref.iloc[0]) / (first - last)
        remaining = 100 * (at_ref.iloc[0] - last) / at_ref.iloc[0]
        log.info("By iteration %d the loss had made %.1f%% of its total decline over %d iterations; "
                 "it fell a further %.2f%% between iteration %d and %d.",
                 reference_iter, 100 * share, int(df["iteration"].iloc[-1]), remaining,
                 reference_iter, int(df["iteration"].iloc[-1]))
        if remaining < 1.0:
            log.info("Reading: training loss was essentially flat by iteration %d (<1%% further decline).", reference_iter)
        else:
            log.info("Reading: training loss was still falling at iteration %d (>=1%% further decline) -- "
                     "check whether validation MAP@12 below benefits.", reference_iter)
    return df


def run_convergence(train, val, factors, regularization, alpha, iterations, skip_eval=False):
    cw_matrix, cw_c2r, cw_i2c = als_mod.build_confidence_weighted_matrix(train)
    log.info("Fitting ALS for %d iterations with loss tracking (factors=%d, reg=%s, alpha=%s)...",
             iterations, factors, regularization, alpha)
    model, loss_df = fit_with_loss(cw_matrix, factors, regularization, alpha, iterations)
    loss_df = summarize_loss(loss_df)

    eval_df = None
    if not skip_eval:
        customers = val["customer_id"].unique()
        segments = base.segment_customers(train, customers)
        fallback_top12 = csp.compute_recent_popularity(train, ex.FALLBACK_DAYS).head(12).index.tolist()
        log.info("Scoring the %d-iteration model on val (exclusion on, 7-day fallback)...", iterations)
        recs = ex.als_recommendations(customers, model, cw_matrix, cw_c2r, cw_i2c, fallback_top12, exclude=True)
        eval_df = pd.DataFrame(iim.score_variant(f"ALS {iterations} iters", recs, val, segments))
    return loss_df, eval_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ALS convergence: training-loss curve + validation check.")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--factors", type=int, default=150)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--skip-eval", action="store_true", help="loss curve only, no validation scoring")
    args = parser.parse_args()

    train = base.load_train()
    val = base.load_val()
    loss_df, eval_df = run_convergence(train, val, args.factors, args.regularization, args.alpha,
                                       args.iterations, skip_eval=args.skip_eval)

    pd.set_option("display.width", 200)
    print("\n=== training loss by iteration ===")
    print(loss_df.round(4).to_string(index=False))
    loss_df.to_csv("convergence_loss_curve.csv", index=False)

    if eval_df is not None:
        print(f"\n=== validation, {args.iterations}-iteration model vs recorded 20-iteration model "
              f"(MAP@12, exclusion on, 7-day fallback) ===")
        for _, row in eval_df.iterrows():
            ref = REFERENCE_20_ITER_MAP[row["segment"]]
            print(f"{row['segment']:>14}:  20 iters {ref:.6f}   {args.iterations} iters {row['map@12']:.6f}   "
                  f"change {100 * (row['map@12'] / ref - 1):+.1f}%")
        eval_df.to_csv("convergence_validation.csv", index=False)
        if (args.factors, args.regularization, args.alpha) != (150, 0.1, 15.0):
            log.warning("Non-default config: the recorded 20-iteration reference numbers do NOT apply to it.")