"""
model.py — Week 5-6 core model (ALS) for coldstart-recsys.

Matrix factorization via implicit's AlternatingLeastSquares, trained on the
full article catalog -- unlike baseline.py's item-item CF, which restricted
to the top 5,000 candidate articles because a pairwise similarity computation
doesn't scale to 105,542 items. ALS is a sparse factorization, not a pairwise
computation, so it's built against the full catalog directly.

LEAK WARNING (same as baseline.py / prepare_data.py): the interaction matrix
is computed from the train split only.

Usage:
    python model.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import threadpoolctl
from implicit.als import AlternatingLeastSquares

import baseline as base
import cold_start_policy as csp

# DESIGN DECISION (Oct 4) -- segment "0 purchases" (no train history, so no
# learned ALS embedding) gets popularity computed over only the last
# COLD_START_WINDOW_DAYS days of train, not all-time popularity. Measured by
# cold_start_policy.py on the 5,395 segment-0 val customers: MAP@12 0.003923
# (all-time) -> 0.006393 (7 days), hit_rate 2.78% -> 5.26%. The 7-day and
# 14-day windows are indistinguishable on MAP (0.006393 vs 0.006399) and 7
# wins the other three metrics, so 7 is chosen -- picked from segment 0's own
# numbers only, not from how the window scored on segments 1-2 / 3+.
COLD_START_WINDOW_DAYS = 7


def fit_als_model(matrix, factors=100, regularization=0.01, alpha=1.0, iterations=15, random_state=42):
    """
    DESIGN DECISION -- ALS hyperparameter starting point (not yet tuned, revisited
    Sat Oct 3): starting from implicit's own defaults rather than guessing custom
    values, and stating why each one is defensible as a starting point, not why
    it's already correct.

      factors=100 (library default): a reasonable starting point for a catalog
      this size (105,542 articles, 1,371,980 customers) -- more expressive than
      what the toy-scale spike tested, but not inflated just because the catalog
      is large. Saturday's tuning pass validates this against real numbers, not
      intuition.

      regularization=0.01 (library default): no dataset-specific reason yet to
      deviate from it.

      alpha=1.0 (library default): historically every train purchase was fed in
      as unweighted binary presence (build_full_interaction_matrix's binary
      collapse), so alpha had nothing to scale. Saturday's test
      (build_confidence_weighted_matrix below) feeds this the same function
      with raw purchase counts instead, so alpha now does real work --
      implicit scales confidence internally as roughly `1 + alpha * count`.
      Swept explicitly below rather than left at the default once that
      matrix is in play.

      iterations=15 (library default): starting point; will check convergence
      via calculate_training_loss=True during Saturday's tuning pass instead of
      assuming 15 is enough.

      num_threads=0 (use all cores), fit wrapped in
      threadpoolctl.threadpool_limits(1, "blas") -- OpenMP/BLAS thread
      contention is a known issue with this library, confirmed locally via a
      spike before touching the real interaction matrix.

      random_state=42 -- added for Saturday's sweep: ALS's factor
      initialization is randomized, so without a fixed seed, two runs of the
      *same* config produce different embeddings and different val scores,
      which would make the sweep table incomparable. Fixed so that score
      differences across sweep rows reflect the hyperparameters changing,
      not initialization noise.
    """
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        num_threads=0,
        random_state=random_state,
    )
    with threadpoolctl.threadpool_limits(1, "blas"):
        model.fit(matrix)
    return model


def build_full_interaction_matrix(train):
    """Binary customer-article interaction matrix, train only, full catalog --
    no candidate-count restriction like baseline.py's item-item CF needed,
    since ALS is a proper sparse factorization, not a pairwise similarity
    computation. Same train-only leak boundary as baseline.py.

    DESIGN DECISION -- rows/cols explicitly cast to int32 numpy arrays, not
    left as pandas.Series.map() output: implicit's Cython solver requires
    exact dtype matches (float32 data, int32 indices) for its fused-type
    dispatch. pandas .map() on a categorical column can return object-dtype
    Python ints, which scipy accepts silently but implicit's compiled solver
    rejects with "ambiguous argument types." baseline.py never hit this
    because it never feeds the matrix into implicit -- only plain scipy ops.
    """
    customers = train["customer_id"].unique()
    articles = train["article_id"].unique()
    customer_to_row = {c: i for i, c in enumerate(customers)}
    item_to_col = {a: i for i, a in enumerate(articles)}

    rows = train["customer_id"].map(customer_to_row).to_numpy(dtype=np.int32)
    cols = train["article_id"].map(item_to_col).to_numpy(dtype=np.int32)
    data = np.ones(len(train), dtype=np.float32)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(customers), len(articles)))
    matrix.data[:] = 1  # collapse duplicate (customer, article) entries to binary presence
    return matrix, customer_to_row, item_to_col


def build_confidence_weighted_matrix(train):
    """Customer-article interaction matrix, train only, full catalog -- SAME
    shape and dtype contract as build_full_interaction_matrix, but data holds
    raw purchase counts instead of being collapsed to binary presence.

    DESIGN DECISION -- Saturday Oct 3 hypothesis test (als_first_run_summary.md
    #2): build_full_interaction_matrix's binary collapse throws away repeat-
    purchase count, but the exclusion ablation already showed repurchases are
    a real positive signal in this catalog (fast-fashion restocking/colorway
    behavior), not noise. implicit's AlternatingLeastSquares is built around
    purchase-count-as-confidence (Hu/Koren/Volinsky): it expects the raw
    interaction matrix in `data` and scales it internally via its own `alpha`
    parameter (confidence ~= 1 + alpha * data), so the fix here is to stop
    overwriting counts with 1 -- NOT to pre-multiply by alpha ourselves, which
    would double-apply it on top of what fit_als_model's alpha already does.

    scipy's csr_matrix constructor sums duplicate (row, col) triplets
    automatically when building from COO-style (data, (row, col)) input, so
    simply NOT calling `matrix.data[:] = 1` (the one line that differs from
    build_full_interaction_matrix) is enough to turn duplicate purchases of
    the same (customer, article) pair into a summed count instead of a flag.
    """
    customers = train["customer_id"].unique()
    articles = train["article_id"].unique()
    customer_to_row = {c: i for i, c in enumerate(customers)}
    item_to_col = {a: i for i, a in enumerate(articles)}

    rows = train["customer_id"].map(customer_to_row).to_numpy(dtype=np.int32)
    cols = train["article_id"].map(item_to_col).to_numpy(dtype=np.int32)
    data = np.ones(len(train), dtype=np.float32)

    # No `matrix.data[:] = 1` here -- duplicate (customer, article) entries
    # are summed by the csr_matrix constructor into a purchase count.
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(customers), len(articles)))
    return matrix, customer_to_row, item_to_col


def get_als_recommendations(val_customers, model, matrix, customer_to_row, item_to_col, popularity_top12, k=12):
    """Per-customer ALS top-k recommendations, falling back to the popularity
    baseline for any customer with no row in the training interaction matrix
    -- including true cold-start customers with zero train purchases. Same
    fallback shape as baseline.py's get_item_item_recommendations: a trained
    ALS model has no learned embedding for a customer it never saw in train,
    so this fallback *is* this model's cold-start policy for segment "0".
    `popularity_top12` is whatever list the caller passes in: as of Oct 4,
    __main__ passes the last-COLD_START_WINDOW_DAYS-days popularity list (see
    the constant above), not all-time popularity.

    `matrix` must be the same matrix `model` was fit on (its rows are passed
    to implicit as `user_items`, which it needs both to filter out
    already-purchased items and to score users correctly in recent
    `implicit` versions).

    DESIGN DECISION -- filter_already_liked_items=True: matches the
    *reported* baseline convention (README's with-exclusion numbers for
    item-item CF -- precision 0.00314/recall 0.01285/hit_rate 3.44%), so
    ALS's sweep numbers are directly comparable to the headline comparison
    this project's README is built around, not the "without exclusion"
    ablation rows. `implicit` applies this internally via `user_items`
    (passed above as `matrix[row_idx]`), rather than this function
    re-implementing baseline.py's manual `scores[...] = -np.inf` approach --
    same effect, library-native mechanism.

    CORRECTION (Oct 4) -- an earlier version of this function had this at
    False, which produced ALS numbers that looked better in isolation but
    weren't comparable to the README's reported item-item CF row at all.
    Flipping this single value is what should happen between any two sweep
    runs being compared -- it should never go back to False silently on a
    future edit of this file.
    """
    col_to_item = {v: item for item, v in item_to_col.items()}
    recommendations = {}

    for customer_id in val_customers:
        row_idx = customer_to_row.get(customer_id)
        if row_idx is None:
            recommendations[customer_id] = popularity_top12
            continue

        item_ids, _scores = model.recommend(
            userid=row_idx,
            user_items=matrix[row_idx],
            N=k,
            filter_already_liked_items=True,
        )
        recommendations[customer_id] = [col_to_item[c] for c in item_ids]

    return recommendations


if __name__ == "__main__":
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading train/val splits...")
    train = base.load_train()
    val = base.load_val()
    val_customers = val["customer_id"].unique()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-only", action="store_true",
                        help="fit only the best config from Oct 3's sweep (factors=50, alpha=15) "
                             "instead of all five -- ~4 min instead of ~25")
    args = parser.parse_args()

    # Cold-start fallback list: last COLD_START_WINDOW_DAYS days of train only.
    # (Replaces the all-time popularity list this variable used to hold.)
    cold_start_top12 = csp.compute_recent_popularity(train, COLD_START_WINDOW_DAYS).head(12).index.tolist()
    log.info("Cold-start fallback (last %dd popularity) top-12: %s", COLD_START_WINDOW_DAYS, cold_start_top12)
    top12 = cold_start_top12

    log.info("Building binary interaction matrix (untuned baseline)...")
    bin_matrix, customer_to_row, item_to_col = build_full_interaction_matrix(train)

    log.info("Building confidence-weighted interaction matrix (Saturday hypothesis)...")
    cw_matrix, cw_customer_to_row, cw_item_to_col = build_confidence_weighted_matrix(train)
    assert cw_matrix.shape == bin_matrix.shape
    assert cw_matrix.nnz == bin_matrix.nnz  # same sparsity pattern, different values
    assert cw_matrix.data.max() > 1, "expected some repeat purchases to produce counts > 1"

    # Saturday Oct 3: small manual sweep. factors/regularization/iterations
    # vary the model itself; alpha varies how hard the confidence-weighted
    # matrix's purchase counts get scaled (see build_confidence_weighted_matrix
    # docstring for why alpha isn't pre-applied to the matrix by hand).
    SWEEP_CONFIGS = [
        {"factors": 100, "regularization": 0.01, "alpha": 1.0, "iterations": 15},
        {"factors": 100, "regularization": 0.01, "alpha": 15.0, "iterations": 15},
        {"factors": 100, "regularization": 0.01, "alpha": 40.0, "iterations": 15},
        {"factors": 50, "regularization": 0.01, "alpha": 15.0, "iterations": 15},
        {"factors": 150, "regularization": 0.1, "alpha": 15.0, "iterations": 20},
    ]
    if args.best_only:
        SWEEP_CONFIGS = [{"factors": 50, "regularization": 0.01, "alpha": 15.0, "iterations": 15}]

    results = []
    for cfg in SWEEP_CONFIGS:
        log.info("Fitting ALS on confidence-weighted matrix, config: %s", cfg)
        model = fit_als_model(
            cw_matrix,
            factors=cfg["factors"],
            regularization=cfg["regularization"],
            alpha=cfg["alpha"],
            iterations=cfg["iterations"],
            random_state=42,
        )

        recs_dict = get_als_recommendations(val_customers, model, cw_matrix, cw_customer_to_row, cw_item_to_col, top12)
        recs = pd.DataFrame({
            "customer_id": list(recs_dict.keys()),
            "recommendations": list(recs_dict.values()),
        })
        precision, recall = base.precision_recall_at_k(recs, val)
        hit_rate = base.hit_rate_at_k(recs, val)
        map_score = base.map_at_k(recs, val)

        row = {**cfg, "precision@12": precision, "recall@12": recall,
               "hit_rate@12": hit_rate, "map@12": map_score, "_recs": recs}
        results.append(row)
        log.info("Result: %s", {k: v for k, v in row.items() if k != "_recs"})

    log.info("Sweep complete. %d configs evaluated.", len(results))
    for row in results:
        log.info({k: v for k, v in row.items() if k != "_recs"})

    # Sun Oct 4 task, done against the best sweep config rather than deferred:
    # does confidence-weighted ALS actually move the needle on segment "0"
    # (true cold start), or is popularity's fallback still doing all the work
    # there -- same question the Week 5-6 plan asks, answered honestly either
    # way. "Best" = highest precision@12 among the configs just swept; this
    # is picking among configs already run above, not a second model search.
    best = max(results, key=lambda r: r["precision@12"])
    best_cfg = {k: v for k, v in best.items() if k not in ("precision@12", "recall@12", "hit_rate@12", "map@12", "_recs")}
    log.info("Best config by precision@12: %s", best_cfg)

    segments = base.segment_customers(train, val_customers)
    log.info("Segment sizes: %s", segments.value_counts().to_dict())

    for segment_label in ["0 purchases", "1-2 purchases", "3+ purchases"]:
        seg_result = base.evaluate_segment(best["_recs"], val, segments, segment_label)
        log.info("ALS (best config) -- segment '%s': %s", segment_label, seg_result)

    # Expected-value check for the Oct 4 cold-start policy: segment 0 gets no
    # ALS signal at all (no embedding), so its row here must now equal
    # cold_start_policy.py's "last 7d" segment-0 row (MAP@12 0.006393,
    # precision@12 0.004557, hit_rate@12 0.052641). If it does, the policy is
    # wired in correctly; if it still shows 0.003923, the all-time list is
    # still being passed through. Segments 1-2 and 3+ must be UNCHANGED from
    # the Oct 4 run (their customers have embeddings and never hit the fallback).
    # Honest framing for the write-up: this policy is a recency-popularity
    # fallback, not a learned cold-start model -- say so plainly.
    seg0 = base.evaluate_segment(best["_recs"], val, segments, "0 purchases")
    if abs(seg0["map@12"] - 0.006393) < 1e-5:
        log.info("CHECK PASSED: segment-0 MAP@12 matches cold_start_policy.py's 7-day row (%.6f)", seg0["map@12"])
    else:
        log.warning("CHECK FAILED: segment-0 MAP@12 is %.6f, expected ~0.006393 -- fallback list not wired through",
                    seg0["map@12"])