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
import scipy.sparse as sp
import threadpoolctl
import pandas as pd
from implicit.als import AlternatingLeastSquares

import baseline as base

def fit_als_model(matrix):
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

      alpha=1.0 (library default): every train purchase in this dataset is
      unweighted binary presence, matching build_full_interaction_matrix's
      binary collapse -- no purchase-count or recency signal is fed in as
      confidence yet, so leaving alpha at its default rather than inventing an
      unjustified weighting scheme.

      iterations=15 (library default): starting point; will check convergence
      via calculate_training_loss=True during Saturday's tuning pass instead of
      assuming 15 is enough.

      num_threads=0 (use all cores), fit wrapped in
      threadpoolctl.threadpool_limits(1, "blas") -- OpenMP/BLAS thread
      contention is a known issue with this library, confirmed locally via a
      spike before touching the real interaction matrix.
    """
    model = AlternatingLeastSquares(
        factors=100, regularization=0.01, alpha=1.0, iterations=15, num_threads=0
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

def get_als_recommendations(val_customers, model, matrix, customer_to_row, item_to_col,
                             popularity_top12, k=12, exclude_purchased=True):
    """Per-customer ALS recommendations, falling back to the popularity
    baseline for any customer with no row in the interaction matrix --
    same cold-start fallback get_item_item_recommendations uses in
    baseline.py, since a trained ALS model has the identical structural
    gap: a customer with zero train interactions has no learned embedding
    to recommend from.

    exclude_purchased -- mirrors get_item_item_recommendations's own
    parameter and reported-default choice (True), via implicit's own
    filter_already_liked_items, for consistency with the rest of this
    repo's baselines.
    """
    col_to_item = {v: item for item, v in item_to_col.items()}
    recommendations = {}

    known_customers = [c for c in val_customers if c in customer_to_row]
    known_rows = np.array([customer_to_row[c] for c in known_customers])

    if len(known_rows) > 0:
        ids, _scores = model.recommend(
            known_rows,
            matrix[known_rows],
            N=k,
            filter_already_liked_items=exclude_purchased,
        )
        for customer_id, item_cols in zip(known_customers, ids):
            recommendations[customer_id] = [col_to_item[c] for c in item_cols]

    for customer_id in val_customers:
        if customer_id not in customer_to_row:
            recommendations[customer_id] = popularity_top12

    return recommendations

def build_confidence_weighted_matrix(train):
    """Customer-article interaction matrix, train only, full catalog --
    identical to build_full_interaction_matrix EXCEPT data is left as
    summed purchase counts instead of collapsed to binary presence.

    DESIGN DECISION -- this is the confidence-weighting test from the
    Oct 1 session notes: implicit's fit() internally computes
    Cui = alpha * <matrix data> (confirmed in implicit/cpu/als.py), so
    whatever values this matrix carries ARE the confidence signal ALS
    optimizes against. The binary collapse in build_full_interaction_matrix
    was throwing away exactly the repurchase signal the exclusion ablation
    already proved matters in this catalog. This is the controlled
    comparison -- same customers, same articles, same hyperparameters as
    the binary run; the only variable that changes is whether a repeat
    purchase counts for more than a single purchase.
    """
    customers = train["customer_id"].unique()
    articles = train["article_id"].unique()
    customer_to_row = {c: i for i, c in enumerate(customers)}
    item_to_col = {a: i for i, a in enumerate(articles)}

    rows = train["customer_id"].map(customer_to_row).to_numpy(dtype=np.int32)
    cols = train["article_id"].map(item_to_col).to_numpy(dtype=np.int32)
    data = np.ones(len(train), dtype=np.float32)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(customers), len(articles)))
    # No binary collapse here -- duplicate (customer, article) entries SUM
    # during construction, giving raw purchase counts as confidence.
    return matrix, customer_to_row, item_to_col

if __name__ == "__main__":
    train = base.load_train()
    matrix, customer_to_row, item_to_col = build_full_interaction_matrix(train)
    print("Interaction matrix shape:", matrix.shape, "nonzero entries:", matrix.nnz)
    print("dtype check:", matrix.dtype, matrix.indices.dtype, matrix.indptr.dtype)

    model = fit_als_model(matrix)
    print("user_factors shape:", model.user_factors.shape)
    print("item_factors shape:", model.item_factors.shape)
    print("any all-zero user vectors:", (np.abs(model.user_factors).sum(axis=1) == 0).sum())
    print("any all-zero item vectors:", (np.abs(model.item_factors).sum(axis=1) == 0).sum())

    val = base.load_val()
    val_customers = val["customer_id"].unique()
    popularity = base.compute_popularity(train)
    popularity_top12 = base.get_popularity_recommendations(popularity)

    als_recs_dict = get_als_recommendations(
        val_customers, model, matrix, customer_to_row, item_to_col, popularity_top12
    )
    als_recs = pd.DataFrame({
        "customer_id": list(als_recs_dict.keys()),
        "recommendations": list(als_recs_dict.values()),
    })

    precision, recall = base.precision_recall_at_k(als_recs, val)
    hit_rate = base.hit_rate_at_k(als_recs, val)
    map12 = base.map_at_k(als_recs, val)
    print(f"ALS -- precision@12: {precision:.5f}, recall@12: {recall:.5f}, "
          f"hit_rate@12: {hit_rate:.5f}, MAP@12: {map12:.5f}")

    print("Testing confidence weighting: raw purchase counts instead of binary presence...")
    cw_matrix, cw_customer_to_row, cw_item_to_col = build_confidence_weighted_matrix(train)
    cw_model = fit_als_model(cw_matrix)

    cw_recs_dict = get_als_recommendations(
        val_customers, cw_model, cw_matrix, cw_customer_to_row, cw_item_to_col, popularity_top12
    )
    cw_recs = pd.DataFrame({
        "customer_id": list(cw_recs_dict.keys()),
        "recommendations": list(cw_recs_dict.values()),
    })
    cw_precision, cw_recall = base.precision_recall_at_k(cw_recs, val)
    cw_hit_rate = base.hit_rate_at_k(cw_recs, val)
    cw_map12 = base.map_at_k(cw_recs, val)
    print(f"ALS (confidence-weighted) -- precision@12: {cw_precision:.5f}, "
          f"recall@12: {cw_recall:.5f}, hit_rate@12: {cw_hit_rate:.5f}, MAP@12: {cw_map12:.5f}")
    print(f"ALS (binary, for comparison) -- MAP@12: {map12:.5f}")