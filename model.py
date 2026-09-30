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