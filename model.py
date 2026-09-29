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

import baseline as base

# TODO (Thu): fit_als_model() -- attach the ALS hyperparameter DESIGN DECISION
# here once the function exists. Starting point: factors=100, regularization=0.01,
# alpha=1.0, iterations=15 (implicit's own defaults), num_threads=0, fit wrapped
# in threadpoolctl.threadpool_limits(1, "blas") per the implicit_spike.py finding.


def build_full_interaction_matrix(train):
    """Binary customer-article interaction matrix, train only, full catalog --
    no candidate-count restriction like baseline.py's item-item CF needed,
    since ALS is a proper sparse factorization, not a pairwise similarity
    computation. Same train-only leak boundary as baseline.py.
    """
    customers = train["customer_id"].unique()
    articles = train["article_id"].unique()
    customer_to_row = {c: i for i, c in enumerate(customers)}
    item_to_col = {a: i for i, a in enumerate(articles)}

    rows = train["customer_id"].map(customer_to_row)
    cols = train["article_id"].map(item_to_col)
    data = np.ones(len(train), dtype=np.float32)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(customers), len(articles)))
    matrix.data[:] = 1  # collapse duplicate (customer, article) entries to binary presence
    return matrix, customer_to_row, item_to_col


if __name__ == "__main__":
    train = base.load_train()
    matrix, customer_to_row, item_to_col = build_full_interaction_matrix(train)
    print("Interaction matrix shape:", matrix.shape, "nonzero entries:", matrix.nnz)