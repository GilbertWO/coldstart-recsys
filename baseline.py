"""
baseline.py — Week 4 baseline models for coldstart-recsys.

Two baselines, per portfolio-project-plan.pdf:
  1. Popularity-based recommender — the floor. Same top-12 list for every
     customer, deliberately NOT personalized. This is the comparison point
     a real model has to beat, so it stays intentionally naive.
  2. Item-item collaborative filtering — added later this week. Personalized
     via co-purchase patterns. That's where per-user history belongs, not
     in the popularity baseline above.

DESIGN DECISION — no already-purchased filtering in the popularity baseline:
    Filtering a user's own past purchases out of their recommendation list
    is a personalization rule. Doing it here would blur the popularity
    baseline into a hybrid, which weakens the two-point comparison the
    write-up depends on ("non-personalized floor" vs. "personalized via
    co-purchase patterns"). That comparison logic belongs to item-item CF.

LEAK WARNING (same as prepare_data.py): popularity is computed from the
train split only. Val and test exist to be evaluated against, not learned
from.

Usage:
    python baseline.py
"""

import logging
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pandas as pd

N_CANDIDATE_ITEMS = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_train(out_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Load the train split produced by prepare_data.py."""
    return pd.read_parquet(out_dir / "transactions_train_split.parquet")


def load_val(out_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Load the val split produced by prepare_data.py."""
    return pd.read_parquet(out_dir / "transactions_val.parquet")


def compute_popularity(train: pd.DataFrame) -> pd.Series:
    """Article purchase counts, train split only — see leak warning above."""
    return train["article_id"].value_counts()


def get_popularity_recommendations(popularity: pd.Series, k: int = 12) -> list:
    """Global top-k most-purchased articles from train. Same list for every
    customer, deliberately not personalized -- that's what makes this the
    floor baseline, not the item-item CF one. See week-03-04 plan doc.

    DESIGN DECISION -- recommendations stay at article_id granularity, not
    deduplicated to distinct products: Kaggle's MAP@12 scores exact
    article_id matches, so recommending multiple color variants of one
    popular product (e.g. three variants of product 706016 in this top-12)
    is a legitimate way to maximize hits, not a bug. Verified: the top-12
    spans only 8 distinct products (article_id // 1000), with the most
    popular repeated 3x. Left as-is deliberately.
    """
    return popularity.head(k).index.tolist()


def build_recommendations(customer_ids, top_k_items: list) -> pd.DataFrame:
    """Same top-k list repeated for every customer_id given."""
    return pd.DataFrame({
        "customer_id": customer_ids,
        "recommendations": [top_k_items] * len(customer_ids),
    })

def precision_recall_at_k(recommendations: pd.DataFrame, val: pd.DataFrame, k: int = 12):
    """Mean precision@k and recall@k over customers with at least one
    actual purchase in val. `recommendations` must have customer_id and
    a recommendations column (list of article_id); `val` is the ground
    truth held-out week from prepare_data.py.
    """
    actual = val.groupby("customer_id")["article_id"].apply(set)

    precisions, recalls = [], []
    for _, row in recommendations.iterrows():
        true_items = actual.get(row["customer_id"], set())
        if not true_items:
            continue
        hits = len(set(row["recommendations"][:k]) & true_items)
        precisions.append(hits / k)
        recalls.append(hits / len(true_items))

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)

def hit_rate_at_k(recommendations: pd.DataFrame, val: pd.DataFrame, k: int = 12) -> float:
    """Fraction of customers with at least one relevant item in their top-k."""
    actual = val.groupby("customer_id")["article_id"].apply(set)
    hits = sum(
        bool(set(row["recommendations"][:k]) & actual.get(row["customer_id"], set()))
        for _, row in recommendations.iterrows()
    )
    return hits / len(recommendations)

def build_interaction_matrix(train: pd.DataFrame, popularity: pd.Series, n_candidates: int = N_CANDIDATE_ITEMS):
    """Binary customer-article interaction matrix, train only, restricted to the
    top n_candidates most popular articles. Binary presence, not purchase count --
    see baseline.py module docstring for why.
    """
    candidate_items = popularity.head(n_candidates).index
    train_restricted = train[train["article_id"].isin(candidate_items)]

    customers = train_restricted["customer_id"].unique()
    customer_to_row = {c: i for i, c in enumerate(customers)}
    item_to_col = {a: i for i, a in enumerate(candidate_items)}

    rows = train_restricted["customer_id"].map(customer_to_row)
    cols = train_restricted["article_id"].map(item_to_col)
    data = np.ones(len(train_restricted), dtype=np.int8)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(customers), n_candidates))
    matrix.data[:] = 1  # collapse duplicate (customer, article) entries to binary presence

    return matrix, customer_to_row, item_to_col

def compute_item_similarity(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Cosine similarity between items, via L2-normalized sparse item vectors --
    never materializes a dense item x item matrix. Output is n_candidates x
    n_candidates (5000x5000 here).
    """
    item_vectors = matrix.T.tocsr()  # items x customers

    norms = np.sqrt(item_vectors.multiply(item_vectors).sum(axis=1))
    norms = np.asarray(norms).flatten()
    norms[norms == 0] = 1  # avoid divide-by-zero for items with no interactions
    inv_norms = sp.diags(1 / norms)
    item_vectors_normalized = inv_norms @ item_vectors

    similarity = item_vectors_normalized @ item_vectors_normalized.T
    return similarity.tocsr()

def get_item_item_recommendations(val_customers, matrix, similarity, customer_to_row, item_to_col, popularity_top12, k=12):
    """Per-customer item-item CF recommendations, falling back to the
    popularity baseline for any customer with no row in the interaction
    matrix -- including true cold-start customers with zero train purchases.
    This fallback is the cold-start handling this project is built around,
    not a patch for a missing case.
    """
    col_to_item = {v: item for item, v in item_to_col.items()}
    recommendations = {}

    for customer_id in val_customers:
        row_idx = customer_to_row.get(customer_id)
        if row_idx is None:
            recommendations[customer_id] = popularity_top12
            continue

        customer_row = matrix[row_idx]
        scores = np.asarray((customer_row @ similarity).todense()).flatten()
        scores[customer_row.indices] = -np.inf  # exclude already-purchased items

        top_k_cols = np.argpartition(scores, -k)[-k:]
        top_k_cols = top_k_cols[np.argsort(-scores[top_k_cols])]
        recommendations[customer_id] = [col_to_item[c] for c in top_k_cols]

    return recommendations
if __name__ == "__main__":
    log.info("Loading train split...")
    train = load_train()

    log.info("Computing popularity (train only)...")
    popularity = compute_popularity(train)

    top12 = get_popularity_recommendations(popularity)
    log.info("Popularity baseline top-12 article_ids: %s", top12)
    from collections import Counter
    product_codes = [aid // 1000 for aid in top12]
    log.info("Distinct products in top-12: %s", Counter(product_codes))

    log.info("Loading val split to get the customer universe for eval...")
    val = load_val()
    val_customers = val["customer_id"].unique()
    log.info("%d unique customers in val split", len(val_customers))

    recs = build_recommendations(val_customers, top12)
    log.info("Built recommendations frame: %d rows", len(recs))
    print(recs.head())

    log.info("Computing precision@12 / recall@12 against val...")
    precision, recall = precision_recall_at_k(recs, val)
    log.info("Popularity baseline -- precision@12: %.5f, recall@12: %.5f", precision, recall)

    hit_rate = hit_rate_at_k(recs, val)
    log.info("Popularity baseline -- hit_rate@12: %.5f (%.2f%% of customers got >=1 hit)",
              hit_rate, hit_rate * 100)

    log.info("Building customer-article interaction matrix (top %d items)...", N_CANDIDATE_ITEMS)
    matrix, customer_to_row, item_to_col = build_interaction_matrix(train, popularity)
    log.info("Interaction matrix shape: %s, nonzero entries: %d", matrix.shape, matrix.nnz)

    log.info("Computing item-item cosine similarity...")
    similarity = compute_item_similarity(matrix)
    log.info("Similarity matrix shape: %s, nonzero entries: %d", similarity.shape, similarity.nnz)

    log.info("Generating item-item CF recommendations for val customers...")
    item_item_recs_dict = get_item_item_recommendations(val_customers, matrix, similarity, customer_to_row, item_to_col, top12)
    item_item_recs = pd.DataFrame({
        "customer_id": list(item_item_recs_dict.keys()),
        "recommendations": list(item_item_recs_dict.values()),
    })
    log.info("Built item-item recommendations frame: %d rows", len(item_item_recs))

    fallback_count = sum(1 for c in val_customers if c not in customer_to_row)
    log.info("%d / %d val customers (%.2f%%) fell back to popularity (no row in interaction matrix)",
              fallback_count, len(val_customers), 100 * fallback_count / len(val_customers))
