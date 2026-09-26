"""
baseline.py — Week 4 baseline models for coldstart-recsys.

Two baselines, per portfolio-project-plan.pdf:
  1. Popularity-based recommender — the floor. Same top-12 list for every
     customer, deliberately NOT personalized. This is the comparison point
     a real model has to beat, so it stays intentionally naive.
  2. Item-item collaborative filtering — personalized via co-purchase
     patterns. That's where per-user history belongs, not in the
     popularity baseline above.

DESIGN DECISION — no already-purchased filtering in the popularity baseline:
    Filtering a user's own past purchases out of their recommendation list
    is a personalization rule. Doing it here would blur the popularity
    baseline into a hybrid, which weakens the two-point comparison the
    write-up depends on ("non-personalized floor" vs. "personalized via
    co-purchase patterns"). That comparison logic belongs to item-item CF.
    Confirmed empirically, not just argued narratively: the exclusion
    ablation (get_popularity_recommendations_excl, see __main__) shows
    adding this filter would have cost popularity 26-37% relative score
    across precision/recall/hit_rate/MAP@12 anyway -- likely because
    repurchase (restocking a basic, buying another colorway) is common
    enough in this catalog that "already bought" is a positive signal,
    not noise.

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


def average_precision_at_k(recommended: list, relevant: set, k: int = 12) -> float:
    """Average precision@k for one customer's ranked recommendation list.
    Kaggle-style: divides by min(len(relevant), k), not len(relevant).
    """
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant), k)


def map_at_k(recommendations: pd.DataFrame, val: pd.DataFrame, k: int = 12) -> float:
    """Mean average precision@k over customers with at least one val purchase."""
    actual = val.groupby("customer_id")["article_id"].apply(set)
    scores = []
    for _, row in recommendations.iterrows():
        true_items = actual.get(row["customer_id"], set())
        if not true_items:
            continue
        scores.append(average_precision_at_k(row["recommendations"], true_items, k))
    return sum(scores) / len(scores)


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


def get_item_item_recommendations(val_customers, matrix, similarity, customer_to_row, item_to_col,
                                   popularity_top12, k=12, exclude_purchased=True):
    """Per-customer item-item CF recommendations, falling back to the
    popularity baseline for any customer with no row in the interaction
    matrix -- including true cold-start customers with zero train purchases.
    This fallback is the cold-start handling this project is built around,
    not a patch for a missing case.

    exclude_purchased -- when True (the reported baseline), already-bought
    items are excluded from a customer's own recommendations. Set False only
    for the exclusion ablation; see get_popularity_recommendations_excl for
    the popularity-side counterpart.

    DESIGN DECISION -- kept True as the reported default despite scoring
    worse: the exclusion ablation showed MAP@12 drops from 0.00839 to
    0.00478 (-43% relative) with exclusion on, most likely because this is
    fast-fashion and a real share of next week's purchases are repeats of
    what a customer already owns. Recommending mostly repurchases is a
    weaker demonstration of co-purchase personalization for this project's
    case study than recommending genuinely new items, even at a lower
    score -- see README's exclusion ablation section for the full number.
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
        if exclude_purchased:
            scores[customer_row.indices] = -np.inf  # exclude already-purchased items

        top_k_cols = np.argpartition(scores, -k)[-k:]
        top_k_cols = top_k_cols[np.argsort(-scores[top_k_cols])]
        recommendations[customer_id] = [col_to_item[c] for c in top_k_cols]

    return recommendations


def compute_recent_popularity(train: pd.DataFrame, window_days: int = 14, date_col: str = "t_dat") -> pd.Series:
    """Article purchase counts restricted to the last `window_days` days of
    train, ending at train's own max date. This is the recency-aware
    alternative to compute_popularity's full-window count -- same
    LEAK WARNING applies: computed from train only.

    DESIGN DECISION -- still not personalized: this stays a single global
    top-k list for every customer, same as compute_popularity. The only
    difference is the purchase-count window. Personalization belongs to
    item-item CF; mixing it in here would blur the three-way comparison.
    """
    cutoff = train[date_col].max() - pd.Timedelta(days=window_days)
    recent = train[train[date_col] >= cutoff]
    return recent["article_id"].value_counts()


def get_popularity_recommendations_excl(popularity, matrix, customer_to_row, item_to_col,
                                         val_customers, k=12, pool_size=200):
    """Ablation variant of get_popularity_recommendations: per-customer,
    excludes items already bought in train. NOT the reported baseline --
    see module docstring for why exclusion stays out of the floor baseline
    by default. This function exists only to measure the effect of adding
    exclusion, for the write-up.

    DESIGN DECISION -- pool_size=200, not the full catalog: a customer
    would need to have bought more than 200 of the globally most popular
    articles for this to run short of k=12 after filtering, which doesn't
    happen in practice at this dataset's purchase-count scale. Capping
    the pool avoids ranking and filtering the full ~105k-article
    popularity list per customer for no accuracy benefit.

    Result this produced (see README exclusion ablation): removing
    exclusion lifted popularity's MAP@12 from 0.00252 to 0.00344, a 37%
    relative gain -- this function is what made that comparison possible.
    """
    ranked_pool = popularity.head(pool_size).index.tolist()
    ranked_pool_cols = [item_to_col[item] for item in ranked_pool]

    recommendations = {}
    for customer_id in val_customers:
        row_idx = customer_to_row.get(customer_id)
        if row_idx is None:
            recommendations[customer_id] = ranked_pool[:k]
            continue
        bought_cols = set(matrix[row_idx].indices)
        filtered = [item for item, col in zip(ranked_pool, ranked_pool_cols) if col not in bought_cols]
        recommendations[customer_id] = filtered[:k]

    return recommendations


def build_customer_purchase_counts(train: pd.DataFrame) -> pd.Series:
    """Train-only row count per customer -- the basis for cold-start
    segmentation below. Same leak boundary as popularity: train only, so
    a customer's segment can't be influenced by what they buy in val.
    Deliberately row count, not distinct-article count: a customer who
    bought the same basic 5 times has a different purchase pattern than
    one who bought 5 different things, and this segmentation is about
    purchase *frequency* (how much history the model has to work with),
    not catalog breadth.
    """
    return train.groupby("customer_id").size()


def segment_customers(customer_ids, purchase_counts: pd.Series) -> dict:
    """Buckets customer_ids into 0 / 1-2 / 3+ train purchases.

    DESIGN DECISION -- these three bands, not quartiles or a continuous
    score: 0 purchases is true cold start (necessarily falls back to
    popularity, see get_item_item_recommendations); 1-2 is "just enough
    history to have a preference signal, not enough to be reliable"; 3+
    is everyone else. The bands aren't arbitrary post-hoc buckets -- they
    were picked because they're the exact groups the plan's cold-start
    policy has to treat differently, so the eval boundary matches the
    policy boundary.
    """
    segments = {"0": [], "1-2": [], "3+": []}
    for cid in customer_ids:
        count = purchase_counts.get(cid, 0)
        if count == 0:
            segments["0"].append(cid)
        elif count <= 2:
            segments["1-2"].append(cid)
        else:
            segments["3+"].append(cid)
    return segments


def evaluate_segment(recommendations_df: pd.DataFrame, val: pd.DataFrame, segment_ids: list, k=12):
    """Precision/recall/hit_rate/MAP@12 restricted to one customer segment.
    Returns None if the segment has no customers in this recommendations frame.
    """
    subset = recommendations_df[recommendations_df["customer_id"].isin(segment_ids)]
    if len(subset) == 0:
        return None
    precision, recall = precision_recall_at_k(subset, val, k)
    hit_rate = hit_rate_at_k(subset, val, k)
    map12 = map_at_k(subset, val, k)
    return precision, recall, hit_rate, map12


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

    map12 = map_at_k(recs, val)
    log.info("Popularity baseline -- MAP@12: %.5f", map12)

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

    log.info("Computing precision@12 / recall@12 / hit_rate@12 for item-item CF...")
    ii_precision, ii_recall = precision_recall_at_k(item_item_recs, val)
    ii_hit_rate = hit_rate_at_k(item_item_recs, val)
    log.info("Item-item CF -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f (%.2f%%)",
              ii_precision, ii_recall, ii_hit_rate, ii_hit_rate * 100)

    ii_map12 = map_at_k(item_item_recs, val)
    log.info("Item-item CF -- MAP@12: %.5f", ii_map12)

    # --- Exclusion ablation ---
    log.info("Running exclusion ablation: popularity and item-item CF, with vs without excluding already-bought items...")

    pop_excl_dict = get_popularity_recommendations_excl(popularity, matrix, customer_to_row, item_to_col, val_customers)
    pop_excl_recs = pd.DataFrame({
        "customer_id": list(pop_excl_dict.keys()),
        "recommendations": list(pop_excl_dict.values()),
    })
    pe_precision, pe_recall = precision_recall_at_k(pop_excl_recs, val)
    pe_hit_rate = hit_rate_at_k(pop_excl_recs, val)
    pe_map12 = map_at_k(pop_excl_recs, val)
    log.info("Popularity WITH exclusion -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f, MAP@12: %.5f",
              pe_precision, pe_recall, pe_hit_rate, pe_map12)
    log.info("Popularity WITHOUT exclusion (reported baseline) -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f, MAP@12: %.5f",
              precision, recall, hit_rate, map12)

    ii_noexcl_dict = get_item_item_recommendations(val_customers, matrix, similarity, customer_to_row, item_to_col, top12, exclude_purchased=False)
    ii_noexcl_recs = pd.DataFrame({
        "customer_id": list(ii_noexcl_dict.keys()),
        "recommendations": list(ii_noexcl_dict.values()),
    })
    iin_precision, iin_recall = precision_recall_at_k(ii_noexcl_recs, val)
    iin_hit_rate = hit_rate_at_k(ii_noexcl_recs, val)
    iin_map12 = map_at_k(ii_noexcl_recs, val)
    log.info("Item-item CF WITHOUT exclusion -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f, MAP@12: %.5f",
              iin_precision, iin_recall, iin_hit_rate, iin_map12)
    log.info("Item-item CF WITH exclusion (reported baseline) -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f, MAP@12: %.5f",
              ii_precision, ii_recall, ii_hit_rate, ii_map12)

    # --- Cold-start segment breakdown ---
    log.info("Segmenting val customers by train purchase count (0 / 1-2 / 3+)...")
    purchase_counts = build_customer_purchase_counts(train)
    segments = segment_customers(val_customers, purchase_counts)
    for seg_name, seg_ids in segments.items():
        log.info("Segment %s: %d customers", seg_name, len(seg_ids))

    for seg_name, seg_ids in segments.items():
        pop_result = evaluate_segment(recs, val, seg_ids)
        ii_result = evaluate_segment(item_item_recs, val, seg_ids)
        if pop_result:
            log.info("Segment %s -- Popularity: precision %.5f, recall %.5f, hit_rate %.5f, MAP@12 %.5f", seg_name, *pop_result)
        if ii_result:
            log.info("Segment %s -- Item-item CF: precision %.5f, recall %.5f, hit_rate %.5f, MAP@12 %.5f", seg_name, *ii_result)

    log.info("Computing recent popularity (last 14 days of train)...")
    recent_popularity = compute_recent_popularity(train, window_days=14)
    recent_top12 = get_popularity_recommendations(recent_popularity)

    log.info("Recent-popularity top-12 article_ids: %s", recent_top12)
    overlap = [a in item_to_col for a in recent_top12]
    log.info("%d / %d recent top-12 items are in the item-item candidate pool", sum(overlap), len(recent_top12))

    recent_recs = build_recommendations(val_customers, recent_top12)
    rp_precision, rp_recall = precision_recall_at_k(recent_recs, val)
    rp_hit_rate = hit_rate_at_k(recent_recs, val)
    rp_map12 = map_at_k(recent_recs, val)
    log.info("Recent-popularity baseline -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f (%.2f%%), MAP@12: %.5f",
              rp_precision, rp_recall, rp_hit_rate, rp_hit_rate * 100, rp_map12)

    log.info("Computing recent popularity (last 7 days of train)...")
    recent_popularity_7d = compute_recent_popularity(train, window_days=7)
    recent_top12_7d = get_popularity_recommendations(recent_popularity_7d)
    log.info("Recent-popularity (7d) top-12 article_ids: %s", recent_top12_7d)

    recent_recs_7d = build_recommendations(val_customers, recent_top12_7d)
    rp7_precision, rp7_recall = precision_recall_at_k(recent_recs_7d, val)
    rp7_hit_rate = hit_rate_at_k(recent_recs_7d, val)
    rp7_map12 = map_at_k(recent_recs_7d, val)
    log.info("Recent-popularity (7d) -- precision@12: %.5f, recall@12: %.5f, hit_rate@12: %.5f (%.2f%%), MAP@12: %.5f",
              rp7_precision, rp7_recall, rp7_hit_rate, rp7_hit_rate * 100, rp7_map12)

    chosen_window = 7 if rp7_map12 > rp_map12 else 14
    log.info("Chosen recent-popularity window: %d days (MAP@12 -- 14d: %.5f, 7d: %.5f)",
              chosen_window, rp_map12, rp7_map12)

    fallback_count = sum(1 for c in val_customers if c not in customer_to_row)
    train_customers = set(train["customer_id"].unique())
    zero_history = sum(1 for c in val_customers if c not in train_customers)
    outside_candidates = fallback_count - zero_history
    log.info("Fallback breakdown -- zero train history: %d (%.2f%%), history outside top-%d candidates: %d (%.2f%%)",
              zero_history, 100 * zero_history / len(val_customers),
              N_CANDIDATE_ITEMS, outside_candidates, 100 * outside_candidates / len(val_customers))
    log.info("%d / %d val customers (%.2f%%) fell back to popularity (no row in interaction matrix)",
              fallback_count, len(val_customers), 100 * fallback_count / len(val_customers))