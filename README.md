# coldstart-recsys
Recommendation engine (ALS + learning-to-rank) trained on real e-commerce transactions, served via Spring Boot + Redis, with explicit cold-start handling for new users.

## Dataset
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 1,371,980 registered customers, 31,788,324 transactions in `transactions_train.csv`.

## EDA Findings

**1. User-level cold start is the dominant case, not an edge case.**
19.58% of all 1,371,980 registered customers — including 9,699 (0.71%) who registered but never made a single purchase — have fewer than 3 purchases in the training window. Repeat-purchase rate (customers with more than 1 purchase) is 89.71%. Cold-start handling has to be a first-class part of the design, not a fallback bolted on afterward.

**2. Item-level imbalance mirrors the user-level problem.**
The top 3 product categories out of 19 (Garment Upper body, Garment Lower body, Garment Full body) account for 72.83% of all transactions, while the bottom 9 categories combined represent under 1%. 5,486 articles (5.20% of the catalog) have zero or one purchase. A naive popularity-based baseline will be dominated by upper-body garments, and item-level cold start (new or rarely-purchased articles) needs the same deliberate handling as user-level cold start.

**3. The article catalog has clean, low-cardinality categorical structure — use it directly, don't engineer around it.**
Fields like `product_group_name` (19), `garment_group_name` (21), `index_group_name` (5), and `colour_group_name` (50) have zero missingness and moderate cardinality, so they feed LightGBM/ALS as categorical features as-is. In contrast, `detail_desc` (43,404 unique values), `prod_name` (45,875), and `product_code` (47,224) — out of 105,542 total articles — carry cardinality close to the row count, meaning they function as identifiers or free text rather than model-usable categories, and are excluded rather than encoded.

## Baseline Models
Two baselines, built before any real model - a floor to measure actual improvement against, not filler.

**1. Popularity-based recommender (floor baseline).** Same top-12 most-purchased articles (by train-window purchase count) recommended to every customer, with no personalization by design — this isolates "how much does personalization actually help" as a question the next baseline and the real model both have to answer. Scored against the held-out validation week: precision@12 = 0.00234, recall@12 = 0.00897, hit_rate@12 = 2.48% (share of customers with at least one relevant item anywhere in their top-12). Low numbers are expected for a fully non-personalized baseline against a 105,542-article catalog — the number that matters isn't this score in isolation, it's the gap the next model has to close. One structural note: the top-12 spans only 8 distinct products, since 3 slots are different color variants of the single most popular item — left this way deliberately, since Kaggle's own MAP@12 scoring is at exact article_id granularity, not product level.

**2. Item-item collaborative filtering.** Personalized via co-purchase patterns: a binary customer-article interaction matrix (train only, restricted to the top 5,000 most popular articles) feeds a cosine item-item similarity matrix, and each customer's top-12 is the highest-scoring unpurchased items given their own purchase history. Customers with no row in the interaction matrix fall back to the popularity baseline -- see the fallback breakdown below.

**3. Recent-popularity baseline.** Same mechanism as the popularity baseline, but purchase counts are restricted to the last N days of train instead of the full window. Tested a 7-day and a 14-day window and picked the winner on validation: 7-day beat 14-day on every metric (MAP@12 0.00677 vs. 0.00657), so 7-day is what's reported below.

**Results on the held-out validation week (72,019 customers):**

| Model | precision@12 | recall@12 | hit_rate@12 | MAP@12 |
|---|---|---|---|---|
| Popularity (floor) | 0.00234 | 0.00897 | 2.48% | 0.00344 |
| Item-item CF | 0.00314 | 0.01285 | 3.44% | 0.00478 |
| Recent-popularity (7-day) | 0.00507 | 0.02230 | 5.78% | 0.00677 |

**The finding that matters here:** item-item CF beats the plain popularity floor on every metric (34-43% relative), but loses to the non-personalized recent-popularity baseline by a wider margin (42-74% relative). The interaction matrix and similarity computation carry no time weighting -- a co-purchase from a year ago counts exactly as much as one from last week. In a fast-fashion catalog where trends visibly turn over in under two weeks, "what's selling right now" predicts next week's purchase better than "what this customer has historically bought," at least until the model itself becomes recency-aware. Time-decayed similarity is the natural next iteration, not attempted yet.

**Fallback breakdown.** 10.51% of validation customers (7,567 of 72,019) have no row in item-item's interaction matrix and fall back to the popularity list: 7.49% (5,395) have zero train purchases at all -- true cold start, unsolvable by any co-purchase model -- and 3.02% (2,172) bought things in train, just nothing that made the top-5,000 candidate cut.