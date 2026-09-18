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
Two baselines, built before anyh real model - a floor to measure actual improvement against, not filler.

**1. Popularity-based recommender (floor baseline).** Same top-12 most-purchased articles (by train-window purchase count) recommended to every customer, with no personalization by design — this isolates "how much does personalization actually help" as a question the next baseline and the real model both have to answer. Scored against the held-out validation week: precision@12 = 0.00234, recall@12 = 0.00897, hit_rate@12 = 2.48% (share of customers with at least one relevant item anywhere in their top-12). Low numbers are expected for a fully non-personalized baseline against a 105,542-article catalog — the number that matters isn't this score in isolation, it's the gap the next model has to close. One structural note: the top-12 spans only 8 distinct products, since 3 slots are different color variants of the single most popular item — left this way deliberately, since Kaggle's own MAP@12 scoring is at exact article_id granularity, not product level.

**2. Item-item collaborative filtering.** *In progress — personalized via co-purchase patterns, comparison numbers pending.*