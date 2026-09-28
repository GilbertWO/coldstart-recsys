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

Two baselines, built before any real model — a floor to measure actual improvement against, not filler. Both are scored against the held-out validation week.

**1. Popularity-based recommender (floor baseline).** Same top-12 most-purchased articles (by train-window purchase count) recommended to every customer, with no personalization by design — this isolates "how much does personalization actually help" as a question the next baseline and the real model both have to answer. precision@12 = 0.00234, recall@12 = 0.00897, hit_rate@12 = 2.48% (share of customers with at least one relevant item anywhere in their top-12), MAP@12 = 0.00344. Low numbers are expected for a fully non-personalized baseline against a 105,542-article catalog — the number that matters isn't this score in isolation, it's the gap the next model has to close. One structural note: the top-12 spans only 8 distinct products, since 3 slots are different color variants of the single most popular item — left this way deliberately, since Kaggle's own MAP@12 scoring is at exact article_id granularity, not product level.

**2. Item-item collaborative filtering.** Personalized via co-purchase patterns computed from a binary customer-article interaction matrix over the top 5,000 candidate articles, with cosine similarity between items. Any val customer with no row in that matrix (including every true-cold-start customer with zero train purchases) falls back to the popularity top-12 — this fallback *is* the cold-start handling this baseline has, not a patch for a missing case. Reported score (already-purchased items excluded from a customer's own recommendations — see the exclusion ablation below for why that's kept as the default despite scoring worse): precision@12 = 0.00314, recall@12 = 0.01285, hit_rate@12 = 3.44%, MAP@12 = 0.00478.

### Exclusion ablation: excluding already-bought items hurts both models

| Model | Metric | With exclusion | Without exclusion | Relative cost of exclusion |
|---|---|---|---|---|
| Item-item CF | precision@12 | 0.00314 | 0.00441 | -29% |
| Item-item CF | recall@12 | 0.01285 | 0.02095 | -39% |
| Item-item CF | hit_rate@12 | 3.44% | 4.65% | -26% |
| Item-item CF | MAP@12 | 0.00478 | 0.00839 | -43% |
| Popularity | precision@12 | 0.00182 | 0.00234 | -22% |
| Popularity | recall@12 | 0.00673 | 0.00897 | -25% |
| Popularity | hit_rate@12 | 1.96% | 2.48% | -21% |
| Popularity | MAP@12 | 0.00252 | 0.00344 | -27% |

Hypothesis: fast-fashion repurchase behavior (restocking basics, buying another colorway) makes "already bought" a positive signal here, not noise — a pattern that matches write-ups from the actual H&M Kaggle competition.

Given that, the honest call would be to report the *unexcluded* numbers, since they score higher across every metric on both models. This project reports the numbers *with* exclusion anyway (item-item CF: precision 0.00314 / recall 0.01285 / hit_rate 3.44% / MAP@12 0.00478 above, not the higher unexcluded figures), a deliberate choice made for one reason: a model that mostly recommends re-buys is a weaker demonstration of collaborative filtering for this case study, even though it scores lower on this dataset's offline metrics. That's a modeling-goals decision, not a data or bug artifact, and it's called out explicitly here rather than left for a reader to notice the gap and wonder.

### A gap the ablation table doesn't cover: recency alone beats the reported item-item number

`baseline.py` also runs a non-personalized recency heuristic (most-purchased articles in just the last 7 days of train, no exclusion, no per-customer logic at all) as a side experiment. It scores MAP@12 = 0.00677 — higher than the reported item-item CF number above (0.00478), on every metric, in aggregate over all val customers. That's worth stating plainly rather than leaving it for a reviewer to find by rerunning the script.

It isn't a fair fight as stated: the recency heuristic was never run through the exclusion filter, and item-item CF *without* exclusion scores 0.00839 — comfortably ahead of it. So on equal terms (neither excluding repurchases), personalization does earn its keep over a recency-only signal. But the specific number this README reports for item-item CF, chosen for the exclusion-tradeoff reasoning above, is not the equal-terms number — and it does lose to a heuristic with no personalization in it at all. The recency comparison hasn't been run per-segment or with exclusion applied to it, so it isn't a full ablation entry yet, just a flag that the case-study narrative ("personalization beats non-personalized baselines") needs the equal-terms comparison to hold, not the as-reported one.

### Cold-start segment breakdown: personalization value is concentrated in the "a little data" band

| Segment | n customers | Popularity MAP@12 | Item-item CF MAP@12 | Relative lift |
|---|---|---|---|---|
| 0 purchases | 5,395 (7.49% of val) | 0.00392 | 0.00392 | 0% |
| 1–2 purchases | 2,066 | 0.00349 | 0.00911 | +161% |
| 3+ purchases | 64,558 | 0.00340 | 0.00471 | +38% |

The 0-purchase segment isn't a rounding error — popularity and item-item CF post *identical* numbers across every metric there (precision 0.00269, recall 0.00968, hit_rate 2.78%, MAP@12 0.00392), because every one of those customers has no row in the interaction matrix and falls straight through to the popularity fallback. That confirms the fallback is wired correctly, but it also means this baseline has no real answer for true cold-start customers yet — it's the popularity floor wearing a different name. Closing that gap is what the planned ALS-plus-cold-start-policy model (method 5) is for.

One-line summary: item-item CF adds the most value for light-history customers (1-2 purchases), meaningfully less for heavy-history customers (3+), and nothing at all for zero-history customers — which is exactly the shape you'd expect from a model that has nothing to work with until a customer has *some* purchase history to compute similarity against.
