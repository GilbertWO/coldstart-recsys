# coldstart-recsys
Recommendation engine (ALS-based collaborative filtering, with a LightGBM re-ranking stage identified as the natural next step) trained on real e-commerce transactions, served via Spring Boot + Redis, with explicit cold-start handling for new users.

## Dataset
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 1,371,980 registered customers, 31,788,324 transactions in `transactions_train.csv`.

## Status

Done: EDA, baselines (popularity, item-item CF), ALS core model with confidence weighting and a cold-start fallback policy, a matched three-way model comparison on a validation week and a test week with paired-bootstrap intervals. Also done: a rolling backtest over four earlier weeks. Not started: LightGBM re-ranking, Spring Boot + Redis serving.

## EDA Findings

**1. User-level cold start is the dominant case, not an edge case.**
19.58% of all 1,371,980 registered customers — including 9,699 (0.71%) who registered but never made a single purchase — have fewer than 3 purchases in the training window. Repeat-purchase rate (customers with more than 1 purchase) is 89.71%. Cold-start handling has to be a first-class part of the design, not a fallback bolted on afterward.

**2. Item-level imbalance mirrors the user-level problem.**
The top 3 product categories out of 19 (Garment Upper body, Garment Lower body, Garment Full body) account for 72.83% of all transactions, while the bottom 9 categories combined represent under 1%. 5,486 articles (5.20% of the catalog) have zero or one purchase. A naive popularity-based baseline will be dominated by upper-body garments, and item-level cold start (new or rarely-purchased articles) needs the same deliberate handling as user-level cold start.

**3. The article catalog has clean, low-cardinality categorical structure — use it directly, don't engineer around it.**
Fields like `product_group_name` (19), `garment_group_name` (21), `index_group_name` (5), and `colour_group_name` (50) have zero missingness and moderate cardinality, so they feed LightGBM/ALS as categorical features as-is. In contrast, `detail_desc` (43,404 unique values), `prod_name` (45,875), and `product_code` (47,224) — out of 105,542 total articles — carry cardinality close to the row count, meaning they function as identifiers or free text rather than model-usable categories, and are excluded rather than encoded.

## Evaluation setup

Temporal split by `prepare_data.py`: the most recent full week is the **test** week (2020-09-16 to 2020-09-22), the week before it is **validation**, everything earlier is **train**. Every model recommends 12 articles per customer who purchased in the target week, scored on precision@12, recall@12, hit_rate@12 (share of customers with at least one relevant item in their top 12) and MAP@12 (Kaggle's definition). Customers are segmented by purchase count in the *history* the model saw: 0, 1-2, or 3+ purchases.

All configuration was chosen on the validation week. The test week was scored **once**, after the configuration was frozen, using train+val as history (so the recency window ends the day before the test week, the same structure the configs were tuned under). `test_evaluation.py` refuses to run twice without an explicit flag.

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

Given that, the honest call would be to report the *unexcluded* numbers, since they score higher across every metric on both models. The baseline numbers above are reported *with* exclusion anyway, a deliberate choice made for one reason: a model that mostly recommends re-buys is a weaker demonstration of collaborative filtering for this case study, even though it scores lower on this dataset's offline metrics. That's a modeling-goals decision, not a data or bug artifact, and it's called out explicitly here rather than left for a reader to notice the gap and wonder. **Update (Oct 4):** the model comparison below reports both exclusion settings in full for every model. Which setting to treat as the headline was not fixed before seeing results, so neither table should be read as pre-registered. The no-exclusion item-item row above is now regenerable from the repo (`exclusion_comparison.py` reproduces it).

### A gap the ablation table doesn't cover: recency alone beats the reported item-item number

`baseline.py` also runs a non-personalized recency heuristic (most-purchased articles in just the last 7 days of train, no exclusion, no per-customer logic at all) as a side experiment. It scores MAP@12 = 0.00677 — higher than the reported item-item CF number above (0.00478), on every metric, in aggregate over all val customers. That's worth stating plainly rather than leaving it for a reviewer to find by rerunning the script.

It isn't a fair fight as stated: the recency heuristic was never run through the exclusion filter, and item-item CF *without* exclusion scores 0.00839 — comfortably ahead of it. So on equal terms (neither excluding repurchases), personalization does earn its keep over a recency-only signal. But the specific number this README reports for item-item CF, chosen for the exclusion-tradeoff reasoning above, is not the equal-terms number — and it does lose to a heuristic with no personalization in it at all. **Update (Oct 4):** the equal-terms comparison has now been run, per segment and with exclusion applied to the recency list; see "Model comparison" below. With exclusion on, the recency list beats both personalized models in the weeks near the end of the data and loses badly in earlier weeks; see the rolling backtest.

### Cold-start segment breakdown: personalization value is concentrated in the "a little data" band

| Segment | n customers | Popularity MAP@12 | Item-item CF MAP@12 | Relative lift |
|---|---|---|---|---|
| 0 purchases | 5,395 (7.49% of val) | 0.00392 | 0.00392 | 0% |
| 1–2 purchases | 2,066 | 0.00349 | 0.00911 | +161% |
| 3+ purchases | 64,558 | 0.00340 | 0.00471 | +38% |

The 0-purchase segment isn't a rounding error — popularity and item-item CF post *identical* numbers across every metric there (precision 0.00269, recall 0.00968, hit_rate 2.78%, MAP@12 0.00392), because every one of those customers has no row in the interaction matrix and falls straight through to the popularity fallback. That confirms the fallback is wired correctly, but it also means this baseline has no real answer for true cold-start customers. The cold-start policy below is the response to that.

One-line summary: item-item CF adds the most value for light-history customers (1-2 purchases) over *all-time* popularity, meaningfully less for heavy-history customers (3+), and nothing at all for zero-history customers. **Caveat added Oct 4:** the +161% compares item-item against all-time popularity. Against a last-14-day popularity list the 1-2 purchase gap is much smaller and not reliably different in the test week (see segment tables below), and in every backtest week 45-48% of the 1-2 purchase customers have no row in item-item's 5,000-article matrix and receive its popularity fallback, so that segment's item-item numbers are partly a popularity list.

## Core model: ALS

Implicit-feedback ALS (`implicit` library, CPU) on a customer-by-article matrix of purchase *counts*, not a binary matrix; `alpha` scales confidence in repeated purchases. Fixed `random_state=42`; runs are deterministic.

What moved validation MAP@12 (exclusion on, all-time fallback, factors=100 unless noted):

| ALS variant | MAP@12 |
|---|---|
| binary matrix, alpha=1 | 0.00354 |
| purchase counts, alpha=1 | 0.003778 |
| purchase counts, alpha=15 | 0.004761 |
| purchase counts, alpha=40 | 0.004470 |
| factors=150, reg=0.1, 20 iterations, alpha=15 | 0.004810 |

Counts alone gave about +7%; scaling confidence with alpha=15 gave most of the rest, and alpha peaks near 15. The five configurations tried are a coarse manual pass, not an ablation (the last one changed factors, regularization and iterations together). Configuration was ranked on MAP@12, the headline metric. An earlier ranking on precision@12 picked a different configuration; that was a selection error and was corrected.

Convergence: with factors=150 / reg=0.1 / alpha=15, 99.0% of the training-loss decline over 40 iterations was done by iteration 20, but validation MAP@12 (exclusion on) still rose 1.6% from 20 to 40 iterations (0.004995 to 0.005076). The frozen configuration therefore uses **factors=150, reg=0.1, alpha=15, 40 iterations**. Validation tables below that were produced before the convergence check use 20 iterations and say so.

## Cold-start policy for zero-history customers

A customer with no history has no ALS embedding, so the model hands them a fallback list. Validation results for that segment (5,395 customers):

| Fallback list | precision@12 | recall@12 | hit_rate@12 | MAP@12 |
|---|---|---|---|---|
| all-time popularity | 0.002688 | 0.009681 | 2.780% | 0.003923 |
| last 28 days | 0.004526 | 0.021942 | 5.246% | 0.006011 |
| last 14 days | 0.004371 | 0.021070 | 5.079% | 0.006399 |
| **last 7 days (chosen)** | 0.004557 | 0.022371 | 5.264% | 0.006393 |

Switching from all-time to last-7-day popularity lifted that segment's validation MAP@12 by 63%. The 7-day and 14-day windows are indistinguishable on MAP; 7 days won the other three metrics. The window was chosen from this one segment on one week, and that week falls in the period where recency lists do well (see the backtest below); how the 7-day fallback behaves in the earlier, recency-hostile weeks has not been examined. This is a recency-popularity fallback, not a learned cold-start model; item-item CF gets the same fallback in every comparison below so the two models are matched.

## Model comparison

Three models, identical fallback (last-7-day popularity) for customers a model cannot personalize: ALS, item-item CF, and a non-personalized last-14-day popularity list ("recency"). Each is scored with already-bought items excluded and with them allowed. "A vs B" gaps are relative to B. 95% intervals come from a paired bootstrap over customers, so they capture customer-sampling noise within one week only.

### Validation week (ALS: 20 iterations)

| Exclusion | Model | MAP@12 | hit_rate@12 |
|---|---|---|---|
| off | ALS | 0.009422 | 5.17% |
| off | item-item CF | 0.008709 | 4.89% |
| off | recency 14d | 0.006995 | 5.51% |
| on | ALS | 0.004995 | 3.32% |
| on | item-item CF | 0.005094 | 3.68% |
| on | recency 14d | 0.006435 | 5.12% |

### Test week (history = train+val; ALS: 40 iterations; 68,984 customers)

| Exclusion | Model | MAP@12 | hit_rate@12 |
|---|---|---|---|
| off | ALS | 0.008802 | 4.94% |
| off | item-item CF | 0.008191 | 4.65% |
| off | recency 14d | 0.007104 | 6.13% |
| on | ALS | 0.004996 | 3.32% |
| on | item-item CF | 0.004705 | 3.51% |
| on | recency 14d | 0.006702 | 5.85% |

Test segments (by history purchase count): 5,572 with 0 purchases, 2,061 with 1-2, 61,351 with 3+. MAP@12 by segment, test week:

| Exclusion | Model | 0 purchases | 1-2 purchases | 3+ purchases |
|---|---|---|---|---|
| off | ALS | 0.007878 | 0.013056 | 0.008743 |
| off | item-item CF | 0.007878 | 0.016237 | 0.007950 |
| off | recency 14d | 0.006726 | 0.008424 | 0.007094 |
| on | ALS | 0.007878 | 0.007349 | 0.004655 |
| on | item-item CF | 0.007878 | 0.009791 | 0.004246 |
| on | recency 14d | 0.006726 | 0.008385 | 0.006643 |

Segment sizes differ from the validation week, so test and validation numbers are two separate tables, not a before/after.

### Rolling backtest: five weeks instead of two

Every comparison above rests on one week. `rolling_backtest.py` scores the same three models, with the same frozen configuration (factors=150, reg=0.1, alpha=15, 40 iterations, 7-day fallback, 14-day recency), on four earlier weeks (targets starting 2020-08-12, 08-19, 08-26, 09-02; history = everything before each week) and combines them with the test week. The validation week (09-09) is left out because the configuration was tuned on it. Cells show A vs B as a percent of B's score; `*` means that week's 95% paired-bootstrap interval (over customers) excludes zero. Aggregate over all customers in the week.

Exclusion on:

| Metric | Comparison | 08-12 | 08-19 | 08-26 | 09-02 | test (09-16) |
|---|---|---|---|---|---|---|
| MAP@12 | ALS vs item-item | +0.6 | -0.2 | +8.0* | +3.5 | +6.2* |
| MAP@12 | ALS vs recency | +60.2* | +94.7* | -14.1* | -9.2* | -25.5* |
| MAP@12 | item-item vs recency | +59.3* | +95.1* | -20.5* | -12.2* | -29.8* |
| hit_rate@12 | ALS vs item-item | -6.6* | -7.9* | -4.3* | -7.1* | -5.4* |
| hit_rate@12 | ALS vs recency | +16.9* | +13.2* | -26.5* | -38.8* | -43.2* |
| hit_rate@12 | item-item vs recency | +25.1* | +22.8* | -23.2* | -34.1* | -39.9* |

Exclusion off:

| Metric | Comparison | 08-12 | 08-19 | 08-26 | 09-02 | test (09-16) |
|---|---|---|---|---|---|---|
| MAP@12 | ALS vs item-item | -2.4 | +2.4 | +15.0* | +9.2* | +7.5* |
| MAP@12 | ALS vs recency | +147.4* | +212.1* | +47.1* | +51.0* | +23.9* |
| MAP@12 | item-item vs recency | +153.3* | +205.0* | +27.9* | +38.3* | +15.3* |
| hit_rate@12 | ALS vs item-item | +2.8* | +5.8* | +11.7* | +7.2* | +6.1* |
| hit_rate@12 | ALS vs recency | +58.5* | +58.4* | -0.2 | -14.2* | -19.4* |
| hit_rate@12 | item-item vs recency | +54.1* | +49.7* | -10.6* | -19.9* | -24.1* |

Absolute MAP@12 with exclusion on, by week (08-12, 08-19, 08-26, 09-02, test): ALS 0.004708, 0.005036, 0.005154, 0.005166, 0.004996; item-item 0.004682, 0.005047, 0.004771, 0.004992, 0.004705; recency 0.002939, 0.002587, 0.006002, 0.005688, 0.006702. The two personalized models stay in a narrow band across all five weeks; the recency list moves by a factor of 2.6.

### What held up across the five weeks

- **Without exclusion, ALS leads item-item on hit rate in all five weeks** (+2.8% to +11.7%, every interval excludes zero) and on MAP@12 in three of five (+7.5% to +15.0%, significant), never significantly behind. Mean MAP gap +6.3%.
- **With exclusion on, item-item reaches more customers than ALS in all five weeks** (hit rate 4.3% to 7.9% higher, every interval excludes zero; -9.7% for ALS on the validation week as well). The same holds in the 3+ purchase segment alone (ALS -4.3% to -8.0%), so it is not an artifact of item-item's fallback on light-history customers.
- **With exclusion on, ALS vs item-item on MAP@12 is a small, week-dependent edge for ALS** (mean +3.6%, significant in two of five weeks, never significantly negative; validation was -1.9% with an interval spanning zero). Treat it as a tie leaning toward ALS, not a result.
- **Without exclusion, both personalized models beat the recency list on MAP@12 in all five weeks**, by 15% to 212%.

### What did not hold: the recency list is regime-dependent

An earlier version of this README, based on the validation and test weeks only, said the non-personalized recency list carries most of the predictive power. The backtest contradicts that as a general statement.

- With exclusion on, the 14-day recency list beat both ALS and item-item on MAP@12 in the three most recent weeks (08-26, 09-02, test; by 9% to 30%) and on validation, but **lost to both by 60% to 95% in the two earliest weeks** (08-12, 08-19), where its hit rate also fell below the personalized models'. Without exclusion the same flip appears on hit rate (recency ahead by 14-19% in the last two weeks, behind by 58% in the first two).
- [Likely] The recency list's quality depends on the calendar: something about late-August assortment or demand made the last two weeks of purchases a poor guide to the next week, and that stopped being true from late August onward. The data here cannot say what. The personalized models, which draw on a customer's whole history, did not show that swing.
- The practical reading: the personalized models are the robust choice; recency is a strong signal when the calendar cooperates and a weak one when it does not. Both the 7-day cold-start fallback and the 14-day recency baseline were selected on a week from the favorable regime.

### Caveats

- Five weeks is a small sample of calendar effects, the weeks share customers, and the earlier weeks differ in season from the later ones. The per-week intervals measure customer-sampling noise only. Read the pattern across weeks rather than any single starred cell, since about 70 intervals were computed with no multiple-comparison correction.
- ALS received a tuning sweep, confidence weighting and a convergence check; item-item CF received none. A tuned item-item might narrow the ALS lead.
- Item-item CF is restricted to its top 5,000 candidate articles. On the test week roughly 7,700 of 68,984 customers (about 5,570 true cold-start plus about 2,100 with history but no purchases in that set) fall back to popularity for this reason, while ALS uses the full catalog. In every backtest week 45-48% of the 1-2 purchase customers fall back, versus under 2% of the 3+ customers.
- The 1-2 purchase segment is small (about 2,000 customers per week), so its intervals are wide and few differences there are distinguishable from noise.
- The "keep exclusion on" framing from the baseline section is a modeling-goals choice, not a metrics-driven one, and the exclusion setting changes which model looks best.

Takeaway as measured: the personalized models are stable across weeks (MAP@12 about 0.0047-0.0052 with exclusion on, about 0.0081-0.0095 without), ALS has a reliable hit-rate and MAP edge over item-item when repeat purchases are allowed, item-item reaches more customers when they are excluded, and the non-personalized recency list can beat or badly lose to either depending on the week. The natural next experiment is a blend of recency and model scores, which is also the plan for the LightGBM re-ranking stage, because the two kinds of signal fail in different weeks. Any such experiment needs development on validation and the earlier folds, and evaluation on weeks not yet used, because the test week has now been used.

## Model artifact

`test_evaluation.py` saves the trained ALS model (user and item factors, row/column id order, the cold-start fallback list, and the config) to `models/`. It is trained on train+val and is gitignored because the factor matrices for every customer are large; the serving stage will regenerate or fetch it.

## Reproducing

Run from the repo root with the processed data in `data/processed/` (output of `prepare_data.py`).

| Script | Purpose |
|---|---|
| `python baseline.py` | popularity and item-item CF baselines, metrics, segments |
| `python model.py` | five-configuration ALS sweep and segment breakdown (`--best-only` fits only the factors=50 / alpha=15 / 15-iteration configuration) |
| `python cold_start_policy.py` | fallback-window comparison for the 0-purchase segment |
| `python item_item_matched.py` | item-item CF re-scored with the 7-day fallback |
| `python exclusion_comparison.py --factors 150 --regularization 0.1 --iterations 20` | ALS vs item-item vs recency, both exclusion settings, validation week |
| `python bootstrap_comparison.py` | paired-bootstrap intervals, validation week |
| `python convergence_check.py` | training-loss curve and 40-iteration validation check |
| `python test_evaluation.py` | the one-time test-week evaluation; also writes `models/` |
| `python rolling_backtest.py` | weekly backtest over four earlier weeks plus the cached test week (about 1 hour; resumable) |

Several scripts reproduce a prior table as a self-check and print CHECK PASSED or CHECK FAILED; a failed check means the pipeline differs from the one that produced the reported numbers.
