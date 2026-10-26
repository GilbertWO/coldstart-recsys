# coldstart-recsys
Recommendation engine (ALS-based collaborative filtering, with a LightGBM re-ranking stage identified as the natural next step) trained on real e-commerce transactions, served via Spring Boot + PostgreSQL (with an optional Redis cache, off by default), with explicit cold-start handling for new users.

## Dataset
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 1,371,980 registered customers, 31,788,324 transactions in `transactions_train.csv`.

## Status

Done: EDA, baselines (popularity, item-item CF), ALS core model with confidence weighting and a cold-start fallback policy, a matched three-way model comparison on a validation week and a test week with paired-bootstrap intervals. Also done: a rolling backtest over four earlier weeks. Also done: the serving data layer, with precomputed top-12 lists for every customer in the saved model, loaded into a local PostgreSQL alongside a Redis container (see Serving layer below). Also done: the Spring Boot endpoint `GET /v1/recommendations/{customerId}`, which reads those lists from PostgreSQL and is covered by 32 automated tests, with the fallback list cached in process memory and a Redis cache for per-customer lists. The Redis layer is correct and survives a Redis outage. A same-session comparison with it switched off (see the verification table) shows it speeds up known customers only when traffic is spread evenly, adds about 0.5 to 0.7 ms to every request for an unknown id, and makes the tail worse on skewed traffic, so it is off by default and can be switched on with `recs.list-cache-enabled=true`. Not started: LightGBM re-ranking.

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

It isn't a fair fight as stated: the recency heuristic was never run through the exclusion filter, and item-item CF *without* exclusion scores 0.00839 — comfortably ahead of it. So on equal terms (neither excluding repurchases), personalization does earn its keep over a recency-only signal. But the specific number this README reports for item-item CF, chosen for the exclusion-tradeoff reasoning above, is not the equal-terms number — and it does lose to a heuristic with no personalization in it at all. **Update (Oct 4):** the equal-terms comparison has now been run, per segment and with exclusion applied to the recency list; see "Model comparison" below. With exclusion on, a 7-day recency list beats both personalized models in four of five backtest weeks and fails in one (a week of complete best-seller turnover); see the rolling backtest.

### Cold-start segment breakdown: against all-time popularity, item-item CF gains most in the 1-2 purchase band

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

Sensitivity to the other hyperparameters: because that first pass was coarse and on one week, `als_sweep.py` varies one knob at a time around the frozen configuration and scores each variant on three earlier weeks (targets starting 2020-09-02, 08-26 and 08-19; validation and test weeks not used), under both exclusion settings. The rule for calling a variant a candidate was fixed beforehand: at least +3% mean MAP@12 in both settings and no significantly worse week. Mean MAP@12 change vs the frozen configuration over those three weeks:

| Variant | Exclusion on | Exclusion off |
|---|---|---|
| alpha 8 | -0.6% | +3.2% |
| alpha 25 | -2.0% | -4.6% |
| regularization 0.03 | +0.1% | -0.0% |
| regularization 0.3 | -0.1% | +0.0% |
| factors 100 | +1.5% | -2.2% |
| factors 200 | -1.0% | +1.6% |

No variant met the rule, so the frozen configuration stands. With exclusion on every variant is within about 2% of it; without exclusion alpha is the only knob with a consistent effect (alpha 8 is slightly better, +4%, +4% and +2% across the three weeks; alpha 25 is clearly worse), and regularization has essentially no effect. Gains of this size are close to what a refit with a different random seed could produce (fits here use a single fixed seed, so seed-to-seed variation was not measured, and the per-week intervals only cover customer-sampling noise). This is a sensitivity analysis, not a re-tuning: the test result was produced beforehand with the frozen configuration. Item-item CF was not tuned at all, so the tuning asymmetry noted below remains.

## Cold-start policy for zero-history customers

A customer with no history has no ALS embedding, so the model hands them a fallback list. Validation results for that segment (5,395 customers):

| Fallback list | precision@12 | recall@12 | hit_rate@12 | MAP@12 |
|---|---|---|---|---|
| all-time popularity | 0.002688 | 0.009681 | 2.780% | 0.003923 |
| last 28 days | 0.004526 | 0.021942 | 5.246% | 0.006011 |
| last 14 days | 0.004371 | 0.021070 | 5.079% | 0.006399 |
| **last 7 days (chosen)** | 0.004557 | 0.022371 | 5.264% | 0.006393 |

Switching from all-time to last-7-day popularity lifted that segment's validation MAP@12 by 63%. The 7-day and 14-day windows are indistinguishable on MAP; 7 days won the other three metrics. The window was chosen from this one segment on one week, and that week falls in the period where recency lists do well (see the backtest below); `fallback_diagnostic.py` later checked the windows on the zero-purchase segment in four earlier weeks as well (`fallback_diagnostic_results.csv`). Across the five weeks the 7-day list had the highest MAP@12 in three; in the other two a 14-day list was ahead by 3.2% and 0.1%, neither significant. All-time popularity was significantly worse than the 7-day list in four of the five weeks. The same check found that in the week starting 2020-08-12 the 7-day list overlapped none of that week's 12 best sellers, which is why the fallback must be refreshed rather than fixed. This is a recency-popularity fallback, not a learned cold-start model; item-item CF gets the same fallback in every comparison below so the two models are matched.

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
- **Without exclusion, both personalized models beat the 14-day recency list on MAP@12 in all five weeks**, by 15% to 212%. That comparison turned out to depend heavily on the recency window; against a 7-day list the lead is much smaller (next section).

### The recency window matters, and so does the week

The 14-day recency list used in the tables above turned out to be a weak version of the baseline. In the week starting 2020-08-19 it overlapped only 1 of that week's 12 best-selling articles (all-customer MAP@12 0.0028), while the 7-day list overlapped 4 of 12 and scored 0.0079 on the same customers. The 7-day window is also the cold-start window and the definition the original baseline used. Which window to call "the" recency baseline was not fixed before seeing results, so both are reported; the 7-day list is treated as the primary baseline because it is the stronger one, and models should be judged against the strongest simple baseline. `recency_window_recheck.py` re-scores it from the saved per-customer scores (it first verifies that its customer alignment reproduces the saved 14-day numbers exactly).

Absolute recency MAP@12 by week (08-12, 08-19, 08-26, 09-02, test), all customers: exclusion off, 14-day 0.003285, 0.002777, 0.006466, 0.006304, 0.007104 and 7-day 0.003174, 0.007867, 0.006828, 0.006732, 0.008748; exclusion on, 14-day 0.002939, 0.002587, 0.006002, 0.005688, 0.006702 and 7-day 0.002907, 0.007630, 0.006399, 0.006308, 0.008353.

Models vs the 7-day recency list, aggregate (08-12, 08-19, 08-26, 09-02, test; `*` = that week's interval excludes zero):

| Exclusion | Metric | Comparison | 08-12 | 08-19 | 08-26 | 09-02 | test |
|---|---|---|---|---|---|---|---|
| on | MAP@12 | ALS vs recency 7d | +62.0* | -34.0* | -19.4* | -18.1* | -40.2* |
| on | MAP@12 | item-item vs recency 7d | +61.1* | -33.8* | -25.4* | -20.9* | -43.7* |
| on | hit_rate@12 | ALS vs recency 7d | +26.4* | -33.5* | -31.4* | -39.8* | -45.8* |
| on | hit_rate@12 | item-item vs recency 7d | +35.2* | -27.8* | -28.3* | -35.2* | -42.7* |
| off | MAP@12 | ALS vs recency 7d | +156.0* | +10.2* | +39.3* | +41.4* | +0.6 |
| off | MAP@12 | item-item vs recency 7d | +162.2* | +7.6 | +21.1* | +29.5* | -6.4 |
| off | hit_rate@12 | ALS vs recency 7d | +79.2* | -5.4* | -5.8* | -15.4* | -23.2* |
| off | hit_rate@12 | item-item vs recency 7d | +74.3* | -10.6* | -15.6* | -21.0* | -27.6* |

What this shows:

- **With exclusion on, the 7-day recency list beats both personalized models on MAP@12 and on hit rate in four of five weeks**, by 18% to 46% (every interval excludes zero). It loses in one week, 08-12, by about 60% on MAP.
- **08-12 is a genuine failure for every window.** The 7-day and 14-day lists overlapped none of that week's 12 best sellers and covered about 1% of purchases, the same as an all-time list; best sellers had turned over. The personalized models, which draw on each customer's whole history, were not affected. A window that spans a turnover goes stale, which is also what happened to the 14-day list in the 08-19 week.
- **Without exclusion, ALS leads the 7-day list on MAP@12 in four of five weeks (+10% to +156%), but the lead is zero on the test week (+0.6%, not significant)**, and item-item is behind it on the test week (-6.4%, significant in the 3+ segment). The 7-day list has the higher hit rate in four of five weeks (ALS -5% to -23%).
- [Likely] The 7-day list's MAP rose across these weeks (0.0032, 0.0079, 0.0068, 0.0067, 0.0087 without exclusion) while the personalized models stayed roughly flat (ALS 0.0081 to 0.0095), so recency has been gaining on the models toward the end of the data. The cause is not known.

### Caveats

- Five weeks is a small sample of calendar effects, the weeks share customers, and the earlier weeks differ in season from the later ones. The per-week intervals measure customer-sampling noise only. Read the pattern across weeks rather than any single starred cell, since about 70 intervals were computed with no multiple-comparison correction.
- ALS received a tuning sweep, confidence weighting and a convergence check; item-item CF received none. A tuned item-item might narrow the ALS lead.
- Item-item CF is restricted to its top 5,000 candidate articles. On the test week roughly 7,700 of 68,984 customers (about 5,570 true cold-start plus about 2,100 with history but no purchases in that set) fall back to popularity for this reason, while ALS uses the full catalog. In every backtest week 45-48% of the 1-2 purchase customers fall back, versus under 2% of the 3+ customers.
- The 1-2 purchase segment is small (about 2,000 customers per week), so its intervals are wide and few differences there are distinguishable from noise.
- The "keep exclusion on" framing from the baseline section is a modeling-goals choice, not a metrics-driven one, and the exclusion setting changes which model looks best.
- The recency window (7 vs 14 days) changed several conclusions, and the choice of which to treat as primary was made after seeing results. It was resolved in the conservative direction (the stronger baseline), but other windows (for example a 10-day list) were not explored, and segment 0 has only about 5,000 customers a week, so the cold-start window comparison cannot separate 7 from 14 or 28 days reliably (all-time popularity is clearly worse in every week).

Takeaway as measured: a 7-day popularity list is a strong baseline on this dataset. With already-bought items excluded it beats ALS and item-item CF on both MAP@12 and hit rate in four of five weeks, and it fails only in a week when best sellers turned over completely. The personalized models are steady week to week (MAP@12 about 0.0047-0.0052 with exclusion on, 0.0081-0.0095 without) and their clearest value is MAP when repeat purchases are allowed (ALS beats item-item in that setting and beats the 7-day list in four of five weeks), a lead that was zero on the test week. So the honest statement is not "personalization beats the baselines", it is that personalization adds ranking quality in specific conditions and that a recency signal and the personalized models fail in different weeks. That is the argument for a blend of recency and model scores, which is also the plan for the LightGBM re-ranking stage. Any such experiment needs development on validation and the earlier folds, and evaluation on weeks not yet used, because the test week has now been used.

## Why ALS for this block, and why LightGBM is the next stage

The project plan budgeted this block for one model: matrix factorization (ALS) or a LightGBM learning-to-rank model, not both, on a schedule with no slack before the buffer week. ALS was chosen because it needs only the purchase matrix, handles the full 105,542-article catalog directly, and gives a clean per-segment story about where personalization does and does not help, which is the cold-start question this project is built around. A LightGBM ranker needs candidate generation, feature engineering and a training-set construction step before it can be evaluated honestly, and splitting two weeks across two under-evaluated models was the worse trade.

The evaluation also argues for the ranker as the next step rather than more ALS tuning. A one-knob sensitivity sweep moved ALS by only a few percent, while a non-personalized 7-day popularity list beat both personalized models in four of five weeks with exclusion on, and failed in the one week when best sellers turned over. The two kinds of signal fail in different weeks, which is the situation where combining candidate sources (recent popularity, previously bought items, ALS scores) in a ranker that sees customer and article features has something to gain. That is a hypothesis, not a result: no blend or ranker has been built or evaluated yet.

## Model artifact

`test_evaluation.py` saves the trained ALS model (user and item factors, row/column id order, the cold-start fallback list, and the config) to `models/`. It is trained on train+val and is gitignored because the factor matrices for every customer are large; the serving layer does not load it at request time, because `export_recommendations.py` reads it once to precompute every customer's recommendation lists (next section).

## Serving layer

The API does not score customers at request time. The saved factors are an 885 MB NumPy file that a Spring Boot service cannot read, and scoring 105,542 articles per request would buy nothing, so `export_recommendations.py` scores every customer in the artifact once and writes two top-12 lists per customer. The service looks them up by primary key in PostgreSQL; an optional Redis cache can sit in front of it, and it is off by default (key `rec:{mode}:{customerId}`, 1-hour TTL, unknown ids are never cached, and any Redis failure falls through to PostgreSQL after a 100 ms client timeout; `recs.list-cache-enabled=true` turns it on, and the A/B row below is why it is off by default). The one list that every unknown customer shares, the fallback list, is cached in the application's memory for 60 seconds (`recs.fallback-cache-ttl`), so unknown-id requests no longer query the database for it. An empty result is never cached, and per-customer lookups are never cached in process. The design and its open decisions are in [`serving-design-draft.md`](serving-design-draft.md).

**Two modes.** `repurchase` (the default) allows articles the customer already bought; `discover` removes them. The default follows the evaluation. Against the 7-day popularity list, ALS leads on MAP@12 in all five evaluation weeks when already-bought items are allowed (significant in four; +0.6% and not significant on the test week), but its hit rate@12 is lower than the popularity list's in four of the five weeks. With already-bought items removed, ALS trails the popularity list on both metrics in four of five weeks, so `discover` stays available but is not the default, and its documentation says so. No model in this project beats the 7-day popularity list on both metrics in either mode.

**Cold start is part of the response contract.** A known customer gets their precomputed list. A well-formed id that is not in the table gets HTTP 200 with the 7-day popularity list and `source: fallback_popularity_7d`. A malformed id (anything other than 64 lowercase hex characters, so ids with spaces or uppercase letters are rejected rather than normalized) or an unknown `mode` gets 400, and a missing `mode` means `repurchase`. If the fallback table is empty, which is a server fault and not a user error, the endpoint returns 503 instead of an empty list. A new user never gets a 404 or 500 from the lookup path. On fallback responses, `modelVersion` records the model run that produced the batch, not the method behind the list; `source` says which method produced it. About 15,300 registered customers (no purchases before the history ended on 2020-09-15) have no embedding and take the fallback path. About 1,660 articles with no purchases in the history window can never be recommended; that is item cold start and is not addressed here.

**What exists and has been verified (October 8-11, 2026).**

| Item | Result |
|---|---|
| Precomputed lists | 1,356,709 customers x 2 modes = 2,713,418 rows in about 21 minutes. The script's checks passed: 12 distinct articles per list, no already-bought article in any `discover` list, and 100% agreement with an independent recomputation on 300 sampled customers |
| PostgreSQL load | the 2,713,418 rows load with `COPY` in about 53 seconds; per-mode counts match the export |
| Primary-key lookup | about 0.13 ms execution time for one lookup (a single EXPLAIN ANALYZE, warm cache). This is database time only, not endpoint latency |
| Local infrastructure | `serving/docker-compose.yml` runs PostgreSQL 16 and Redis 7, both bound to localhost |
| Endpoint contract tests | 32 automated tests pass (33 with Spring's context-load test): 8 web-layer contract tests with a mocked service, 5 service tests with a fake repository, 4 tests of the in-memory fallback cache with a controllable clock, 9 tests of the Redis layer with a fake store, 4 wiring tests that check which caching repositories the service receives with the Redis layer on, off and unset, and 2 tests that the Redis health check is registered only when the layer is on. The contract and service tests cover a known customer, an unknown valid id, a malformed id, an id with a leading space, an uppercase id, an invalid mode, a missing mode defaulting to `repurchase`, and an empty fallback table returning 503 |
| Live check against the local database | For one sampled customer, the endpoint's `repurchase` list matched the PostgreSQL array exactly, in order. The `discover` list for that customer differed from it (it was not compared against the database). An all-zero id returned the fallback list in both modes with the requested mode echoed back. A malformed id, a leading-space id, an uppercase id and an invalid mode each returned 400. One customer was checked, not a sample |
| Endpoint vs export parity | `serving/parity_check_api.py` sampled 600 random rows from `recommendations.csv` (295 `repurchase`, 305 `discover`, seed 42) and called the running endpoint for each. All 600 matched the export exactly: same 12 article ids in the same order, `source` `als`, same mode and model version. 20 random well-formed ids not in the table all returned the fallback list with the requested mode echoed, and 4 malformed requests returned 400. 624 checks, 0 failures. This is a sample, not every customer |
| Latency baseline, PostgreSQL only (before Redis) | `serving/latency_baseline.py`, October 12: one client, 5,000 requests per run, 3 runs after a 3,000-request warm-up, 90% known customers and 10% unknown ids, both modes. Uniform traffic: median 2.10 ms, p95 2.92 ms, p99 3.57 ms (median across the three runs). Skewed (zipf) traffic: 1.65 / 2.66 / 3.28 ms. Uniform traffic with 8 concurrent clients: 2.24 / 3.45 / 4.19 ms at roughly 3,000 to 3,300 requests per second. Requests for unknown ids (the fallback path) are slower than known-customer requests by about 0.6 to 1.0 ms. The client, the app, PostgreSQL and Redis all run on one laptop and the numbers include the Python client's own overhead; TCP connect time is excluded. Skewed traffic is faster than uniform traffic even with no cache, so any Redis comparison has to use the same workload on both sides |
| Latency with the fallback list cached in process memory | Same method and workloads as the row above, October 13, after the endpoint-vs-export parity check passed again (624 checks, 0 failures). Median latency of requests for unknown ids: 2.642 to 1.957 ms (uniform), 2.568 to 1.807 ms (zipf), 3.073 to 2.367 ms (8 clients). The drop is about 0.7 ms in all three workloads, which is consistent with the second database query that unknown ids used to cause. Afterwards, unknown-id requests cost about the same as known-customer requests; the remaining gap changes sign between workloads and is within run-to-run noise (one workload's median drifted by 0.24 ms between its first and third run). Overall latency did not improve: medians are slightly higher in all three workloads and the tails are mixed, which is what a saving of about 0.7 ms on 10% of requests cannot be expected to show against this much noise. Median / p95 / p99, uniform: 2.12 / 2.98 / 3.61 ms against 2.10 / 2.92 / 3.57 ms before. Zipf: 1.70 / 2.44 / 2.88 against 1.65 / 2.66 / 3.28. 8 clients: 2.36 / 3.44 / 4.53 against 2.24 / 3.45 / 4.19, where the higher p99 was not investigated. One laptop and one session per variant, so a difference of about 0.1 ms in a median should not be read as real |
| Redis cache layer: correctness and outage (October 14) | Correctness: with Redis cleared, the parity script created exactly 600 entries (one per sampled customer and mode, none for unknown or malformed ids); an entry had a TTL just under 3,600 s and held the model version plus 12 ids; a second parity pass, served from Redis, matched the export again (624 checks, 0 failures each time). Outage: with the Redis container stopped, 8 of 8 requests returned 200 at about 0.11 s each (one 100 ms client timeout; an earlier version also waited on the cache write and took about 0.21 s), and the app recovered within about 40 s of Redis returning, with no restart. The overall health status reports DOWN while Redis is down even though requests succeed; with the layer off (the default) Redis is not part of the health check, and when the layer is switched on this still applies. The latency measured that day is superseded by the next row: its comparison runs without Redis were taken the evening before, and the same configuration moved by up to 8% between days, so the conclusion drawn then (no benefit, and a Redis round trip costing about as much as the PostgreSQL lookup) was not supported by those numbers. |
| Redis cache layer: same-session A/B (October 17) | The layer was switched off with `recs.list-cache-enabled=false` (checked: 624 parity checks passed and Redis held 0 keys afterwards), measured, then switched on and measured with a cold and a warm cache, then switched off again and measured a second time, so the two off runs show how far the same configuration drifts within one session. Same method and workloads as above. Overall median / p95 / p99 in ms, one client. Uniform: off 2.027 / 2.769 / 3.199, then 2.018 / 2.794 / 3.312; on, warm (hit ratio about 89.5%) 1.598 / 2.560 / 3.246; on, cold 3.829 / 5.122 / 5.891. Zipf: off 1.561 / 2.330 / 2.757, then 1.557 / 2.246 / 2.700; on, warm (about 89.7%) 1.566 / 2.481 / 3.154; on, cold 1.724 / 4.117 / 5.112. The two off runs agree within 0.01 ms at the median and within about 0.1 ms in the tails, so larger differences are read as real. Known-customer median: uniform 2.049 and 2.047 off against 1.561 on (0.49 ms faster); zipf 1.550 and 1.544 off against 1.533 on (no difference). Unknown-id median, which always misses in Redis: uniform 1.878 and 1.813 off against 2.401 on; zipf 1.689 and 1.689 off against 2.361 on, so 0.5 to 0.7 ms slower. On uniform traffic the warm layer improved the median by about 0.4 ms and p95 by about 0.2 ms and left p99 unchanged; on zipf traffic it left the median unchanged and made the tail worse (p95 +0.15 to +0.24 ms, p99 about +0.4 ms). An empty cache is expensive: about 1.9 times the off median on uniform traffic. Eight clients: no conclusion. The two off runs differed by 0.45 ms at the median (2.174 and 2.622 ms, about 3,400 and 2,900 requests per second), which is larger than any on/off difference, so no concurrency claim is made; the raw files are kept in `serving/latency_results/`. Reading (inferred from the differences, not measured directly): a Redis lookup costs about 0.6 ms here; a PostgreSQL primary-key lookup costs about the same when its pages are hot (skewed traffic) and about 1.1 ms when they are not (evenly spread traffic). Conclusion: on this machine the layer pays off only for evenly spread traffic, which is the less realistic case for retail, and otherwise costs every unknown-id request and the tail; it is therefore off by default (`recs.list-cache-enabled=false`), and the Redis health check is registered only when the layer is on. Checked on October 17: with the layer on, 624 parity checks passed and Redis held exactly 600 keys; with the default and Redis stopped, health stayed UP with no Redis component and 624 parity checks passed. Caveats: one laptop, everything over loopback with the 2.7-million-row table in a warm local PostgreSQL, so a much larger table or a database on another host could change the answer; a database outage stays hidden while entries are cached; with Redis down each request waits for a 100 ms timeout because there is no circuit breaker, and every failed read logs a warning |
| Fallback list refresh job (October 17) | `refresh_fallback.py --as-of YYYY-MM-DD [--dry-run]` computes the 7-day popularity list for the window ending at `as_of` (that day included) from the train, validation and test transactions and inserts one row into `fallback_lists`, copying `model_version` from the latest row; running the same date again updates that row. It refuses an `as_of` after the last day in the data, one whose window would start before the data, and one older than the row the API serves (the API takes the highest `as_of`, so an older row would be written but never served). Nine Python unit tests (`python -m unittest test_refresh_fallback`, standard library only) cover the window, tie-breaking, the validation rules and the SQL; one checks that `as_of` 2020-09-15 reproduces the stored fallback list exactly. Live check: a dry run for 2020-09-22 kept 7 of the 12 articles; 65 seconds after the real write the API returned the new list for an unknown id, with the same `source` and `modelVersion`; after the 2020-09-22 row was deleted the parity script passed again, including its 20 fallback checks. The 2020-09-22 list uses test-week purchases, so it is a serving demonstration and does not touch any evaluation number. One pair of adjacent weeks says nothing about how fast popularity moves in general, and some of the new entries may be variants of the same product (not checked). The job runs by hand; there is no scheduler. To remove a refreshed row: `DELETE FROM fallback_lists WHERE as_of = DATE '...'` (never the original 2020-09-15 row) |
| Postman collection (October 18) | `serving/postman/coldstart-recs.postman_collection.json`, run in Postman's collection runner against the running app with the default configuration: 10 requests, 25 assertions, 0 failed. It sends a known customer in default, `repurchase` and `discover` mode and compares the exact lists with the export, sends an unknown id in two modes and compares with the fallback list, sends four bad requests (too short, uppercase, leading space, unknown mode) and expects 400, and checks that health reports UP with the database UP. The 503 case (empty fallback table) is not covered here; the controller tests cover it. The expected lists are copied from the current export, so they need updating if the export is rebuilt |

**Not built yet:** any scheduling of the refresh job. Because of the 60-second fallback cache, a refreshed popularity list reaches the API up to a minute late. The endpoint has only been run locally. Run instructions, the response contract and the limitations are collected in `serving/README.md`.

**Running the data layer locally** (Docker Desktop running; the export needs the model artifact from `test_evaluation.py` and the processed data):

```
python export_recommendations.py
cd serving
copy .env.example .env      # then set POSTGRES_PASSWORD in .env (gitignored)
docker compose up -d
docker compose exec postgres psql -U coldstart -d coldstart -f /sql/load.sql
```

**Running the API locally** (containers up, and `POSTGRES_PASSWORD` set in the shell environment from `serving/.env`; the app reads it from the environment and never from a tracked file):

```
# Windows, from the repo root: loads POSTGRES_PASSWORD from serving/.env (never printed), then starts the API;
# extra arguments go to the app, e.g. --recs.list-cache-enabled=true
powershell -NoProfile -ExecutionPolicy Bypass -File .\serving\run-api.ps1

# or by hand:
cd serving/api
./mvnw spring-boot:run        # listens on port 8081
./mvnw spring-boot:run "-Dspring-boot.run.arguments=--recs.list-cache-enabled=true"   # same, with the Redis list cache turned on
curl "http://localhost:8081/v1/recommendations/<customer_id>?mode=repurchase"
./mvnw test "-Dtest=DefaultRecommendationServiceTest,RecommendationControllerContractTest"
```

## Reproducing

Run from the repo root with the processed data in `data/processed/` (output of `prepare_data.py`).

| Script | Purpose |
|---|---|
| `python baseline.py` | popularity and item-item CF baselines, metrics, segments |
| `python model.py` | five-configuration ALS sweep and segment breakdown (`--best-only` fits only the frozen configuration: factors=150, regularization=0.1, alpha=15, 40 iterations) |
| `python cold_start_policy.py` | fallback-window comparison for the 0-purchase segment |
| `python item_item_matched.py` | item-item CF re-scored with the 7-day fallback |
| `python exclusion_comparison.py --factors 150 --regularization 0.1 --iterations 20` | ALS vs item-item vs recency, both exclusion settings, validation week |
| `python bootstrap_comparison.py` | paired-bootstrap intervals, validation week |
| `python convergence_check.py` | training-loss curve and 40-iteration validation check |
| `python test_evaluation.py` | the one-time test-week evaluation; also writes `models/` |
| `python als_sweep.py` | one-knob-at-a-time ALS sensitivity sweep over three earlier weeks (about 1.5 hours; resumable) |
| `python rolling_backtest.py` | weekly backtest over four earlier weeks plus the cached test week (about 1 hour; resumable) |
| `python fallback_diagnostic.py` | popularity-window comparison by week, with overlap against each week's best sellers |
| `python recency_window_recheck.py` | re-scores the recency baseline with a 7-day window from the cached backtest scores |
| `python smoke_test_artifact.py` | reloads the saved artifact and checks that it reproduces the reported test-week MAP@12 under both exclusion settings |
| `python export_recommendations.py` | precomputes both top-12 lists for every customer plus the fallback list into `models/serving_export/` (about 21 minutes; `--limit N --out-dir ...` for a trial run) |
| `python serving/parity_check_api.py` | with the API running, samples rows from the export and checks that the endpoint returns exactly the same lists, plus fallback and 400 cases (about 10 seconds plus the request time) |
| `python serving/latency_baseline.py` | measures endpoint latency (median, p95, p99) over repeated runs with a fixed method and saves the raw numbers to `serving/latency_results/`; `--workload zipf`, `--concurrency N` and `--label` select the variant |
| `python refresh_fallback.py --as-of YYYY-MM-DD` | computes the 7-day popularity list for that day and writes it to `fallback_lists` in the running PostgreSQL container (`--dry-run` only prints it and the checks); needs the containers up |
| `python -m unittest test_refresh_fallback` | the nine tests for the refresh job; the parity test loads the parquet files and takes about 30 seconds |

Several scripts reproduce a prior table as a self-check and print CHECK PASSED or CHECK FAILED; a failed check means the pipeline differs from the one that produced the reported numbers.
