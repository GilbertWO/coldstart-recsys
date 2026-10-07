# Serving design (DRAFT, written Oct 7, 2026, before the Week 7-8 block)

Status: nothing in this document is built yet. Read every decision below and rewrite it in
your own words before it goes into the real serving README. Items marked CONFIRM are still
open. The default mode was chosen by Claude on Oct 7 from the evaluation numbers; it is
cheap to reverse (see decision 2).

## What the service does

`GET /recommendations/{customerId}?mode=discover|repurchase` returns 12 article ids for a
customer. Recommendations are computed offline from the saved ALS model and stored in
PostgreSQL; the service looks them up (Redis in front as a cache).

## Decisions

1. Precompute offline, serve lookups. The saved factors are an 885 MB NumPy file that Spring
   Boot cannot read, and scoring 105k articles per request in Java buys nothing. Measured on
   Oct 6: 35 s for 63,412 customers (0.55 ms each); about 13 minutes per list for all
   1,356,709 customers (an estimate for the full run, not measured).
2. Two lists per customer, default `repurchase`. `repurchase` allows articles the customer
   already bought; `discover` removes them. If `mode` is missing the default is used; an
   invalid `mode` value returns 400. The default is a single config value, and both lists
   are precomputed, so changing it later needs no recomputation.
   Why `repurchase`: it is the only mode where ALS beats the 7-day popularity list on the
   primary metric.
   - repurchase: ALS is ahead on MAP@12 in all 5 weeks (+10% to +156% in the four backtest
     weeks, 4 of 5 significant; +0.6%, not significant, on the test week). Its hit rate@12
     is LOWER than the popularity list's in 4 of 5 weeks (5% to 23% lower). It ranks hits
     better but hits fewer customers; this is not a clean win.
   - discover: ALS is behind the 7-day popularity list on MAP@12 in 4 of 5 weeks including
     the test week (18% to 40% lower, all significant) and on hit rate@12 (31% to 46% lower).
     It leads only in the week of 2020-08-12, when the fallback list had gone stale.
   `discover` stays available, but its documentation says ALS does not beat the popularity
   list there. The API never claims the personalized list beats the fallback in general.
3. Cold start is part of the response contract, not an error path:

   | Request | Response | `source` |
   |---|---|---|
   | known customer id | 200, precomputed list for the requested mode | `als` |
   | valid-format id not in the table | 200, fallback list (same for both modes; the response echoes the requested mode) | `fallback_popularity_7d` |
   | malformed id or invalid `mode` | 400 | n/a |

   A new user never gets 404 or 500.
4. Id format. `customer_id` is a 64-character lowercase hex string. Verified on Oct 7 over
   the full customer table: all 1,371,980 ids are unique and match `^[0-9a-f]{64}$`.
   Malformed means "does not match that pattern"; an uppercase id returns 400 and is not
   normalized, so cache keys stay unambiguous. `article_id` is a 32-bit integer in the data
   and a string in the model artifact. Store it as an integer; if the frontend later needs
   product images, format it as a zero-padded 10-digit string at the API boundary only.
   CONFIRM.
5. The fallback list is refreshed, not hard-coded. The evaluation showed a stale list can
   miss every one of a week's 12 best sellers (week of 2020-08-12). It is recomputed by a job
   that takes an `as_of` date; its Redis TTL is short. TTL value: CONFIRM.
6. Redis: cache-aside, key `rec:{mode}:{customerId}`, with negative caching for unknown ids so
   unknown-id floods do not reach PostgreSQL. TTL values: CONFIRM and justify.
7. Every response carries `modelVersion` (config and history end date from the artifact's
   metadata), so a stale table is visible.

## Known limits

- The artifact covers 1,356,709 customers and 103,880 articles; the data has 1,371,980
  customers and 105,542 articles. About 15,300 customers have no embedding (consistent with
  having no purchases before the history ended on 2020-09-15) and take the fallback path.
  About 1,660 articles had no purchases in the history window and can never be recommended;
  this is item cold start and is not solved here.
- Evaluation numbers come from one test week plus a four-fold rolling backtest, on
  customer-sampling noise only.
- In the evaluation, no model beats the 7-day popularity list on both MAP@12 and hit rate@12
  in either mode. The serving layer is the deliverable of this block; model quality is
  reported, not oversold.

## Environment verified

Oct 6, 2026: Windows, WSL 3.0.1 with a WSL2 backend, Docker 29.8.2; `postgres:16` answered a
query and `redis:7` answered PONG in throwaway containers. Java 21.0.2 LTS present; Maven or
Gradle come from the project wrapper.

## How to run

TODO after the service exists (docker-compose up, load script, start Spring Boot, example
requests).
