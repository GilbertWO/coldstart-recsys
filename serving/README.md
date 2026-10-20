# Serving layer: Spring Boot + PostgreSQL

This folder serves precomputed recommendations from the ALS model over HTTP. Nothing is scored at request time. `export_recommendations.py` computes two top-12 lists per customer offline, `db/load.sql` loads them into PostgreSQL, and the API looks them up.

## The endpoint

```
GET /v1/recommendations/{customerId}?mode=repurchase|discover
```

`customerId` is a 64-character lowercase hex string. `mode` is optional and defaults to `repurchase`. The app listens on port 8081.

| Request | Response | `source` |
|---|---|---|
| known customer | 200, precomputed list for the mode | `als` |
| well-formed id not in the table | 200, latest popularity list, requested mode echoed | `fallback_popularity_7d` |
| malformed id or unknown `mode` | 400 | not applicable |
| fallback table empty (a server fault, not a user error) | 503 | not applicable |

A malformed id is anything other than 64 lowercase hex characters. Uppercase ids and ids with spaces are rejected, not normalized, so a cache key can only mean one customer.

The body has `customerId`, `mode`, `source`, `modelVersion` and `items` (12 integer article ids, best first). A new customer never gets a 404 or 500 from the lookup path. On a fallback response, `modelVersion` is the model run that produced the batch and `source` says which method produced the list.

`repurchase` is the default because of the evaluation (root README, "Two modes"). With already-bought items allowed, ALS leads the 7-day popularity list on MAP@12 in all five evaluation weeks (significant in four), but its hit rate@12 is lower than the popularity list's in four of the five. With already-bought items removed (`discover`), ALS trails the popularity list on both metrics in four of five weeks. The endpoint serves `discover` anyway; it is not the default.

## Components

PostgreSQL 16 holds two tables, defined in `db/schema.sql`: `recommendations` (primary key customer and mode, 12 ids each) and `fallback_lists` (primary key source and `as_of` date). The API reads the fallback list with the highest `as_of`. Redis 7 runs in the same compose file and is used only when the list cache is switched on. Both containers listen on 127.0.0.1 only.

Two caching layers sit in front of the database as wrappers around the repository. The fallback list is cached in process memory for 60 seconds (`recs.fallback-cache-ttl`), always. Per-customer lists can be cached in Redis under the key `rec:{mode}:{customerId}` for one hour (`recs.list-cache-ttl`); that layer is off by default.

## Run it

You need Docker Desktop, Java 21 (the Maven wrapper is in `serving/api`) and the model export. From the repo root:

```
python export_recommendations.py          # about 21 minutes, writes models/serving_export/
cd serving
copy .env.example .env                    # set POSTGRES_PASSWORD in .env; the file is gitignored
docker compose up -d                      # creates the tables on first start only
docker compose exec postgres psql -U coldstart -d coldstart -f /sql/load.sql
cd ..
powershell -NoProfile -ExecutionPolicy Bypass -File .\serving\run-api.ps1
```

`run-api.ps1` reads `serving/.env`, puts `POSTGRES_PASSWORD` into the environment without printing it, and starts the app, which reads the password from that variable. Start the API this way. A bare `mvnw spring-boot:run` has no password, and the health check then reports the database as DOWN. To check a running app: `curl.exe -s http://localhost:8081/actuator/health`. With the default configuration it returns `UP` with a `db` component and no `redis` component.

Two cautions. `load.sql` replaces the contents of both tables, so re-running it also deletes any fallback rows added by the refresh job (the original 2020-09-15 row comes back from the export). And `docker compose down -v` wipes the database, while plain `docker compose down` keeps it.

## The Redis list cache

Off by default. Switch it on with `--recs.list-cache-enabled=true`, for example `.\serving\run-api.ps1 --recs.list-cache-enabled=true`. The Redis health check is registered only when the cache is on.

It is off because of the same-session comparison on October 17 (root README, "Redis cache layer: same-session A/B"). With one client, the layer made known customers about 0.5 ms faster at the median on evenly spread traffic and made no difference on skewed traffic. It added 0.5 to 0.7 ms to every request for an unknown id, made the tail worse on skewed traffic, and an empty cache cost about 1.9 times the median without it. These are numbers from one laptop over loopback. The eight-client runs could not be interpreted, because the same configuration differed by 0.45 ms between two runs, so there is no concurrency claim.

## Refreshing the fallback list

```
python refresh_fallback.py --as-of 2020-09-15 --dry-run    # prints the list and the checks, writes nothing
python refresh_fallback.py --as-of 2020-09-15              # writes one row to fallback_lists
```

Run it from the repo root with the project's virtual-environment Python and the containers up. The list is the 12 most-bought articles in the 7 days ending on `as_of`, that day included, with ties broken by the lower article id. The job refuses an `as_of` after the last day in the data, one whose window would start before the data, and one older than the row the API currently serves. The API takes the highest `as_of`, so an older row would be written but never served, and for the same reason a wrongly future-dated row would block every later refresh. Running the same date twice updates the row. Because of the 60-second cache, a new row reaches the API up to about a minute later.

The job runs by hand and nothing schedules it. To remove a refreshed row, run `DELETE FROM fallback_lists WHERE as_of = DATE '...'` and never delete the original 2020-09-15 row.

## Tests

| Layer | Command | What it covers |
|---|---|---|
| Java | `.\mvnw test` in `serving/api` | 33 tests (32 plus Spring's context-load test): controller contract, service logic, both cache layers, wiring with the Redis layer on, off and unset, and the Redis health switch |
| Python | `.\venv\Scripts\python.exe -m unittest test_refresh_fallback` from the repo root | 9 tests of the refresh job, including that 2020-09-15 reproduces the stored fallback list exactly (about 40 seconds) |
| Parity | `python serving/parity_check_api.py` with the API running | 600 sampled rows against the export, 20 unknown ids and 4 malformed requests: 624 checks, 0 failures |
| Postman | `serving/postman/coldstart-recs.postman_collection.json` | 10 requests, 25 assertions, run in Postman's collection runner with the API running |

The Postman collection sends a known customer in default, `repurchase` and `discover` mode and checks the exact lists, sends an unknown id in two modes and checks the fallback list, sends four bad requests (too short, uppercase, leading space, unknown mode) and checks the health endpoint. On October 18 against the running app it passed: 10 requests, 25 assertions, 0 failed. It does not cover the 503 case, which only the controller tests cover. Its variables (`knownId` and the two expected lists) were copied from the current export, so update them if the export is rebuilt.

## When the data is stale

The service never checks how old its data is. It returns whatever the tables hold, and the only data fault it reports is an empty fallback table (503). Two things can be old, and they show up in different places.

The personalized lists carry a `modelVersion` in every response, for example `als-f150-r0.1-a15-i40-h2020-09-15`. `export_recommendations.py` builds it from the model's settings: 150 factors, regularization 0.1, alpha 15, 40 iterations, and `h`, the last day of purchase history the model was trained on. Purchases after that day have no effect on any list. To see what the table holds:

```
docker compose exec postgres psql -U coldstart -d coldstart -c "SELECT model_version, count(*) FROM recommendations GROUP BY model_version;"
```

On October 20 this returned one version for all 2,713,418 rows (both modes for 1,356,709 customers), with `h2020-09-15`. A table is stale when that date is well behind the newest transactions you have.

The fallback list is different: its age is not visible in a response. The refresh job copies `model_version` from the latest row, so a freshly refreshed list and the original one report the same `modelVersion`, and the API does not select `as_of` at all. The only place to see it is the table:

```
docker compose exec postgres psql -U coldstart -d coldstart -c "SELECT source, as_of, window_days, model_version FROM fallback_lists ORDER BY as_of DESC;"
```

On October 20 this returned one row, `as_of` 2020-09-15 with a 7-day window. This matters because a stale popularity list can miss most of a week's best sellers; in the evaluation, the week of 2020-08-12 was one where the list overlapped none of them (root README). Exposing `as_of` in a response or in the health details would close this gap; it has not been built.

## Limitations

- Everything ran on one machine over loopback against a warm local PostgreSQL. A larger table or a database on another host could change every latency number.
- With the Redis layer on, a database outage stays hidden while entries are cached, health reports DOWN while Redis is down even though requests succeed, each request waits for a 100 ms timeout while Redis is down (there is no circuit breaker), and each failed read logs a warning.
- The refresh job is manual.
- The service does not detect stale data, and the age of the fallback list is not visible in a response (see "When the data is stale").
- The 2020-09-22 row used to check the refresh job end to end contained test-week purchases. It was a serving demonstration and was deleted afterwards; no evaluation number depends on it.
- About 15,300 customers have no embedding and always take the fallback path. About 1,660 articles can never be recommended; item cold start is not solved here.
- The API has been run only locally. It has no authentication, no rate limiting and no container image.
- `serving-design-draft.md` is the plan written before the build. Where it differs from this file or the code (it omits the `/v1` prefix and describes negative caching of unknown ids in Redis, which was not built), this file and the code are correct.
