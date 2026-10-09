-- Loads the files written by export_recommendations.py into the tables from schema.sql.
-- Run from serving/ with the containers up (use your POSTGRES_USER and POSTGRES_DB from .env):
--   docker compose exec postgres psql -U coldstart -d coldstart -f /sql/load.sql
-- Safe to re-run: it replaces the table contents inside one transaction, so a failed load leaves
-- the previous data in place. Expect roughly a minute or two for the 2.7 million rows.
\set ON_ERROR_STOP on
\timing on

BEGIN;
TRUNCATE recommendations;
COPY recommendations (customer_id, mode, items, model_version)
    FROM '/import/recommendations.csv' WITH (FORMAT csv, HEADER true);
TRUNCATE fallback_lists;
COPY fallback_lists (source, as_of, window_days, items, model_version)
    FROM '/import/fallback.csv' WITH (FORMAT csv, HEADER true);
COMMIT;

ANALYZE recommendations;

-- Sanity checks: expect 1356709 rows per mode (full export), one fallback row, one model version.
SELECT mode, count(*) AS customers FROM recommendations GROUP BY mode ORDER BY mode;
SELECT count(*) AS fallback_rows FROM fallback_lists;
SELECT model_version, count(*) FROM recommendations GROUP BY model_version;
