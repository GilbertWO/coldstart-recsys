-- Schema for the precomputed recommendation lookups (see serving-design-draft.md).
-- Created automatically on the first start of the postgres container.

-- One row per customer per mode: the 12 article ids, best first, written by
-- export_recommendations.py. (customer_id, mode) is the lookup key the API uses.
CREATE TABLE recommendations (
    customer_id   text     NOT NULL CHECK (customer_id ~ '^[0-9a-f]{64}$'),
    mode          text     NOT NULL CHECK (mode IN ('repurchase', 'discover')),
    items         integer[] NOT NULL CHECK (cardinality(items) = 12),
    model_version text     NOT NULL,
    PRIMARY KEY (customer_id, mode)
);

-- The popularity list served for unknown customers. Kept as rows keyed by as_of so the refresh
-- job (Week B) can add a newer list without touching the model's lists; the API reads the latest.
CREATE TABLE fallback_lists (
    source        text     NOT NULL,
    as_of         date     NOT NULL,
    window_days   integer  NOT NULL CHECK (window_days > 0),
    items         integer[] NOT NULL CHECK (cardinality(items) = 12),
    model_version text     NOT NULL,
    PRIMARY KEY (source, as_of)
);
