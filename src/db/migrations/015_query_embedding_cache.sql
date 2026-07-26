-- Cache for query embeddings.
--
-- A query embedding is a pure function of (active model, query text), so it is
-- safe to memoize indefinitely. Persisting it in Postgres means the cache
-- survives restarts and is shared by every process pointed at this database,
-- which matters on small hosts where the embedding round trip dominates search
-- latency.
--
-- The query text itself is deliberately NOT stored: entries are keyed by a
-- sha256 of the text so user queries are not retained at rest.
--
-- `embedding` is an unmodified `vector` rather than `vector(1536)` so the table
-- tolerates whichever dimensionality the active model uses. Lookups go through
-- the primary key, so no ANN index (and therefore no fixed dimension) is needed.

CREATE TABLE query_embedding_cache (
  model_id uuid NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  query_hash bytea NOT NULL,
  embedding vector NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_id, query_hash)
);

-- Supports age-based pruning. There is deliberately no last_used_at/hit_count:
-- maintaining those would turn every cache read into a write, which is exactly
-- the cost this table exists to avoid.
CREATE INDEX idx_query_embedding_cache_created_at
  ON query_embedding_cache (created_at);
