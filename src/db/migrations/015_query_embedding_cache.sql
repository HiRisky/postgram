-- Cache for query embeddings.
--
-- A query embedding is a pure function of (active model, query text), so it is
-- safe to memoize indefinitely. Persisting it in Postgres means the cache
-- survives restarts and is shared by every process pointed at this database,
-- which matters on small hosts where the embedding round trip dominates search
-- latency.
--
-- The query text itself is deliberately NOT stored: entries are keyed by a
-- digest of the text so user queries are not retained at rest.
--
-- Entries are scoped by client_id as well as model. Without that scope every
-- client shares one cache, and a client can probe a guessed query and tell a
-- cache hit from a provider-backed miss by latency alone — an information leak
-- that does not depend on result authorization, which happens later. Scoping
-- costs little in practice: the dominant hit is an agent repeating its own
-- query, which stays within one client.
--
-- The digest is a keyed HMAC when QUERY_EMBEDDING_CACHE_SECRET is configured.
-- Without it the digest is an unkeyed sha256, which someone with read access to
-- this table can dictionary-test to confirm whether a guessed query was run.
-- That is documented and accepted by default: such a reader already has every
-- entity in the corpus. Operators who consider query text more sensitive than
-- entity content should set the secret, which must live outside the database
-- for the keying to mean anything.
--
-- `embedding` is an unmodified `vector` rather than `vector(1536)` so the table
-- tolerates whichever dimensionality the active model uses. Lookups go through
-- the primary key, so no ANN index (and therefore no fixed dimension) is needed.

CREATE TABLE query_embedding_cache (
  client_id text NOT NULL,
  model_id uuid NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
  query_hash bytea NOT NULL,
  embedding vector NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (client_id, model_id, query_hash)
);

-- Supports age-based pruning. There is deliberately no last_used_at/hit_count:
-- maintaining those would turn every cache read into a write, which is exactly
-- the cost this table exists to avoid.
CREATE INDEX idx_query_embedding_cache_created_at
  ON query_embedding_cache (created_at);
