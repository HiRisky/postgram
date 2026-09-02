-- Supports bounded per-client eviction after query embedding cache writes.
CREATE INDEX idx_query_embedding_cache_client_created_at
  ON query_embedding_cache (client_id, created_at DESC, model_id, query_hash);
