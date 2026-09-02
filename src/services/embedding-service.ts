import { createHash, createHmac } from 'node:crypto';

import type { Pool } from 'pg';

import { AppError, ErrorCode } from '../util/errors.js';
import type { EmbeddingProvider } from './embeddings/providers.js';

const DEFAULT_DIMENSIONS = 1536;
const QUERY_EMBEDDING_CACHE_SIZE = 512;
const QUERY_EMBEDDING_CACHE_MAX_ENTRIES_PER_SCOPE = 2_000;
const ACTIVE_MODEL_TTL_MS = 60 * 1000;

type EmbeddingMode = 'deterministic' | 'provider';

type ActiveModelRow = {
  id: string;
  name: string;
  provider: string;
  dimensions: number;
  chunk_size: number;
  chunk_overlap: number;
  metadata: Record<string, unknown>;
  created_at: Date;
};

export type ActiveEmbeddingModel = {
  id: string;
  name: string;
  provider: string;
  dimensions: number;
  chunkSize: number;
  chunkOverlap: number;
  metadata: Record<string, unknown>;
  createdAt: string;
};

type EmbeddingServiceOptions = {
  mode?: EmbeddingMode | undefined;
  provider?: EmbeddingProvider | undefined;
  embedBatch?:
    | ((texts: string[], model?: ActiveEmbeddingModel) => Promise<number[][]>)
    | undefined;
  embedQuery?:
    | ((text: string, model?: ActiveEmbeddingModel) => Promise<number[]>)
    | undefined;
  queryCacheMaxSize?: number | undefined;
  queryCacheSecret?: string | Buffer | undefined;
  activeModelTtlMs?: number | undefined;
  now?: (() => number) | undefined;
};

export type QueryEmbeddingCacheStatus =
  | 'bypass'
  | 'memory_hit'
  | 'database_hit'
  | 'miss';

export type QueryEmbeddingOptions = {
  /**
   * When supplied, query embeddings are read from and written to the
   * `query_embedding_cache` table so the cache survives restarts and is shared
   * across processes. Without it only the in-process cache is consulted.
   */
  pool?: Pool | undefined;
  /**
   * Client the query belongs to. Cache entries are partitioned by it so one
   * client cannot detect another client's queries by timing a cache hit.
   * Without it the query is not cached at all.
   */
  cacheScope?: string | undefined;
  onCacheStatus?: ((status: QueryEmbeddingCacheStatus) => void) | undefined;
};

export type EmbeddingService = ReturnType<typeof createEmbeddingService>;

/**
 * Keys cache entries by a one-way digest of the query text so the cache never
 * retains plaintext queries — neither in memory nor in Postgres.
 *
 * With a secret this is a keyed HMAC, which prevents someone holding a copy of
 * the table from dictionary-testing guessed queries. The secret must come from
 * configuration rather than the database: a key stored next to the digests it
 * protects would defeat exactly that threat.
 *
 * Without a secret it is an unkeyed sha256 and offers no protection against
 * that offline attack — an accepted default, since a reader of this table can
 * already read every entity in the corpus. It still keeps plaintext queries out
 * of the database at rest.
 */
export function createQueryEmbeddingCacheKey(
  text: string,
  secret?: string | Buffer
): Buffer {
  if (secret !== undefined && secret.length > 0) {
    return createHmac('sha256', secret).update(text, 'utf8').digest();
  }
  return createHash('sha256').update(text, 'utf8').digest();
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

function hashToken(token: string): number {
  let hash = 0;
  for (const character of token) {
    hash = (hash * 31 + character.charCodeAt(0)) % DEFAULT_DIMENSIONS;
  }
  return Math.abs(hash);
}

function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(
    vector.reduce((sum, value) => sum + value ** 2, 0)
  );

  if (magnitude === 0) {
    return vector;
  }

  return vector.map((value) => value / magnitude);
}

function embedDeterministically(text: string): number[] {
  const vector = new Array<number>(DEFAULT_DIMENSIONS).fill(0);

  for (const token of tokenize(text)) {
    const index = hashToken(token);
    vector[index] = (vector[index] ?? 0) + 1;
  }

  return normalizeVector(vector);
}

function resolveDefaultMode(options: EmbeddingServiceOptions): EmbeddingMode {
  if (options.provider) {
    return 'provider';
  }

  if (process.env.NODE_ENV === 'test' || process.env.VITEST) {
    return 'deterministic';
  }

  return 'provider';
}

function mapActiveModel(row: ActiveModelRow): ActiveEmbeddingModel {
  return {
    id: row.id,
    name: row.name,
    provider: row.provider,
    dimensions: row.dimensions,
    chunkSize: row.chunk_size,
    chunkOverlap: row.chunk_overlap,
    metadata: row.metadata,
    createdAt: row.created_at.toISOString()
  };
}

export function vectorToSql(vector: number[]): string {
  return `[${vector.join(',')}]`;
}

/**
 * pgvector columns come back from node-postgres as their text representation
 * (`[0.1,0.2,...]`). Returns null for anything that does not parse cleanly so a
 * corrupt row degrades to a cache miss rather than poisoning search results.
 */
export function sqlToVector(value: unknown): number[] | null {
  if (typeof value !== 'string') {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) {
    return null;
  }

  const body = trimmed.slice(1, -1);
  if (body.length === 0) {
    return null;
  }

  const parsed: number[] = [];
  for (const part of body.split(',')) {
    const component = Number(part);
    if (!Number.isFinite(component)) {
      return null;
    }
    parsed.push(component);
  }

  return parsed;
}

/**
 * Deletes cache entries older than `maxAgeDays`. Age-based rather than
 * usage-based eviction is deliberate: tracking last-used timestamps would turn
 * every cache read into a write. Re-embedding a still-hot query once per
 * retention window is far cheaper than that.
 */
export async function pruneQueryEmbeddingCache(
  pool: Pool,
  maxAgeDays: number,
  maxEntriesPerScope = QUERY_EMBEDDING_CACHE_MAX_ENTRIES_PER_SCOPE
): Promise<number> {
  const expired = await pool.query(
    `DELETE FROM query_embedding_cache
     WHERE created_at < now() - ($1::integer * interval '1 day')`,
    [maxAgeDays]
  );
  const overflow = await pool.query(
    `WITH ranked AS MATERIALIZED (
       SELECT
         client_id,
         model_id,
         query_hash,
         ROW_NUMBER() OVER (
           PARTITION BY client_id
           ORDER BY created_at DESC, model_id, query_hash
         ) AS entry_rank
       FROM query_embedding_cache
     )
     DELETE FROM query_embedding_cache AS cached
     USING ranked
     WHERE ranked.entry_rank > $1::integer
       AND cached.client_id = ranked.client_id
       AND cached.model_id = ranked.model_id
       AND cached.query_hash = ranked.query_hash`,
    [maxEntriesPerScope]
  );
  return (expired.rowCount ?? 0) + (overflow.rowCount ?? 0);
}

export function createEmbeddingService(options: EmbeddingServiceOptions = {}) {
  const mode = options.mode ?? resolveDefaultMode(options);

  const embedBatchImpl =
    options.embedBatch ??
    (async (texts: string[], model?: ActiveEmbeddingModel) => {
      if (mode === 'deterministic') {
        return texts.map((text) => embedDeterministically(text));
      }

      if (!options.provider) {
        throw new AppError(
          ErrorCode.EMBEDDING_FAILED,
          'No embedding provider configured'
        );
      }

      if (model && model.dimensions !== options.provider.dimensions) {
        throw new AppError(
          ErrorCode.EMBEDDING_FAILED,
          'Active model dimensions do not match provider dimensions',
          {
            activeModel: model.dimensions,
            provider: options.provider.dimensions
          }
        );
      }

      return options.provider.embedBatch(texts);
    });

  const embedQueryImpl =
    options.embedQuery ??
    (async (text: string, model?: ActiveEmbeddingModel) => {
      const [vector] = await embedBatchImpl([text], model);
      if (!vector) {
        throw new AppError(
          ErrorCode.EMBEDDING_FAILED,
          'Failed to embed query text'
        );
      }
      return vector;
    });

  // L1: resolved embeddings, insertion-ordered for LRU eviction. There is no
  // TTL — an embedding is a pure function of (model, text), so an entry can
  // only be evicted for space, never for staleness.
  const memoryCache = new Map<string, number[]>();
  // Coalesces concurrent requests for the same query onto one provider call.
  const inFlight = new Map<string, Promise<number[]>>();
  const pendingWrites = new Set<Promise<void>>();
  const queryCacheMaxSize =
    options.queryCacheMaxSize ?? QUERY_EMBEDDING_CACHE_SIZE;
  const queryCacheSecret = options.queryCacheSecret;
  const activeModelTtlMs = options.activeModelTtlMs ?? ACTIVE_MODEL_TTL_MS;
  const now = options.now ?? Date.now;

  let activeModelCache: { model: ActiveEmbeddingModel; expiresAt: number } | null =
    null;

  function rememberInMemory(cacheKey: string, embedding: number[]): void {
    memoryCache.delete(cacheKey);
    memoryCache.set(cacheKey, embedding);

    while (memoryCache.size > queryCacheMaxSize) {
      const oldestKey = memoryCache.keys().next().value;
      if (oldestKey === undefined) {
        break;
      }
      memoryCache.delete(oldestKey);
    }
  }

  async function readFromDatabase(
    pool: Pool,
    cacheScope: string,
    model: ActiveEmbeddingModel,
    queryHash: Buffer
  ): Promise<number[] | null> {
    const result = await pool.query<{ embedding: unknown }>(
      `SELECT embedding
       FROM query_embedding_cache
       WHERE client_id = $1 AND model_id = $2 AND query_hash = $3`,
      [cacheScope, model.id, queryHash]
    );

    const embedding = sqlToVector(result.rows[0]?.embedding);
    if (!embedding) {
      return null;
    }

    // Guard against a model row being mutated in place under a stable id.
    if (embedding.length !== model.dimensions) {
      return null;
    }

    return embedding;
  }

  function writeToDatabase(
    pool: Pool,
    cacheScope: string,
    model: ActiveEmbeddingModel,
    queryHash: Buffer,
    embedding: number[]
  ): void {
    // Deliberately not awaited by the caller: the row is a pure optimisation,
    // and on a contended host waiting for the INSERT would add the very
    // latency this cache exists to remove. Tracked so tests and shutdown can
    // drain it.
    const write = pool
      .query(
        `INSERT INTO query_embedding_cache
           (client_id, model_id, query_hash, embedding)
         VALUES ($1, $2, $3, $4::vector)
         ON CONFLICT (client_id, model_id, query_hash) DO NOTHING`,
        [cacheScope, model.id, queryHash, vectorToSql(embedding)]
      )
      .then(
        () => undefined,
        () => undefined
      )
      .finally(() => {
        pendingWrites.delete(write);
      });
    pendingWrites.add(write);
  }

  function embedCachedQuery(
    text: string,
    model?: ActiveEmbeddingModel,
    queryOptions: QueryEmbeddingOptions = {}
  ): Promise<number[]> {
    const cacheScope = queryOptions.cacheScope;
    // Without both a model to key on and a client to partition by, caching
    // would either be incorrect or would pool entries across clients.
    if (!model || !cacheScope) {
      queryOptions.onCacheStatus?.('bypass');
      return embedQueryImpl(text, model);
    }

    const queryHash = createQueryEmbeddingCacheKey(text, queryCacheSecret);
    const cacheKey = `${cacheScope}:${model.id}:${queryHash.toString('hex')}`;

    const cached = memoryCache.get(cacheKey);
    if (cached) {
      rememberInMemory(cacheKey, cached);
      queryOptions.onCacheStatus?.('memory_hit');
      return Promise.resolve(cached);
    }

    const existing = inFlight.get(cacheKey);
    if (existing) {
      queryOptions.onCacheStatus?.('memory_hit');
      return existing;
    }

    const { pool } = queryOptions;
    const resolution = (async () => {
      if (pool) {
        // A failed cache read must not fail the search.
        const stored = await readFromDatabase(
          pool,
          cacheScope,
          model,
          queryHash
        ).catch(() => null);
        if (stored) {
          rememberInMemory(cacheKey, stored);
          queryOptions.onCacheStatus?.('database_hit');
          return stored;
        }
      }

      queryOptions.onCacheStatus?.('miss');
      const embedding = await embedQueryImpl(text, model);
      rememberInMemory(cacheKey, embedding);
      if (pool) {
        writeToDatabase(pool, cacheScope, model, queryHash, embedding);
      }
      return embedding;
    })();

    inFlight.set(cacheKey, resolution);
    // Always clear the in-flight entry: on success the value now lives in the
    // memory cache, and on failure the next caller must be free to retry.
    void resolution.then(
      () => inFlight.delete(cacheKey),
      () => inFlight.delete(cacheKey)
    );

    return resolution;
  }

  return {
    dimensions: options.provider?.dimensions ?? DEFAULT_DIMENSIONS,
    async embedBatch(
      texts: string[],
      model?: ActiveEmbeddingModel
    ): Promise<number[][]> {
      return embedBatchImpl(texts, model);
    },
    async embedQuery(
      text: string,
      model?: ActiveEmbeddingModel,
      queryOptions?: QueryEmbeddingOptions
    ): Promise<number[]> {
      return embedCachedQuery(text, model, queryOptions);
    },
    /** Resolves once every write-behind cache INSERT has settled. */
    async flushPendingWrites(): Promise<void> {
      await Promise.all(Array.from(pendingWrites));
    },
    /** Drops the memoized active model, forcing the next read to hit the database. */
    invalidateActiveModel(): void {
      activeModelCache = null;
    },
    /**
     * Memoized variant for read paths, where this lookup used to add a database
     * round trip to the critical path of every search.
     *
     * Deliberately NOT used by writers. A writer that acts on a stale model
     * would attach chunks to the wrong (or a deleted) `embedding_models` row;
     * a reader that does so at worst searches against a model that changed
     * moments ago, and the query embedding cache is keyed by model id so no
     * mismatched vector can be served. Background writes pay the extra SELECT.
     */
    async getActiveModelForQuery(pool: Pool): Promise<ActiveEmbeddingModel> {
      if (activeModelCache && activeModelCache.expiresAt > now()) {
        return activeModelCache.model;
      }

      const model = await this.getActiveModel(pool);
      activeModelCache = {
        model,
        expiresAt: now() + activeModelTtlMs
      };
      return model;
    },
    async getActiveModel(pool: Pool): Promise<ActiveEmbeddingModel> {
      const result = await pool.query<ActiveModelRow>(
        `
          SELECT
            id,
            name,
            provider,
            dimensions,
            chunk_size,
            chunk_overlap,
            metadata,
            created_at
          FROM embedding_models
          WHERE is_active = true
          LIMIT 1
        `
      );

      const model = result.rows[0];
      if (!model) {
        throw new AppError(
          ErrorCode.INTERNAL,
          'No active embedding model configured'
        );
      }

      return mapActiveModel(model);
    }
  };
}
