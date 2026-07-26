import { describe, expect, it, vi } from 'vitest';

import type { Pool } from 'pg';

import {
  createEmbeddingService,
  createQueryEmbeddingCacheKey
} from '../../src/services/embedding-service.js';
import type { EmbeddingProvider } from '../../src/services/embeddings/providers.js';

type ProviderWithMocks = EmbeddingProvider & {
  embedBatchMock: ReturnType<typeof vi.fn>;
  embedMock: ReturnType<typeof vi.fn>;
};

function makeProvider(): ProviderWithMocks {
  const embedBatchMock = vi.fn().mockResolvedValue([[0.25, 0.75]]);
  const embedMock = vi.fn().mockResolvedValue([0.25, 0.75]);
  return {
    name: 'openai',
    model: 'text-embedding-3-small',
    dimensions: 2,
    embed: embedMock as unknown as EmbeddingProvider['embed'],
    embedBatch: embedBatchMock as unknown as EmbeddingProvider['embedBatch'],
    embedBatchMock,
    embedMock
  };
}

/**
 * Minimal stand-in for the `query_embedding_cache` table, keyed the same way
 * the real primary key is: (model_id, query_hash).
 */
function makeCachePool(stored: Map<string, string>): Pool {
  return {
    query: vi.fn((text: string, values: unknown[]) => {
      const modelId = values[0] as string;
      const hash = (values[1] as Buffer).toString('hex');
      const key = `${modelId}:${hash}`;

      if (text.includes('INSERT INTO query_embedding_cache')) {
        if (!stored.has(key)) {
          stored.set(key, values[2] as string);
        }
        return Promise.resolve({ rows: [], rowCount: 1 });
      }

      const embedding = stored.get(key);
      return Promise.resolve({
        rows: embedding ? [{ embedding }] : [],
        rowCount: 0
      });
    })
  } as unknown as Pool;
}

describe('embedding-service', () => {
  const activeModel = {
    id: 'model-1',
    name: 'text-embedding-3-small',
    provider: 'openai',
    dimensions: 2,
    chunkSize: 300,
    chunkOverlap: 100,
    metadata: {},
    createdAt: new Date().toISOString()
  };

  it('delegates embedBatch to the injected provider when given one', async () => {
    const provider = makeProvider();

    const service = createEmbeddingService({ provider });

    const vectors = await service.embedBatch(['hello world'], {
      id: 'model-1',
      name: 'text-embedding-3-small',
      provider: 'openai',
      dimensions: 2,
      chunkSize: 300,
      chunkOverlap: 100,
      metadata: {},
      createdAt: new Date().toISOString()
    });

    expect(provider.embedBatchMock).toHaveBeenCalledWith(['hello world']);
    expect(vectors).toEqual([[0.25, 0.75]]);
  });

  it('uses deterministic mode by default under vitest', async () => {
    const service = createEmbeddingService();
    const vectors = await service.embedBatch(['hello']);
    expect(vectors).toHaveLength(1);
    expect(vectors[0]).toHaveLength(1536);
  });

  it('rejects when the active model dimensions disagree with the provider', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await expect(
      service.embedBatch(['hi'], {
        id: 'model-1',
        name: 'text-embedding-3-small',
        provider: 'openai',
        dimensions: 1536,
        chunkSize: 300,
        chunkOverlap: 100,
        metadata: {},
        createdAt: new Date().toISOString()
      })
    ).rejects.toMatchObject({
      message: 'Active model dimensions do not match provider dimensions'
    });
  });

  it('reuses query embeddings for the same text and active model', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    const [first, second] = await Promise.all([
      service.embedQuery('postgres search', activeModel),
      service.embedQuery('postgres search', activeModel)
    ]);
    const third = await service.embedQuery('postgres search', activeModel);

    expect(first).toEqual([0.25, 0.75]);
    expect(second).toEqual(first);
    expect(third).toEqual(first);
    expect(provider.embedBatchMock).toHaveBeenCalledTimes(1);
  });

  it('does not reuse query embeddings across active models', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await service.embedQuery('postgres search', activeModel);
    await service.embedQuery('postgres search', {
      ...activeModel,
      id: 'model-2'
    });

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('retries query embeddings after a provider request fails', async () => {
    const provider = makeProvider();
    provider.embedBatchMock
      .mockRejectedValueOnce(new Error('provider unavailable'))
      .mockResolvedValueOnce([[0.25, 0.75]]);
    const service = createEmbeddingService({ provider });

    await expect(
      service.embedQuery('postgres search', activeModel)
    ).rejects.toThrow('provider unavailable');
    await expect(
      service.embedQuery('postgres search', activeModel)
    ).resolves.toEqual([0.25, 0.75]);

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('bypasses the cache when there is no active model to key on', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await service.embedQuery('private roadmap');
    await service.embedQuery('private roadmap');

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('evicts the least recently used query when the cache is full', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({
      provider,
      queryCacheMaxSize: 2
    });

    await service.embedQuery('query one', activeModel);
    await service.embedQuery('query two', activeModel);
    await service.embedQuery('query one', activeModel);
    await service.embedQuery('query three', activeModel);
    await service.embedQuery('query two', activeModel);

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(4);
  });

  it('serves a repeat query from the database cache in a fresh process', async () => {
    const provider = makeProvider();
    const stored = new Map<string, string>();
    const pool = makeCachePool(stored);

    const first = createEmbeddingService({ provider });
    await first.embedQuery('postgres search', activeModel, { pool });
    await first.flushPendingWrites();

    // A second service instance stands in for a restarted process: its
    // in-memory cache is empty, so a hit can only come from Postgres.
    const second = createEmbeddingService({ provider });
    const statuses: string[] = [];
    const result = await second.embedQuery('postgres search', activeModel, {
      pool,
      onCacheStatus: (status) => statuses.push(status)
    });

    expect(result).toEqual([0.25, 0.75]);
    expect(statuses).toEqual(['database_hit']);
    expect(provider.embedBatchMock).toHaveBeenCalledTimes(1);
  });

  it('treats a cached vector of the wrong dimension as a miss', async () => {
    const provider = makeProvider();
    const stored = new Map<string, string>();
    const pool = makeCachePool(stored);

    const first = createEmbeddingService({ provider });
    await first.embedQuery('postgres search', activeModel, { pool });
    await first.flushPendingWrites();

    // Simulate the model row being mutated in place under a stable id.
    for (const key of stored.keys()) {
      stored.set(key, '[0.1,0.2,0.3,0.4]');
    }

    const second = createEmbeddingService({ provider });
    const statuses: string[] = [];
    await second.embedQuery('postgres search', activeModel, {
      pool,
      onCacheStatus: (status) => statuses.push(status)
    });

    expect(statuses).toEqual(['miss']);
    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('still returns an embedding when the database cache is unreadable', async () => {
    const provider = makeProvider();
    const pool = {
      query: vi.fn().mockRejectedValue(new Error('cache table missing'))
    } as unknown as Pool;

    const service = createEmbeddingService({ provider });

    await expect(
      service.embedQuery('postgres search', activeModel, { pool })
    ).resolves.toEqual([0.25, 0.75]);
  });

  it('reports memory hits, database hits, misses and bypasses', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });
    const statuses: string[] = [];
    const onCacheStatus = (status: string) => statuses.push(status);

    await service.embedQuery('private roadmap', activeModel, { onCacheStatus });
    await service.embedQuery('private roadmap', activeModel, { onCacheStatus });
    await service.embedQuery('private roadmap', undefined, { onCacheStatus });

    expect(statuses).toEqual(['miss', 'memory_hit', 'bypass']);
  });

  it('memoizes the active model for reads until its TTL elapses', async () => {
    const provider = makeProvider();
    let now = 1_000;
    const service = createEmbeddingService({
      provider,
      activeModelTtlMs: 100,
      now: () => now
    });
    const query = vi.fn().mockResolvedValue({
      rows: [
        {
          id: 'model-1',
          name: 'text-embedding-3-small',
          provider: 'openai',
          dimensions: 2,
          chunk_size: 300,
          chunk_overlap: 100,
          metadata: {},
          created_at: new Date()
        }
      ]
    });
    const pool = { query } as unknown as Pool;

    await service.getActiveModelForQuery(pool);
    await service.getActiveModelForQuery(pool);
    expect(query).toHaveBeenCalledTimes(1);

    now += 101;
    await service.getActiveModelForQuery(pool);
    expect(query).toHaveBeenCalledTimes(2);

    service.invalidateActiveModel();
    await service.getActiveModelForQuery(pool);
    expect(query).toHaveBeenCalledTimes(3);
  });

  it('never memoizes the active model for writers', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });
    const query = vi.fn().mockResolvedValue({
      rows: [
        {
          id: 'model-1',
          name: 'text-embedding-3-small',
          provider: 'openai',
          dimensions: 2,
          chunk_size: 300,
          chunk_overlap: 100,
          metadata: {},
          created_at: new Date()
        }
      ]
    });
    const pool = { query } as unknown as Pool;

    // Enrichment writes chunks against this row. Acting on a stale model would
    // attach them to the wrong (or a deleted) embedding_models row, so writers
    // always re-read.
    await service.getActiveModel(pool);
    await service.getActiveModel(pool);

    expect(query).toHaveBeenCalledTimes(2);
  });

  it('keys cache entries by a digest that does not retain the query text', () => {
    const key = createQueryEmbeddingCacheKey('private roadmap');

    expect(key).toHaveLength(32);
    expect(key.toString('hex')).toMatch(/^[a-f0-9]{64}$/u);
    expect(key.toString('utf8')).not.toContain('private roadmap');
    expect(createQueryEmbeddingCacheKey('private roadmap')).toEqual(key);
    expect(createQueryEmbeddingCacheKey('other query')).not.toEqual(key);
  });
});
