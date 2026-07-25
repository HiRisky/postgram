import { describe, expect, it, vi } from 'vitest';

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
      service.embedQuery('postgres search', activeModel, {
        cacheScope: 'key-a'
      }),
      service.embedQuery('postgres search', activeModel, {
        cacheScope: 'key-a'
      })
    ]);
    const third = await service.embedQuery('postgres search', activeModel, {
      cacheScope: 'key-a'
    });

    expect(first).toEqual([0.25, 0.75]);
    expect(second).toEqual(first);
    expect(third).toEqual(first);
    expect(provider.embedBatchMock).toHaveBeenCalledTimes(1);
  });

  it('does not reuse query embeddings across active models', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await service.embedQuery('postgres search', activeModel, {
      cacheScope: 'key-a'
    });
    await service.embedQuery('postgres search', {
      ...activeModel,
      id: 'model-2'
    }, {
      cacheScope: 'key-a'
    });

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('retries query embeddings after a cached provider request fails', async () => {
    const provider = makeProvider();
    provider.embedBatchMock
      .mockRejectedValueOnce(new Error('provider unavailable'))
      .mockResolvedValueOnce([[0.25, 0.75]]);
    const service = createEmbeddingService({ provider });

    await expect(
      service.embedQuery('postgres search', activeModel, {
        cacheScope: 'key-a'
      })
    ).rejects.toThrow('provider unavailable');
    await expect(
      service.embedQuery('postgres search', activeModel, {
        cacheScope: 'key-a'
      })
    ).resolves.toEqual([0.25, 0.75]);

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('does not share cached query embeddings across API key scopes', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await service.embedQuery('private roadmap', activeModel, {
      cacheScope: 'key-a'
    });
    await service.embedQuery('private roadmap', activeModel, {
      cacheScope: 'key-b'
    });

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('bypasses the query cache without an authenticated scope', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });

    await service.embedQuery('private roadmap', activeModel);
    await service.embedQuery('private roadmap', activeModel);

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('expires cached query embeddings after the configured TTL', async () => {
    const provider = makeProvider();
    let now = 1_000;
    const service = createEmbeddingService({
      provider,
      queryCacheTtlMs: 100,
      now: () => now
    });

    await service.embedQuery('private roadmap', activeModel, {
      cacheScope: 'key-a'
    });
    now += 101;
    await service.embedQuery('private roadmap', activeModel, {
      cacheScope: 'key-a'
    });

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(2);
  });

  it('evicts the least recently used query when the cache is full', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({
      provider,
      queryCacheMaxSize: 2
    });
    const options = { cacheScope: 'key-a' };

    await service.embedQuery('query one', activeModel, options);
    await service.embedQuery('query two', activeModel, options);
    await service.embedQuery('query one', activeModel, options);
    await service.embedQuery('query three', activeModel, options);
    await service.embedQuery('query two', activeModel, options);

    expect(provider.embedBatchMock).toHaveBeenCalledTimes(4);
  });

  it('reports cache misses, hits, and unauthenticated bypasses', async () => {
    const provider = makeProvider();
    const service = createEmbeddingService({ provider });
    const statuses: string[] = [];
    const options = {
      cacheScope: 'key-a',
      onCacheStatus: (status: string) => statuses.push(status)
    };

    await service.embedQuery('private roadmap', activeModel, options);
    await service.embedQuery('private roadmap', activeModel, options);
    await service.embedQuery('private roadmap', activeModel, {
      onCacheStatus: (status) => statuses.push(status)
    });

    expect(statuses).toEqual(['miss', 'hit', 'bypass']);
  });

  it('uses an opaque cache key that does not retain plaintext query text', () => {
    const key = createQueryEmbeddingCacheKey(
      'test-secret',
      'key-a',
      activeModel,
      'private roadmap'
    );

    expect(key).toMatch(/^[a-f0-9]{64}$/u);
    expect(key).not.toContain('private roadmap');
    expect(key).not.toContain('key-a');
  });
});
