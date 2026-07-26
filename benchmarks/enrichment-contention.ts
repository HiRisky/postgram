/**
 * Measures what a draining enrichment backlog does to search latency.
 *
 * The enrichment worker runs in-process, on the same event loop as the HTTP
 * handlers. Node's JS thread is single-threaded, so the worker's *awaits* are
 * free — but every synchronous stretch between them (parsing a provider
 * response full of floats, joining 1536 numbers into a `vector` literal,
 * chunking text) delays whatever request continuation is queued behind it.
 * The worker also holds a pooled connection, inside an open transaction,
 * across its provider call.
 *
 * This benchmark runs the same search load twice — once against an idle
 * worker, once against a worker draining a backlog — and attributes the
 * difference to event-loop delay vs connection-pool wait vs SQL, so the fix
 * targets whichever one actually dominates.
 *
 * Run: npx tsx benchmarks/enrichment-contention.ts
 * Gate: npx tsx benchmarks/enrichment-contention.ts --assert
 */
import { monitorEventLoopDelay, performance } from 'node:perf_hooks';

import type { Pool } from 'pg';

import type { AuthContext } from '../src/auth/types.js';
import { createEmbeddingService } from '../src/services/embedding-service.js';
import { createEnrichmentWorker } from '../src/services/enrichment-worker.js';
import { searchEntities } from '../src/services/search-service.js';
import {
  createTestDatabase,
  resetTestDatabase
} from '../tests/helpers/postgres.js';

/** Corpus the searches run against. Matches benchmarks/search-latency.ts. */
const ENTITY_COUNT = 2_000;
/** Entities left `enrichment_status = 'pending'` for the worker to drain. */
const BACKLOG_SIZE = 400;
const DIMENSIONS = 1536;

/**
 * Latency of one embedding round trip. The worker's calls and the search
 * path's calls are stubbed with the same delay: this benchmark is about who
 * waits behind whom, not about any particular provider's speed.
 */
const EMBEDDING_CALL_DELAY_MS = 80;

/** Open-loop search arrival rate — requests are fired on a timer whether or
 * not the previous one finished, which is how real traffic queues up. */
const SEARCH_INTERVAL_MS = 50;
const SEARCH_SAMPLES = 120;
const WARMUP_SEARCHES = 10;

const auth: AuthContext = {
  apiKeyId: '00000000-0000-0000-0000-00000000b001',
  keyName: 'benchmark-key',
  clientId: 'benchmark-key',
  scopes: ['read'],
  allowedTypes: null,
  allowedVisibility: ['personal', 'work', 'shared']
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.floor((sorted.length - 1) * fraction)] ?? 0;
}

type Profile = {
  count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  max_ms: number;
};

function summarize(samples: number[]): Profile {
  const sorted = samples.slice().sort((left, right) => left - right);
  const round = (value: number) => Number(value.toFixed(2));
  return {
    count: sorted.length,
    p50_ms: round(percentile(sorted, 0.5)),
    p95_ms: round(percentile(sorted, 0.95)),
    p99_ms: round(percentile(sorted, 0.99)),
    max_ms: round(sorted.at(-1) ?? 0)
  };
}

/**
 * A stub that reproduces both halves of a real embedding call: the network
 * wait (which yields the event loop) and the response parse (which does not).
 * The parse is the part that matters here — `JSON.parse` of a batch of
 * 1536-float arrays is hundreds of KB of synchronous work that nothing else
 * on the loop can interleave with.
 */
function createStubProvider(delayMs: number) {
  const payloadCache = new Map<number, string>();
  const stats = { calls: 0, vectors: 0 };

  const payloadFor = (count: number): string => {
    const cached = payloadCache.get(count);
    if (cached) return cached;
    const vector = Array.from(
      { length: DIMENSIONS },
      (_, index) => Number((((index % 200) - 100) / 1000).toFixed(6))
    );
    const built = JSON.stringify({
      data: Array.from({ length: count }, (_, index) => ({
        index,
        embedding: vector
      }))
    });
    payloadCache.set(count, built);
    return built;
  };

  const embedBatch = async (texts: string[]): Promise<number[][]> => {
    if (texts.length === 0) return [];
    stats.calls += 1;
    stats.vectors += texts.length;
    const payload = payloadFor(texts.length);
    await sleep(delayMs);
    // Synchronous, exactly as it is inside the provider SDK.
    const parsed = JSON.parse(payload) as {
      data: Array<{ index: number; embedding: number[] }>;
    };
    return parsed.data.map((item) => item.embedding);
  };

  return { embedBatch, stats };
}

async function seed(pool: Pool): Promise<void> {
  await resetTestDatabase(pool);
  await pool.query(
    `
      INSERT INTO entities (
        id, type, content, visibility, status, enrichment_status,
        tags, metadata, created_at, updated_at
      )
      SELECT
        gen_random_uuid(),
        'memory',
        'postgres enrichment contention benchmark entity ' || value || ' ' ||
          repeat('knowledge memory context ', 40),
        CASE
          WHEN value % 3 = 0 THEN 'shared'
          WHEN value % 3 = 1 THEN 'personal'
          ELSE 'work'
        END,
        'active',
        'completed',
        ARRAY['benchmark'],
        '{}'::jsonb,
        now() - (value || ' minutes')::interval,
        now()
      FROM generate_series(1, $1::integer) AS value
    `,
    [ENTITY_COUNT]
  );

  await pool.query(
    `
      INSERT INTO chunks (
        entity_id, chunk_index, content, embedding, model_id, token_count
      )
      SELECT
        e.id,
        0,
        left(e.content, 500),
        array_fill(0.01::real, ARRAY[1536])::vector,
        (SELECT id FROM embedding_models WHERE is_active = true),
        100
      FROM entities e
    `
  );

  await pool.query('ANALYZE entities');
  await pool.query('ANALYZE chunks');
}

/**
 * Queues `BACKLOG_SIZE` entities for enrichment. Content is long enough to
 * chunk into several pieces, so each entity costs the worker one batched
 * provider call plus one INSERT per chunk — the shape of a real import.
 */
async function queueBacklog(pool: Pool): Promise<void> {
  await pool.query(
    `
      INSERT INTO entities (
        id, type, content, visibility, status, enrichment_status,
        tags, metadata, created_at, updated_at
      )
      SELECT
        gen_random_uuid(),
        'memory',
        'backlog entity ' || value || ' ' ||
          repeat('imported document body with enough prose to chunk ', 60),
        'personal',
        'active',
        'pending',
        ARRAY['backlog'],
        '{}'::jsonb,
        now(),
        now()
      FROM generate_series(1, $1::integer) AS value
    `,
    [BACKLOG_SIZE]
  );
}

async function clearBacklog(pool: Pool): Promise<void> {
  await pool.query("DELETE FROM entities WHERE tags @> ARRAY['backlog']");
}

async function pendingCount(pool: Pool): Promise<number> {
  const result = await pool.query<{ count: string }>(
    "SELECT count(*)::text AS count FROM entities WHERE enrichment_status = 'pending'"
  );
  return Number(result.rows[0]?.count ?? '0');
}

/**
 * Wraps the pool so time spent waiting for a free connection is separable
 * from time spent executing SQL. If the worker were starving the pool, this
 * is the number that would move.
 */
function instrumentPool(pool: Pool): {
  pool: Pool;
  waits: number[];
  reset: () => void;
} {
  const waits: number[] = [];
  const instrumented = new Proxy(pool, {
    get(target, property, receiver) {
      if (property === 'connect') {
        return async (...args: unknown[]) => {
          const started = performance.now();
          const client = await (
            target.connect as (...a: unknown[]) => Promise<unknown>
          )(...args);
          waits.push(performance.now() - started);
          return client;
        };
      }
      const value: unknown = Reflect.get(target, property, receiver);
      return typeof value === 'function'
        ? (value.bind(target) as unknown)
        : value;
    }
  });
  return {
    pool: instrumented,
    waits,
    reset: () => {
      waits.length = 0;
    }
  };
}

type PhaseResult = {
  search: Profile;
  event_loop_delay: { mean_ms: number; p95_ms: number; max_ms: number };
  pool_wait: Profile;
  entities_drained: number;
  worker_embedding_calls: number;
  worker_vectors_embedded: number;
  search_embedding_calls: number;
};

/**
 * Fires searches on a fixed interval and records end-to-end latency. Every
 * query is unique so each one pays a provider round trip — the same path a
 * cold cache takes. `repeated` reuses one query instead, which the cache
 * serves without any provider call: that variant isolates contention, since
 * nothing but SQL and event-loop time is left in the measurement.
 */
async function runSearchLoad(
  pool: Pool,
  input: { repeated: boolean; label: string }
): Promise<{ samples: number[]; embeddingCalls: number }> {
  const provider = createStubProvider(EMBEDDING_CALL_DELAY_MS);
  const embeddingService = createEmbeddingService({
    embedQuery: async () => {
      const [vector] = await provider.embedBatch(['query']);
      if (!vector) throw new Error('stub returned no vector');
      return vector;
    }
  });

  const samples: number[] = [];
  const inflight: Array<Promise<void>> = [];

  const fire = async (index: number): Promise<void> => {
    // The query embedding cache is persisted in Postgres and survives across
    // phases, so a miss-path phase has to use query text no earlier phase has
    // seen. Without the label in the string, the second uncached phase would
    // be served entirely from cache and look 6x faster than the first.
    const query = input.repeated
      ? 'postgres enrichment contention benchmark entity 42'
      : `postgres enrichment contention benchmark ${input.label} entity ${index}`;
    const started = performance.now();
    const result = await searchEntities(
      pool,
      auth,
      { query, limit: 10, threshold: 0 },
      { embeddingService }
    );
    const elapsed = performance.now() - started;
    if (result.isErr()) throw result.error;
    samples.push(elapsed);
  };

  for (let index = 0; index < WARMUP_SEARCHES; index += 1) {
    await fire(10_000 + index);
  }
  samples.length = 0;
  provider.stats.calls = 0;

  for (let index = 0; index < SEARCH_SAMPLES; index += 1) {
    inflight.push(fire(index));
    await sleep(SEARCH_INTERVAL_MS);
  }
  await Promise.all(inflight);
  return { samples, embeddingCalls: provider.stats.calls };
}

async function runPhase(
  rawPool: Pool,
  instrumented: { pool: Pool; waits: number[]; reset: () => void },
  input: { withWorker: boolean; repeated: boolean; label: string }
): Promise<PhaseResult> {
  const workerProvider = createStubProvider(EMBEDDING_CALL_DELAY_MS);
  const workerEmbeddingService = createEmbeddingService({
    embedBatch: async (texts: string[]) => workerProvider.embedBatch(texts)
  });

  await clearBacklog(rawPool);
  if (input.withWorker) {
    await queueBacklog(rawPool);
  }
  const pendingBefore = await pendingCount(rawPool);

  const worker = createEnrichmentWorker({
    pool: instrumented.pool,
    embeddingService: workerEmbeddingService,
    extractionEnabled: false
  });

  let workerActive = input.withWorker;
  const workerLoop = async () => {
    while (workerActive) {
      try {
        await worker.runOnce();
      } catch {
        // Drain errors are surfaced through the drained count below.
      }
      await sleep(50);
    }
  };
  const workerDone = input.withWorker ? workerLoop() : Promise.resolve();

  const histogram = monitorEventLoopDelay({ resolution: 1 });
  instrumented.reset();
  histogram.enable();

  const { samples, embeddingCalls } = await runSearchLoad(instrumented.pool, {
    repeated: input.repeated,
    label: input.label
  });

  histogram.disable();
  workerActive = false;
  await workerDone;

  const pendingAfter = await pendingCount(rawPool);
  const toMs = (nanoseconds: number) =>
    Number((nanoseconds / 1_000_000).toFixed(2));

  return {
    search: summarize(samples),
    event_loop_delay: {
      mean_ms: toMs(histogram.mean),
      p95_ms: toMs(histogram.percentile(95)),
      max_ms: toMs(histogram.max)
    },
    pool_wait: summarize(instrumented.waits),
    entities_drained: Math.max(0, pendingBefore - pendingAfter),
    worker_embedding_calls: workerProvider.stats.calls,
    worker_vectors_embedded: workerProvider.stats.vectors,
    search_embedding_calls: embeddingCalls
  };
}

const database = await createTestDatabase();
let report: unknown;
try {
  await seed(database.pool);
  const instrumented = instrumentPool(database.pool);

  // Cache-hit searches first: with no provider round trip in the measurement,
  // any latency the worker adds has nowhere to hide.
  const cachedIdle = await runPhase(database.pool, instrumented, {
    withWorker: false,
    repeated: true,
    label: 'cached_idle'
  });
  const cachedDraining = await runPhase(database.pool, instrumented, {
    withWorker: true,
    repeated: true,
    label: 'cached_draining'
  });
  // Cache-miss searches: the realistic worst case, where the request is also
  // waiting on its own provider call.
  const uncachedIdle = await runPhase(database.pool, instrumented, {
    withWorker: false,
    repeated: false,
    label: 'uncached_idle'
  });
  const uncachedDraining = await runPhase(database.pool, instrumented, {
    withWorker: true,
    repeated: false,
    label: 'uncached_draining'
  });

  const delta = (idle: PhaseResult, draining: PhaseResult) => ({
    p50_ms: Number((draining.search.p50_ms - idle.search.p50_ms).toFixed(2)),
    p95_ms: Number((draining.search.p95_ms - idle.search.p95_ms).toFixed(2)),
    p95_ratio: Number(
      (draining.search.p95_ms / Math.max(idle.search.p95_ms, 0.01)).toFixed(2)
    )
  });

  report = {
    config: {
      entities: ENTITY_COUNT,
      backlog: BACKLOG_SIZE,
      embedding_call_delay_ms: EMBEDDING_CALL_DELAY_MS,
      search_interval_ms: SEARCH_INTERVAL_MS,
      search_samples: SEARCH_SAMPLES
    },
    phases: {
      cached_idle: cachedIdle,
      cached_draining: cachedDraining,
      uncached_idle: uncachedIdle,
      uncached_draining: uncachedDraining
    },
    contention: {
      cached: delta(cachedIdle, cachedDraining),
      uncached: delta(uncachedIdle, uncachedDraining)
    }
  };

  if (process.argv.includes('--assert')) {
    const failures: string[] = [];

    // The phases only mean something if the worker actually drained the
    // backlog while the searches were running.
    if (cachedDraining.entities_drained < BACKLOG_SIZE) {
      failures.push(
        `worker drained ${cachedDraining.entities_drained}/${BACKLOG_SIZE} entities during the cached phase`
      );
    }
    // A miss-path phase that made no provider calls was served from the
    // persisted query-embedding cache, which would make it silently bogus.
    if (uncachedDraining.search_embedding_calls < SEARCH_SAMPLES) {
      failures.push(
        `uncached phase made ${uncachedDraining.search_embedding_calls}/${SEARCH_SAMPLES} embedding calls; phase query text is colliding with an earlier phase`
      );
    }

    // The actual regression gate: draining a backlog must not inflate search
    // p95. Stated as a ratio so it survives being run on a slower host.
    const maxP95Ratio = 1.5;
    for (const [name, measured] of Object.entries({
      cached: delta(cachedIdle, cachedDraining),
      uncached: delta(uncachedIdle, uncachedDraining)
    })) {
      if (measured.p95_ratio > maxP95Ratio) {
        failures.push(
          `${name} search p95 grew ${measured.p95_ratio}x while the backlog drained (limit ${maxP95Ratio}x)`
        );
      }
    }

    if (failures.length > 0) {
      throw new Error(failures.join('; '));
    }
  }
} finally {
  await database.close();
  if (report !== undefined) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  }
}
