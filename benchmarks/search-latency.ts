import { performance } from 'node:perf_hooks';

import type { Pool, PoolClient, QueryResult } from 'pg';

import type { AuthContext } from '../src/auth/types.js';
import { createEmbeddingService } from '../src/services/embedding-service.js';
import { searchEntities } from '../src/services/search-service.js';
import {
  createTestDatabase,
  resetTestDatabase
} from '../tests/helpers/postgres.js';

const ENTITY_COUNT = 5_000;
const CHUNK_COUNT = 6_000;
const WARMUP_RUNS = 5;
const SAMPLE_RUNS = 20;

type CapturedQuery = { text: string; values: unknown[] };
type Profile = {
  p50_ms: number;
  p95_ms: number;
  min_ms: number;
  max_ms: number;
};

const auth: AuthContext = {
  apiKeyId: '00000000-0000-0000-0000-00000000b001',
  keyName: 'benchmark-key',
  clientId: 'benchmark-key',
  scopes: ['read'],
  allowedTypes: null,
  allowedVisibility: ['personal', 'work', 'shared']
};

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.floor((sorted.length - 1) * fraction)] ?? 0;
}

function summarize(samples: number[]): Profile {
  const sorted = samples.slice().sort((left, right) => left - right);
  return {
    p50_ms: Number(percentile(sorted, 0.5).toFixed(2)),
    p95_ms: Number(percentile(sorted, 0.95).toFixed(2)),
    min_ms: Number((sorted[0] ?? 0).toFixed(2)),
    max_ms: Number((sorted.at(-1) ?? 0).toFixed(2))
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function seed(pool: Pool): Promise<void> {
  await resetTestDatabase(pool);
  await pool.query(
    `
      INSERT INTO entities (
        id,
        type,
        content,
        visibility,
        status,
        enrichment_status,
        tags,
        metadata,
        created_at,
        updated_at
      )
      SELECT
        gen_random_uuid(),
        'memory',
        'postgres search latency benchmark entity ' || value || ' ' ||
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
        entity_id,
        chunk_index,
        content,
        embedding,
        model_id,
        token_count
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

  await pool.query(
    `
      INSERT INTO chunks (
        entity_id,
        chunk_index,
        content,
        embedding,
        model_id,
        token_count
      )
      SELECT
        e.id,
        1,
        right(e.content, 500),
        array_fill(0.011::real, ARRAY[1536])::vector,
        (SELECT id FROM embedding_models WHERE is_active = true),
        100
      FROM entities e
      ORDER BY e.id
      LIMIT $1
    `,
    [CHUNK_COUNT - ENTITY_COUNT]
  );
  await pool.query('ANALYZE entities');
  await pool.query('ANALYZE chunks');
}

function captureSearchQueries(pool: Pool): {
  pool: Pool;
  captured: { hybrid?: CapturedQuery };
  counts: { total: number };
} {
  const captured: { hybrid?: CapturedQuery } = {};
  const counts = { total: 0 };
  const capture = (text: string, values?: unknown[]): void => {
    counts.total += 1;
    if (text.includes('ROW_NUMBER() OVER')) {
      captured.hybrid = { text, values: values ?? [] };
    }
  };
  const instrumentClient = (client: PoolClient): PoolClient =>
    new Proxy(client, {
      get(target, property, receiver) {
        if (property === 'query') {
          return (text: string, values?: unknown[]) => {
            capture(text, values);
            return target.query(text, values);
          };
        }
        const value: unknown = Reflect.get(target, property, receiver);
        return typeof value === 'function'
          ? (value.bind(target) as unknown)
          : value;
      }
    });
  const instrumented = new Proxy(pool, {
    get(target, property, receiver) {
      if (property === 'query') {
        return (text: string, values?: unknown[]) => {
          capture(text, values);
          return target.query(text, values);
        };
      }
      if (property === 'connect') {
        return async () => instrumentClient(await target.connect());
      }
      const value: unknown = Reflect.get(target, property, receiver);
      return typeof value === 'function'
        ? (value.bind(target) as unknown)
        : value;
    }
  });
  return { pool: instrumented, captured, counts };
}

/**
 * The embedding provider is stubbed with a fixed delay so the gate measures
 * what this repository controls: SQL time, and how often that provider delay is
 * paid at all. It deliberately does NOT claim to measure real provider latency
 * — that varies by provider and host, and a green gate here says nothing about
 * it. `embeddingCallDelayMs` is set well above any plausible SQL time so a
 * regression in cache hit rate shows up as a latency cliff rather than noise.
 */
const EMBEDDING_CALL_DELAY_MS = 250;

async function runProfile(
  pool: Pool,
  input: {
    name: string;
    /** Reuse one query across runs so every call after the first is a cache hit. */
    repeated?: boolean;
    /** Share the cache across runs; false simulates a cold process. */
    freshServicePerRun?: boolean;
  }
): Promise<[string, Profile, { embeddingCalls: number }]> {
  const vector = new Array<number>(1536).fill(0.01);
  let embeddingCalls = 0;
  const embedQuery = async () => {
    embeddingCalls += 1;
    await sleep(EMBEDDING_CALL_DELAY_MS);
    return vector;
  };
  let embeddingService = createEmbeddingService({ embedQuery });

  const run = async (iteration: number): Promise<number> => {
    if (input.freshServicePerRun) {
      embeddingService = createEmbeddingService({ embedQuery });
    }
    const query = input.repeated
      ? 'postgres search latency benchmark entity 42'
      : `postgres search latency benchmark entity ${iteration + 1}`;
    const started = performance.now();
    const result = await searchEntities(
      pool,
      auth,
      { query, limit: 10, threshold: 0 },
      { embeddingService }
    );
    if (result.isErr()) {
      throw result.error;
    }
    return performance.now() - started;
  };

  for (let index = 0; index < WARMUP_RUNS; index += 1) {
    await run(100 + index);
  }
  embeddingCalls = 0;
  const samples: number[] = [];
  for (let index = 0; index < SAMPLE_RUNS; index += 1) {
    samples.push(await run(index));
  }
  return [input.name, summarize(samples), { embeddingCalls }];
}

type ExplainPlanNode = {
  'Node Type'?: string;
  'Index Name'?: string;
  'Shared Hit Blocks'?: number;
  Plans?: ExplainPlanNode[];
};

type ExplainRow = {
  'QUERY PLAN'?: Array<{
    'Planning Time'?: number;
    'Execution Time'?: number;
    Plan?: ExplainPlanNode;
  }>;
};

type ExplainSummary = {
  planning_ms: number | null;
  execution_ms: number | null;
  root_node: string | null;
  shared_hit_blocks: number | null;
  indexes: string[];
};

function collectIndexes(
  node: ExplainPlanNode | undefined,
  indexes: Set<string>
): void {
  if (!node) return;
  if (node['Index Name']) indexes.add(node['Index Name']);
  for (const child of node.Plans ?? []) {
    collectIndexes(child, indexes);
  }
}

function summarizeExplain(result: QueryResult<ExplainRow>): ExplainSummary {
  const plan = result.rows[0]?.['QUERY PLAN']?.[0];
  const indexes = new Set<string>();
  collectIndexes(plan?.Plan, indexes);
  return {
    planning_ms: plan?.['Planning Time'] ?? null,
    execution_ms: plan?.['Execution Time'] ?? null,
    root_node: plan?.Plan?.['Node Type'] ?? null,
    shared_hit_blocks: plan?.Plan?.['Shared Hit Blocks'] ?? null,
    indexes: Array.from(indexes).sort()
  };
}

async function explain(
  pool: Pool,
  query: CapturedQuery | undefined
): Promise<ExplainSummary | null> {
  if (!query) return null;
  const result = await pool.query<ExplainRow>(
    `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ${query.text}`,
    query.values
  );
  return summarizeExplain(result);
}

const database = await createTestDatabase();
let report: unknown;
try {
  await seed(database.pool);
  const { pool, captured, counts } = captureSearchQueries(database.pool);

  const runs = [
    // Every query is unique, so every search pays the provider round trip.
    // This is the worst case and the upper bound on search latency.
    await runProfile(pool, { name: 'cold_unique_queries' }),
    // The same query repeatedly: served from the in-process cache.
    await runProfile(pool, { name: 'memory_cache_hit', repeated: true }),
    // The same query repeatedly, but with a fresh service each run so the
    // in-process cache is always empty. Only the Postgres-backed cache can
    // satisfy these — this is what a restarted process sees.
    await runProfile(pool, {
      name: 'database_cache_hit',
      repeated: true,
      freshServicePerRun: true
    })
  ];

  const profiles = Object.fromEntries(
    runs.map(([name, profile]) => [name, profile])
  ) as Record<string, Profile>;
  const embeddingCalls = Object.fromEntries(
    runs.map(([name, , stats]) => [name, stats.embeddingCalls])
  ) as Record<string, number>;

  const hybridExplain = await explain(database.pool, captured.hybrid);
  report = {
    dataset: { entities: ENTITY_COUNT, chunks: CHUNK_COUNT },
    samples: SAMPLE_RUNS,
    embedding_call_delay_ms: EMBEDDING_CALL_DELAY_MS,
    profiles,
    embedding_calls: embeddingCalls,
    database_queries: counts.total,
    explain: { hybrid: hybridExplain }
  };

  if (process.argv.includes('--assert')) {
    if (!hybridExplain?.indexes.includes('idx_chunks_embedding')) {
      throw new Error('hybrid search plan did not use idx_chunks_embedding');
    }

    const maxP95: Record<string, number> = {
      // One provider round trip plus SQL.
      cold_unique_queries: EMBEDDING_CALL_DELAY_MS + 150,
      // No provider round trip at all: this must stay pure SQL time.
      memory_cache_hit: 125,
      // One extra indexed SELECT versus a memory hit, and still no round trip.
      database_cache_hit: 150
    };
    for (const [name, maximum] of Object.entries(maxP95)) {
      const actual = profiles[name]?.p95_ms;
      if (actual === undefined || actual > maximum) {
        throw new Error(
          `${name} p95 ${actual ?? 'missing'}ms exceeded ${maximum}ms`
        );
      }
    }

    // Guards the cache itself rather than just its latency effect: a cache that
    // silently stopped working would still pass a wall-clock threshold on a
    // fast CI box, but cannot pass this.
    const maxEmbeddingCalls: Record<string, number> = {
      cold_unique_queries: SAMPLE_RUNS,
      memory_cache_hit: 0,
      database_cache_hit: 0
    };
    for (const [name, maximum] of Object.entries(maxEmbeddingCalls)) {
      const actual = embeddingCalls[name];
      if (actual === undefined || actual > maximum) {
        throw new Error(
          `${name} made ${actual ?? 'missing'} embedding calls; expected at most ${maximum}`
        );
      }
    }
  }
} finally {
  await database.close();
  // Written in `finally` so a failed assertion still emits the numbers that
  // explain the failure rather than an empty artifact.
  if (report !== undefined) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  }
}
