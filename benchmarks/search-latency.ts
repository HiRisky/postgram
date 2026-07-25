import { performance } from 'node:perf_hooks';

import type { Pool, QueryResult } from 'pg';

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
  captured: { hybrid?: CapturedQuery; lexical?: CapturedQuery };
} {
  const captured: { hybrid?: CapturedQuery; lexical?: CapturedQuery } = {};
  const instrumented = new Proxy(pool, {
    get(target, property, receiver) {
      if (property === 'query') {
        return (text: string, values?: unknown[]) => {
          if (text.includes('search_tsvector @@')) {
            captured.lexical = { text, values: values ?? [] };
          } else if (text.includes('ROW_NUMBER() OVER')) {
            captured.hybrid = { text, values: values ?? [] };
          }
          return target.query(text, values);
        };
      }
      const value = Reflect.get(target, property, receiver) as unknown;
      return typeof value === 'function' ? value.bind(target) : value;
    }
  }) as Pool;
  return { pool: instrumented, captured };
}

async function runProfile(
  pool: Pool,
  input: {
    name: string;
    embeddingDelayMs: number;
    embeddingBudgetMs: number;
    repeated?: boolean;
    expectedMode: 'hybrid' | 'lexical_fallback';
  }
): Promise<[string, Profile]> {
  const vector = new Array<number>(1536).fill(0.01);
  const embeddingService = createEmbeddingService({
    embedQuery: async () => {
      if (input.embeddingDelayMs > 0) {
        await sleep(input.embeddingDelayMs);
      }
      return vector;
    }
  });

  const run = async (iteration: number): Promise<number> => {
    const query = input.repeated
      ? 'postgres search latency benchmark entity 42'
      : `postgres search latency benchmark entity ${iteration + 1}`;
    const started = performance.now();
    const result = await searchEntities(
      pool,
      auth,
      { query, limit: 10, threshold: 0 },
      {
        embeddingService,
        embeddingBudgetMs: input.embeddingBudgetMs
      }
    );
    if (result.isErr()) {
      throw result.error;
    }
    if (result.value.searchMode !== input.expectedMode) {
      throw new Error(
        `${input.name} returned ${result.value.searchMode}; expected ${input.expectedMode}`
      );
    }
    return performance.now() - started;
  };

  for (let index = 0; index < WARMUP_RUNS; index += 1) {
    await run(100 + index);
  }
  const samples: number[] = [];
  for (let index = 0; index < SAMPLE_RUNS; index += 1) {
    samples.push(await run(index));
  }
  return [input.name, summarize(samples)];
}

function summarizeExplain(result: QueryResult): Record<string, unknown> {
  const payload = result.rows[0]?.['QUERY PLAN'] as
    | Array<{
        'Planning Time'?: number;
        'Execution Time'?: number;
        Plan?: { 'Node Type'?: string; 'Shared Hit Blocks'?: number };
      }>
    | undefined;
  const plan = payload?.[0];
  return {
    planning_ms: plan?.['Planning Time'] ?? null,
    execution_ms: plan?.['Execution Time'] ?? null,
    root_node: plan?.Plan?.['Node Type'] ?? null,
    shared_hit_blocks: plan?.Plan?.['Shared Hit Blocks'] ?? null
  };
}

async function explain(
  pool: Pool,
  query: CapturedQuery | undefined
): Promise<Record<string, unknown> | null> {
  if (!query) return null;
  const result = await pool.query(
    `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ${query.text}`,
    query.values
  );
  return summarizeExplain(result);
}

const database = await createTestDatabase();
try {
  await seed(database.pool);
  const { pool, captured } = captureSearchQueries(database.pool);
  const profileEntries: Array<[string, Profile]> = [];
  profileEntries.push(
    await runProfile(pool, {
      name: 'sql_only_unique',
      embeddingDelayMs: 0,
      embeddingBudgetMs: 350,
      expectedMode: 'hybrid'
    })
  );
  profileEntries.push(
    await runProfile(pool, {
      name: 'hybrid_unique_250ms_embedding',
      embeddingDelayMs: 250,
      embeddingBudgetMs: 350,
      expectedMode: 'hybrid'
    })
  );
  profileEntries.push(
    await runProfile(pool, {
      name: 'lexical_fallback_500ms_embedding',
      embeddingDelayMs: 500,
      embeddingBudgetMs: 350,
      expectedMode: 'lexical_fallback'
    })
  );
  profileEntries.push(
    await runProfile(pool, {
      name: 'cache_hit_repeated',
      embeddingDelayMs: 250,
      embeddingBudgetMs: 350,
      repeated: true,
      expectedMode: 'hybrid'
    })
  );
  const profiles = Object.fromEntries(profileEntries) as Record<
    string,
    Profile
  >;

  const report = {
    dataset: { entities: ENTITY_COUNT, chunks: CHUNK_COUNT },
    samples: SAMPLE_RUNS,
    profiles,
    explain: {
      hybrid: await explain(database.pool, captured.hybrid),
      lexical: await explain(database.pool, captured.lexical)
    }
  };

  if (process.argv.includes('--assert')) {
    const thresholds: Record<string, number> = {
      sql_only_unique: 125,
      hybrid_unique_250ms_embedding: 450,
      lexical_fallback_500ms_embedding: 500,
      cache_hit_repeated: 125
    };
    for (const [name, maximum] of Object.entries(thresholds)) {
      const actual = profiles[name]?.p95_ms;
      if (actual === undefined || actual > maximum) {
        throw new Error(
          `${name} p95 ${actual ?? 'missing'}ms exceeded ${maximum}ms`
        );
      }
    }
  }

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} finally {
  await database.close();
}
