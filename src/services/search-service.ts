import { ResultAsync } from 'neverthrow';
import type { Pool } from 'pg';
import type { Logger } from 'pino';

import { requireScope } from '../auth/key-service.js';
import type { AuthContext } from '../auth/types.js';
import type { ServiceResult } from '../types/common.js';
import type {
  Entity,
  EntityStatus,
  EntityType,
  EnrichmentStatus,
  Visibility
} from '../types/entities.js';
import { AppError, ErrorCode } from '../util/errors.js';
import { ownerSqlCondition } from './owner-filter.js';
import {
  createEmbeddingService,
  type EmbeddingService,
  vectorToSql
} from './embedding-service.js';
import type { MemoryRole } from './memory-role-service.js';

type EntityRow = {
  id: string;
  type: EntityType;
  content: string | null;
  visibility: Visibility;
  owner: string | null;
  status: EntityStatus | null;
  enrichment_status: EnrichmentStatus;
  version: number;
  tags: string[];
  source: string | null;
  metadata: Record<string, unknown>;
  created_at: Date;
  updated_at: Date;
};

type SearchRow = EntityRow & {
  chunk_content: string;
  similarity: number;
  score: number;
};

export type SearchResult = {
  entity: Entity;
  entityId: string;
  chunkContent: string;
  similarity: number;
  score: number;
  edges?: SearchEdgeSummary | undefined;
  related?: Array<{
    entity: { id: string; type: string; content: string | null; metadata: Record<string, unknown> };
    relation: string;
    direction: 'outgoing' | 'incoming';
  }> | undefined;
};

export type SearchEdgeSummary = {
  count: number;
  relations: Array<{ relation: string; count: number }>;
};

export type SearchEdgeSummaryRow = {
  result_entity_id: string;
  relation: string;
};

type SearchInput = {
  query: string;
  type?: EntityType | undefined;
  tags?: string[] | undefined;
  visibility?: Visibility | undefined;
  owner?: string | undefined;
  memoryRole?: MemoryRole | undefined;
  limit?: number | undefined;
  threshold?: number | undefined;
  recencyWeight?: number | undefined;
  expandGraph?: boolean | undefined;
  includeArchived?: boolean | undefined;
};

type SearchOptions = {
  embeddingService?: EmbeddingService | undefined;
  now?: (() => Date) | undefined;
  embeddingBudgetMs?: number | undefined;
  logger?: Pick<Logger, 'debug' | 'warn'> | undefined;
};

export type SearchMode = 'hybrid' | 'lexical_fallback';
export type SearchFallbackReason = 'embedding_timeout' | 'embedding_error';

export type SearchResponse = {
  results: SearchResult[];
  searchMode: SearchMode;
  fallbackReason?: SearchFallbackReason | undefined;
};

function toAppError(error: unknown, fallbackMessage: string): AppError {
  if (error instanceof AppError) {
    return error;
  }

  if (error instanceof Error) {
    return new AppError(ErrorCode.INTERNAL, fallbackMessage, {
      cause: error.message
    });
  }

  return new AppError(ErrorCode.INTERNAL, fallbackMessage);
}

function mapEntity(row: EntityRow): Entity {
  return {
    id: row.id,
    type: row.type,
    content: row.content,
    visibility: row.visibility,
    owner: row.owner,
    status: row.status,
    enrichmentStatus: row.enrichment_status,
    version: row.version,
    tags: row.tags,
    source: row.source,
    metadata: row.metadata,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString()
  };
}

export function scopedMemoryVisibilitySql(
  metadataColumn: string,
  clientIdPlaceholder: string
): string {
  return `(
    (
      COALESCE(${metadataColumn}->>'memory_role', 'durable_memory') = 'session_context'
      AND ${metadataColumn} #>> '{session_scope,client_id}' = ${clientIdPlaceholder}
    )
    OR (
      COALESCE(${metadataColumn}->>'memory_role', 'durable_memory') <> 'session_context'
      AND (
        ${metadataColumn} #>> '{session_scope,client_id}' IS NULL
        OR ${metadataColumn} #>> '{session_scope,client_id}' = ${clientIdPlaceholder}
      )
    )
  )`;
}

export function applyRecencyBoost({
  similarity,
  ageDays,
  recencyWeight,
  halfLifeDays
}: {
  similarity: number;
  ageDays: number;
  recencyWeight: number;
  halfLifeDays: number;
}): number {
  return similarity * (1 + recencyWeight * Math.exp(-ageDays / halfLifeDays));
}

export function deduplicateResults<T extends { entityId: string; score: number }>(
  results: T[]
): T[] {
  const bestByEntity = new Map<string, T>();

  for (const result of results) {
    const existing = bestByEntity.get(result.entityId);
    if (!existing || result.score > existing.score) {
      bestByEntity.set(result.entityId, result);
    }
  }

  return Array.from(bestByEntity.values()).sort((left, right) => right.score - left.score);
}

export function buildSearchEdgeSummaries(
  rows: SearchEdgeSummaryRow[]
): Map<string, SearchEdgeSummary> {
  const relationsByEntityId = new Map<string, Map<string, number>>();

  for (const row of rows) {
    const relationCounts =
      relationsByEntityId.get(row.result_entity_id) ?? new Map<string, number>();
    relationCounts.set(row.relation, (relationCounts.get(row.relation) ?? 0) + 1);
    relationsByEntityId.set(row.result_entity_id, relationCounts);
  }

  const summaries = new Map<string, SearchEdgeSummary>();
  for (const [entityId, relationCounts] of relationsByEntityId) {
    const relations = Array.from(relationCounts.entries())
      .map(([relation, count]) => ({ relation, count }))
      .sort(
        (left, right) =>
          right.count - left.count || left.relation.localeCompare(right.relation)
      );

    summaries.set(entityId, {
      count: relations.reduce((total, entry) => total + entry.count, 0),
      relations
    });
  }

  return summaries;
}

const VECTOR_WEIGHT = 0.6;
const BM25_WEIGHT = 0.4;

export function normalizeBm25Scores<T extends { bm25: number }>(
  results: T[]
): T[] {
  const maxBm25 = Math.max(...results.map((r) => r.bm25));

  if (maxBm25 === 0) {
    return results;
  }

  return results.map((r) => ({
    ...r,
    bm25: r.bm25 / maxBm25
  }));
}

export function blendScores(
  vectorScore: number,
  normalizedBm25Score: number
): number {
  return VECTOR_WEIGHT * vectorScore + BM25_WEIGHT * normalizedBm25Score;
}

type SearchContext = {
  threshold: number;
  recencyWeight: number;
  limit: number;
  now: Date;
};

// Bound the vector candidate set before SQL-side BM25 and recency reranking.
// This keeps ranking work and intermediate tuples predictable as the corpus grows.
const CANDIDATE_CAP = 500;
const DEFAULT_EMBEDDING_BUDGET_MS = 350;

function mapSearchRows(rows: SearchRow[]): SearchResult[] {
  return rows.map((row) => {
    const entity = mapEntity(row);
    return {
      entity,
      entityId: entity.id,
      chunkContent: row.chunk_content,
      similarity: Number(row.similarity),
      score: Number(row.score)
    };
  });
}

async function runHybridSearch(
  pool: Pool,
  auth: AuthContext,
  input: SearchInput,
  ctx: SearchContext & { queryEmbedding: number[]; queryText: string }
): Promise<{ results: SearchResult[] }> {
  const candidateLimit = Math.min(ctx.limit * 20, CANDIDATE_CAP);

  const rows = await pool.query<SearchRow>(
    `
      WITH candidates AS MATERIALIZED (
        SELECT
          e.id AS entity_id,
          c.id AS chunk_id,
          e.created_at,
          1 - (c.embedding <=> $1::vector) AS similarity,
          c.embedding <=> $1::vector AS distance,
          ts_rank(e.search_tsvector, plainto_tsquery('simple', $8)) AS bm25
        FROM chunks c
        JOIN entities e ON e.id = c.entity_id
        WHERE ($10::boolean = true OR e.status IS DISTINCT FROM 'archived')
          AND ($2::text IS NULL OR e.type = $2)
          AND ($3::text[] IS NULL OR e.tags @> $3)
          AND ($4::text[] IS NULL OR e.type = ANY($4))
          AND e.visibility = ANY($5)
          AND ($6::text IS NULL OR e.visibility = $6)
          AND ${ownerSqlCondition('e.owner', '$7')}
          AND (
            $11::text IS NULL
            OR (
              e.type = 'memory'
              AND COALESCE(e.metadata->>'memory_role', 'durable_memory') = $11
            )
          )
          AND ${scopedMemoryVisibilitySql('e.metadata', '$12')}
        ORDER BY distance
        LIMIT $9
      ),
      normalized AS (
        SELECT
          candidates.*,
          CASE
            WHEN MAX(bm25) OVER () = 0 THEN bm25
            ELSE bm25 / MAX(bm25) OVER ()
          END AS normalized_bm25
        FROM candidates
      ),
      scored AS (
        SELECT
          normalized.*,
          (
            0.6 * similarity + 0.4 * normalized_bm25
          ) * (
            1 + $13::double precision * EXP(
              -EXTRACT(EPOCH FROM ($14::timestamptz - created_at))
              / 86400.0
              / 30.0
            )
          ) AS score
        FROM normalized
      ),
      deduplicated AS (
        SELECT
          scored.*,
          ROW_NUMBER() OVER (
            PARTITION BY entity_id
            ORDER BY score DESC, chunk_id
          ) AS entity_rank
        FROM scored
        WHERE score >= $15
      ),
      top_results AS MATERIALIZED (
        SELECT *
        FROM deduplicated
        WHERE entity_rank = 1
        ORDER BY score DESC
        LIMIT $16
      )
      SELECT
        e.*,
        c.content AS chunk_content,
        top_results.similarity,
        top_results.score
      FROM top_results
      JOIN entities e ON e.id = top_results.entity_id
      JOIN chunks c ON c.id = top_results.chunk_id
      ORDER BY top_results.score DESC
    `,
    [
      vectorToSql(ctx.queryEmbedding),
      input.type ?? null,
      input.tags?.length ? input.tags : null,
      auth.allowedTypes,
      auth.allowedVisibility,
      input.visibility ?? null,
      input.owner ?? null,
      ctx.queryText,
      candidateLimit,
      input.includeArchived ?? false,
      input.memoryRole ?? null,
      auth.clientId,
      ctx.recencyWeight,
      ctx.now,
      ctx.threshold,
      ctx.limit
    ]
  );

  return {
    results: mapSearchRows(rows.rows)
  };
}

async function runLexicalSearch(
  pool: Pool,
  auth: AuthContext,
  input: SearchInput,
  ctx: SearchContext & { queryText: string }
): Promise<{ results: SearchResult[] }> {
  const candidateLimit = Math.min(ctx.limit * 20, CANDIDATE_CAP);
  const rows = await pool.query<SearchRow>(
    `
      WITH search_query AS (
        SELECT plainto_tsquery('simple', $1) AS value
      ),
      candidates AS MATERIALIZED (
        SELECT
          e.id AS entity_id,
          e.created_at,
          ts_rank(e.search_tsvector, search_query.value) AS bm25
        FROM entities e
        CROSS JOIN search_query
        WHERE ($9::boolean = true OR e.status IS DISTINCT FROM 'archived')
          AND ($2::text IS NULL OR e.type = $2)
          AND ($3::text[] IS NULL OR e.tags @> $3)
          AND ($4::text[] IS NULL OR e.type = ANY($4))
          AND e.visibility = ANY($5)
          AND ($6::text IS NULL OR e.visibility = $6)
          AND ${ownerSqlCondition('e.owner', '$7')}
          AND (
            $10::text IS NULL
            OR (
              e.type = 'memory'
              AND COALESCE(e.metadata->>'memory_role', 'durable_memory') = $10
            )
          )
          AND ${scopedMemoryVisibilitySql('e.metadata', '$11')}
          AND e.search_tsvector @@ search_query.value
          AND EXISTS (SELECT 1 FROM chunks c WHERE c.entity_id = e.id)
        ORDER BY bm25 DESC, e.id
        LIMIT $8
      ),
      normalized AS (
        SELECT
          candidates.*,
          CASE
            WHEN MAX(bm25) OVER () = 0 THEN bm25
            ELSE bm25 / MAX(bm25) OVER ()
          END AS normalized_bm25
        FROM candidates
      ),
      scored AS (
        SELECT
          normalized.*,
          normalized_bm25 * (
            1 + $12::double precision * EXP(
              -EXTRACT(EPOCH FROM ($13::timestamptz - created_at))
              / 86400.0
              / 30.0
            )
          ) AS score
        FROM normalized
      ),
      top_results AS MATERIALIZED (
        SELECT *
        FROM scored
        WHERE score >= $14
        ORDER BY score DESC, entity_id
        LIMIT $15
      )
      SELECT
        e.*,
        first_chunk.content AS chunk_content,
        0::double precision AS similarity,
        top_results.score
      FROM top_results
      JOIN entities e ON e.id = top_results.entity_id
      JOIN LATERAL (
        SELECT c.content
        FROM chunks c
        WHERE c.entity_id = e.id
        ORDER BY c.chunk_index, c.id
        LIMIT 1
      ) first_chunk ON true
      ORDER BY top_results.score DESC, top_results.entity_id
    `,
    [
      ctx.queryText,
      input.type ?? null,
      input.tags?.length ? input.tags : null,
      auth.allowedTypes,
      auth.allowedVisibility,
      input.visibility ?? null,
      input.owner ?? null,
      candidateLimit,
      input.includeArchived ?? false,
      input.memoryRole ?? null,
      auth.clientId,
      ctx.recencyWeight,
      ctx.now,
      ctx.threshold,
      ctx.limit
    ]
  );

  return { results: mapSearchRows(rows.rows) };
}

async function fetchSearchEdgeSummaries(
  pool: Pool,
  auth: AuthContext,
  input: SearchInput,
  resultEntityIds: string[]
): Promise<Map<string, SearchEdgeSummary>> {
  if (resultEntityIds.length === 0) {
    return new Map();
  }

  const rows = await pool.query<SearchEdgeSummaryRow>(
    `
      WITH result_edges AS (
        SELECT e.source_id AS result_entity_id, e.source_id, e.target_id, e.relation
        FROM unnest($1::uuid[]) AS anchor(id)
        JOIN edges e ON e.source_id = anchor.id
        UNION ALL
        SELECT e.target_id AS result_entity_id, e.source_id, e.target_id, e.relation
        FROM unnest($1::uuid[]) AS anchor(id)
        JOIN edges e ON e.target_id = anchor.id
      )
      SELECT result_edges.result_entity_id, result_edges.relation
      FROM result_edges
      JOIN entities src ON src.id = result_edges.source_id
      JOIN entities tgt ON tgt.id = result_edges.target_id
      WHERE src.status IS DISTINCT FROM 'archived'
        AND tgt.status IS DISTINCT FROM 'archived'
        AND ($2::text[] IS NULL OR src.type = ANY($2))
        AND ($2::text[] IS NULL OR tgt.type = ANY($2))
        AND src.visibility = ANY($3)
        AND tgt.visibility = ANY($3)
        AND ${ownerSqlCondition('src.owner', '$4')}
        AND ${ownerSqlCondition('tgt.owner', '$4')}
        AND ${scopedMemoryVisibilitySql('src.metadata', '$5')}
        AND ${scopedMemoryVisibilitySql('tgt.metadata', '$5')}
    `,
    [
      resultEntityIds,
      auth.allowedTypes,
      auth.allowedVisibility,
      input.owner ?? null,
      auth.clientId
    ]
  );

  return buildSearchEdgeSummaries(rows.rows);
}

export function searchEntities(
  pool: Pool,
  auth: AuthContext,
  input: SearchInput,
  options: SearchOptions = {}
): ServiceResult<SearchResponse> {
  return ResultAsync.fromPromise(
    (async () => {
      const startedAt = Date.now();
      requireScope(auth, 'read');

      const query = input.query.trim();
      if (!query) {
        throw new AppError(ErrorCode.VALIDATION, 'Query must not be empty');
      }

      const threshold = input.threshold ?? 0.35;
      const recencyWeight = input.recencyWeight ?? 0.1;
      const limit = input.limit ?? 10;
      const now = options.now?.() ?? new Date();
      const embeddingBudgetMs =
        options.embeddingBudgetMs ?? DEFAULT_EMBEDDING_BUDGET_MS;
      const timings: Record<string, number> = {};
      let cacheStatus: 'bypass' | 'hit' | 'miss' = 'bypass';

      const embeddingService =
        options.embeddingService ?? createEmbeddingService();
      const searchContext = {
        queryText: query,
        threshold,
        recencyWeight,
        limit,
        now
      };

      const lexicalStartedAt = Date.now();
      const lexicalTask = runLexicalSearch(
        pool,
        auth,
        input,
        searchContext
      ).then(
        (value) => {
          timings['lexicalSqlMs'] = Date.now() - lexicalStartedAt;
          return { ok: true as const, value };
        },
        (error: unknown) => {
          timings['lexicalSqlMs'] = Date.now() - lexicalStartedAt;
          return { ok: false as const, error };
        }
      );

      const embeddingTask = (async () => {
        const modelStartedAt = Date.now();
        const activeModel = await embeddingService.getActiveModel(pool);
        timings['activeModelMs'] = Date.now() - modelStartedAt;
        const embeddingStartedAt = Date.now();
        const queryEmbedding = await embeddingService.embedQuery(
          query,
          activeModel,
          {
            ...(auth.apiKeyId ? { cacheScope: auth.apiKeyId } : {}),
            onCacheStatus: (status) => {
              cacheStatus = status;
            }
          }
        );
        timings['embeddingMs'] = Date.now() - embeddingStartedAt;
        return queryEmbedding;
      })();

      type EmbeddingOutcome =
        | { kind: 'ready'; queryEmbedding: number[] }
        | { kind: 'error' }
        | { kind: 'timeout' };

      let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
      const timeoutTask = new Promise<EmbeddingOutcome>((resolve) => {
        timeoutHandle = setTimeout(
          () => resolve({ kind: 'timeout' }),
          embeddingBudgetMs
        );
        timeoutHandle.unref?.();
      });
      const embeddingOutcome = await Promise.race([
        embeddingTask.then(
          (queryEmbedding): EmbeddingOutcome => ({
            kind: 'ready',
            queryEmbedding
          }),
          (): EmbeddingOutcome => ({ kind: 'error' })
        ),
        timeoutTask
      ]);
      if (embeddingOutcome.kind !== 'timeout' && timeoutHandle) {
        clearTimeout(timeoutHandle);
      }

      let results: { results: SearchResult[] };
      let searchMode: SearchMode;
      let fallbackReason: SearchFallbackReason | undefined;

      if (embeddingOutcome.kind === 'ready') {
        const hybridStartedAt = Date.now();
        results = await runHybridSearch(pool, auth, input, {
          ...searchContext,
          queryEmbedding: embeddingOutcome.queryEmbedding
        });
        timings['hybridSqlMs'] = Date.now() - hybridStartedAt;
        searchMode = 'hybrid';
      } else {
        const lexicalOutcome = await lexicalTask;
        if (!lexicalOutcome.ok) {
          throw lexicalOutcome.error;
        }
        results = lexicalOutcome.value;
        searchMode = 'lexical_fallback';
        fallbackReason =
          embeddingOutcome.kind === 'timeout'
            ? 'embedding_timeout'
            : 'embedding_error';
      }

      const edgeStartedAt = Date.now();
      const resultEntityIds = results.results.map((r) => r.entityId);
      if (!input.expandGraph) {
        const edgeSummaries = await fetchSearchEdgeSummaries(
          pool,
          auth,
          input,
          resultEntityIds
        );
        for (const result of results.results) {
          const summary = edgeSummaries.get(result.entityId);
          if (summary) {
            result.edges = summary;
          }
        }
      }

      if (input.expandGraph && results.results.length > 0) {
        // Batch graph expansion: 2 queries total instead of 2N
        const allEdges = await pool.query<{
          source_id: string; target_id: string; relation: string;
        }>(
          'SELECT source_id, target_id, relation FROM edges WHERE source_id = ANY($1) OR target_id = ANY($1)',
          [resultEntityIds]
        );

        // Collect all neighbor IDs and build per-entity edge info
        const edgesByEntityId = new Map<string, Array<{ entityId: string; relation: string; direction: 'outgoing' | 'incoming' }>>();
        const allNeighborIds = new Set<string>();

        for (const edge of allEdges.rows) {
          for (const entityId of resultEntityIds) {
            if (edge.source_id === entityId) {
              if (!edgesByEntityId.has(entityId)) edgesByEntityId.set(entityId, []);
              edgesByEntityId.get(entityId)!.push({ entityId: edge.target_id, relation: edge.relation, direction: 'outgoing' });
              allNeighborIds.add(edge.target_id);
            } else if (edge.target_id === entityId) {
              if (!edgesByEntityId.has(entityId)) edgesByEntityId.set(entityId, []);
              edgesByEntityId.get(entityId)!.push({ entityId: edge.source_id, relation: edge.relation, direction: 'incoming' });
              allNeighborIds.add(edge.source_id);
            }
          }
        }

        if (allNeighborIds.size > 0) {
          const neighbors = await pool.query<{
            id: string; type: string; content: string | null; metadata: Record<string, unknown>;
          }>(
            `SELECT id, type, content, metadata FROM entities
             WHERE id = ANY($1)
               AND status IS DISTINCT FROM 'archived'
               AND ($2::text[] IS NULL OR type = ANY($2))
               AND visibility = ANY($3)
               AND ${ownerSqlCondition('owner', '$4')}
               AND ${scopedMemoryVisibilitySql('metadata', '$5')}`,
            [
              Array.from(allNeighborIds),
              auth.allowedTypes,
              auth.allowedVisibility,
              input.owner ?? null,
              auth.clientId
            ]
          );

          const neighborMap = new Map(neighbors.rows.map((n) => [n.id, n]));
          const edgeSummaryRows: SearchEdgeSummaryRow[] = [];

          for (const result of results.results) {
            const edgeInfo = edgesByEntityId.get(result.entityId);
            if (!edgeInfo) continue;
            const related = edgeInfo
              .map((info) => {
                const entity = neighborMap.get(info.entityId);
                if (!entity) return null;
                return { entity, relation: info.relation, direction: info.direction };
              })
              .filter((r): r is NonNullable<typeof r> => r !== null);
            result.related = related;

            for (const entry of related) {
              edgeSummaryRows.push({
                result_entity_id: result.entityId,
                relation: entry.relation
              });
            }
          }

          const edgeSummaries = buildSearchEdgeSummaries(edgeSummaryRows);
          for (const result of results.results) {
            const summary = edgeSummaries.get(result.entityId);
            if (summary) {
              result.edges = summary;
            }
          }
        }
      }

      timings['edgeMs'] = Date.now() - edgeStartedAt;
      timings['totalMs'] = Date.now() - startedAt;
      const logContext = {
        event: 'search.completed',
        searchMode,
        ...(fallbackReason ? { fallbackReason } : {}),
        cacheStatus,
        resultCount: results.results.length,
        timings
      };
      if (fallbackReason) {
        options.logger?.warn(logContext, 'search used lexical fallback');
      } else {
        options.logger?.debug(logContext, 'search completed');
      }

      return {
        results: results.results,
        searchMode,
        ...(fallbackReason ? { fallbackReason } : {})
      };
    })(),
    (error) => toAppError(error, 'Failed to search entities')
  );
}
