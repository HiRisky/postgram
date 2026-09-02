# Compact Search Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make default agent-facing search results return only discovery metadata and matched chunks, while retaining explicit access to complete entity content through full responses and `recall`.

**Architecture:** Add an `includeContent` hydration control to the search service and expose it as `include_content` on REST. REST keeps its current full-content default for compatibility, while MCP compact search and the current CLI explicitly disable content hydration unless `full_response` / `--full-response` is requested. Compact JSON and TOON omit result and related-entity content unconditionally; agents select IDs from chunks and call `recall` for complete content.

**Tech Stack:** TypeScript 5.9, Node.js 22, PostgreSQL 16 with pgvector, Hono REST, MCP SDK, Commander CLI, Vitest.

**Spec:** `docs/superpowers/specs/2026-06-27-search-edge-affordances-design.md`, with the result-content disclosure amendment recorded under Global Constraints below and approved by Ivo on 2026-09-02.

## Global Constraints

- Preserve search authorization, filtering, ranking, recall, HNSW/exact selection, graph edge summaries, and fallback semantics.
- Default MCP search, `pgm search --json`, `pgm search --toon`, and human-readable `pgm search` must not return full result or related-entity content.
- Keep `chunk` / `chunk_content` in compact output as the evidence used to choose a result.
- `full_response: true` and `pgm search --json --full-response` must retain the legacy API-shaped response, including complete entity content.
- `POST /api/search` without `include_content` must retain its existing full-content behavior for backwards compatibility.
- `POST /api/search` with `include_content: false` must not hydrate or serialize result or related-entity content.
- A new CLI talking to an old server may still download legacy content, but must strip it before model-visible output. An old CLI talking to a new server must retain legacy behavior.
- Full content remains available through `pgm recall <entity-id>` and the MCP `recall` tool.
- Do not add a database migration or runtime configuration value.
- Keep the two copies of search output formatting, `src/util/search-output.ts` and `cli/src/search-output.ts`, behaviorally identical.

---

### Task 1: Selective Search Content Hydration

**Files:**

- Modify: `src/services/search-service.ts`
- Test: `tests/integration/search-service.test.ts`

**Interfaces:**

- Consumes: existing `searchEntities(pool, auth, input, options)` callers.
- Produces: `SearchInput.includeContent?: boolean`; omission means `true` for compatibility. `SearchResult.entity.content` and `SearchResult.related[].entity.content` become optional and are absent when hydration is disabled.

- [ ] **Step 1: Write the failing direct-service test**

Add this case to `tests/integration/search-service.test.ts` using the existing database, auth, and enrichment helpers:

```ts
it('omits entity and related content when hydration is disabled', async () => {
  if (!database) {
    throw new Error('test database not initialized');
  }

  const auth = makeAuthContext();
  const largeContent = `compact retrieval marker ${'x'.repeat(100_000)}`;
  const stored = (
    await storeEntity(database.pool, auth, {
      type: 'document',
      content: largeContent
    })
  )._unsafeUnwrap();
  const neighbor = (
    await storeEntity(database.pool, auth, {
      type: 'project',
      content: `related content ${'y'.repeat(20_000)}`
    })
  )._unsafeUnwrap();
  await createEdge(database.pool, auth, {
    sourceId: stored.id,
    targetId: neighbor.id,
    relation: 'part_of'
  });

  const embeddingService = createEmbeddingService();
  await createEnrichmentWorker({
    pool: database.pool,
    embeddingService
  }).runOnce();

  const compact = await searchEntities(
    database.pool,
    auth,
    {
      query: 'compact retrieval marker',
      threshold: 0,
      expandGraph: true,
      includeContent: false
    },
    { embeddingService }
  );
  const compactHit = compact
    ._unsafeUnwrap()
    .results.find((entry) => entry.entity.id === stored.id);

  expect(compactHit?.chunkContent).toContain('compact retrieval marker');
  expect(compactHit?.entity).not.toHaveProperty('content');
  expect(compactHit?.related?.[0]?.entity).not.toHaveProperty('content');

  const legacy = await searchEntities(
    database.pool,
    auth,
    {
      query: 'compact retrieval marker',
      threshold: 0,
      expandGraph: true
    },
    { embeddingService }
  );
  const legacyHit = legacy
    ._unsafeUnwrap()
    .results.find((entry) => entry.entity.id === stored.id);

  expect(legacyHit?.entity.content).toBe(largeContent);
  expect(legacyHit?.related?.[0]?.entity.content).toContain('related content');
});
```

Use the existing camelCase `createEdge` service input exactly as shown; REST and MCP adapters remain responsible for snake_case conversion.

- [ ] **Step 2: Run the new service test to verify RED**

Run:

```bash
npm test -- tests/integration/search-service.test.ts -t "omits entity and related content when hydration is disabled"
```

Expected: TypeScript or runtime failure because `includeContent` is not supported and content is still present.

- [ ] **Step 3: Add an optional search entity shape**

In `src/services/search-service.ts`, export an entity shape that represents deliberate non-hydration without pretending the stored value is `null`:

```ts
export type SearchEntity = Omit<Entity, 'content'> & {
  content?: string | null;
};

export type SearchResult = {
  entity: SearchEntity;
  entityId: string;
  chunkContent: string;
  similarity: number;
  score: number;
  edges?: SearchEdgeSummary | undefined;
  related?:
    | Array<{
        entity: {
          id: string;
          type: string;
          content?: string | null;
          metadata: Record<string, unknown>;
        };
        relation: string;
        direction: 'outgoing' | 'incoming';
      }>
    | undefined;
};
```

Add `includeContent?: boolean | undefined` to `SearchInput`.

- [ ] **Step 4: Stop projecting result content when it is not requested**

Change `buildHybridSearchSql` to accept `includeContent: boolean` and define the trusted projection before the existing SQL template:

```ts
function buildHybridSearchSql(
  candidateSql: string,
  includeContent: boolean
): string {
  const contentProjection = includeContent ? ', e.content' : '';
```

Inside the existing final `SELECT`, replace only `e.*` with these explicit columns. Leave the surrounding CTEs, joins, and ordering unchanged:

```sql
      e.id,
      e.type,
      e.visibility,
      e.owner,
      e.status,
      e.enrichment_status,
      e.version,
      e.tags,
      e.source,
      e.metadata,
      e.created_at,
      e.updated_at${contentProjection},
```

Pass `input.includeContent ?? true` from `executeHybridSearch`. Update `EntityRow.content` and `mapEntity` so the returned object includes `content` only when the row has that property:

```ts
const entity: SearchEntity = {
  id: row.id,
  type: row.type,
  visibility: row.visibility,
  owner: row.owner,
  status: row.status,
  enrichmentStatus: row.enrichment_status,
  version: row.version,
  tags: row.tags,
  source: row.source,
  metadata: row.metadata,
  createdAt: row.created_at.toISOString(),
  updatedAt: row.updated_at.toISOString(),
  ...('content' in row ? { content: row.content } : {})
};
```

Keep candidate selection and scoring SQL byte-for-byte unchanged.

- [ ] **Step 5: Stop projecting expanded neighbor content when disabled**

Build the neighbor projection from trusted constants, not user input:

```ts
const neighborProjection =
  (input.includeContent ?? true)
    ? 'id, type, content, metadata'
    : 'id, type, metadata';
```

Change the existing query's first line from:

```sql
SELECT id, type, content, metadata FROM entities
```

to:

```ts
`SELECT ${neighborProjection} FROM entities
```

Keep all existing neighbor authorization predicates and query values unchanged. Change the neighbor row type to `content?: string | null`, then map neighbor entities with conditional object spread so `content` is absent rather than `null` when not selected.

- [ ] **Step 6: Run focused tests and typecheck**

Run:

```bash
npm test -- tests/integration/search-service.test.ts
npm run typecheck
```

Expected: all search-service tests and both application/CLI typechecks pass.

- [ ] **Step 7: Commit the service change**

```bash
git add src/services/search-service.ts tests/integration/search-service.test.ts
git commit -m "feat(search): support selective content hydration"
```

---

### Task 2: Backwards-Compatible REST Content Negotiation

**Files:**

- Modify: `src/transport/rest.ts`
- Modify: `cli/src/client.ts`
- Test: `tests/contract/rest-api.test.ts`
- Test: `cli/tests/unit/client.test.ts`

**Interfaces:**

- Consumes: `SearchInput.includeContent` from Task 1.
- Produces: REST request field `include_content?: boolean`; omitted defaults to legacy `true`. `createPgmClient().searchEntities()` accepts and sends the same field. Search response entity content is optional only for search responses; recall/store types remain unchanged.

- [ ] **Step 1: Write failing REST compatibility tests**

Extend the existing `/api/search` contract test with a large document and these assertions:

```ts
const compactResponse = await app.request('/api/search', {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'postgres search',
    tags: ['search'],
    include_content: false
  })
});
const compactBody = (await compactResponse.json()) as {
  results: Array<{ entity: Record<string, unknown>; chunk_content: string }>;
};
expect(compactBody.results[0]?.chunk_content).toContain('postgres');
expect(compactBody.results[0]?.entity).not.toHaveProperty('content');

const legacyResponse = await app.request('/api/search', {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ query: 'postgres search', tags: ['search'] })
});
const legacyBody = (await legacyResponse.json()) as {
  results: Array<{ entity: { content?: string | null } }>;
};
expect(legacyBody.results[0]?.entity.content).toContain('postgres');
```

Also send `include_content: 'false'` and assert HTTP 400 so REST does not silently coerce malformed JSON values.

- [ ] **Step 2: Run the REST test to verify RED**

Run:

```bash
npm test -- tests/contract/rest-api.test.ts -t "search"
```

Expected: FAIL because the schema strips `include_content` and content remains present.

- [ ] **Step 3: Add REST request and response handling**

Add the field to `searchEntitiesSchema`:

```ts
include_content: z.boolean().optional();
```

Pass `includeContent: body.include_content ?? true` to `searchEntities`. Add a search-specific serializer that preserves the legacy entity shape while omitting only `content` when requested:

```ts
function toSearchStoredEntity(entity: SearchEntity, includeContent: boolean) {
  return {
    id: entity.id,
    type: entity.type,
    ...(includeContent ? { content: entity.content ?? null } : {}),
    visibility: entity.visibility,
    owner: entity.owner,
    status: entity.status,
    enrichment_status: entity.enrichmentStatus,
    version: entity.version,
    tags: entity.tags,
    source: entity.source,
    metadata: entity.metadata,
    created_at: entity.createdAt,
    updated_at: entity.updatedAt
  };
}
```

Use the same boolean to omit `related[].entity.content`. Do not change `toStoredEntity`, because store, recall, update, list, and task endpoints still return complete entities.

- [ ] **Step 4: Add the CLI REST-client request contract test**

Create `cli/tests/unit/client.test.ts`. Stub `globalThis.fetch`, call `createPgmClient`, and verify the body is passed exactly:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createPgmClient } from '../../src/client.js';

describe('pgm REST client', () => {
  afterEach(() => vi.restoreAllMocks());

  it('sends include_content for search requests', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ results: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    );
    const client = createPgmClient({
      apiUrl: 'https://postgram.test',
      apiKey: 'test-key'
    });

    await client.searchEntities({
      query: 'compact retrieval',
      include_content: false
    });

    const init = fetchMock.mock.calls[0]?.[1];
    expect(JSON.parse(String(init?.body))).toMatchObject({
      query: 'compact retrieval',
      include_content: false
    });
  });
});
```

Add `include_content?: boolean | undefined` to the CLI client input and make search response `entity.content` and related `entity.content` optional. Do not loosen `StoredEntityResponse`; define a search-only entity response type.

- [ ] **Step 5: Run REST and CLI-client checks**

Run:

```bash
npm test -- tests/contract/rest-api.test.ts -t "search"
npm test --workspace @ivotoby/postgram-cli -- tests/unit/client.test.ts
npm run typecheck
```

Expected: all pass.

- [ ] **Step 6: Commit REST negotiation**

```bash
git add src/transport/rest.ts cli/src/client.ts tests/contract/rest-api.test.ts cli/tests/unit/client.test.ts
git commit -m "feat(search): negotiate content hydration over rest"
```

---

### Task 3: Truly Compact Search JSON And TOON

**Files:**

- Modify: `src/util/search-output.ts`
- Modify: `cli/src/search-output.ts`
- Test: `tests/unit/search-output.test.ts`
- Test: `tests/contract/mcp-tools.test.ts`

**Interfaces:**

- Consumes: search entities whose content may be absent from Tasks 1 and 2.
- Produces: compact result shape `{ id, type, score, chunk, tags?, edges?, related? }`. Compact related rows contain `{ id, type, relation, direction }`. Full response bypasses compact formatting.

- [ ] **Step 1: Change unit expectations to the intended compact contract**

In `tests/unit/search-output.test.ts`, change the compact expected result to omit both content fields:

```ts
expect(compactSearchResponse(fullSearchResponse)).toEqual({
  results: [
    {
      id: '01234567-89ab-cdef-0123-456789abcdef',
      type: 'memory',
      score: 0.88,
      chunk: 'token compact search response shape',
      tags: ['tokens'],
      edges: {
        count: 3,
        relations: [
          { relation: 'mentioned_in', count: 2 },
          { relation: 'depends_on', count: 1 }
        ]
      },
      related: [
        {
          id: 'fedcba98-7654-3210-fedc-ba9876543210',
          type: 'project',
          relation: 'part_of',
          direction: 'outgoing'
        }
      ]
    }
  ]
});
```

Change the TOON header expectation to:

```ts
'results[1]{id,type,score,chunk,tags,edges,related}:';
```

Change the related header expectation to:

```ts
'related[1]{id,type,relation,direction}:';
```

Add assertions that neither direct nor related full content appears in TOON.

- [ ] **Step 2: Run formatter tests to verify RED**

Run:

```bash
npm test -- tests/unit/search-output.test.ts
```

Expected: FAIL because compact JSON and TOON still serialize `content`.

- [ ] **Step 3: Remove content from both compact formatters**

Apply the same type and mapper change to `src/util/search-output.ts` and `cli/src/search-output.ts`:

```ts
export type CompactSearchResult = {
  id: string;
  type: string;
  score: number;
  chunk: string;
  tags?: string[];
  edges?: CompactSearchEdgeSummary;
  related?: CompactRelatedResult[];
};

type CompactRelatedResult = {
  id: string;
  type: string;
  relation: string;
  direction: string;
};
```

Delete `content: entry.entity.content` and `content: related.entity.content` from `compactSearchResponse`. Remove the two content columns and values from `searchResponseToToon`.

After editing, verify both copies remain identical:

```bash
diff -u src/util/search-output.ts cli/src/search-output.ts
```

Expected: no diff.

- [ ] **Step 4: Write failing MCP disclosure assertions**

In the existing MCP test `supports full-response and TOON search output via MCP arguments`:

```ts
expect(fullHit?.entity.content).toBe('token compact search response shape');
expect(compactHit).not.toHaveProperty('content');
expect(toonText).not.toContain(',content,');
expect(toonText).toContain(',chunk,');
```

In its expanded-search assertions, require compact related rows not to have `content`. Keep the full-response metadata, similarity, and chunk assertions.

- [ ] **Step 5: Disable hydration for default MCP search**

In `src/transport/mcp.ts`, pass:

```ts
includeContent: args.full_response === true;
```

into the `searchEntities` input. Compact and TOON paths continue through `compactSearchResponse`; `full_response` continues through `toToolSuccess(payload)`. Update the MCP tool description for `full_response` to say it includes full entity content and tell agents to prefer `recall` after selecting a result.

- [ ] **Step 6: Run formatter and MCP tests**

Run:

```bash
npm test -- tests/unit/search-output.test.ts
npm test -- tests/contract/mcp-tools.test.ts -t "supports full-response and TOON search output via MCP arguments"
npm run typecheck
```

Expected: all pass.

- [ ] **Step 7: Commit compact MCP output**

```bash
git add src/util/search-output.ts cli/src/search-output.ts src/transport/mcp.ts tests/unit/search-output.test.ts tests/contract/mcp-tools.test.ts
git commit -m "feat(search): omit content from compact results"
```

---

### Task 4: CLI Search-To-Recall Flow

**Files:**

- Modify: `cli/src/pgm.ts`
- Test: `tests/integration/cli-pgm.test.ts`
- Test: `tests/unit/cli-help.test.ts`

**Interfaces:**

- Consumes: REST `include_content` and compact formatters from Tasks 2 and 3.
- Produces: all default CLI search modes send `include_content: false`; `--full-response` sends `include_content: true`. Existing CLI syntax remains valid and no new flag is added.

- [ ] **Step 1: Write a failing CLI request-body contract test**

Import `Hono` in `tests/integration/cli-pgm.test.ts`, then add a fake search server which records request bodies:

```ts
it('requests chunk-only search unless full response is selected', async () => {
  const requestBodies: Array<Record<string, unknown>> = [];
  const app = new Hono();
  app.post('/api/search', async (context) => {
    requestBodies.push((await context.req.json()) as Record<string, unknown>);
    return context.json({ results: [] });
  });

  let fakeBaseUrl = '';
  const fakeServer = serve(
    { fetch: app.fetch, hostname: '127.0.0.1', port: 0 },
    (info) => {
      fakeBaseUrl = `http://${info.address}:${info.port}`;
    }
  );
  await vi.waitFor(() => expect(fakeBaseUrl).not.toBe(''));

  try {
    const env = {
      PGM_API_URL: fakeBaseUrl,
      PGM_API_KEY: 'test-key'
    };
    await runPgm(['search', 'compact retrieval', '--json'], env);
    await runPgm(
      ['search', 'compact retrieval', '--json', '--full-response'],
      env
    );
  } finally {
    await new Promise<void>((resolve, reject) => {
      fakeServer.close((error) => {
        if (error) reject(error);
        else resolve();
      });
    });
  }

  expect(requestBodies.map((body) => body.include_content)).toEqual([
    false,
    true
  ]);
});
```

Add `vi` to the Vitest import in this file.

- [ ] **Step 2: Change CLI output expectations**

Update `supports full-response, TOON, and discoverable help for search output` so the compact JSON result type has no `content`, while chunk and score remain:

```ts
expect(firstCompactResult).toMatchObject({
  id: stored.id,
  type: 'memory',
  chunk: 'token compact search response shape'
});
expect(firstCompactResult).not.toHaveProperty('content');
```

Require the full response to include the stored content:

```ts
expect(full.results[0]?.entity.content).toBe(
  'token compact search response shape'
);
```

Require TOON to use the content-free header and still include the chunk:

```ts
expect(toonResult.stdout).toContain(
  'results[1]{id,type,score,chunk,tags,edges,related}:'
);
expect(toonResult.stdout).toContain('token compact search response shape');
```

Add a long-tail marker after the searchable first sentence and assert the marker is absent from compact JSON, TOON, and human-readable output but present in `--json --full-response`.

- [ ] **Step 3: Run the request-body test to verify RED**

Run:

```bash
npm test -- tests/integration/cli-pgm.test.ts -t "requests chunk-only search unless full response is selected"
```

Expected: FAIL because both request bodies omit `include_content`.

- [ ] **Step 4: Send the hydration intent from `pgm search`**

Add this field to the call in `cli/src/pgm.ts`:

```ts
include_content: options.fullResponse === true;
```

Keep `--full-response` as the only opt-in for full search content. Do not add `--include-content`; it would duplicate the established disclosure control.

Update the search description and option help:

```ts
.description('Search stored entities (chunks by default; full content with --json --full-response)')
```

```ts
.option(
  '--full-response',
  'emit the full API response with complete entity content when used with --json'
)
```

Update `tests/unit/cli-help.test.ts` for the new description.

- [ ] **Step 5: Run CLI checks**

Run:

```bash
npm test -- tests/integration/cli-pgm.test.ts -t "supports full-response, TOON, and discoverable help for search output"
npm test -- tests/unit/cli-help.test.ts
npm run typecheck
```

Expected: all pass.

- [ ] **Step 6: Commit CLI retrieval behavior**

```bash
git add cli/src/pgm.ts tests/integration/cli-pgm.test.ts tests/unit/cli-help.test.ts
git commit -m "feat(cli): default search to matched chunks"
```

---

### Task 5: Retrieval Guidance And Regression Validation

**Files:**

- Modify: `skill/postgram/SKILL.md`
- Modify: `README.md`
- Modify: `cli/README.md`
- Modify: `templates/AGENTS.md`
- Modify: `templates/CLAUDE.md`
- Modify: `docs/optimized-system-profile.md`

**Interfaces:**

- Consumes: compact search and explicit full-response behavior from Tasks 1–4.
- Produces: one consistent agent workflow: narrow search, inspect chunks, recall only selected IDs.

- [ ] **Step 1: Update the repository skill guidance**

Under `### Search` in `skill/postgram/SKILL.md`, add:

```markdown
Search is a discovery step. Compact JSON and TOON contain matched chunks, not
complete entity bodies. Start with a specific query and `--limit 5`, inspect
`id`, `score`, `chunk`, tags, and edge summaries, then run `pgm recall <id>`
only for the results needed to answer the user.

Do not request `--full-response` just to read a search hit. It can inline large
documents and graph-neighbor content. Use it only when a machine consumer needs
the complete legacy search envelope.
```

Add the equivalent MCP wording: default `search`, then `recall`; use `full_response: true` only for explicit envelope consumers.

- [ ] **Step 2: Update product and CLI documentation**

In `README.md` and `cli/README.md`, document these exact examples:

```bash
pgm search "database decisions" --limit 5 --toon
pgm recall <selected-entity-id>
pgm search "database decisions" --json --full-response
```

State that:

- compact JSON, TOON, and human output contain matched chunks and identifiers;
- `--full-response` includes complete entity bodies;
- REST remains backwards compatible and accepts `include_content: false` for chunk-only retrieval;
- `expand_graph` exposes neighbor identity/relation in compact output, not neighbor bodies.

Update the MCP compact-output section with the same search-to-recall guidance.

- [ ] **Step 3: Update bundled agent templates**

In `templates/AGENTS.md`, `templates/CLAUDE.md`, and `docs/optimized-system-profile.md`, replace guidance that implies compact search contains complete stored content with:

```markdown
Use search to rank compact matched chunks. Recall a selected entity by ID when
you need its complete content. Avoid full_response for normal retrieval.
```

Keep existing visibility, memory-role, graph expansion, and TOON instructions.

- [ ] **Step 4: Run documentation and repository checks**

Run:

```bash
rg -n "search.*recall|full.response|include_content|matched chunk" \
  README.md cli/README.md skill/postgram/SKILL.md \
  templates/AGENTS.md templates/CLAUDE.md docs/optimized-system-profile.md
npm run lint
npm run typecheck
npm test
npm run build
```

Expected: documentation references the same contract; lint, typecheck, all tests, and build pass.

- [ ] **Step 5: Verify payload behavior against `10.0.1.3` after deployment approval**

Run only after the change is deployed to the internal VM:

```bash
export PGM_API_URL='http://10.0.1.3:3100'
QUERY='postgram production external docker container internal large dataset testing reduce search latency'

pgm --json search "$QUERY" --limit 10 \
  | jq '{bytes: (tostring | length), has_content: any(.results[]; has("content")), ids: [.results[].id]}'

pgm --json search "$QUERY" --limit 10 --full-response \
  | jq '{bytes: (tostring | length), has_content: any(.results[]; .entity | has("content")), ids: [.results[].entity.id]}'
```

Expected:

- both calls return the same ranked IDs in the same order;
- compact `has_content` is false;
- full `has_content` is true;
- compact bytes are bounded by matched chunks and metadata, not the 265,532 characters of entity bodies observed on 2026-09-02;
- no request is sent to `postgram.cloud.toby.nu` during this deployment check;
- document bodies are consumed by `jq` and never printed or saved.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md cli/README.md skill/postgram/SKILL.md templates/AGENTS.md templates/CLAUDE.md docs/optimized-system-profile.md
git commit -m "docs(search): document progressive content retrieval"
```

---

## Final Review Gate

- [ ] Confirm `git diff --check` passes.
- [ ] Confirm compact and full searches produce identical IDs, scores, chunks, edge summaries, and ordering.
- [ ] Confirm only content hydration/serialization differs; authorization and search strategy code are unchanged.
- [ ] Confirm REST requests that omit `include_content` still receive the legacy response.
- [ ] Confirm compact MCP, CLI JSON, CLI TOON, and human output never contain complete entity or related content.
- [ ] Confirm `full_response` and `recall` still return complete content.
- [ ] Run Codex review and resolve all P0, P1, and P2 findings before push.
