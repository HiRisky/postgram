import { afterEach, describe, expect, it, vi } from 'vitest';

import { createPgmClient } from '../../src/client.js';

describe('pgm REST client', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('sends the selected content mode with search requests', async () => {
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

    const request = fetchMock.mock.calls[0]?.[1];
    expect(JSON.parse(String(request?.body))).toMatchObject({
      query: 'compact retrieval',
      include_content: false
    });
  });
});
