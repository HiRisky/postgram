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
    const requestBody = request?.body;
    if (typeof requestBody !== 'string') {
      throw new Error('expected a JSON request body');
    }
    const parsedBody: unknown = JSON.parse(requestBody);
    expect(parsedBody).toMatchObject({
      query: 'compact retrieval',
      include_content: false
    });
  });
});
