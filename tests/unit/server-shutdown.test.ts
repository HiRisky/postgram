import { describe, expect, it } from 'vitest';

import { closeServerResources } from '../../src/index.js';

describe('closeServerResources', () => {
  it('drains cache writes after requests stop and before closing the pool', async () => {
    const events: string[] = [];

    await closeServerResources({
      closeHttpServer: () => {
        events.push('http');
        return Promise.resolve();
      },
      flushPendingWrites: () => {
        events.push('cache');
        return Promise.resolve();
      },
      closePool: () => {
        events.push('pool');
        return Promise.resolve();
      }
    });

    expect(events).toEqual(['http', 'cache', 'pool']);
  });
});
