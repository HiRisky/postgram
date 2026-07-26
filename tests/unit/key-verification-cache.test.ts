import { describe, expect, it } from 'vitest';

import { createKeyVerificationCache } from '../../src/auth/key-verification-cache.js';

const HASH_A = '$argon2id$v=19$m=65536,t=3,p=4$c2FsdEFBQUE$aaaaaaaaaaaaaaaaaaaa';
const HASH_B = '$argon2id$v=19$m=65536,t=3,p=4$c2FsdEJCQkI$bbbbbbbbbbbbbbbbbbbb';
const SECRET = 'pgm-agent-0123456789ABCDEFGHIJKLMNOPQRSTUV';

function countingVerifier(result: (hash: string, secret: string) => boolean) {
  const calls: Array<{ hash: string; secret: string }> = [];
  return {
    calls,
    verify: (hash: string, secret: string) => {
      calls.push({ hash, secret });
      return Promise.resolve(result(hash, secret));
    }
  };
}

describe('key-verification-cache', () => {
  it('runs the underlying verification only once for a repeated key', async () => {
    const verifier = countingVerifier(() => true);
    const cache = createKeyVerificationCache({ verify: verifier.verify });

    expect(await cache.verify(HASH_A, SECRET)).toBe(true);
    expect(await cache.verify(HASH_A, SECRET)).toBe(true);
    expect(await cache.verify(HASH_A, SECRET)).toBe(true);

    expect(verifier.calls).toHaveLength(1);
    expect(cache.stats).toMatchObject({ hits: 2, misses: 1 });
  });

  it('caches negative results so a wrong secret cannot force repeated hashing', async () => {
    const verifier = countingVerifier(() => false);
    const cache = createKeyVerificationCache({ verify: verifier.verify });

    expect(await cache.verify(HASH_A, 'wrong-secret')).toBe(false);
    expect(await cache.verify(HASH_A, 'wrong-secret')).toBe(false);

    expect(verifier.calls).toHaveLength(1);
  });

  it('treats a different secret against the same hash as a separate entry', async () => {
    const verifier = countingVerifier((_hash, secret) => secret === SECRET);
    const cache = createKeyVerificationCache({ verify: verifier.verify });

    expect(await cache.verify(HASH_A, SECRET)).toBe(true);
    expect(await cache.verify(HASH_A, 'other-secret')).toBe(false);

    expect(verifier.calls).toHaveLength(2);
  });

  it('re-verifies when the stored hash changes, so a rotated key is not served from cache', async () => {
    const verifier = countingVerifier((hash) => hash === HASH_A);
    const cache = createKeyVerificationCache({ verify: verifier.verify });

    expect(await cache.verify(HASH_A, SECRET)).toBe(true);
    // Same secret presented, but the row now stores a different hash.
    expect(await cache.verify(HASH_B, SECRET)).toBe(false);

    expect(verifier.calls).toHaveLength(2);
  });

  it('collapses concurrent verifications of the same key into one call', async () => {
    let resolveVerification: ((value: boolean) => void) | undefined;
    let calls = 0;
    const cache = createKeyVerificationCache({
      verify: () => {
        calls += 1;
        return new Promise<boolean>((resolve) => {
          resolveVerification = resolve;
        });
      }
    });

    const inflight = Promise.all(
      Array.from({ length: 16 }, () => cache.verify(HASH_A, SECRET))
    );
    // All sixteen callers must be waiting on the same verification.
    expect(calls).toBe(1);

    resolveVerification?.(true);
    expect(await inflight).toEqual(Array.from({ length: 16 }, () => true));
    expect(calls).toBe(1);
  });

  it('re-verifies once an entry has expired', async () => {
    const verifier = countingVerifier(() => true);
    let clock = 1_000;
    const cache = createKeyVerificationCache({
      verify: verifier.verify,
      ttlMs: 500,
      now: () => clock
    });

    await cache.verify(HASH_A, SECRET);
    clock += 499;
    await cache.verify(HASH_A, SECRET);
    expect(verifier.calls).toHaveLength(1);

    clock += 2;
    await cache.verify(HASH_A, SECRET);
    expect(verifier.calls).toHaveLength(2);
  });

  it('evicts least-recently-used entries beyond the size limit', async () => {
    const verifier = countingVerifier(() => true);
    const cache = createKeyVerificationCache({
      verify: verifier.verify,
      maxEntries: 2
    });

    await cache.verify(HASH_A, 'secret-1');
    await cache.verify(HASH_A, 'secret-2');
    // Touch secret-1 so secret-2 becomes the least recently used.
    await cache.verify(HASH_A, 'secret-1');
    await cache.verify(HASH_A, 'secret-3');

    expect(cache.stats.size).toBe(2);
    // secret-2 was evicted and must be verified again.
    await cache.verify(HASH_A, 'secret-2');
    expect(verifier.calls.map((call) => call.secret)).toEqual([
      'secret-1',
      'secret-2',
      'secret-3',
      'secret-2'
    ]);
  });

  it('does not cache a failed verification attempt', async () => {
    let attempt = 0;
    const cache = createKeyVerificationCache({
      verify: () => {
        attempt += 1;
        return attempt === 1
          ? Promise.reject(new Error('argon2 unavailable'))
          : Promise.resolve(true);
      }
    });

    await expect(cache.verify(HASH_A, SECRET)).rejects.toThrow(
      'argon2 unavailable'
    );
    // The transient failure must not have poisoned the entry.
    expect(await cache.verify(HASH_A, SECRET)).toBe(true);
    expect(attempt).toBe(2);
  });
});
