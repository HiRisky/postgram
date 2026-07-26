import { createHash } from 'node:crypto';

import argon2 from 'argon2';

/**
 * Memoizes the argon2 comparison performed on every authenticated request.
 *
 * `argon2id` at the library defaults (m=64MiB, t=3, p=4) costs ~55ms of CPU on
 * a 2 vCPU host, and node-argon2 runs it on the libuv threadpool — which has
 * four slots and is shared with `dns.lookup()`. A handful of concurrent
 * requests therefore delays DNS resolution for outbound provider calls by
 * hundreds of milliseconds, which shows up as embedding latency rather than as
 * auth latency. See benchmarks/auth-verification.ts.
 *
 * Password stretching exists to make low-entropy secrets expensive to guess.
 * API keys here are 32 characters drawn from a 62-symbol alphabet (~190 bits),
 * so the stretching buys nothing against a brute-force attacker that a single
 * hash would not already buy. Rather than migrate the stored hash format, this
 * caches the comparison.
 *
 * The cache entry is keyed by the stored hash *and* the presented secret, so
 * it cannot outlive the credential it describes:
 *
 * - Revoking or deactivating a key removes it from the `is_active` lookup that
 *   runs before this, so revocation stays immediate.
 * - Rotating a key changes `key_hash`, which changes the cache key, so the new
 *   secret is verified from scratch.
 * - Scope and visibility changes are read from the row on every request and
 *   are never cached here.
 *
 * Negative results are cached too. Without that, a caller repeatedly presenting
 * a wrong secret whose prefix matches a real key would force a fresh argon2
 * verification per request — 55ms of CPU each on the deployment host, which is
 * a cheap way to saturate two cores.
 */

type CacheEntry = {
  /**
   * Stored as the in-flight promise rather than the resolved value so that
   * concurrent requests presenting the same key collapse into one argon2 call.
   * Eight simultaneous cold verifications cost 651ms of wall time on 2 vCPUs;
   * deduplicated they cost one verification.
   */
  result: Promise<boolean>;
  expiresAt: number;
};

const DEFAULT_MAX_ENTRIES = 1_000;
const DEFAULT_TTL_MS = 5 * 60_000;

export type KeyVerificationCacheOptions = {
  maxEntries?: number;
  ttlMs?: number;
  /** Injectable for tests; defaults to the real argon2 comparison. */
  verify?: (keyHash: string, plaintextKey: string) => Promise<boolean>;
  now?: () => number;
};

export type KeyVerificationCache = {
  verify: (keyHash: string, plaintextKey: string) => Promise<boolean>;
  clear: () => void;
  readonly stats: { hits: number; misses: number; size: number };
};

function cacheKeyFor(keyHash: string, plaintextKey: string): string {
  // The stored hash embeds its own salt, which is what binds an entry to one
  // specific credential. A NUL separator keeps the two fields unambiguous.
  return createHash('sha256')
    .update(keyHash)
    .update('\0')
    .update(plaintextKey)
    .digest('hex');
}

export function createKeyVerificationCache(
  options: KeyVerificationCacheOptions = {}
): KeyVerificationCache {
  const maxEntries = options.maxEntries ?? DEFAULT_MAX_ENTRIES;
  const ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
  const now = options.now ?? Date.now;
  const verifyImpl =
    options.verify ??
    ((keyHash: string, plaintextKey: string) =>
      argon2.verify(keyHash, plaintextKey));

  // Insertion-ordered, so the first key returned by `keys()` is the least
  // recently used once hits re-insert their entry.
  const entries = new Map<string, CacheEntry>();
  const counters = { hits: 0, misses: 0 };

  const evictIfNeeded = (): void => {
    while (entries.size > maxEntries) {
      const oldest = entries.keys().next();
      if (oldest.done) return;
      entries.delete(oldest.value);
    }
  };

  return {
    async verify(keyHash: string, plaintextKey: string): Promise<boolean> {
      const cacheKey = cacheKeyFor(keyHash, plaintextKey);
      const timestamp = now();
      const existing = entries.get(cacheKey);

      if (existing && existing.expiresAt > timestamp) {
        counters.hits += 1;
        // Re-insert to move this entry to the most-recently-used end.
        entries.delete(cacheKey);
        entries.set(cacheKey, existing);
        return existing.result;
      }

      counters.misses += 1;
      const result = verifyImpl(keyHash, plaintextKey);
      const entry: CacheEntry = { result, expiresAt: timestamp + ttlMs };
      entries.delete(cacheKey);
      entries.set(cacheKey, entry);
      evictIfNeeded();

      try {
        return await result;
      } catch (error) {
        // A failed verification attempt says nothing about the credential —
        // don't let a transient error pin a rejected promise in the cache.
        if (entries.get(cacheKey) === entry) {
          entries.delete(cacheKey);
        }
        throw error;
      }
    },
    clear(): void {
      entries.clear();
      counters.hits = 0;
      counters.misses = 0;
    },
    get stats() {
      return { hits: counters.hits, misses: counters.misses, size: entries.size };
    }
  };
}

/** Process-wide cache used by the request path. */
export const keyVerificationCache = createKeyVerificationCache();
