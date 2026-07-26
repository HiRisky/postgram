/**
 * Measures what API-key verification costs, and what it costs everything else.
 *
 * `validateKey` runs `argon2.verify` on every authenticated request. argon2 is
 * deliberately expensive in CPU *and* memory, and node-argon2 executes it on
 * the libuv threadpool — four slots by default, shared with `dns.lookup()`.
 * Outbound provider calls (embeddings) resolve their hostname through that same
 * threadpool, so concurrent authentication delays them without ever showing up
 * as event-loop lag. That makes it invisible to the usual "is the loop
 * blocked?" instrumentation, and makes the delay land in `embeddingMs` rather
 * than in any auth metric.
 *
 * The `dns.lookup` probe below is the load-bearing measurement: it stands in
 * for the DNS resolution an outbound embedding call has to perform, and it is
 * reported next to event-loop delay to show that the two disagree.
 *
 * Run:  npx tsx benchmarks/auth-verification.ts
 * Gate: npx tsx benchmarks/auth-verification.ts --assert
 */
import argon2 from 'argon2';
import dns from 'node:dns';
import { monitorEventLoopDelay, performance } from 'node:perf_hooks';

import { createKeyVerificationCache } from '../src/auth/key-verification-cache.js';

const CONCURRENCY_LEVELS = [1, 2, 4, 8, 16];
const SAMPLE_KEY = 'pgm-benchmark-0123456789ABCDEFGHIJKLMNOPQRSTUV';

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.floor((sorted.length - 1) * fraction)] ?? 0;
}

function summarize(samples: number[]): {
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
} {
  const sorted = samples.slice().sort((left, right) => left - right);
  const round = (value: number) => Number(value.toFixed(2));
  return {
    p50_ms: round(percentile(sorted, 0.5)),
    p95_ms: round(percentile(sorted, 0.95)),
    max_ms: round(sorted.at(-1) ?? 0)
  };
}

/** Stands in for the DNS resolution an outbound embedding call must perform. */
function timedLookup(): Promise<number> {
  const started = performance.now();
  return new Promise((resolve) =>
    dns.lookup('localhost', () => resolve(performance.now() - started))
  );
}

type Verifier = (keyHash: string, plaintextKey: string) => Promise<boolean>;

async function measure(
  verify: Verifier,
  keyHash: string,
  concurrency: number
): Promise<{
  auth: { p50_ms: number; p95_ms: number; max_ms: number };
  wall_ms: number;
  dns_ms: number;
  event_loop_max_ms: number;
}> {
  const samples: number[] = [];
  const histogram = monitorEventLoopDelay({ resolution: 1 });
  histogram.enable();

  const started = performance.now();
  const inflight = Array.from({ length: concurrency }, async () => {
    const requestStarted = performance.now();
    const valid = await verify(keyHash, SAMPLE_KEY);
    if (!valid) throw new Error('benchmark key failed to verify');
    samples.push(performance.now() - requestStarted);
  });
  // Issued while the verifications are in flight, exactly as an embedding call
  // would be if it were triggered by one of those requests.
  const dnsMs = await timedLookup();
  await Promise.all(inflight);
  const wall = performance.now() - started;

  histogram.disable();
  return {
    auth: summarize(samples),
    wall_ms: Number(wall.toFixed(2)),
    dns_ms: Number(dnsMs.toFixed(2)),
    event_loop_max_ms: Number((histogram.max / 1_000_000).toFixed(2))
  };
}

const keyHash = await argon2.hash(SAMPLE_KEY, { type: argon2.argon2id });
const hashParams = keyHash.split('$').slice(0, 4).join('$');

// Warm the native binding so the first measured call isn't paying for it.
await argon2.verify(keyHash, SAMPLE_KEY);

const uncached: Verifier = (hash, plaintext) => argon2.verify(hash, plaintext);

const baselineDns = await timedLookup();

const before: Record<string, unknown> = {};
for (const concurrency of CONCURRENCY_LEVELS) {
  before[`concurrency_${concurrency}`] = await measure(
    uncached,
    keyHash,
    concurrency
  );
}

const after: Record<string, unknown> = {};
for (const concurrency of CONCURRENCY_LEVELS) {
  // A fresh cache per level, so each level pays exactly one cold verification
  // and the rest are hits — the steady state a long-lived process sees.
  const cache = createKeyVerificationCache();
  await cache.verify(keyHash, SAMPLE_KEY);
  after[`concurrency_${concurrency}`] = await measure(
    (hash, plaintext) => cache.verify(hash, plaintext),
    keyHash,
    concurrency
  );
}

// Cold cache under concurrency: proves the in-flight promise is shared rather
// than every caller starting its own argon2 verification.
const herdCache = createKeyVerificationCache();
const herd = await measure((hash, plaintext) => herdCache.verify(hash, plaintext), keyHash, 16);

const report = {
  argon2_params: hashParams,
  threadpool_size: process.env.UV_THREADPOOL_SIZE ?? '4 (default)',
  cpus: (await import('node:os')).cpus().length,
  dns_lookup_idle_ms: Number(baselineDns.toFixed(2)),
  before_cache: before,
  after_cache: after,
  cold_cache_16_concurrent: {
    ...herd,
    argon2_verifications: herdCache.stats.misses
  }
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);

if (process.argv.includes('--assert')) {
  const failures: string[] = [];

  // The cache must remove argon2 from the warm path entirely: a warm
  // verification is a SHA-256 and a map lookup, not a key-stretching function.
  const warm = after['concurrency_16'] as { auth: { p95_ms: number } };
  if (warm.auth.p95_ms > 5) {
    failures.push(`warm verification p95 ${warm.auth.p95_ms}ms exceeded 5ms`);
  }

  // Concurrent cold verifications must collapse into one argon2 call.
  if (herdCache.stats.misses !== 1) {
    failures.push(
      `16 concurrent cold verifications ran ${herdCache.stats.misses} argon2 calls; expected 1`
    );
  }

  if (failures.length > 0) {
    throw new Error(failures.join('; '));
  }
}
