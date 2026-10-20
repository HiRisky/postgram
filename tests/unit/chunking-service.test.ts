import { describe, expect, it } from 'vitest';

import { chunkText } from '../../src/services/chunking-service.js';

describe('chunkText', () => {
  it('returns a single chunk for short content', () => {
    const chunks = chunkText('short note about pgvector');

    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({
      chunkIndex: 0,
      content: 'short note about pgvector'
    });
    expect(chunks[0]?.tokenCount).toBeGreaterThan(0);
  });

  it('splits long content with overlap and stable ordering', () => {
    const sentence = 'postgres vector search makes retrieval fast and local. ';
    const content = sentence.repeat(20);

    const chunks = chunkText(content, {
      chunkSize: 120,
      overlap: 30
    });

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.map((chunk) => chunk.chunkIndex)).toEqual(
      chunks.map((_, index) => index)
    );
    expect(chunks[1]?.content).toContain(
      chunks[0]?.content.slice(-20).trim().split(/\s+/)[0] ?? ''
    );
  });

  describe('surrogate pairs', () => {
    const isWellFormed = (text: string): boolean =>
      !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(text);

    it('never splits an emoji across a chunk boundary', () => {
      // Every 2nd UTF-16 unit boundary is eligible; sweep sizes and overlaps so
      // some boundary lands between the two halves of a pair.
      const content = '😀'.repeat(400);
      for (const chunkSize of [99, 100, 101, 150, 301]) {
        for (const overlap of [0, 7, 30, 51]) {
          const chunks = chunkText(content, { chunkSize, overlap });
          expect(chunks.length).toBeGreaterThan(1);
          for (const chunk of chunks) {
            expect(isWellFormed(chunk.content)).toBe(true);
            expect(chunk.content).toMatch(/^(?:😀)+$/u);
          }
        }
      }
    });

    it('terminates when chunkSize is smaller than a code point', () => {
      const chunks = chunkText('😀😀😀', { chunkSize: 1, overlap: 0 });

      expect(chunks.map((chunk) => chunk.content)).toEqual(['😀', '😀', '😀']);
    });

    it('keeps JSON payloads free of lone surrogate escapes', () => {
      const chunks = chunkText('😀'.repeat(400), {
        chunkSize: 101,
        overlap: 33
      });
      expect(chunks.length).toBeGreaterThan(1);
      for (const chunk of chunks) {
        expect(JSON.stringify(chunk.content)).not.toMatch(/\\ud[89ab]/i);
        expect(JSON.stringify(chunk.content)).not.toMatch(/\\ud[c-f]/i);
      }
    });

    it('replaces lone surrogates already present in the input', () => {
      const chunks = chunkText('before \uD83D after \uDE00 end');

      expect(chunks).toHaveLength(1);
      expect(isWellFormed(chunks[0]?.content ?? '')).toBe(true);
      expect(chunks[0]?.content).toBe('before \uFFFD after \uFFFD end');
    });

    it('leaves valid text unchanged', () => {
      expect(chunkText('plain 😀 text ✓')[0]?.content).toBe('plain 😀 text ✓');
    });
  });
});
