export type ChunkTextOptions = {
  chunkSize?: number;
  overlap?: number;
  separators?: string[];
};

export type ChunkDraft = {
  chunkIndex: number;
  content: string;
  tokenCount: number;
};

const DEFAULT_CHUNK_SIZE = 300;
const DEFAULT_OVERLAP = 100;
const DEFAULT_SEPARATORS = ['\n\n', '\n', '. ', ' '];

const LONE_SURROGATE =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

/**
 * Replaces unpaired UTF-16 surrogates with U+FFFD. JSON.stringify serialises a
 * lone surrogate as a `\ud83d` escape, which strict API servers (Mistral, for
 * one) reject with a 400, failing every chunk that carries one.
 */
function replaceLoneSurrogates(text: string): string {
  return text.replace(LONE_SURROGATE, '\uFFFD');
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

/**
 * Chunk boundaries are UTF-16 indexes, so a boundary can land between the two
 * halves of a pair (an emoji, say). Step back so the pair stays whole.
 */
function alignToCodePoint(text: string, index: number): number {
  if (
    index > 0
    && index < text.length
    && isHighSurrogate(text.charCodeAt(index - 1))
  ) {
    return index - 1;
  }
  return index;
}

export function estimateTokenCount(text: string): number {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  return Math.max(tokens.length, 1);
}

function findSplitPoint(
  text: string,
  start: number,
  maxEnd: number,
  separators: string[]
): number {
  for (const separator of separators) {
    const candidate = text.lastIndexOf(separator, maxEnd);
    if (candidate > start + Math.floor((maxEnd - start) / 2)) {
      return candidate + separator.length;
    }
  }

  return maxEnd;
}

export function chunkText(
  text: string,
  options: ChunkTextOptions = {}
): ChunkDraft[] {
  const normalized = replaceLoneSurrogates(text).trim();
  if (!normalized) {
    return [];
  }

  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = options.overlap ?? DEFAULT_OVERLAP;
  const separators = options.separators ?? DEFAULT_SEPARATORS;

  const chunks: ChunkDraft[] = [];
  let start = 0;
  let chunkIndex = 0;

  while (start < normalized.length) {
    const rawEnd = Math.min(start + chunkSize, normalized.length);
    let end =
      rawEnd === normalized.length
        ? rawEnd
        : alignToCodePoint(
            normalized,
            findSplitPoint(normalized, start, rawEnd, separators)
          );
    if (end <= start) {
      // chunkSize smaller than one code point: take the whole pair rather than
      // making no progress.
      end = Math.min(start + 2, normalized.length);
    }
    const content = normalized.slice(start, end).trim();

    if (content) {
      chunks.push({
        chunkIndex,
        content,
        tokenCount: estimateTokenCount(content)
      });
      chunkIndex += 1;
    }

    if (end >= normalized.length) {
      break;
    }

    const nextStart = alignToCodePoint(normalized, Math.max(0, end - overlap));
    start = nextStart > start ? nextStart : end;
  }

  return chunks;
}
