import type { QdrantClient, Schemas } from '@qdrant/js-client-rest';
import { MockEmbeddingModelV4 } from 'ai/test';
import { vi } from 'vitest';

/**
 * The number of dimensions of the test embedding model.
 */
export const TEST_DIMENSIONS = 256;

/**
 * Creates a deterministic embedding for a text. Texts that share words get
 * similar vectors. This lets tests run without an embedding API key.
 */
export function hashEmbedding(text: string): number[] {
  const vector = new Array<number>(TEST_DIMENSIONS).fill(0);
  for (const word of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let hash = 0;
    for (const char of word) {
      hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    }
    vector[hash % TEST_DIMENSIONS]! += 1;
  }
  const norm = Math.hypot(...vector) || 1;
  return vector.map(value => value / norm);
}

/**
 * Creates a mock AI SDK embedding model that uses `hashEmbedding`.
 */
export function createTestEmbeddingModel(
  maxEmbeddingsPerCall: number | null = null,
) {
  return new MockEmbeddingModelV4({
    maxEmbeddingsPerCall,
    doEmbed: async ({ values }) => ({
      embeddings: values.map(hashEmbedding),
      warnings: [],
    }),
  });
}

type StoredPoint = {
  id: string | number;
  vector: Schemas['VectorStruct'];
  payload: Record<string, unknown>;
};

/**
 * Creates a mock Qdrant client that keeps upserted points in memory and
 * returns all of them, with a score of 1, for each query.
 */
export function createMockClient() {
  const points: StoredPoint[] = [];

  const query = vi.fn(
    async (_collectionName: string, _request: Record<string, unknown>) => ({
      points: points.map(point => ({
        id: point.id,
        version: 0,
        score: 1,
        payload: point.payload,
      })),
    }),
  );

  const upsert = vi.fn(
    async (
      _collectionName: string,
      request: { wait?: boolean; points: StoredPoint[] },
    ) => {
      points.push(...request.points);
      return { operation_id: 1, status: 'completed' as const };
    },
  );

  const client = { query, upsert } as unknown as QdrantClient;

  return { client, query, upsert, points };
}
