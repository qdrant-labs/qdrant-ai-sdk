/**
 * End-to-end tests against a running Qdrant server.
 *
 * These tests run only when `QDRANT_URL` is set. For example:
 *
 *   docker run -p 6333:6333 qdrant/qdrant
 *   QDRANT_URL=http://localhost:6333 npm run test:e2e
 *
 * The Cloud Inference test also needs `QDRANT_API_KEY` and
 * `QDRANT_INFERENCE_MODEL`, with a Qdrant Cloud cluster URL in `QDRANT_URL`.
 * Set `QDRANT_INFERENCE_DIMENSIONS` to the vector size of that model.
 */
import { randomUUID } from 'node:crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createQdrantStore, createQdrantTools } from '../src/index.js';
import { createTestEmbeddingModel, TEST_DIMENSIONS } from './helpers.js';

const url = process.env.QDRANT_URL;
const apiKey = process.env.QDRANT_API_KEY;
const inferenceModel = process.env.QDRANT_INFERENCE_MODEL;
const inferenceDimensions = Number(process.env.QDRANT_INFERENCE_DIMENSIONS);

const toolOptions = { toolCallId: 't', messages: [], context: {} };

async function find(
  tools: ReturnType<typeof createQdrantTools>,
  query: string,
) {
  const output = await tools['qdrant-find'].execute({ query }, toolOptions);
  if (Symbol.asyncIterator in output) {
    throw new Error('The find tool returned a stream.');
  }
  return output.results;
}

describe.skipIf(!url)('Qdrant server', () => {
  let client: QdrantClient;
  const collectionName = `ai-sdk-e2e-${randomUUID()}`;

  beforeAll(async () => {
    client = new QdrantClient({ url: url!, apiKey });
    await client.createCollection(collectionName, {
      vectors: { dense: { size: TEST_DIMENSIONS, distance: 'Cosine' } },
    });
    await client.createPayloadIndex(collectionName, {
      field_name: 'metadata.userId',
      field_schema: { type: 'keyword', is_tenant: true },
      wait: true,
    });
  });

  afterAll(async () => {
    await client.deleteCollection(collectionName);
  });

  it('keeps the data of each scope apart', async () => {
    const settings = {
      client,
      collectionName,
      embeddingModel: createTestEmbeddingModel(),
      vectorName: 'dense',
    };
    const alice = createQdrantTools({
      ...settings,
      scope: { userId: 'alice' },
    });
    const bob = createQdrantTools({ ...settings, scope: { userId: 'bob' } });

    await alice['qdrant-store'].execute(
      { information: 'Alice likes green tea', metadata: { userId: 'bob' } },
      toolOptions,
    );
    await bob['qdrant-store'].execute(
      { information: 'Bob likes black coffee' },
      toolOptions,
    );

    const aliceResults = await find(alice, 'what the user likes to drink');
    expect(aliceResults.map(r => r.content)).toEqual(['Alice likes green tea']);
    expect(aliceResults[0]!.metadata).toEqual({ userId: 'alice' });

    const bobResults = await find(bob, 'what the user likes to drink');
    expect(bobResults.map(r => r.content)).toEqual(['Bob likes black coffee']);
  });

  it('ranks the most similar entry first', async () => {
    const store = createQdrantStore({
      client,
      collectionName,
      embeddingModel: createTestEmbeddingModel(),
      vectorName: 'dense',
      scope: { userId: 'carol' },
    });

    await store.store([
      { content: 'The burrito costs 9 dollars' },
      { content: 'The office opens at 8 in the morning' },
    ]);

    const results = await store.find('burrito price', { limit: 1 });
    expect(results).toHaveLength(1);
    expect(results[0]!.content).toBe('The burrito costs 9 dollars');
  });
});

describe.skipIf(!url || !apiKey || !inferenceModel || !inferenceDimensions)(
  'Qdrant Cloud Inference',
  () => {
    let client: QdrantClient;
    const collectionName = `ai-sdk-e2e-inference-${randomUUID()}`;

    beforeAll(async () => {
      client = new QdrantClient({ url: url!, apiKey });
      await client.createCollection(collectionName, {
        vectors: { size: inferenceDimensions, distance: 'Cosine' },
      });
    });

    afterAll(async () => {
      await client.deleteCollection(collectionName);
    });

    it('stores and finds text with server-side vectors', async () => {
      const store = createQdrantStore({
        client,
        collectionName,
        cloudInference: { model: inferenceModel! },
      });

      await store.store([
        { content: 'Qdrant is a vector search engine' },
        { content: 'Bananas are yellow' },
      ]);

      const results = await store.find('What is Qdrant?', { limit: 1 });
      expect(results[0]!.content).toBe('Qdrant is a vector search engine');
    });
  },
);
