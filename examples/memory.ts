/**
 * Agent memory with Qdrant Cloud Inference. Qdrant creates the vectors on the
 * server, so the agent needs no embedding model.
 *
 * Set QDRANT_URL and QDRANT_API_KEY for a Qdrant Cloud cluster, set
 * AI_GATEWAY_API_KEY, and run:
 *
 *   npx tsx examples/memory.ts
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { ToolLoopAgent, isStepCount } from 'ai';
import { createQdrantTools } from 'qdrant-ai-sdk';

const client = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY,
});

const collectionName = 'memories';

if (!(await client.collectionExists(collectionName)).exists) {
  await client.createCollection(collectionName, {
    vectors: { size: 384, distance: 'Cosine' },
  });
  await client.createPayloadIndex(collectionName, {
    field_name: 'metadata.userId',
    field_schema: { type: 'keyword', is_tenant: true },
  });
}

// Get the user ID from your authentication, not from the model.
const userId = 'user-123';

const agent = new ToolLoopAgent({
  model: 'anthropic/claude-sonnet-4.5',
  instructions: 'You are a helpful assistant with long-term memory.',
  tools: createQdrantTools({
    client,
    collectionName,
    cloudInference: { model: 'sentence-transformers/all-minilm-l6-v2' },
    scope: { userId },
  }),
  stopWhen: isStepCount(5),
});

await agent.generate({ prompt: 'Remember that my favorite editor is Neovim.' });

const { text } = await agent.generate({ prompt: 'Which editor do I use?' });
console.log(text);
