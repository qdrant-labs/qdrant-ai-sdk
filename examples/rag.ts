/**
 * Agentic RAG with an AI SDK embedding model.
 *
 * Run a local Qdrant server, set AI_GATEWAY_API_KEY, and run:
 *
 *   QDRANT_URL=http://localhost:6333 npx tsx examples/rag.ts
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { generateText, isStepCount } from 'ai';
import { createQdrantStore, qdrantFindTool } from 'qdrant-ai-sdk';

const client = new QdrantClient({
  url: process.env.QDRANT_URL ?? 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY,
});

const collectionName = 'menu';
const settings = {
  client,
  collectionName,
  embeddingModel: 'openai/text-embedding-3-small',
};

if (!(await client.collectionExists(collectionName)).exists) {
  await client.createCollection(collectionName, {
    vectors: { size: 1536, distance: 'Cosine' },
  });

  await createQdrantStore(settings).store([
    { content: 'A burrito costs 9 dollars.' },
    { content: 'A taco costs 4 dollars.' },
    { content: 'The restaurant opens at 11 in the morning.' },
  ]);
}

const { text } = await generateText({
  model: 'openai/gpt-5-mini',
  tools: {
    search: qdrantFindTool({
      ...settings,
      description: 'Search the menu for prices and opening hours.',
    }),
  },
  stopWhen: isStepCount(3),
  prompt: 'What is the price of a burrito?',
});

console.log(text);
