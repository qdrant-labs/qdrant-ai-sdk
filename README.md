# Qdrant for the AI SDK

`qdrant-ai-sdk` gives [AI SDK](https://ai-sdk.dev) agents semantic search and long-term memory in [Qdrant](https://qdrant.tech). It provides two tools. The `qdrant-find` tool searches a collection. The `qdrant-store` tool adds text to a collection.

- Use any AI SDK embedding model, or let [Qdrant Cloud Inference](https://qdrant.tech/documentation/inference/cloud-inference/) create the vectors on the server.
- Keep the data of each user or tenant apart with `scope`. The model cannot change the scope.
- Pass the tools to `generateText`, `streamText`, or `ToolLoopAgent`.
- Share a collection with the [Qdrant MCP server](https://github.com/qdrant/mcp-server-qdrant). Both use the same payload format.

## Installation

```bash
npm install qdrant-ai-sdk @qdrant/js-client-rest ai zod
```

The package needs Node.js 22 or later, AI SDK 7, and `@qdrant/js-client-rest` 1.14 or later.

## Quickstart

Create a collection with a vector size that matches your embedding model. Then pass the tools to `generateText`.

```ts
import { QdrantClient } from '@qdrant/js-client-rest';
import { generateText, isStepCount } from 'ai';
import { openai } from '@ai-sdk/openai';
import { createQdrantTools } from 'qdrant-ai-sdk';

const client = new QdrantClient({ url: 'http://localhost:6333' });

if (!(await client.collectionExists('notes')).exists) {
  await client.createCollection('notes', {
    vectors: { size: 1536, distance: 'Cosine' },
  });
}

const tools = createQdrantTools({
  client,
  collectionName: 'notes',
  embeddingModel: openai.embedding('text-embedding-3-small'),
});

const { text } = await generateText({
  model: openai('gpt-5-mini'),
  tools,
  stopWhen: isStepCount(5),
  prompt:
    'Remember that the team stand-up is at 9:30. Then tell me when it is.',
});
```

The tool names contain a hyphen, so use brackets to get one tool: `tools['qdrant-find']`.

## Qdrant Cloud Inference

Set `cloudInference` instead of `embeddingModel` to let your Qdrant Cloud cluster create the vectors. The tools then send text to Qdrant, and your application needs no embedding provider.

```ts
const tools = createQdrantTools({
  client,
  collectionName: 'notes',
  cloudInference: { model: 'sentence-transformers/all-minilm-l6-v2' },
});
```

The collection's vector size must match the model. The `sentence-transformers/all-minilm-l6-v2` model makes vectors with 384 dimensions. The Inference tab of your cluster in the Qdrant Cloud Console lists the models and their dimensions.

To use an [external provider](https://qdrant.tech/documentation/inference/external-inference-providers/) such as OpenAI through Qdrant, put the provider key in `options`:

```ts
cloudInference: {
  model: 'openai/text-embedding-3-small',
  options: { 'openai-api-key': process.env.OPENAI_API_KEY },
},
```

## Keep users apart with `scope`

Set `scope` to the values that identify the current user, agent, or tenant. The store tool writes these values into the metadata of each point. The find tool adds them as required conditions to each search. If the model sends metadata with the same key, the scope value replaces it. Use scope keys with only letters, digits, underscores, and hyphens. Use strings, integers, or booleans as scope values.

```ts
const tools = createQdrantTools({
  client,
  collectionName: 'memories',
  embeddingModel: openai.embedding('text-embedding-3-small'),
  scope: { userId: session.userId },
});
```

Get the scope values from your authentication, and create the tools for each request. The package throws an error when a scope value is `undefined`, `null`, an empty string, or another wrong type. The package cannot know which scope keys your application needs, so it accepts a missing key or a missing `scope`. Without a scope, the tools read and write all points in the collection. Check that the user is signed in before you create the tools. The scope keeps tenants apart only when every writer to the collection sets the same scope keys.

Create a payload index on each scope key. Qdrant Cloud turns on strict mode for new collections, and strict mode rejects filters on fields that have no index. Use a `keyword` index for a string value, an `integer` index for an integer value, and a `bool` index for a boolean value. Scope keys are stored under `metadata`, so the index for `userId` is on `metadata.userId`:

```ts
await client.createPayloadIndex('memories', {
  field_name: 'metadata.userId',
  field_schema: { type: 'keyword', is_tenant: true },
});
```

The [multitenancy guide](https://qdrant.tech/documentation/manage-data/multitenancy/) explains `is_tenant` and other tenant settings.

## Add a fixed filter

Set `filter` to apply a [Qdrant filter](https://qdrant.tech/documentation/search/filtering/) to each search. The model cannot change it.

```ts
const tools = createQdrantTools({
  client,
  collectionName: 'docs',
  embeddingModel: openai.embedding('text-embedding-3-small'),
  filter: {
    must: [{ key: 'metadata.language', match: { value: 'en' } }],
  },
});
```

Create a payload index on each field in the filter, such as a `keyword` index on `metadata.language`. Strict mode rejects filters on fields that have no index.

## Use one tool

Use `qdrantFindTool` to give the model search access only. Use `qdrantStoreTool` to give it write access only. Each function takes a `description` that tells the model when to use the tool.

```ts
import { qdrantFindTool } from 'qdrant-ai-sdk';

const { text } = await generateText({
  model: openai('gpt-5-mini'),
  tools: {
    search: qdrantFindTool({
      client,
      collectionName: 'menu',
      embeddingModel: openai.embedding('text-embedding-3-small'),
      description: 'Search the menu for prices and opening hours.',
    }),
  },
  stopWhen: isStepCount(3),
  prompt: 'What is the price of a burrito?',
});
```

## Load documents

Use `createQdrantStore` to add documents or to search from your own code. It takes the same settings as the tools.

```ts
import { createQdrantStore } from 'qdrant-ai-sdk';

const store = createQdrantStore({
  client,
  collectionName: 'menu',
  embeddingModel: openai.embedding('text-embedding-3-small'),
});

await store.store([
  { content: 'A burrito costs 9 dollars.', metadata: { category: 'food' } },
  { content: 'A taco costs 4 dollars.', metadata: { category: 'food' } },
]);

const results = await store.find('burrito price', { limit: 3 });
```

`store` sends large loads in requests of 256 points, and it creates the vectors for each request just before it sends the request. Set `batchSize` in the second argument when your collection's strict mode allows fewer points per upsert. `store` checks all entry IDs before the first request. If one request fails, the points from the earlier requests stay in the collection. Give each entry an `id`, such as a UUID v5 of the document path, so that a retry replaces those points instead of adding copies.

Point IDs are unique in the whole collection. With a `scope`, `store` makes each point ID from the scope and the entry ID, so two tenants can load the same document with the same entry ID. `store` returns the point IDs.

## Settings

All functions take these settings:

| Setting          | Type                                          | Description                                                                             |
| ---------------- | --------------------------------------------- | --------------------------------------------------------------------------------------- |
| `client`         | `QdrantClient`                                | The Qdrant client.                                                                      |
| `collectionName` | `string`                                      | The collection that holds the points.                                                   |
| `embeddingModel` | `EmbeddingModel`                              | An AI SDK embedding model. Set this or `cloudInference`.                                |
| `cloudInference` | `{ model: string; options?: object }`         | A Qdrant Cloud Inference model and its options. Set this or `embeddingModel`.           |
| `vectorName`     | `string`                                      | The name of the vector to use. Leave it empty for a collection with one unnamed vector. |
| `scope`          | `Record<string, string \| number \| boolean>` | Values that the tools store in the metadata of each point and require in each search.   |

The find tool and `createQdrantTools` also take these settings:

| Setting          | Type     | Description                                                                                      |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------ |
| `limit`          | `number` | The maximum number of results. The default is 5.                                                 |
| `scoreThreshold` | `number` | The minimum score of a result. For Euclid and Manhattan collections, it is the maximum distance. |
| `filter`         | `Filter` | A Qdrant filter that applies to each search.                                                     |

`createQdrantTools` takes `findDescription` and `storeDescription`. `qdrantFindTool` and `qdrantStoreTool` take `description`.

## Tools

| Tool           | Input                                                               | Output                                                                                        |
| -------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `qdrant-find`  | `query`: the text to search for                                     | `results`: a list of `content`, `metadata`, and `score`                                       |
| `qdrant-store` | `information`: the text to store. `metadata`: optional string pairs | `stored`, the `id` of the new point, and the stored `metadata`, with the scope values applied |

## Payload format

Each point has this payload:

```json
{
  "document": "The text that the vector represents.",
  "metadata": { "userId": "user-123" }
}
```

The Qdrant MCP server uses the same format. To share a collection with it, use the same embedding model and set `vectorName` to the vector name that the MCP server uses.

## License

Apache 2.0
