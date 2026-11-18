import { tool } from 'ai';
import { z } from 'zod';
import { createQdrantStore, type QdrantStore } from './store.js';
import type {
  QdrantFindTool,
  QdrantFindToolSettings,
  QdrantSettings,
  QdrantStoreTool,
  QdrantStoreToolOutput,
  QdrantStoreToolSettings,
  QdrantTools,
  QdrantToolsSettings,
} from './types.js';

/**
 * The default description of the find tool.
 */
export const DEFAULT_FIND_DESCRIPTION =
  'Search the Qdrant knowledge base for information that is relevant to the query. ' +
  'Use this tool when you need facts, documents, or memories that you stored before.';

/**
 * The default description of the store tool.
 */
export const DEFAULT_STORE_DESCRIPTION =
  'Store information in the Qdrant knowledge base for later use. ' +
  'Use this tool when you are asked to remember something.';

const findInputSchema = z.object({
  query: z
    .string()
    .describe('A natural language description of the information to find.'),
});

const storeInputSchema = z.object({
  information: z.string().describe('The information to store.'),
  metadata: z
    .record(z.string(), z.string())
    .optional()
    .describe('Optional key-value pairs to store with the information.'),
});

/**
 * Creates a tool that searches a Qdrant collection.
 */
export function qdrantFindTool(
  settings: QdrantFindToolSettings,
): QdrantFindTool {
  const store = createQdrantStore(settings);
  return createFindTool(store, settings, settings.description);
}

/**
 * Creates a tool that stores text in a Qdrant collection.
 */
export function qdrantStoreTool(
  settings: QdrantStoreToolSettings,
): QdrantStoreTool {
  const store = createQdrantStore(settings);
  return createStoreTool(store, settings.description);
}

/**
 * Creates the find tool and the store tool for one Qdrant collection.
 *
 * The tool names are `qdrant-find` and `qdrant-store`. They match the tool
 * names of the Qdrant MCP server.
 */
export function createQdrantTools(settings: QdrantToolsSettings): QdrantTools {
  const store = createQdrantStore(settings);
  return {
    'qdrant-find': createFindTool(store, settings, settings.findDescription),
    'qdrant-store': createStoreTool(store, settings.storeDescription),
  };
}

function createFindTool(
  store: QdrantStore,
  settings: Omit<QdrantFindToolSettings, keyof QdrantSettings>,
  description: string | undefined,
): QdrantFindTool {
  const { limit, scoreThreshold, filter } = settings;
  return tool({
    description: description ?? DEFAULT_FIND_DESCRIPTION,
    inputSchema: findInputSchema,
    execute: async ({ query }, { abortSignal }) => {
      const results = await store.find(query, {
        limit,
        scoreThreshold,
        filter,
        abortSignal,
      });
      return {
        results: results.map(({ content, metadata, score }) => ({
          content,
          metadata,
          score,
        })),
      };
    },
  });
}

function createStoreTool(
  store: QdrantStore,
  description: string | undefined,
): QdrantStoreTool {
  return tool({
    description: description ?? DEFAULT_STORE_DESCRIPTION,
    inputSchema: storeInputSchema,
    execute: async ({ information, metadata }, { abortSignal }) => {
      const [id] = await store.store([{ content: information, metadata }], {
        abortSignal,
      });
      const output: QdrantStoreToolOutput = {
        stored: true,
        id: id!,
        metadata: { ...metadata, ...store.scope },
      };
      return output;
    },
  });
}
