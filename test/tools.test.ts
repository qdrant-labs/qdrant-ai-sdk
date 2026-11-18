import { generateText, isStepCount } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import type { LanguageModelV4GenerateResult } from '@ai-sdk/provider';
import { describe, expect, it } from 'vitest';
import {
  createQdrantTools,
  DEFAULT_FIND_DESCRIPTION,
  DEFAULT_STORE_DESCRIPTION,
  qdrantFindTool,
  qdrantStoreTool,
} from '../src/index.js';
import { createMockClient, createTestEmbeddingModel } from './helpers.js';

const usage: LanguageModelV4GenerateResult['usage'] = {
  inputTokens: {
    total: 1,
    noCache: 1,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 1, text: 1, reasoning: undefined },
};

function toolCallStep(
  toolCallId: string,
  toolName: string,
  input: unknown,
): LanguageModelV4GenerateResult {
  return {
    content: [
      {
        type: 'tool-call',
        toolCallId,
        toolName,
        input: JSON.stringify(input),
      },
    ],
    finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
    usage,
    warnings: [],
  };
}

function textStep(text: string): LanguageModelV4GenerateResult {
  return {
    content: [{ type: 'text', text }],
    finishReason: { unified: 'stop', raw: 'stop' },
    usage,
    warnings: [],
  };
}

describe('createQdrantTools', () => {
  it('returns the find tool and the store tool with default descriptions', () => {
    const { client } = createMockClient();
    const tools = createQdrantTools({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });

    expect(Object.keys(tools)).toEqual(['qdrant-find', 'qdrant-store']);
    expect(tools['qdrant-find'].description).toBe(DEFAULT_FIND_DESCRIPTION);
    expect(tools['qdrant-store'].description).toBe(DEFAULT_STORE_DESCRIPTION);
  });

  it('uses custom descriptions', () => {
    const { client } = createMockClient();
    const tools = createQdrantTools({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
      findDescription: 'Search the product docs.',
      storeDescription: 'Save a user preference.',
    });

    expect(tools['qdrant-find'].description).toBe('Search the product docs.');
    expect(tools['qdrant-store'].description).toBe('Save a user preference.');
  });

  it('runs a store call and a find call in a generateText tool loop', async () => {
    const { client, query, upsert } = createMockClient();
    const model = new MockLanguageModelV4({
      doGenerate: [
        toolCallStep('call-1', 'qdrant-store', {
          information: 'The user prefers dark mode.',
          metadata: { userId: 'mallory', kind: 'preference' },
        }),
        toolCallStep('call-2', 'qdrant-find', { query: 'display preferences' }),
        textStep('You prefer dark mode.'),
      ],
    });

    const result = await generateText({
      model,
      prompt: 'Remember that I prefer dark mode, then tell me my preference.',
      tools: createQdrantTools({
        client,
        collectionName: 'memories',
        embeddingModel: createTestEmbeddingModel(),
        scope: { userId: 'alice' },
        limit: 3,
      }),
      stopWhen: isStepCount(5),
    });

    expect(result.text).toBe('You prefer dark mode.');
    expect(result.steps).toHaveLength(3);

    const storeResult = result.steps[0]!.toolResults[0]!;
    expect(storeResult.toolName).toBe('qdrant-store');
    expect(storeResult.output).toEqual({
      stored: true,
      id: expect.any(String),
      metadata: { userId: 'alice', kind: 'preference' },
    });

    const findResult = result.steps[1]!.toolResults[0]!;
    expect(findResult.toolName).toBe('qdrant-find');
    expect(findResult.output).toEqual({
      results: [
        {
          content: 'The user prefers dark mode.',
          metadata: { userId: 'alice', kind: 'preference' },
          score: 1,
        },
      ],
    });

    expect(upsert).toHaveBeenCalledTimes(1);
    expect(query.mock.calls[0]![1]).toMatchObject({
      limit: 3,
      filter: { must: [{ key: 'metadata.userId', match: { value: 'alice' } }] },
    });
  });

  it('sends the tool input schemas to the model', async () => {
    const { client } = createMockClient();
    const model = new MockLanguageModelV4({ doGenerate: textStep('ok') });

    await generateText({
      model,
      prompt: 'hi',
      tools: createQdrantTools({
        client,
        collectionName: 'memories',
        embeddingModel: createTestEmbeddingModel(),
      }),
    });

    const tools = model.doGenerateCalls[0]!.tools ?? [];
    const byName = Object.fromEntries(
      tools.map(t => [t.name, t] as const),
    ) as Record<string, { inputSchema?: { properties?: object } }>;

    expect(Object.keys(byName).sort()).toEqual(['qdrant-find', 'qdrant-store']);
    expect(
      Object.keys(byName['qdrant-find']!.inputSchema!.properties!),
    ).toEqual(['query']);
    expect(
      Object.keys(byName['qdrant-store']!.inputSchema!.properties!).sort(),
    ).toEqual(['information', 'metadata']);
  });
});

describe('single tools', () => {
  it('creates a read-only find tool', async () => {
    const { client, query } = createMockClient();
    const find = qdrantFindTool({
      client,
      collectionName: 'docs',
      cloudInference: { model: 'sentence-transformers/all-minilm-l6-v2' },
      description: 'Search the docs.',
    });

    expect(find.description).toBe('Search the docs.');
    const output = await find.execute(
      { query: 'pricing' },
      { toolCallId: 't', messages: [], context: {} },
    );
    expect(output).toEqual({ results: [] });
    expect(query.mock.calls[0]![1].query).toEqual({
      text: 'pricing',
      model: 'sentence-transformers/all-minilm-l6-v2',
    });
  });

  it('creates a store tool', async () => {
    const { client, upsert } = createMockClient();
    const store = qdrantStoreTool({
      client,
      collectionName: 'docs',
      embeddingModel: createTestEmbeddingModel(),
    });

    const output = await store.execute(
      { information: 'note' },
      { toolCallId: 't', messages: [], context: {} },
    );
    expect(output).toEqual({
      stored: true,
      id: expect.any(String),
      metadata: {},
    });
    expect(upsert).toHaveBeenCalledTimes(1);
  });
});
