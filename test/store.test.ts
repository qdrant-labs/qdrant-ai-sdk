import { describe, expect, it } from 'vitest';
import { buildFilter, toPointId, uuidV5 } from '../src/store.js';
import { createQdrantStore } from '../src/index.js';
import {
  createMockClient,
  createTestEmbeddingModel,
  hashEmbedding,
} from './helpers.js';

describe('createQdrantStore settings', () => {
  it('throws when neither embedding setting is set', () => {
    const { client } = createMockClient();
    expect(() => createQdrantStore({ client, collectionName: 'c' })).toThrow(
      /exactly one of `embeddingModel` or `cloudInference`/,
    );
  });

  it('throws when both embedding settings are set', () => {
    const { client } = createMockClient();
    expect(() =>
      createQdrantStore({
        client,
        collectionName: 'c',
        embeddingModel: createTestEmbeddingModel(),
        cloudInference: { model: 'm' },
      }),
    ).toThrow(/exactly one/);
  });

  it('throws when the collection name is empty', () => {
    const { client } = createMockClient();
    expect(() =>
      createQdrantStore({
        client,
        collectionName: '',
        embeddingModel: createTestEmbeddingModel(),
      }),
    ).toThrow(/collectionName/);
  });
});

describe('scope validation', () => {
  it.each(['user.id', 'groups[]', 'a"b', ''])(
    'throws for the scope key %j',
    key => {
      const { client } = createMockClient();
      expect(() =>
        createQdrantStore({
          client,
          collectionName: 'c',
          embeddingModel: createTestEmbeddingModel(),
          scope: { [key]: 'alice' },
        }),
      ).toThrow(/scope key/);
    },
  );

  it.each([
    1.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    undefined,
    null,
    { a: 1 },
    ['x'],
  ])('throws for the scope value %s', value => {
    const { client } = createMockClient();
    expect(() =>
      createQdrantStore({
        client,
        collectionName: 'c',
        embeddingModel: createTestEmbeddingModel(),
        scope: { tenant: value as never },
      }),
    ).toThrow(/scope value/);
  });

  it('accepts string, integer, and boolean scope values', () => {
    const { client } = createMockClient();
    expect(() =>
      createQdrantStore({
        client,
        collectionName: 'c',
        embeddingModel: createTestEmbeddingModel(),
        scope: { user_id: 'alice', 'tenant-id': 7, active: true },
      }),
    ).not.toThrow();
  });
});

describe('createQdrantStore with an AI SDK embedding model', () => {
  it('stores entries with the vector and the payload', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });

    const ids = await store.store([
      { content: 'I like green tea', metadata: { topic: 'drinks' } },
    ]);

    expect(ids).toHaveLength(1);
    expect(upsert).toHaveBeenCalledTimes(1);
    const [collectionName, request] = upsert.mock.calls[0]!;
    expect(collectionName).toBe('notes');
    expect(request.wait).toBe(true);
    expect(request.points[0]).toEqual({
      id: ids[0],
      vector: hashEmbedding('I like green tea'),
      payload: {
        document: 'I like green tea',
        metadata: { topic: 'drinks' },
      },
    });
  });

  it('finds points and maps the payload', async () => {
    const { client, query } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });
    await store.store([{ content: 'I like green tea' }]);

    const results = await store.find('green tea', { scoreThreshold: 0.2 });

    expect(query).toHaveBeenCalledWith('notes', {
      query: hashEmbedding('green tea'),
      using: undefined,
      filter: undefined,
      limit: 5,
      score_threshold: 0.2,
      with_payload: true,
    });
    expect(results).toEqual([
      {
        id: expect.any(String),
        content: 'I like green tea',
        metadata: {},
        score: 1,
      },
    ]);
  });

  it('uses the vector name for queries and points', async () => {
    const { client, query, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
      vectorName: 'dense',
    });

    await store.store([{ content: 'hello' }]);
    await store.find('hello', { limit: 2 });

    expect(upsert.mock.calls[0]![1].points[0]!.vector).toEqual({
      dense: hashEmbedding('hello'),
    });
    expect(query.mock.calls[0]![1]).toMatchObject({ using: 'dense', limit: 2 });
  });

  it('splits large loads into several upsert requests', async () => {
    const { client, upsert, points } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });

    const entries = Array.from({ length: 600 }, (_, i) => ({
      content: `entry ${i}`,
    }));
    const ids = await store.store(entries);

    expect(upsert.mock.calls.map(call => call[1].points.length)).toEqual([
      256, 256, 88,
    ]);
    expect(points.map(point => point.id)).toEqual(ids);
  });

  it('uses a custom batch size and rejects an invalid one', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });
    const entries = Array.from({ length: 5 }, (_, i) => ({ content: `${i}` }));

    await store.store(entries, { batchSize: 2 });
    expect(upsert.mock.calls.map(call => call[1].points.length)).toEqual([
      2, 2, 1,
    ]);

    for (const batchSize of [0, -1, 1.5, Number.NaN]) {
      await expect(store.store(entries, { batchSize })).rejects.toThrow(
        /batchSize/,
      );
    }
    expect(upsert).toHaveBeenCalledTimes(3);
  });

  it('stops before Qdrant requests when the signal is aborted', async () => {
    const { client, query, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });
    const controller = new AbortController();
    controller.abort();

    await expect(
      store.store([{ content: 'x' }], { abortSignal: controller.signal }),
    ).rejects.toThrow();
    await expect(
      store.find('x', { abortSignal: controller.signal }),
    ).rejects.toThrow();
    expect(upsert).not.toHaveBeenCalled();
    expect(query).not.toHaveBeenCalled();
  });

  it('uses the entry IDs when they are set', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });
    const id = '0b6f3b9e-8d8e-4f4b-9a3a-3f5c2a1b7e10';

    const ids = await store.store([
      { id, content: 'a' },
      { id: 42, content: 'b' },
      { content: 'c' },
    ]);

    expect(ids[0]).toBe(id);
    expect(ids[1]).toBe(42);
    expect(ids[2]).toMatch(/^[0-9a-f-]{36}$/);
    expect(upsert.mock.calls[0]![1].points.map(point => point.id)).toEqual(ids);
  });

  it('makes no requests when there are no entries to store', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });

    expect(await store.store([])).toEqual([]);
    expect(upsert).not.toHaveBeenCalled();
  });
});

describe('createQdrantStore with Qdrant Cloud Inference', () => {
  it('sends documents instead of vectors', async () => {
    const { client, query, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      cloudInference: {
        model: 'sentence-transformers/all-minilm-l6-v2',
        options: { lowercase: true },
      },
    });

    await store.store([{ content: 'hello' }]);
    await store.find('hi');

    expect(upsert.mock.calls[0]![1].points[0]!.vector).toEqual({
      text: 'hello',
      model: 'sentence-transformers/all-minilm-l6-v2',
      options: { lowercase: true },
    });
    expect(query.mock.calls[0]![1].query).toEqual({
      text: 'hi',
      model: 'sentence-transformers/all-minilm-l6-v2',
      options: { lowercase: true },
    });
  });

  it('puts the document under the vector name', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      cloudInference: { model: 'qdrant/bm25' },
      vectorName: 'bm25',
    });

    await store.store([{ content: 'hello' }]);

    expect(upsert.mock.calls[0]![1].points[0]!.vector).toEqual({
      bm25: { text: 'hello', model: 'qdrant/bm25' },
    });
  });
});

describe('scope', () => {
  it('replaces metadata keys that have the same name as a scope key', async () => {
    const { client, upsert } = createMockClient();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
      scope: { userId: 'alice' },
    });

    await store.store([
      { content: 'secret', metadata: { userId: 'mallory', topic: 'x' } },
    ]);

    expect(upsert.mock.calls[0]![1].points[0]!.payload).toEqual({
      document: 'secret',
      metadata: { userId: 'alice', topic: 'x' },
    });
  });

  it('adds the scope and the fixed filter to each search', async () => {
    const { client, query } = createMockClient();
    const fixedFilter = {
      must_not: [{ key: 'metadata.archived', match: { value: true } }],
    };
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
      scope: { userId: 'alice', agent: 7 },
    });

    await store.find('anything', { filter: fixedFilter });

    expect(query.mock.calls[0]![1].filter).toEqual({
      must: [
        { key: 'metadata.userId', match: { value: 'alice' } },
        { key: 'metadata.agent', match: { value: 7 } },
        fixedFilter,
      ],
    });
  });
});

describe('buildFilter', () => {
  it('returns undefined when there is no scope and no filter', () => {
    expect(buildFilter(undefined, undefined)).toBeUndefined();
    expect(buildFilter({}, undefined)).toBeUndefined();
  });

  it('wraps a fixed filter without a scope', () => {
    const filter = { must: [{ key: 'a', match: { value: 'b' } }] };
    expect(buildFilter(undefined, filter)).toEqual({ must: [filter] });
  });
});

describe('review fixes', () => {
  const settings = () => {
    const { client, upsert } = createMockClient();
    const embeddingModel = createTestEmbeddingModel();
    return { client, upsert, embeddingModel };
  };

  it('rejects an invalid batch size before any embedding, even with no entries', async () => {
    const { client, upsert, embeddingModel } = settings();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
    });

    await expect(store.store([], { batchSize: 0 })).rejects.toThrow(
      /batchSize/,
    );
    await expect(
      store.store([{ content: 'x' }], { batchSize: -1 }),
    ).rejects.toThrow(/batchSize/);
    expect(embeddingModel.doEmbedCalls).toHaveLength(0);
    expect(upsert).not.toHaveBeenCalled();
  });

  it('creates the vectors for each batch just before its request', async () => {
    const { client, upsert } = settings();
    const embeddingModel = createTestEmbeddingModel(100);
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
    });

    await store.store(
      Array.from({ length: 5 }, (_, i) => ({ content: `${i}` })),
      { batchSize: 2 },
    );

    expect(embeddingModel.doEmbedCalls.map(call => call.values)).toEqual([
      ['0', '1'],
      ['2', '3'],
      ['4'],
    ]);
    expect(upsert).toHaveBeenCalledTimes(3);
  });

  it('checks every ID before the first embedding or request', async () => {
    const { client, upsert, embeddingModel } = settings();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
    });
    const entries = Array.from({ length: 300 }, (_, i) => ({
      id: i,
      content: `${i}`,
    }));

    for (const bad of ['doc-250', -1, 1.5, 2 ** 53, '']) {
      const withBad = entries.map((entry, i) =>
        i === 250 ? { ...entry, id: bad } : entry,
      );
      await expect(store.store(withBad, { batchSize: 100 })).rejects.toThrow(
        /entry ID/,
      );
    }
    expect(embeddingModel.doEmbedCalls).toHaveLength(0);
    expect(upsert).not.toHaveBeenCalled();
  });

  it('returns UUID IDs in the form that Qdrant returns', () => {
    const uuid = '0b6f3b9e-8d8e-4f4b-9a3a-3f5c2a1b7e10';
    for (const form of [
      uuid,
      uuid.toUpperCase(),
      uuid.replace(/-/g, ''),
      `{${uuid}}`,
      `urn:uuid:${uuid}`,
    ]) {
      expect(toPointId(form, undefined)).toBe(uuid);
    }
    expect(toPointId(0, undefined)).toBe(0);
    expect(toPointId(Number.MAX_SAFE_INTEGER, {})).toBe(
      Number.MAX_SAFE_INTEGER,
    );
  });

  it('makes UUID v5 values that match the standard', () => {
    expect(
      uuidV5('www.example.com', '6ba7b810-9dad-11d1-80b4-00c04fd430c8'),
    ).toBe('2ed6657d-e927-568b-95e1-2665a8aea6a2');
  });

  it('makes scoped point IDs from the scope and the entry ID', () => {
    const a = toPointId(1, { userId: 'alice' });
    const b = toPointId(1, { userId: 'bob' });
    expect(a).not.toBe(b);
    expect(a).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
    expect(toPointId(1, { userId: 'alice' })).toBe(a);
    expect(toPointId(1, { b: 2, a: 'x' })).toBe(toPointId(1, { a: 'x', b: 2 }));
    expect(toPointId(1, { n: 1 })).not.toBe(toPointId(1, { n: true }));

    const uuid = '0b6f3b9e-8d8e-4f4b-9a3a-3f5c2a1b7e10';
    expect(toPointId(uuid.toUpperCase(), { u: 'x' })).toBe(
      toPointId(uuid, { u: 'x' }),
    );
  });

  it('stores scoped entries under scoped point IDs', async () => {
    const { client, upsert, embeddingModel } = settings();
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
      scope: { userId: 'alice' },
    });

    const ids = await store.store([{ id: 7, content: 'x' }]);

    expect(ids).toEqual([toPointId(7, { userId: 'alice' })]);
    expect(upsert.mock.calls[0]![1].points[0]!.id).toBe(ids[0]);
  });

  it.each(['', '   '])('rejects the empty scope value %j', value => {
    const { client, embeddingModel } = settings();
    expect(() =>
      createQdrantStore({
        client,
        collectionName: 'notes',
        embeddingModel,
        scope: { userId: value },
      }),
    ).toThrow(/scope value/);
  });

  it('keeps the validated scope when the settings object changes later', async () => {
    const { client, upsert, embeddingModel } = settings();
    const query = (
      client as unknown as { query: { mock: { calls: unknown[][] } } }
    ).query;
    const scope: Record<string, string | undefined> = { userId: 'alice' };
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
      scope: scope as Record<string, string>,
    });

    scope.userId = undefined;
    scope.other = 'x';
    await store.store([{ content: 'x' }]);
    await store.find('x');

    expect(upsert.mock.calls[0]![1].points[0]!.payload).toEqual({
      document: 'x',
      metadata: { userId: 'alice' },
    });
    expect((query.mock.calls[0]![1] as { filter: unknown }).filter).toEqual({
      must: [{ key: 'metadata.userId', match: { value: 'alice' } }],
    });
  });
});

describe('Qdrant errors', () => {
  it('adds the reason that Qdrant gives to the error message', async () => {
    const { client, embeddingModel } = {
      ...createMockClient(),
      embeddingModel: createTestEmbeddingModel(),
    };
    const apiError = Object.assign(new Error('Bad Request'), {
      status: 400,
      data: {
        status: {
          error:
            'Bad request: Index required but not found for "metadata.userId"',
        },
      },
    });
    (client as unknown as { query: () => Promise<never> }).query = async () => {
      throw apiError;
    };
    (client as unknown as { upsert: () => Promise<never> }).upsert =
      async () => {
        throw apiError;
      };
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel,
    });

    for (const call of [
      () => store.find('x'),
      () => store.store([{ content: 'x' }]),
    ]) {
      const error = await call().catch((e: unknown) => e);
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toBe(
        'Qdrant request failed with status 400: Bad request: Index required but not found for "metadata.userId"',
      );
      expect((error as Error).cause).toBe(apiError);
    }
  });

  it('passes other errors through unchanged', async () => {
    const { client } = createMockClient();
    const networkError = new TypeError('fetch failed');
    (client as unknown as { query: () => Promise<never> }).query = async () => {
      throw networkError;
    };
    const store = createQdrantStore({
      client,
      collectionName: 'notes',
      embeddingModel: createTestEmbeddingModel(),
    });

    await expect(store.find('x')).rejects.toBe(networkError);
  });
});
