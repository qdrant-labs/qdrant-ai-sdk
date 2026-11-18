import { createHash, randomUUID } from 'node:crypto';
import type { Schemas } from '@qdrant/js-client-rest';
import { embed, embedMany } from 'ai';
import type {
  QdrantEntry,
  QdrantFindResult,
  QdrantFindSettings,
  QdrantScopeValue,
  QdrantSettings,
} from './types.js';

/**
 * The payload key that holds the stored text.
 */
export const DOCUMENT_KEY = 'document';

/**
 * The payload key that holds the stored metadata.
 */
export const METADATA_KEY = 'metadata';

const DEFAULT_LIMIT = 5;

/**
 * The default number of points in one upsert request. Large loads use
 * several requests so that each one stays below the Qdrant request size
 * limit.
 */
const DEFAULT_BATCH_SIZE = 256;

const SCOPE_KEY_PATTERN = /^[A-Za-z0-9_-]+$/;

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SIMPLE_UUID_PATTERN = /^[0-9a-f]{32}$/i;

/**
 * The UUID v5 namespace for point IDs that the store makes from a scope and
 * an entry ID.
 */
const SCOPED_ID_NAMESPACE = 'a3c2e3c4-6a0f-4d5e-9b1e-7f2d8c4b6a10';

type VectorInput = number[] | Schemas['Document'];

/**
 * Options for one call to `find` or `store`.
 */
export interface QdrantCallOptions {
  /**
   * A signal that stops the call. It cancels the embedding request and
   * stops the call before the next Qdrant request.
   */
  abortSignal?: AbortSignal;
}

/**
 * Options for one call to `store`.
 */
export interface QdrantStoreOptions extends QdrantCallOptions {
  /**
   * The number of points in one upsert request. The default is 256. Set a
   * lower value when the collection's strict mode limits the upsert batch
   * size.
   */
  batchSize?: number;
}

/**
 * Searches and stores text in one Qdrant collection.
 */
export interface QdrantStore {
  /**
   * Finds the points that are most similar to the query.
   */
  find(
    query: string,
    options?: QdrantFindSettings & QdrantCallOptions,
  ): Promise<QdrantFindResult[]>;

  /**
   * Stores the entries and returns the IDs of the points, in the order of
   * the entries.
   *
   * Large loads use several upsert requests. The store creates the vectors
   * for each request just before it sends the request. If one request
   * fails or the call stops, the points from the earlier requests stay in
   * the collection. Set an `id` on each entry to make a retry replace those
   * points.
   */
  store(
    entries: QdrantEntry[],
    options?: QdrantStoreOptions,
  ): Promise<Array<string | number>>;

  /**
   * The validated scope of this store. The store writes these values into
   * the metadata of each point, after the entry metadata.
   */
  readonly scope: Readonly<Record<string, QdrantScopeValue>> | undefined;
}

/**
 * Creates a store that searches and stores text in one Qdrant collection.
 * The find and store tools use this store. You can also use it to load
 * documents or to search without a tool.
 */
export function createQdrantStore(settings: QdrantSettings): QdrantStore {
  validateSettings(settings);

  const { client, collectionName, embeddingModel, cloudInference, vectorName } =
    settings;

  // Copy the validated scope, so that later changes to the settings object
  // cannot bypass the validation.
  const scope = settings.scope
    ? Object.freeze({ ...settings.scope })
    : undefined;

  async function toQueryVector(
    text: string,
    abortSignal: AbortSignal | undefined,
  ): Promise<VectorInput> {
    if (cloudInference) {
      return toDocument(text);
    }
    const { embedding } = await embed({
      model: embeddingModel!,
      value: text,
      abortSignal,
    });
    return embedding;
  }

  async function toPointVectors(
    texts: string[],
    abortSignal: AbortSignal | undefined,
  ): Promise<VectorInput[]> {
    if (cloudInference) {
      return texts.map(toDocument);
    }
    const { embeddings } = await embedMany({
      model: embeddingModel!,
      values: texts,
      abortSignal,
    });
    return embeddings;
  }

  function toDocument(text: string): Schemas['Document'] {
    const document: Schemas['Document'] = {
      text,
      model: cloudInference!.model,
    };
    if (cloudInference!.options) {
      document.options = cloudInference!.options;
    }
    return document;
  }

  function toVectorStruct(vector: VectorInput): Schemas['VectorStruct'] {
    return vectorName ? { [vectorName]: vector } : vector;
  }

  return {
    scope,

    async find(query, options = {}) {
      const { limit, scoreThreshold, filter, abortSignal } = options;

      const queryVector = await toQueryVector(query, abortSignal);
      abortSignal?.throwIfAborted();

      const response = await withQdrantError(() =>
        client.query(collectionName, {
          query: queryVector,
          using: vectorName,
          filter: buildFilter(scope, filter),
          limit: limit ?? DEFAULT_LIMIT,
          score_threshold: scoreThreshold,
          with_payload: true,
        }),
      );

      return response.points.map(point => {
        const payload = point.payload ?? {};
        const content = payload[DOCUMENT_KEY];
        const metadata = payload[METADATA_KEY];
        return {
          id: point.id,
          content: typeof content === 'string' ? content : '',
          metadata: isRecord(metadata) ? metadata : {},
          score: point.score,
        };
      });
    },

    async store(entries, options = {}) {
      const { abortSignal } = options;
      const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
      if (!Number.isSafeInteger(batchSize) || batchSize < 1) {
        throw new Error('Set `batchSize` to a positive integer.');
      }

      // Check every ID before the first request, so that an invalid ID
      // cannot fail a later request after earlier requests wrote points.
      const ids = entries.map(entry =>
        entry.id === undefined ? randomUUID() : toPointId(entry.id, scope),
      );

      for (let start = 0; start < entries.length; start += batchSize) {
        const batch = entries.slice(start, start + batchSize);
        const vectors = await toPointVectors(
          batch.map(entry => entry.content),
          abortSignal,
        );
        abortSignal?.throwIfAborted();

        await withQdrantError(() =>
          client.upsert(collectionName, {
            wait: true,
            points: batch.map((entry, index) => ({
              id: ids[start + index]!,
              vector: toVectorStruct(vectors[index]!),
              payload: {
                [DOCUMENT_KEY]: entry.content,
                // The scope comes last, so it replaces any metadata key with
                // the same name.
                [METADATA_KEY]: { ...entry.metadata, ...scope },
              },
            })),
          }),
        );
      }

      return ids;
    },
  };
}

/**
 * Combines the scope and the fixed filter into one Qdrant filter. Each scope
 * value becomes a required condition on the matching metadata key.
 */
export function buildFilter(
  scope: QdrantSettings['scope'],
  filter: Schemas['Filter'] | undefined,
): Schemas['Filter'] | undefined {
  const must: Schemas['Condition'][] = Object.entries(scope ?? {}).map(
    ([key, value]) => ({
      key: `${METADATA_KEY}.${key}`,
      match: { value },
    }),
  );

  if (filter) {
    must.push(filter);
  }

  return must.length > 0 ? { must } : undefined;
}

function validateSettings(settings: QdrantSettings): void {
  const hasModel = settings.embeddingModel != null;
  const hasInference = settings.cloudInference != null;

  if (hasModel === hasInference) {
    throw new Error(
      'Set exactly one of `embeddingModel` or `cloudInference` in the Qdrant settings.',
    );
  }

  if (!settings.collectionName) {
    throw new Error('Set `collectionName` in the Qdrant settings.');
  }

  for (const [key, value] of Object.entries(settings.scope ?? {})) {
    if (!SCOPE_KEY_PATTERN.test(key)) {
      throw new Error(
        `The scope key "${key}" is invalid. Use only letters, digits, underscores, and hyphens.`,
      );
    }
    const valid =
      (typeof value === 'string' && value.trim() !== '') ||
      typeof value === 'boolean' ||
      Number.isSafeInteger(value);
    if (!valid) {
      throw new Error(
        `The scope value for "${key}" is invalid. Use a non-empty string, a boolean, or an integer.`,
      );
    }
  }
}

/**
 * Checks an entry ID and returns the point ID to store.
 *
 * Without a scope, the point ID is the entry ID. UUID strings are returned
 * in the lowercase hyphenated form that Qdrant returns.
 *
 * With a scope, the point ID is a UUID v5 of the scope and the entry ID. Two
 * scopes can then use the same entry ID without replacing each other's
 * points, and a retry in the same scope replaces its own points.
 */
export function toPointId(
  id: string | number,
  scope: QdrantSettings['scope'],
): string | number {
  const pointId = normalizePointId(id);
  if (!scope || Object.keys(scope).length === 0) {
    return pointId;
  }
  const sortedScope = Object.keys(scope)
    .sort()
    .map(key => [key, scope[key]]);
  return uuidV5(JSON.stringify([sortedScope, pointId]), SCOPED_ID_NAMESPACE);
}

function normalizePointId(id: string | number): string | number {
  if (typeof id === 'number' && Number.isSafeInteger(id) && id >= 0) {
    return id;
  }
  if (typeof id === 'string') {
    let text = id;
    if (text.startsWith('urn:uuid:')) {
      text = text.slice('urn:uuid:'.length);
    } else if (text.startsWith('{') && text.endsWith('}')) {
      text = text.slice(1, -1);
    }
    if (UUID_PATTERN.test(text)) {
      return text.toLowerCase();
    }
    if (SIMPLE_UUID_PATTERN.test(text)) {
      return formatUuid(text.toLowerCase());
    }
  }
  throw new Error(
    `The entry ID ${JSON.stringify(id)} is invalid. Use a UUID string or an unsigned safe integer.`,
  );
}

export function uuidV5(name: string, namespace: string): string {
  const hash = createHash('sha1')
    .update(Buffer.from(namespace.replace(/-/g, ''), 'hex'))
    .update(name, 'utf8')
    .digest();
  hash[6] = (hash[6]! & 0x0f) | 0x50;
  hash[8] = (hash[8]! & 0x3f) | 0x80;
  return formatUuid(hash.subarray(0, 16).toString('hex'));
}

function formatUuid(hex: string): string {
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join('-');
}

/**
 * Runs a Qdrant request. When Qdrant rejects it, throws an error whose
 * message includes the reason that Qdrant gives, such as a missing payload
 * index. The original error is the `cause`.
 */
async function withQdrantError<T>(request: () => Promise<T>): Promise<T> {
  try {
    return await request();
  } catch (error) {
    const detail = qdrantErrorDetail(error);
    if (detail === undefined) {
      throw error;
    }
    const status = (error as { status?: unknown }).status;
    const prefix =
      typeof status === 'number'
        ? `Qdrant request failed with status ${status}`
        : 'Qdrant request failed';
    throw new Error(`${prefix}: ${detail}`, { cause: error });
  }
}

function qdrantErrorDetail(error: unknown): string | undefined {
  if (!isRecord(error) || !isRecord(error.data)) {
    return undefined;
  }
  const status = error.data.status;
  if (isRecord(status) && typeof status.error === 'string') {
    return status.error;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
