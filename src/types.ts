import type { QdrantClient, Schemas } from '@qdrant/js-client-rest';
import type { EmbeddingModel, Tool } from 'ai';

/**
 * A value that can scope stored points and searches. Numbers must be
 * integers, because Qdrant matches only strings, integers, and booleans.
 */
export type QdrantScopeValue = string | number | boolean;

/**
 * Settings for Qdrant Cloud Inference. Qdrant creates the vectors on the
 * server, so you do not need an AI SDK embedding model.
 *
 * @see https://qdrant.tech/documentation/inference/cloud-inference/
 */
export interface QdrantCloudInferenceSettings {
  /**
   * The model that Qdrant uses to create vectors, for example
   * `sentence-transformers/all-minilm-l6-v2`.
   */
  model: string;

  /**
   * Model options that Qdrant passes to the inference service as-is.
   */
  options?: Record<string, unknown>;
}

/**
 * Settings that the find tool, the store tool, and `createQdrantStore` share.
 */
export interface QdrantSettings {
  /**
   * The Qdrant client to use.
   */
  client: QdrantClient;

  /**
   * The name of the collection that holds the points.
   */
  collectionName: string;

  /**
   * The AI SDK embedding model that creates vectors on the client.
   * Set this or `cloudInference`, but not both.
   */
  embeddingModel?: EmbeddingModel;

  /**
   * The Qdrant Cloud Inference model that creates vectors on the server.
   * Set this or `embeddingModel`, but not both.
   */
  cloudInference?: QdrantCloudInferenceSettings;

  /**
   * The name of the vector in the collection. Leave it empty when the
   * collection uses a single unnamed vector.
   */
  vectorName?: string;

  /**
   * Values that the tools write into the metadata of each stored point and
   * add as required conditions to each search. Use it to keep the data of
   * one user, agent, or tenant apart from the data of others.
   *
   * The model cannot change or remove these values.
   */
  scope?: Record<string, QdrantScopeValue>;
}

/**
 * Settings for searches.
 */
export interface QdrantFindSettings {
  /**
   * The maximum number of points to return. The default is 5.
   */
  limit?: number;

  /**
   * The score limit for results. For Cosine and Dot collections, it is the
   * minimum score. For Euclid and Manhattan collections, it is the maximum
   * distance.
   */
  scoreThreshold?: number;

  /**
   * An extra Qdrant filter that applies to each search. The model cannot
   * change or remove it.
   */
  filter?: Schemas['Filter'];
}

/**
 * Settings that change how a tool describes itself to the model.
 */
export interface QdrantToolDescriptionSettings {
  /**
   * The tool description that the model reads. Use it to tell the model
   * when to use the tool.
   */
  description?: string;
}

/**
 * Settings for the find tool.
 */
export type QdrantFindToolSettings = QdrantSettings &
  QdrantFindSettings &
  QdrantToolDescriptionSettings;

/**
 * Settings for the store tool.
 */
export type QdrantStoreToolSettings = QdrantSettings &
  QdrantToolDescriptionSettings;

/**
 * Settings for `createQdrantTools`.
 */
export type QdrantToolsSettings = QdrantSettings &
  QdrantFindSettings & {
    /**
     * The description of the find tool.
     */
    findDescription?: string;

    /**
     * The description of the store tool.
     */
    storeDescription?: string;
  };

/**
 * A text entry to store in Qdrant.
 */
export interface QdrantEntry {
  /**
   * The ID of the entry: a UUID string or an unsigned safe integer. Set IDs
   * when you load documents, so that a retry replaces the points it already
   * wrote. The default is a new random UUID.
   *
   * Without a scope, the entry ID is the point ID, and an existing point
   * with this ID is replaced. With a scope, the store makes the point ID
   * from the scope and the entry ID, so two scopes can use the same entry
   * ID without replacing each other's points. `store` returns the point
   * IDs.
   */
  id?: string | number;

  /**
   * The text to store. The tools create the vector from this text.
   */
  content: string;

  /**
   * Metadata to store with the text.
   */
  metadata?: Record<string, unknown>;
}

/**
 * A point that a search returned.
 */
export interface QdrantFindResult {
  /**
   * The ID of the point.
   */
  id: string | number;

  /**
   * The stored text.
   */
  content: string;

  /**
   * The stored metadata.
   */
  metadata: Record<string, unknown>;

  /**
   * The similarity score of the point.
   */
  score: number;
}

/**
 * The input of the find tool.
 */
export interface QdrantFindToolInput {
  query: string;
}

/**
 * The output of the find tool.
 */
export interface QdrantFindToolOutput {
  results: Array<Omit<QdrantFindResult, 'id'>>;
}

/**
 * The input of the store tool.
 */
export interface QdrantStoreToolInput {
  information: string;
  metadata?: Record<string, string>;
}

/**
 * The output of the store tool.
 */
export interface QdrantStoreToolOutput {
  stored: true;
  id: string | number;
  /**
   * The metadata that was stored. The scope replaces any metadata key with
   * the same name, so the model can see the values that were saved.
   */
  metadata: Record<string, unknown>;
}

type WithExecute<TOOL extends Tool> = TOOL & {
  execute: NonNullable<TOOL['execute']>;
};

/**
 * The find tool.
 */
export type QdrantFindTool = WithExecute<
  Tool<QdrantFindToolInput, QdrantFindToolOutput>
>;

/**
 * The store tool.
 */
export type QdrantStoreTool = WithExecute<
  Tool<QdrantStoreToolInput, QdrantStoreToolOutput>
>;

/**
 * The tools that `createQdrantTools` returns.
 */
export type QdrantTools = {
  'qdrant-find': QdrantFindTool;
  'qdrant-store': QdrantStoreTool;
};
