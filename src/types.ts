export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  /** visible answer text (think block already stripped out) */
  content: string;
  /** chain-of-thought captured from <think>…</think>, if any */
  think?: string;
  ts: number;
  /** still streaming */
  pending?: boolean;
}

export interface Artifact {
  id: string;
  title: string;
  html: string;
  ts: number;
}

export interface Project {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  modelId: string;
  messages: ChatMessage[];
  artifacts: Artifact[];
}

/** Reasoning effort understood by SIQ-1 (trained-in control). */
export type ReasoningEffort = "low" | "medium" | "high";

/**
 * A remote OpenAI-compatible model — inference runs server-side (e.g. SIQ-1-35B
 * on RunPod serverless, reached through the Railway proxy) instead of in-browser.
 */
export interface RemoteModel {
  /** OpenAI-compatible base, e.g. "https://…/api/siq". `/v1/chat/completions` is appended. */
  endpoint: string;
  /** model tag sent in the request body. */
  model: string;
  /** SIQ-1 exposes a thinking toggle + reasoning-effort control per request. */
  reasoning?: boolean;
  /** Context window the serving endpoint is configured with (server-side; the
   *  client context-length slider doesn't apply to remote models). */
  contextWindow?: number;
}

export interface ModelPreset {
  id: string;
  label: string;
  /** GGUF repo/file — only for in-browser (local) presets. */
  repo?: string;
  file?: string;
  /** When set, this is a remote model served over an OpenAI-compatible endpoint. */
  remote?: RemoteModel;
  sizeLabel: string;
  note: string;
  /** confirmed to run on the WebGPU backend (local) or live on the endpoint (remote) */
  verified: boolean;
  accent?: string;
}

export interface GenParams {
  temperature: number;
  maxTokens: number;
  contextLength: number;
  /** Remote-only: SIQ-1 thinking mode (emit <think>…</think>). Ignored by local models. */
  thinking?: boolean;
  /** Remote-only: SIQ-1 reasoning effort. Ignored by local models. */
  effort?: ReasoningEffort;
}
