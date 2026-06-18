/**
 * Model descriptor for the in-browser local provider.
 *
 * pi's agent loop needs a `Model` object to pass to the stream function. Our
 * stream function ignores the network-oriented fields (baseUrl, cost, headers)
 * and only uses contextWindow / maxTokens / id, but the object must satisfy the
 * pi-ai `Model` shape. `api`/`provider` use custom string tags — pi-ai's `Api`
 * and `Provider` types accept arbitrary strings — so the built-in provider
 * registry is never consulted (we always inject our own streamFn).
 */
import type { Model } from "@earendil-works/pi-ai";
import type { GenParams } from "../types";

export const LOCAL_API = "local-webgpu";
export const LOCAL_PROVIDER = "local";

/** Custom key on the Model carrying remote-only reasoning controls (SIQ-1). */
export const REASONING_OPTS = "__reasoningOpts" as const;

export interface ReasoningOpts {
  thinking?: boolean;
  effort?: GenParams["effort"];
}

export function buildLocalModel(descriptor: { id: string; label: string }, params: GenParams): Model<typeof LOCAL_API> {
  const model: Model<typeof LOCAL_API> = {
    id: descriptor.id,
    name: descriptor.label,
    api: LOCAL_API,
    provider: LOCAL_PROVIDER,
    baseUrl: "local://webgpu",
    reasoning: true, // Soyuz reasons in <think>…</think> before acting.
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: params.contextLength,
    maxTokens: params.maxTokens,
  };
  // Stash reasoning controls for the stream fn; the engine forwards them only on
  // the remote (SIQ-1) path and ignores them in-browser.
  (model as unknown as Record<string, unknown>)[REASONING_OPTS] = {
    thinking: params.thinking,
    effort: params.effort,
  } satisfies ReasoningOpts;
  return model;
}
