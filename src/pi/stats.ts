/**
 * Tiny live-stats store for the model readout in the top bar: tokens/sec of the
 * last generation, context occupancy, and an estimated memory footprint.
 *
 * It's a hand-rolled external store (useSyncExternalStore-friendly) so both the
 * stream function (a plain module, not a component) and React can write/read it.
 */
import { useSyncExternalStore } from "react";

export interface LiveStats {
  /** Tokens/sec — live during generation, final after. */
  tps: number | null;
  /** Estimated tokens occupying the context window after the last turn. */
  contextUsed: number | null;
  /** The model's context window (tokens). */
  contextWindow: number | null;
  /** GGUF weight size in bytes (≈ in-memory weights). */
  modelBytes: number | null;
  /** True while the model is actively generating (drives the live readout). */
  generating: boolean;
  /** Tokens produced so far in the current generation (ticks up live). */
  liveTokens: number;
  /** What the model is doing right now, e.g. "thinking", "writing index.html". */
  phase: string | null;
}

let stats: LiveStats = {
  tps: null,
  contextUsed: null,
  contextWindow: null,
  modelBytes: null,
  generating: false,
  liveTokens: 0,
  phase: null,
};

const listeners = new Set<() => void>();

export function getStats(): LiveStats {
  return stats;
}

export function setStats(patch: Partial<LiveStats>): void {
  stats = { ...stats, ...patch };
  for (const l of listeners) l();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useStats(): LiveStats {
  return useSyncExternalStore(subscribe, getStats, getStats);
}

/**
 * Rough memory footprint estimate (bytes). Real GPU/WebGPU memory isn't
 * queryable from JS, so we estimate: weights (from the GGUF size) + KV cache.
 * The qwen3next hybrid linear-attention arch keeps KV growth modest, so we use
 * a conservative ~0.09 MB/token. Returns null until the model size is known.
 */
export function estimateModelBytes(modelBytes: number | null, contextWindow: number | null): number | null {
  if (!modelBytes) return null;
  const kvPerToken = 0.09 * 1024 * 1024;
  const kv = contextWindow ? contextWindow * kvPerToken : 0;
  return modelBytes + kv;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}
