import { Wllama } from "@reeselevine/wllama-webgpu";
import WasmFromPackage from "@reeselevine/wllama-webgpu/esm/wasm-from-package.js";
import type { ChatMessage, ReasoningEffort, RemoteModel } from "../types";

import { LLM_SERVER } from "../config";
import { clientId } from "../lib/logger";
import { cutAtStop } from "./stops";

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as any).gpu;
}

/** What `load` is asked to make active: either an in-browser GGUF or a remote endpoint. */
export interface LoadTarget {
  /** GGUF url for the in-browser WebGPU path. */
  url?: string;
  /** OpenAI-compatible endpoint for the remote path. */
  remote?: RemoteModel;
}

export interface LoadOpts {
  contextLength: number;
  onProgress?: (frac: number, loaded: number, total: number) => void;
}

export interface ChatOpts {
  temperature: number;
  maxTokens: number;
  signal?: AbortSignal;
  onToken: (fullText: string) => void;
  /** Remote-only (SIQ-1): thinking mode + reasoning effort. */
  thinking?: boolean;
  effort?: ReasoningEffort;
}

/**
 * Thin singleton wrapper around the WebGPU build of llama.cpp
 * (@reeselevine/wllama-webgpu). One model loaded at a time.
 */
class LlamaEngine {
  private wllama: Wllama | null = null;
  loadedUrl: string | null = null;
  private loadedCtx: number | null = null;
  /** When set, the active model runs against this OpenAI-compatible endpoint. */
  private remote: RemoteModel | null = null;
  readonly backend: "webgpu" | "cpu" = hasWebGPU() ? "webgpu" : "cpu";

  /**
   * The remote target for the *current* model, if any. An explicitly-selected
   * remote model (e.g. SIQ-1) wins; otherwise a build-time VITE_LLM_SERVER
   * (the DFlash llama-server dev path) applies to local models.
   */
  private activeRemote(): RemoteModel | null {
    if (this.remote) return this.remote;
    if (LLM_SERVER) return { endpoint: LLM_SERVER, model: "soyuz" };
    return null;
  }

  /** True when the active model serves over the network rather than in-browser. */
  get serverMode(): boolean {
    return !!this.activeRemote();
  }

  get ready(): boolean {
    return this.serverMode || (!!this.wllama && !!this.loadedUrl);
  }

  /**
   * Make a model active. Remote targets need no download — we just record the
   * endpoint (and free any in-browser model holding GPU memory). GGUF targets
   * download + load on WebGPU; a no-op if the same url is already loaded at the
   * same context length; reloads (from cache) when the context length changes.
   */
  async load(target: LoadTarget, opts: LoadOpts): Promise<void> {
    // remote mode: nothing to download/load in the browser — the endpoint holds the model
    if (target.remote) {
      // drop any in-browser model so a 35B cloud model doesn't sit next to GPU weights
      if (this.wllama) await this.unload();
      this.remote = target.remote;
      this.loadedUrl = `remote:${target.remote.endpoint}`;
      opts.onProgress?.(1, 1, 1);
      return;
    }
    // switching back to a local model: clear any explicit remote target
    this.remote = null;
    // build-time DFlash dev server still short-circuits local loads
    if (this.activeRemote()) {
      this.loadedUrl = `server:${LLM_SERVER}`;
      opts.onProgress?.(1, 1, 1);
      return;
    }
    const url = target.url!;
    if (this.loadedUrl === url && this.wllama && this.loadedCtx === opts.contextLength) return;
    // free any previous model first
    await this.unload();

    console.debug(`[pi] loading model on ${this.backend}, n_ctx=${opts.contextLength}…`);
    const t0 = performance.now();
    const wllama = new Wllama(WasmFromPackage, { backend: this.backend });
    await wllama.loadModelFromUrl(url, {
      n_ctx: opts.contextLength,
      n_batch: 256,
      useCache: true,
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        opts.onProgress?.(total > 0 ? loaded / total : 0, loaded, total);
      },
    });

    this.wllama = wllama;
    this.loadedUrl = url;
    this.loadedCtx = opts.contextLength;
    console.debug(`[pi] model ready in ${((performance.now() - t0) / 1000).toFixed(1)}s (n_ctx=${opts.contextLength})`);
  }

  async unload(): Promise<void> {
    if (this.wllama) {
      try {
        await this.wllama.exit();
      } catch {
        /* ignore */
      }
    }
    this.wllama = null;
    this.loadedUrl = null;
    this.loadedCtx = null;
  }

  /** Run a streaming chat completion. Resolves with the final full text. */
  async chat(messages: Pick<ChatMessage, "role" | "content">[], opts: ChatOpts): Promise<string> {
    const remote = this.activeRemote();
    if (remote) return this.chatViaServer(remote, messages, opts);
    if (!this.wllama) throw new Error("No model loaded");

    const internal = new AbortController();
    const forward = () => internal.abort();
    opts.signal?.addEventListener("abort", forward, { once: true });
    let cut: string | null = null;

    const handleToken = (currentText: string) => {
      if (cut === null) {
        const truncated = cutAtStop(currentText);
        if (truncated !== null) {
          cut = truncated;
          internal.abort(); // our own stop — not a user abort
        }
      }
      opts.onToken(cut ?? currentText);
    };

    try {
      const result = await this.wllama.createChatCompletion(
        messages.map((m) => ({ role: m.role, content: m.content })),
        {
          nPredict: opts.maxTokens,
          abortSignal: internal.signal,
          // Empirically-tuned sampling for Soyuz: temp 0.8 (greedy/low → empty
          // <think> + loops), soft penalties only (repeat ≥1.2 or presence ≥0.8
          // garble code), no frequency penalty.
          sampling: {
            temp: opts.temperature,
            top_p: 0.9,
            top_k: 40,
            min_p: 0,
            penalty_repeat: 1.1,
            penalty_present: 0.4,
            penalty_freq: 0,
          },
          onNewToken: (_token: number, _piece: Uint8Array, currentText: string) => handleToken(currentText),
        },
      );
      const text = typeof result === "string" ? result : String(result ?? "");
      return cut ?? text;
    } catch (err) {
      if (cut !== null) return cut; // generation was stopped by our stop-sequence
      throw err; // genuine error or user abort
    } finally {
      opts.signal?.removeEventListener("abort", forward);
    }
  }

  /**
   * Stream a chat completion from an OpenAI-compatible endpoint (SSE). Covers
   * both the DFlash llama-server (speculative decoding server-side) and the
   * SIQ-1 RunPod-serverless proxy. The client consumes the stream and applies
   * the same stop-sequence cut as the in-browser path.
   *
   * For SIQ-1 (a reasoning model) the chain-of-thought may arrive in
   * `delta.reasoning_content` (vLLM) rather than inline `<think>…</think>`
   * (llama.cpp). We fold reasoning back into a single `<think>…</think>` prefix
   * so the downstream parser (parseOutput) handles both backends identically.
   */
  private async chatViaServer(
    remote: RemoteModel,
    messages: Pick<ChatMessage, "role" | "content">[],
    opts: ChatOpts,
  ): Promise<string> {
    let cut: string | null = null;
    let reasoning = "";
    let answer = "";

    // Reconstruct one cumulative text, with separated reasoning wrapped in <think>.
    const compose = () => {
      if (!reasoning) return answer;
      // close the think block once the answer has started; otherwise leave it open
      return answer ? `<think>\n${reasoning}\n</think>\n${answer}` : `<think>\n${reasoning}`;
    };
    const emit = () => {
      const text = compose();
      if (cut === null) {
        const truncated = cutAtStop(text);
        if (truncated !== null) cut = truncated;
      }
      opts.onToken(cut ?? text);
    };

    const thinking = remote.reasoning ? opts.thinking !== false : false;
    const body: Record<string, unknown> = {
      model: remote.model,
      // stable per-browser id so the proxy can rate-limit per client (not just IP)
      client_id: clientId(),
      stream: true,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      temperature: opts.temperature,
      max_tokens: opts.maxTokens,
      top_p: thinking ? 0.95 : 0.9,
      top_k: 40,
      // greedy verify on the server engages when this is a deterministic sample;
      // these match the in-browser sampling so output style is consistent.
      repeat_penalty: 1.1,
      presence_penalty: 0.4,
    };
    if (remote.reasoning) {
      // SIQ-1 toggles thinking via the chat template; effort is injected as a
      // system line by the proxy when thinking is on (see server/index.js).
      body.chat_template_kwargs = { enable_thinking: thinking };
      if (thinking && opts.effort) body.effort = opts.effort;
    }

    const res = await fetch(`${remote.endpoint}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: opts.signal,
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`inference endpoint ${res.status} ${res.statusText}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    try {
      while (cut === null) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const s = line.trim();
          if (!s.startsWith("data:")) continue;
          const data = s.slice(5).trim();
          if (data === "[DONE]") return cut ?? compose();
          try {
            const delta = JSON.parse(data)?.choices?.[0]?.delta ?? {};
            const rc = delta.reasoning_content ?? "";
            const c = delta.content ?? "";
            if (rc) reasoning += rc;
            if (c) answer += c;
            if (rc || c) emit();
          } catch {
            /* partial JSON across chunk boundary — wait for more */
          }
          if (cut !== null) break;
        }
      }
      try {
        await reader.cancel();
      } catch {
        /* ignore */
      }
      return cut ?? compose();
    } catch (err) {
      if (cut !== null) return cut; // stopped by our stop-sequence
      if ((err as { name?: string })?.name === "AbortError") return compose(); // user abort
      throw err;
    }
  }
}

export const engine = new LlamaEngine();
