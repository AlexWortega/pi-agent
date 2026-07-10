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

export interface NativeTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface ChatOpts {
  temperature: number;
  maxTokens: number;
  signal?: AbortSignal;
  onToken: (fullText: string) => void;
  /** Remote-only (SIQ-1): thinking mode + reasoning effort. */
  thinking?: boolean;
  effort?: ReasoningEffort;
  /** Remote-only: RunPod job lifecycle (IN_QUEUE/IN_PROGRESS) for a live status. */
  onStatus?: (status: string) => void;
  /** Tools to pass natively when the endpoint supports OpenAI tool_calls (e.g. OpenRouter). */
  tools?: NativeTool[];
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

    // wllama's own default is hardwareConcurrency/2 when n_threads is unset —
    // leave just one core free for UI/rendering instead of giving up half the
    // machine. Only takes effect when cross-origin isolation (see
    // coi-serviceworker in index.html) actually unlocks the multi-thread build.
    const n_threads = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
    console.debug(`[pi] loading model on ${this.backend}, n_ctx=${opts.contextLength}, n_threads=${n_threads}…`);
    const t0 = performance.now();
    const wllama = new Wllama(WasmFromPackage, { backend: this.backend });
    await wllama.loadModelFromUrl(url, {
      n_ctx: opts.contextLength,
      n_batch: 256,
      n_threads,
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
          // Reuse the KV cache for the unchanged history prefix instead of
          // re-prefilling the whole conversation every turn — without this,
          // an agent loop with growing tool-call history gets quadratically
          // slower as the conversation goes on.
          useCache: true,
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
    // Direct API calls (apiKey set = OpenRouter / Claude) use native tool_calls;
    // proxy calls (SIQ-1 via Railway) rely on the <tool_call> text format.
    const useNativeTools = !!remote.apiKey && !!opts.tools?.length;
    const body: Record<string, unknown> = {
      model: remote.model,
      stream: true,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      temperature: opts.temperature,
      max_tokens: opts.maxTokens,
    };
    if (remote.apiKey) {
      // Direct API (OpenRouter): frontier models sample fine on their provider
      // defaults — the llama.cpp anti-loop knobs below would only distort them.
    } else {
      body.client_id = clientId();
      body.top_p = thinking ? 0.95 : 0.9;
      body.top_k = 40;
      body.repeat_penalty = 1.1;
      body.presence_penalty = 0.4;
    }
    if (useNativeTools) {
      body.tools = opts.tools!.map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
      body.tool_choice = "auto";
    }
    if (remote.reasoning) {
      body.chat_template_kwargs = { enable_thinking: thinking };
      if (thinking && opts.effort) body.effort = opts.effort;
    }

    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (remote.apiKey) {
      headers["Authorization"] = `Bearer ${remote.apiKey}`;
      // OpenRouter recommends these for usage attribution
      headers["HTTP-Referer"] = "https://alexwortega.github.io/pi-agent/";
      headers["X-Title"] = "Pi Agent";
    }
    const res = await fetch(`${remote.endpoint}/v1/chat/completions`, {
      method: "POST",
      headers,
      signal: opts.signal,
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`inference endpoint ${res.status} ${res.statusText}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    // Accumulate native OpenAI tool_calls across SSE chunks (keyed by index).
    const pendingCalls: Record<number, { name: string; args: string }> = {};
    // Queue watchdog (see below): flags set inside the parse-try (whose catch
    // exists to swallow partial-JSON errors, so we must not throw in there).
    let queuedSince = 0;
    let queueTimedOut = false;
    const QUEUE_LIMIT_MS = 180_000;
    try {
      while (cut === null) {
        const { done, value } = await reader.read();
        if (done) break;
        if (queueTimedOut) {
          throw new Error(
            "The SIQ-1 cloud endpoint is not picking up jobs (queued for over 3 minutes). " +
              "It may be cold-starting or out of capacity — try again in a few minutes, " +
              "or pick another model in the picker (OpenRouter with your key, or the local Soyuz).",
          );
        }
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const s = line.trim();
          if (!s.startsWith("data:")) continue;
          const data = s.slice(5).trim();
          if (data === "[DONE]") {
            // Serialize any native tool calls as <tool_call> text blocks.
            if (useNativeTools) {
              const indices = Object.keys(pendingCalls).map(Number).sort((a, b) => a - b);
              for (const idx of indices) {
                const tc = pendingCalls[idx];
                if (!tc.name) continue;
                let args: unknown;
                try { args = JSON.parse(tc.args); } catch { args = {}; }
                answer += `\n<tool_call>\n${JSON.stringify({ name: tc.name, arguments: args })}\n</tool_call>`;
              }
              if (indices.length) emit();
            }
            return cut ?? compose();
          }
          try {
            const obj = JSON.parse(data);
            if (obj?.siq_status) {
              // Fail fast when the job never leaves the queue: a healthy
              // endpoint assigns a worker within a couple of minutes; endless
              // IN_QUEUE means it's dead/out of capacity — surface a real,
              // actionable error instead of an infinite "starting cloud GPU…".
              if (obj.siq_status === "IN_QUEUE") {
                if (queuedSince === 0) queuedSince = Date.now();
                else if (Date.now() - queuedSince > QUEUE_LIMIT_MS) queueTimedOut = true;
              } else {
                queuedSince = 0; // worker picked it up (IN_PROGRESS etc.)
              }
              opts.onStatus?.(obj.siq_status);
              continue;
            }
            const delta = obj?.choices?.[0]?.delta ?? {};
            const rc = delta.reasoning_content ?? "";
            const c = delta.content ?? "";
            if (rc) reasoning += rc;
            if (c) answer += c;
            if (rc || c) emit();
            // Accumulate native tool_calls fragments.
            if (useNativeTools && Array.isArray(delta.tool_calls)) {
              for (const tc of delta.tool_calls as Array<{ index: number; function?: { name?: string; arguments?: string } }>) {
                if (!pendingCalls[tc.index]) pendingCalls[tc.index] = { name: "", args: "" };
                if (tc.function?.name) pendingCalls[tc.index].name += tc.function.name;
                if (tc.function?.arguments) pendingCalls[tc.index].args += tc.function.arguments;
              }
            }
          } catch {
            /* partial JSON across chunk boundary */
          }
          if (cut !== null) break;
        }
      }
      try { await reader.cancel(); } catch { /* ignore */ }
      // Flush any tool calls if stream ended without [DONE].
      if (useNativeTools) {
        const indices = Object.keys(pendingCalls).map(Number).sort((a, b) => a - b);
        for (const idx of indices) {
          const tc = pendingCalls[idx];
          if (!tc.name) continue;
          let args: unknown;
          try { args = JSON.parse(tc.args); } catch { args = {}; }
          answer += `\n<tool_call>\n${JSON.stringify({ name: tc.name, arguments: args })}\n</tool_call>`;
        }
      }
      return cut ?? compose();
    } catch (err) {
      if (cut !== null) return cut;
      if ((err as { name?: string })?.name === "AbortError") return compose();
      throw err;
    }
  }
}

export const engine = new LlamaEngine();
