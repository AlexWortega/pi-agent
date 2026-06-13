import { Wllama } from "@reeselevine/wllama-webgpu";
import WasmFromPackage from "@reeselevine/wllama-webgpu/esm/wasm-from-package.js";
import type { ChatMessage } from "../types";

import { LLM_SERVER } from "../config";
import { cutAtStop } from "./stops";

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as any).gpu;
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
}

/**
 * Thin singleton wrapper around the WebGPU build of llama.cpp
 * (@reeselevine/wllama-webgpu). One model loaded at a time.
 */
class LlamaEngine {
  private wllama: Wllama | null = null;
  loadedUrl: string | null = null;
  private loadedCtx: number | null = null;
  readonly backend: "webgpu" | "cpu" = hasWebGPU() ? "webgpu" : "cpu";

  /** Server mode: inference runs against a DFlash llama-server instead of in-browser. */
  readonly serverMode: boolean = !!LLM_SERVER;

  get ready(): boolean {
    return this.serverMode || (!!this.wllama && !!this.loadedUrl);
  }

  /**
   * Download + load a GGUF. No-op if the same url is already loaded at the same
   * context length; reloads (from cache) when the context length changes.
   */
  async load(url: string, opts: LoadOpts): Promise<void> {
    // server mode: nothing to download/load in the browser — the DFlash server holds the model
    if (this.serverMode) {
      this.loadedUrl = `server:${LLM_SERVER}`;
      opts.onProgress?.(1, 1, 1);
      return;
    }
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
    if (this.serverMode) return this.chatViaServer(messages, opts);
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
   * Stream a chat completion from a DFlash llama-server (OpenAI-compatible SSE).
   * Speculative decoding runs server-side; the client just consumes the stream and
   * applies the same stop-sequence cut as the in-browser path.
   */
  private async chatViaServer(
    messages: Pick<ChatMessage, "role" | "content">[],
    opts: ChatOpts,
  ): Promise<string> {
    let cut: string | null = null;
    let full = "";
    const onChunk = (text: string) => {
      if (cut === null) {
        const truncated = cutAtStop(text);
        if (truncated !== null) cut = truncated;
      }
      opts.onToken(cut ?? text);
    };

    const res = await fetch(`${LLM_SERVER}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: opts.signal,
      body: JSON.stringify({
        model: "soyuz",
        stream: true,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        top_p: 0.9,
        top_k: 40,
        // greedy verify on the server engages when this is a deterministic sample;
        // these match the in-browser sampling so output style is consistent.
        repeat_penalty: 1.1,
        presence_penalty: 0.4,
      }),
    });
    if (!res.ok || !res.body) throw new Error(`llama-server ${res.status} ${res.statusText}`);

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
          if (data === "[DONE]") return cut ?? full;
          try {
            const delta = JSON.parse(data)?.choices?.[0]?.delta?.content ?? "";
            if (delta) {
              full += delta;
              onChunk(full);
            }
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
      return cut ?? full;
    } catch (err) {
      if (cut !== null) return cut; // stopped by our stop-sequence
      if ((err as { name?: string })?.name === "AbortError") return full; // user abort
      throw err;
    }
  }
}

export const engine = new LlamaEngine();
