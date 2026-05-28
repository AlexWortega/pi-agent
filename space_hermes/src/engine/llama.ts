import { Wllama } from "@reeselevine/wllama-webgpu";
import WasmFromPackage from "@reeselevine/wllama-webgpu/esm/wasm-from-package.js";
import type { ChatMessage } from "../types";

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

export interface RawCompleteOpts {
  temperature: number;
  maxTokens: number;
  signal?: AbortSignal;
  /** stop generation as soon as one of these substrings appears in the output */
  stopStrings?: string[];
  onToken?: (fullText: string) => void;
}

export type StoppedBy = "tool_call" | "stop" | "eos" | "abort";

export interface RawCompleteResult {
  text: string;
  stoppedBy: StoppedBy;
}

/**
 * Thin singleton wrapper around the WebGPU build of llama.cpp
 * (@reeselevine/wllama-webgpu). One model loaded at a time.
 *
 * In addition to the base `chat()` path it exposes `rawComplete()`, which feeds
 * a hand-built ChatML prompt straight to llama.cpp (no chat template) so the
 * Hermes agent loop can inject a `tool` role, the <tools> block, and stop the
 * stream the instant a </tool_call> closes.
 */
class LlamaEngine {
  private wllama: Wllama | null = null;
  loadedUrl: string | null = null;
  readonly backend: "webgpu" | "cpu" = hasWebGPU() ? "webgpu" : "cpu";
  /** token ids that end a turn (EOS / EOT / <|im_end|>), resolved on load */
  private stopTokens: number[] = [];

  get ready(): boolean {
    return !!this.wllama && !!this.loadedUrl;
  }

  /** Download + load a GGUF. No-op if the same url is already loaded. */
  async load(url: string, opts: LoadOpts): Promise<void> {
    if (this.loadedUrl === url && this.wllama) return;
    // free any previous model first
    await this.unload();

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

    // resolve end-of-turn token ids once for the raw-completion path
    const stops = new Set<number>();
    try {
      const eos = wllama.getEOS();
      if (typeof eos === "number" && eos >= 0) stops.add(eos);
      const eot = wllama.getEOT();
      if (typeof eot === "number" && eot >= 0) stops.add(eot);
    } catch {
      /* ignore */
    }
    try {
      const im = await wllama.lookupToken("<|im_end|>");
      if (im >= 0) stops.add(im);
    } catch {
      /* ignore */
    }
    this.stopTokens = [...stops];
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
    this.stopTokens = [];
  }

  /** Run a streaming chat completion (uses the GGUF template). Kept for any
   *  non-agent direct calls. */
  async chat(messages: Pick<ChatMessage, "role" | "content">[], opts: ChatOpts): Promise<string> {
    if (!this.wllama) throw new Error("No model loaded");

    const result = await this.wllama.createChatCompletion(
      messages.map((m) => ({ role: m.role as "system" | "user" | "assistant", content: m.content })),
      {
        nPredict: opts.maxTokens,
        abortSignal: opts.signal,
        sampling: {
          temp: opts.temperature,
          top_k: 40,
          top_p: 0.9,
          min_p: 0.05,
          penalty_repeat: 1.12,
          penalty_last_n: 128,
        },
        onNewToken: (_token: number, _piece: Uint8Array, currentText: string) => {
          opts.onToken(currentText);
        },
      },
    );
    return typeof result === "string" ? result : String(result ?? "");
  }

  /**
   * Raw completion over a pre-formatted ChatML prompt. Stops on end-of-turn
   * token ids and, since llama.cpp here only supports token-id stops, also on
   * any `stopStrings` substring (detected in the cumulative text → abort). The
   * accumulated text up to and including the stop string is returned.
   */
  async rawComplete(prompt: string, opts: RawCompleteOpts): Promise<RawCompleteResult> {
    if (!this.wllama) throw new Error("No model loaded");

    const inner = new AbortController();
    const onOuterAbort = () => inner.abort();
    opts.signal?.addEventListener("abort", onOuterAbort);

    const stopStrings = opts.stopStrings ?? [];
    let text = "";
    let stoppedBy: StoppedBy = "eos";

    try {
      const result = await this.wllama.createCompletion(prompt, {
        nPredict: opts.maxTokens,
        abortSignal: inner.signal,
        stopTokens: this.stopTokens,
        sampling: {
          temp: opts.temperature,
          top_k: 40,
          top_p: 0.9,
          min_p: 0.05,
          penalty_repeat: 1.12,
          penalty_last_n: 128,
        },
        onNewToken: (_t: number, _p: Uint8Array, currentText: string) => {
          text = currentText;
          opts.onToken?.(currentText);
          for (const s of stopStrings) {
            const i = currentText.indexOf(s);
            if (i >= 0) {
              text = currentText.slice(0, i + s.length);
              stoppedBy = s === "</tool_call>" ? "tool_call" : "stop";
              inner.abort();
              return;
            }
          }
        },
      });
      // normal end-of-turn (stop token / length): trust the returned string
      if (stoppedBy === "eos" && typeof result === "string") text = result;
    } catch (e: any) {
      if (inner.signal.aborted) {
        // a stop-string hit (tool_call/stop) keeps its reason; an external
        // abort with no stop-string is a user cancel.
        if (stoppedBy === "eos") stoppedBy = "abort";
      } else {
        throw e;
      }
    } finally {
      opts.signal?.removeEventListener("abort", onOuterAbort);
    }

    return { text, stoppedBy };
  }
}

export const engine = new LlamaEngine();
