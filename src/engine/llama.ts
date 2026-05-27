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

/**
 * Thin singleton wrapper around the WebGPU build of llama.cpp
 * (@reeselevine/wllama-webgpu). One model loaded at a time.
 */
class LlamaEngine {
  private wllama: Wllama | null = null;
  loadedUrl: string | null = null;
  readonly backend: "webgpu" | "cpu" = hasWebGPU() ? "webgpu" : "cpu";

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
  }

  /** Run a streaming chat completion. Resolves with the final full text. */
  async chat(messages: Pick<ChatMessage, "role" | "content">[], opts: ChatOpts): Promise<string> {
    if (!this.wllama) throw new Error("No model loaded");

    const result = await this.wllama.createChatCompletion(
      messages.map((m) => ({ role: m.role, content: m.content })),
      {
        nPredict: opts.maxTokens,
        abortSignal: opts.signal,
        sampling: {
          temp: opts.temperature,
          top_k: 40,
          top_p: 0.9,
        },
        onNewToken: (_token: number, _piece: Uint8Array, currentText: string) => {
          opts.onToken(currentText);
        },
      },
    );
    return typeof result === "string" ? result : String(result ?? "");
  }
}

export const engine = new LlamaEngine();
