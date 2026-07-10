import type { ModelPreset, GenParams } from "./types";

const HF = "https://huggingface.co";

export function ggufUrl(repo: string, file: string): string {
  return `${HF}/${repo}/resolve/main/${file}`;
}

/**
 * Logging API base (Railway). Public endpoint — fine to ship in the bundle.
 * Override at build time with VITE_LOG_API; set to "" to disable logging.
 */
export const LOG_API: string =
  import.meta.env.VITE_LOG_API ?? "https://api-production-bd22.up.railway.app";

/**
 * Optional DFlash llama-server endpoint (OpenAI-compatible). When set, the agent
 * runs inference against this server — speculative decoding (DFlash) happens
 * server-side — instead of the in-browser WebGPU model. Set with VITE_LLM_SERVER,
 * e.g. "https://my-dflash-host". Empty (default) = in-browser wllama-webgpu.
 *
 * Run the server with the DFlash target+drafter:
 *   llama-server -m Qwen3.5-4B-Q8_0.gguf -md Qwen3.5-4B-DFlash-f16.gguf --dflash \
 *     --draft-max 16 -ngl 99 -ngld 99 -np 1 --host 0.0.0.0 --port 8080
 */
export const LLM_SERVER: string =
  (import.meta.env.VITE_LLM_SERVER ?? "").replace(/\/+$/, "");

/**
 * OpenAI-compatible base for the SIQ-1-35B RunPod serverless endpoint. The
 * browser can't call RunPod directly (API key + CORS + async polling), so it
 * talks to a streaming proxy that holds RUNPOD_API_KEY server-side and bridges
 * to RunPod. `/v1/chat/completions` is appended by the engine.
 *
 * Default: the Railway server's /api/siq route (same host as LOG_API, already in
 * the bundle). A mirror also runs as an HF Space (siq_proxy_space/, repo
 * AlexWortega/siq-proxy) at https://alexwortega-siq-proxy.hf.space/api/siq —
 * point VITE_SIQ_API there as a fallback if Railway is down.
 */
export const SIQ_API: string = (
  // Default to the HF Space proxy — it holds the CURRENT RunPod endpoint id
  // (the Railway deployment still points at a deleted endpoint, which queued
  // jobs forever and looked like "the model never edits anything").
  import.meta.env.VITE_SIQ_API ?? "https://alexwortega-siq-proxy.hf.space/api/siq"
).replace(/\/+$/, "");

/**
 * Local endpoint for SIQ-1 when the model picker's local/remote toggle is set to
 * "local": an OpenAI-compatible llama-server you run yourself with the SIQ-1 GGUF
 * (e.g. `llama-server -m SIQ-1-35B.Q4_K_M.gguf --jinja --host 0.0.0.0 --port 8080`).
 * `/v1/chat/completions` is appended by the engine. Override with VITE_SIQ_LOCAL.
 */
export const SIQ_LOCAL: string = (
  import.meta.env.VITE_SIQ_LOCAL ?? "http://localhost:8080"
).replace(/\/+$/, "");

/**
 * The Pi Agent runs the real Soyuz model — Qwen3.5-4B (hybrid linear-attention,
 * qwen3next gguf arch). The @reeselevine/wllama-webgpu WebGPU build ships the
 * qwen3next arch + its ops (gated_delta / linear_attn / ssm_scan), so it loads
 * and runs in-browser on your GPU.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "siq1-35b",
    label: "SIQ-1-35B (cloud)",
    // contextWindow matches the serving worker's N_CTX (65536).
    remote: { endpoint: SIQ_API, model: "siq", reasoning: true, contextWindow: 65536 },
    sizeLabel: "cloud · 35B-A3B",
    note: "Big reasoning model (Qwen3.6-35B-A3B MoE + Soyuz + RFT) on RunPod serverless — nothing downloads, no WebGPU needed. Cold starts can take minutes; if it stays queued the run aborts with a clear error.",
    verified: false,
    accent: "#7c5cff",
  },
  {
    id: "soyuz-4b",
    label: "Soyuz Qwen3.5-4B (vibeapps)",
    repo: "AlexWortega/qwen35-4b-soyuz-vibeapps-merged",
    // Split into 7×~512MB shards (llama-gguf-split) so wllama downloads them
    // in parallel (parallelDownloads, default 3) instead of crawling a single
    // 2.7GB file on one connection. Pass the first shard; wllama auto-detects
    // the rest from the gguf-split naming convention.
    file: "vibeapps.Q4_K_M-00001-of-00007.gguf",
    sizeLabel: "~2.5 GB",
    note: "Soyuz vibeapps checkpoint — fine-tuned for self-contained web apps. Runs in your browser on WebGPU.",
    verified: true,
    accent: "#ff7a45",
  },
];

// Default to the LOCAL model: it needs no server and always works on a WebGPU
// browser. The SIQ cloud endpoint proved operationally flaky (deleted RunPod
// endpoint → jobs queued forever and the agent "never edited anything");
// it stays in the picker, but a broken default is worse than a heavy one.
export const DEFAULT_MODEL_ID = "soyuz-4b";

/**
 * OpenRouter (BYO key): the browser calls https://openrouter.ai/api/v1/chat/completions
 * directly with the user's key — no proxy. `/v1/chat/completions` is appended by
 * the engine, so the base stops at /api.
 */
export const OPENROUTER_API = "https://openrouter.ai/api";

/** localStorage key for the user's OpenRouter API key (never shipped in the bundle). */
export const OPENROUTER_KEY_STORAGE = "pi_openrouter_key";

export interface CloudPreset {
  /** OpenRouter model id, e.g. "anthropic/claude-sonnet-5". */
  id: string;
  label: string;
  note: string;
  accent: string;
  contextWindow: number;
}

/**
 * Frontier models reached through OpenRouter with the user's own key. Same pi
 * agent loop and tools — the engine's remote path does the streaming, with
 * native OpenAI tool_calls (serialized back into the <tool_call> text protocol
 * that the rest of the pipeline already parses).
 */
export const CLOUD_PRESETS: CloudPreset[] = [
  {
    id: "anthropic/claude-sonnet-5",
    label: "Claude Sonnet 5",
    note: "Excellent coding agent; the reference cloud brain.",
    accent: "#d97757",
    contextWindow: 200_000,
  },
  {
    id: "openai/gpt-5.5",
    label: "GPT-5.5",
    note: "OpenAI flagship, strong at tools and UI work.",
    accent: "#4aa8ff",
    contextWindow: 256_000,
  },
  {
    id: "google/gemini-3.5-flash",
    label: "Gemini 3.5 Flash",
    note: "Fast and cheap, great for quick iterations.",
    accent: "#8b5cf6",
    contextWindow: 1_000_000,
  },
  {
    id: "moonshotai/kimi-k2.7-code",
    label: "Kimi K2.7 Code",
    note: "Open-weights coding model, very good agentic behavior.",
    accent: "#2fe6b0",
    contextWindow: 256_000,
  },
  {
    id: "deepseek/deepseek-v3.2",
    label: "DeepSeek V3.2",
    note: "Budget-friendly generalist with solid coding.",
    accent: "#ffb454",
    contextWindow: 160_000,
  },
];

export const SYSTEM_PROMPT = `You are Soyuz, the Pi Agent — a sharp, fast coding assistant.
Reason briefly inside <think> ... </think>, then answer.
When the user asks for a web app, page, game, tool, or any visual UI, output ONE complete, self-contained HTML file with inline CSS and JavaScript inside a single \`\`\`html code block. Do not use external files. Only use CDNs if strictly necessary. Make it polished, responsive and genuinely nice-looking.
Keep the thinking short so you have room for the code.`;

export const DEFAULT_PARAMS: GenParams = {
  // Tuned against the real model on the headless rig: at temp 0.1 Soyuz
  // deterministically does a short <think> then calls `write` with the full
  // file as clean, valid JSON (temp 0.8 garbled the long JSON — "f xj nfr").
  // maxTokens must be large enough to finish the file inside one write call.
  temperature: 0.1,
  maxTokens: 8192,
  contextLength: 16384,
  // Remote-only (SIQ-1): thinking on, medium effort — the sweet spot from the
  // GPQA effort sweep (medium/high ≈ 79% on the 24-q slice). Ignored by local.
  thinking: true,
  effort: "medium",
  // SIQ-1 runs on the cloud proxy by default; flip to "local" to hit a local llama-server.
  endpointMode: "remote",
};
