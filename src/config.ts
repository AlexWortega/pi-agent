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
 * The Pi Agent runs the real Soyuz model — Qwen3.5-4B (hybrid linear-attention,
 * qwen3next gguf arch). The @reeselevine/wllama-webgpu WebGPU build ships the
 * qwen3next arch + its ops (gated_delta / linear_attn / ssm_scan), so it loads
 * and runs in-browser on your GPU.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "soyuz-4b",
    label: "Soyuz Qwen3.5-4B (vibeapps)",
    repo: "AlexWortega/qwen35-4b-soyuz-vibeapps-merged",
    file: "vibeapps.Q4_K_M.gguf",
    sizeLabel: "~2.5 GB",
    note: "Soyuz vibeapps checkpoint — fine-tuned for self-contained web apps.",
    verified: true,
    accent: "#ff7a45",
  },
];

export const DEFAULT_MODEL_ID = "soyuz-4b";

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
};
