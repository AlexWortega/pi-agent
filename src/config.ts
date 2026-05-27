import type { ModelPreset, GenParams } from "./types";

const HF = "https://huggingface.co";

export function ggufUrl(repo: string, file: string): string {
  return `${HF}/${repo}/resolve/main/${file}`;
}

/**
 * The Pi Agent runs the real Soyuz model — Qwen3.5-4B (hybrid linear-attention,
 * qwen3next gguf arch). The @reeselevine/wllama-webgpu WebGPU build ships the
 * qwen3next arch + its ops (gated_delta / linear_attn / ssm_scan), so it loads
 * and runs in-browser on your GPU.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "soyuz-4b",
    label: "Soyuz Qwen3.5-4B",
    repo: "AlexWortega/qwen35-4b-soyuz-merged-gguf",
    file: "qwen35-4b-soyuz-merged.nomtp.Q4_K_M.gguf",
    sizeLabel: "~2.5 GB",
    note: "The real Pi Agent brain — Soyuz, fine-tuned for self-contained web apps.",
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
  temperature: 0.6,
  maxTokens: 4096,
  contextLength: 8192,
};
