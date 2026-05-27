import type { ModelPreset, GenParams } from "./types";

const HF = "https://huggingface.co";

export function ggufUrl(repo: string, file: string): string {
  return `${HF}/${repo}/resolve/main/${file}`;
}

/**
 * Model presets. All run on the @reeselevine/wllama-webgpu WebGPU backend.
 * `verified: true` = tested working in the llamas-on-the-web paper/demo or
 * confirmed by us. The Soyuz 4B is Qwen3.5 hybrid linear-attention
 * (qwen3next gguf arch) which is NOT yet in the verified set — try it, but
 * it may fail to load if the backend lacks an op it needs.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "soyuz-4b",
    label: "Soyuz Qwen3.5-4B",
    repo: "AlexWortega/qwen35-4b-soyuz-merged-gguf",
    file: "qwen35-4b-soyuz-merged.nomtp.Q4_K_M.gguf",
    sizeLabel: "~2.5 GB",
    note: "The real Pi Agent brain. Hybrid linear-attn — experimental on WebGPU.",
    verified: false,
    accent: "#ff7a45",
  },
  {
    id: "qwen35-2b",
    label: "Qwen3.5-2B",
    repo: "unsloth/Qwen3.5-2B-GGUF",
    file: "Qwen3.5-2B-Q4_K_M.gguf",
    sizeLabel: "~1.4 GB",
    note: "Dense Qwen3.5. Verified on the WebGPU backend — safe default.",
    verified: true,
    accent: "#7c5cff",
  },
  {
    id: "qwen3-06b",
    label: "Qwen3-0.6B",
    repo: "unsloth/Qwen3-0.6B-GGUF",
    file: "Qwen3-0.6B-Q4_K_M.gguf",
    sizeLabel: "~0.4 GB",
    note: "Tiny & fast. Great for testing the pipeline on weak GPUs.",
    verified: true,
    accent: "#2fe6b0",
  },
  {
    id: "gemma3-270m",
    label: "Gemma-3 270M",
    repo: "unsloth/gemma-3-270m-it-GGUF",
    file: "gemma-3-270m-it-Q4_K_M.gguf",
    sizeLabel: "~0.2 GB",
    note: "Smallest. Loads in seconds. Output quality is limited.",
    verified: true,
    accent: "#9aa0b8",
  },
];

export const DEFAULT_MODEL_ID = "qwen35-2b";

export const SYSTEM_PROMPT = `You are Soyuz, the Pi Agent — a sharp, fast coding assistant.
Reason briefly inside <think> ... </think>, then answer.
When the user asks for a web app, page, game, tool, or any visual UI, output ONE complete, self-contained HTML file with inline CSS and JavaScript inside a single \`\`\`html code block. Do not use external files. Only use CDNs if strictly necessary. Make it polished, responsive and genuinely nice-looking.
Keep the thinking short so you have room for the code.`;

export const DEFAULT_PARAMS: GenParams = {
  temperature: 0.6,
  maxTokens: 4096,
  contextLength: 8192,
};
