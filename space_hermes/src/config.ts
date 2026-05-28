import type { ModelPreset, GenParams } from "./types";

const HF = "https://huggingface.co";

export function ggufUrl(repo: string, file: string): string {
  return `${HF}/${repo}/resolve/main/${file}`;
}

/**
 * Logging API base. Disabled by default for the Hermes space (set VITE_LOG_API
 * at build time to re-enable). Logging must never affect the chat.
 */
export const LOG_API: string = import.meta.env.VITE_LOG_API ?? "";

/**
 * Same brain as the Pi Agent demo: the Soyuz Qwen3.5-4B GGUF (hybrid
 * linear-attention, qwen3next arch). The @reeselevine/wllama-webgpu WebGPU
 * build ships that arch + its ops, so it loads and runs in-browser on the GPU.
 * Here we wrap it in a Hermes-style tool-calling agent loop instead of the
 * single-shot HTML coder.
 */
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "soyuz-4b",
    label: "Soyuz Qwen3.5-4B",
    repo: "AlexWortega/qwen35-4b-soyuz-merged-gguf",
    file: "qwen35-4b-soyuz-merged.nomtp.Q4_K_M.gguf",
    sizeLabel: "~2.5 GB",
    note: "Runs the Hermes agent loop in your browser — tools, skills, memory, all local.",
    verified: true,
    accent: "#16b3a7",
  },
];

export const DEFAULT_MODEL_ID = "soyuz-4b";

/** The reader proxy used as a CORS fallback by web_extract (configurable). */
export const READER_PROXY = "https://r.jina.ai/";

/**
 * Hermes function-calling preamble (ChatML). The registry appends the live
 * <tools> JSON block; relevant skills + memories are injected separately.
 * A short few-shot nudges the (non-Hermes-tuned) Soyuz model to emit
 * well-formed <tool_call> JSON.
 */
const HERMES_PERSONA = `You are Hermes, a capable, self-improving AI agent running entirely in the user's browser. You are direct, resourceful and friendly.

You think step by step inside <think>…</think>, then act. To use a tool, emit a JSON object inside <tool_call></tool_call> tags using this schema:
{"name": <tool-name>, "arguments": <args-object>}

Rules:
- Call a tool ONLY when it genuinely helps (math, current time, the web, memory, saved skills, building a UI, running code, delegating). Don't invent argument values — ask the user with the "clarify" tool if something is missing.
- After a tool runs you receive a <tool_response>. Use it, then either call another tool or give your final answer. When you have enough to answer, just write the answer with NO tool_call.
- To build a web app/page/game/visual UI, call the "render_html" tool with one complete self-contained HTML file (inline CSS+JS). It renders live in the canvas on the right.
- Keep reasoning brief. Never fabricate tool outputs.
- If the user just asks a normal question that needs no tool, answer directly — DO NOT call a tool.
- Don't repeat yourself. Don't pad. Don't list duplicates.`;

/** Worked example as REAL ChatML turns (not embedded text), so the tokenizer
 *  sees proper turn boundaries instead of literal "<|im_start|>" strings. */
export const FEW_SHOT_WIRE = [
  { role: "user" as const, content: "what's 12.5% of 840, and what's today's date?" },
  {
    role: "assistant" as const,
    content:
      '<think>Two quick lookups.</think>\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "840 * 0.125"}}\n</tool_call>',
  },
  {
    role: "tool" as const,
    content: '<tool_response>\n{"name": "calculator", "content": {"result": 105}}\n</tool_response>',
  },
  {
    role: "assistant" as const,
    content: '<tool_call>\n{"name": "datetime", "arguments": {}}\n</tool_call>',
  },
  {
    role: "tool" as const,
    content:
      '<tool_response>\n{"name": "datetime", "content": {"iso": "2026-05-27T10:00:00Z"}}\n</tool_response>',
  },
  { role: "assistant" as const, content: "12.5% of 840 is **105**, and today is **2026-05-27**." },
];

/** Compose the system message: persona + live tools + memory. The few-shot
 *  example is injected as separate wire turns, not embedded inside system. */
export function buildSystemPrompt(toolsJson: string, memoryBlock?: string): string {
  let s = HERMES_PERSONA + `\n\nHere are the available tools: <tools>${toolsJson}</tools>`;
  if (memoryBlock && memoryBlock.trim()) {
    s += `\n\nWhat you remember about this user (from past sessions):\n${memoryBlock.trim()}`;
  }
  return s;
}

export const DEFAULT_PARAMS: GenParams = {
  temperature: 0.5,
  maxTokens: 2048,
  contextLength: 8192,
  maxIterations: 6,
};
