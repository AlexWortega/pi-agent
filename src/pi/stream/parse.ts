/**
 * Parse Soyuz / Qwen-style assistant output into structured regions.
 *
 * The model emits a single text stream that interleaves three things:
 *   - reasoning wrapped in <think> … </think>
 *   - normal prose / explanation
 *   - tool calls in the Hermes/Qwen format:
 *       <tool_call>
 *       {"name": "...", "arguments": { ... }}
 *       </tool_call>
 *
 * We scan the cumulative text left-to-right, classifying each region. The scan
 * is streaming-tolerant: an unterminated <think> or <tool_call> at the end is
 * reported via `open`, so the live UI can keep showing partial reasoning/text
 * without prematurely treating a half-written tool call as prose.
 */

export interface ParsedToolCall {
  name: string;
  arguments: Record<string, unknown>;
  /** Raw JSON text inside the tool_call block (for diagnostics / repair). */
  raw: string;
  /** True when the JSON could not be parsed into {name, arguments}. */
  malformed: boolean;
}

export interface ParsedOutput {
  /** Concatenation of all <think> content seen so far. */
  thinking: string;
  /** Visible prose with think/tool_call regions removed. */
  text: string;
  /** Completed tool_call blocks (only closed ones). */
  toolCalls: ParsedToolCall[];
  /** Which tag, if any, is currently open at the end of the stream. */
  open: "think" | "tool_call" | null;
}

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";
const TOOL_OPEN = "<tool_call>";
const TOOL_CLOSE = "</tool_call>";

/** Best-effort JSON repair for small-model output: strip fences, trailing commas. */
function looseJsonParse(raw: string): Record<string, unknown> | null {
  const trimmed = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  const candidates = [trimmed, trimmed.replace(/,(\s*[}\]])/g, "$1")];
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
    } catch {
      /* try next */
    }
  }
  return null;
}

function toToolCall(raw: string): ParsedToolCall {
  const parsed = looseJsonParse(raw);
  const name = parsed && typeof parsed.name === "string" ? parsed.name : "";
  let args: Record<string, unknown> = {};
  if (parsed && parsed.arguments && typeof parsed.arguments === "object") {
    args = parsed.arguments as Record<string, unknown>;
  } else if (parsed) {
    // Some outputs inline args at the top level alongside "name".
    const { name: _drop, ...rest } = parsed;
    args = rest;
  }
  return { name, arguments: args, raw: raw.trim(), malformed: !parsed || name === "" };
}

/**
 * Find the earliest occurrence of any of the given markers at or after `from`.
 * Returns the marker and its index, or null if none found.
 */
function nextMarker(text: string, from: number, markers: string[]): { marker: string; index: number } | null {
  let best: { marker: string; index: number } | null = null;
  for (const marker of markers) {
    const index = text.indexOf(marker, from);
    if (index !== -1 && (best === null || index < best.index)) {
      best = { marker, index };
    }
  }
  return best;
}

export function parseOutput(full: string): ParsedOutput {
  let thinking = "";
  let text = "";
  const toolCalls: ParsedToolCall[] = [];
  let open: "think" | "tool_call" | null = null;

  let i = 0;
  while (i < full.length) {
    const marker = nextMarker(full, i, [THINK_OPEN, TOOL_OPEN]);
    if (!marker) {
      text += full.slice(i);
      break;
    }
    // Prose before the next tagged region.
    text += full.slice(i, marker.index);

    if (marker.marker === THINK_OPEN) {
      const contentStart = marker.index + THINK_OPEN.length;
      const close = full.indexOf(THINK_CLOSE, contentStart);
      if (close === -1) {
        thinking += full.slice(contentStart);
        open = "think";
        break;
      }
      thinking += full.slice(contentStart, close);
      i = close + THINK_CLOSE.length;
    } else {
      const contentStart = marker.index + TOOL_OPEN.length;
      const close = full.indexOf(TOOL_CLOSE, contentStart);
      if (close === -1) {
        open = "tool_call";
        break;
      }
      toolCalls.push(toToolCall(full.slice(contentStart, close)));
      i = close + TOOL_CLOSE.length;
    }
  }

  return { thinking, text: text.replace(/\s+$/, (m) => (m.includes("\n") ? "\n" : "")), toolCalls, open };
}
