import type { ToolCall, ToolResult } from "../types";
import { splitThink, stripToolCalls } from "../lib/parse";

/** One ChatML turn fed to the raw-completion engine. */
export interface WireMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

/** Build a ChatML prompt string (Qwen/Hermes style) ending ready for the
 *  assistant to continue. We hand-build this instead of using the GGUF chat
 *  template so we can inject a `tool` role and the <tools> block. */
export function buildPrompt(messages: WireMessage[]): string {
  let out = "";
  for (const m of messages) {
    out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  }
  out += "<|im_start|>assistant\n";
  return out;
}

/** Tolerant JSON parse: trims, strips trailing commas, and falls back to the
 *  first balanced {...} block. Returns null if nothing parses. */
function looseParse(raw: string): any | null {
  const tryIt = (s: string) => {
    try {
      return JSON.parse(s);
    } catch {
      return undefined;
    }
  };
  let t = raw.trim();
  let v = tryIt(t);
  if (v !== undefined) return v;

  // strip trailing commas
  v = tryIt(t.replace(/,\s*([}\]])/g, "$1"));
  if (v !== undefined) return v;

  // grab the first {...} span
  const start = t.indexOf("{");
  const end = t.lastIndexOf("}");
  if (start >= 0 && end > start) {
    v = tryIt(t.slice(start, end + 1).replace(/,\s*([}\]])/g, "$1"));
    if (v !== undefined) return v;
  }
  return null;
}

let callSeq = 0;

/** Parse every <tool_call>…</tool_call> (tolerating a missing closing tag at
 *  the very end of a stream). Returns [] if the model called nothing. */
export function parseToolCalls(text: string): ToolCall[] {
  const calls: ToolCall[] = [];
  const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const obj = looseParse(m[1]);
    if (obj && typeof obj.name === "string") {
      calls.push({
        id: `tc-${Date.now().toString(36)}-${callSeq++}`,
        name: obj.name,
        arguments: obj.arguments && typeof obj.arguments === "object" ? obj.arguments : {},
      });
    }
  }
  if (calls.length === 0) {
    const open = text.match(/<tool_call>\s*([\s\S]*)$/i);
    if (open) {
      const obj = looseParse(open[1]);
      if (obj && typeof obj.name === "string") {
        calls.push({
          id: `tc-${Date.now().toString(36)}-${callSeq++}`,
          name: obj.name,
          arguments: obj.arguments && typeof obj.arguments === "object" ? obj.arguments : {},
        });
      }
    }
  }
  return calls;
}

/** Strip reasoning + tool-call tags → the plain prose the user should read. */
export function visibleText(text: string): { visible: string; think: string } {
  const { think, answer } = splitThink(text);
  return { visible: stripToolCalls(answer), think };
}

/** Reconstruct an assistant turn for replay into a later prompt: visible prose
 *  plus its tool calls re-serialised as <tool_call> blocks. Reasoning is
 *  dropped to save context (Hermes does the same). */
export function assistantWire(visible: string, calls: ToolCall[]): string {
  const parts: string[] = [];
  if (visible.trim()) parts.push(visible.trim());
  for (const c of calls) {
    parts.push(`<tool_call>\n${JSON.stringify({ name: c.name, arguments: c.arguments })}\n</tool_call>`);
  }
  return parts.join("\n");
}

/** Serialise a tool result as a Hermes <tool_response> turn. */
export function toolResponseWire(r: ToolResult): string {
  const content = r.ok ? r.content : { error: String(r.content) };
  return `<tool_response>\n${JSON.stringify({ name: r.name, content })}\n</tool_response>`;
}
