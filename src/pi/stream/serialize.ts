/**
 * Serialize a pi-ai `Context` (systemPrompt + transcript + tool specs) into the
 * flat {role, content} message array the wllama chat-completion wrapper expects.
 *
 * Tool calling is done text-side (not via the GGUF template's native tools
 * kwarg): the tool catalogue is appended to the system prompt and the model is
 * asked to emit Qwen/Hermes-style <tool_call> blocks, which parse.ts recovers.
 * This keeps us independent of whatever tool-template support the wllama build
 * happens to ship, and matches how Qwen3.5/Soyuz was trained to call tools.
 */
import type { Context, Tool } from "@earendil-works/pi-ai";
import type { Role } from "../../types";

export interface FlatMessage {
  role: Role;
  content: string;
}

function joinTextParts(parts: Array<{ type: string; text?: string; mimeType?: string }>): string {
  return parts
    .map((p) => {
      if (p.type === "text") return p.text ?? "";
      if (p.type === "image") return `[image omitted: ${p.mimeType ?? "image"} — this model is text-only]`;
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

interface JsonSchemaLike {
  type?: string;
  properties?: Record<string, { type?: string; description?: string }>;
  required?: string[];
}

/** Compact, human-readable signature like `write(path: string, content: string)`. */
function toolSignature(tool: Tool): string {
  const schema = tool.parameters as unknown as JsonSchemaLike;
  const props = schema.properties ?? {};
  const required = new Set(schema.required ?? []);
  const params = Object.entries(props)
    .map(([name, def]) => `${name}: ${def.type ?? "any"}${required.has(name) ? "" : "?"}`)
    .join(", ");
  return `${tool.name}(${params})`;
}

/**
 * Concrete, fully-filled call examples per tool. Small models pattern-match on
 * examples, so these MUST show real argument values (the previous abstract
 * "{ ... }" skeleton made Soyuz emit empty arguments {}).
 */
const TOOL_EXAMPLES: Record<string, string> = {
  write:
    '<tool_call>{"name": "write", "arguments": {"path": "/workspace/index.html", "content": "<!doctype html>\\n<html><head><style>body{margin:0}</style></head><body><h1>Hi</h1><script>console.log(1)</script></body></html>"}}</tool_call>',
  edit:
    '<tool_call>{"name": "edit", "arguments": {"path": "/workspace/index.html", "edits": [{"oldText": "<h1>Hello</h1>", "newText": "<h1>Hello, world</h1>"}]}}</tool_call>',
  read: '<tool_call>{"name": "read", "arguments": {"path": "/workspace/index.html"}}</tool_call>',
  ls: '<tool_call>{"name": "ls", "arguments": {"path": "/workspace"}}</tool_call>',
};

function serializeToolSpecs(tools: Tool[] | undefined): string {
  if (!tools || tools.length === 0) return "";

  const lines: string[] = [
    "",
    "# Tool call format",
    "",
    "To call a tool, output a block with these EXACT tags containing ONE JSON object with `name` and a filled `arguments`:",
    "",
    '<tool_call>{"name": "edit", "arguments": {"path": "/workspace/index.html", "edits": [{"oldText": "old", "newText": "new"}]}}</tool_call>',
    "",
    "Rules: include every required argument (never empty {}); write the whole file content in one write call; after a tool call, STOP — the result is sent back to you, then you continue. Never write the result yourself.",
    "",
    "## Tools",
  ];

  for (const tool of tools) {
    lines.push(`- ${toolSignature(tool)} — ${tool.description}`);
    const example = TOOL_EXAMPLES[tool.name];
    if (example) lines.push(`  example: ${example}`);
  }

  return lines.join("\n");
}

/** Render one pi Message into the model-visible transcript text for its role. */
function serializeAssistant(content: Array<{ type: string; text?: string; name?: string; arguments?: unknown }>): string {
  const parts: string[] = [];
  for (const block of content) {
    if (block.type === "text" && block.text) {
      parts.push(block.text);
    } else if (block.type === "toolCall") {
      // Replay prior tool calls so the model sees its own action history.
      parts.push(`<tool_call>\n${JSON.stringify({ name: block.name, arguments: block.arguments ?? {} })}\n</tool_call>`);
    }
    // Thinking blocks are intentionally not replayed (CoT is ephemeral).
  }
  return parts.join("\n");
}

export function buildSystemPrompt(context: Context): string {
  const base = context.systemPrompt ?? "";
  const tools = serializeToolSpecs(context.tools);
  return tools ? `${base}\n${tools}` : base;
}

export function buildChatMessages(context: Context): FlatMessage[] {
  const out: FlatMessage[] = [];
  const system = buildSystemPrompt(context);
  if (system.trim()) out.push({ role: "system", content: system });

  for (const message of context.messages) {
    if (message.role === "user") {
      const content =
        typeof message.content === "string" ? message.content : joinTextParts(message.content as any);
      out.push({ role: "user", content });
    } else if (message.role === "assistant") {
      const content = serializeAssistant(message.content as any);
      if (content.trim()) out.push({ role: "assistant", content });
    } else if (message.role === "toolResult") {
      const body = joinTextParts(message.content as any);
      // Plain, tag-free format — feeding back a literal <tool_response> tag
      // trains the model to hallucinate one. Fed as a user turn.
      const wrapped = `Result of ${message.toolName}${message.isError ? " (error)" : ""}:\n${body}`;
      out.push({ role: "user", content: wrapped });
    }
  }

  return out;
}
