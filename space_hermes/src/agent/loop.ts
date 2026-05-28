import { engine } from "../engine/llama";
import { buildPrompt, parseToolCalls, visibleText, assistantWire, toolResponseWire } from "./chatml";
import type { WireMessage } from "./chatml";
import type { Tool, ToolContext } from "./registry";
import type { ToolCall, ToolResult } from "../types";

export interface AgentCallbacks {
  onStatus?: (s: { iteration: number; maxIterations: number; phase: "thinking" | "tools" | "done" }) => void;
  /** streaming partial of the current assistant turn */
  onStream?: (iteration: number, visible: string, think: string) => void;
  /** a turn finished generating (with any tool calls it emitted) */
  onAssistant?: (iteration: number, visible: string, think: string, calls: ToolCall[]) => void;
  onToolStart?: (call: ToolCall) => void;
  onToolResult?: (result: ToolResult) => void;
}

export interface RunAgentOpts {
  wire: WireMessage[];
  tools: Tool[];
  ctx: ToolContext;
  params: { temperature: number; maxTokens: number; maxIterations: number };
  signal: AbortSignal;
  cb?: AgentCallbacks;
}

/**
 * The synchronous Hermes tool-calling loop (browser port of run_agent.py):
 * call the model with the <tools> block, stop at </tool_call>, run the tools,
 * append <tool_response> turns, repeat — until a turn has no tool call or the
 * iteration budget is spent. Returns the final visible answer.
 */
export async function runAgent(opts: RunAgentOpts): Promise<string> {
  const { tools, ctx, params, signal, cb } = opts;
  const wire = [...opts.wire];
  const byName = new Map(tools.map((t) => [t.name, t]));
  let finalVisible = "";

  for (let iter = 0; iter < params.maxIterations; iter++) {
    if (signal.aborted) break;
    cb?.onStatus?.({ iteration: iter + 1, maxIterations: params.maxIterations, phase: "thinking" });

    const { text } = await engine.rawComplete(buildPrompt(wire), {
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      signal,
      stopStrings: ["</tool_call>"],
      onToken: (full) => {
        const v = visibleText(full);
        cb?.onStream?.(iter, v.visible, v.think);
      },
    });

    const { visible, think } = visibleText(text);
    const calls = parseToolCalls(text);
    finalVisible = visible;
    cb?.onAssistant?.(iter, visible, think, calls);
    wire.push({ role: "assistant", content: assistantWire(visible, calls) });

    if (calls.length === 0) {
      cb?.onStatus?.({ iteration: iter + 1, maxIterations: params.maxIterations, phase: "done" });
      return finalVisible;
    }

    cb?.onStatus?.({ iteration: iter + 1, maxIterations: params.maxIterations, phase: "tools" });
    for (const call of calls) {
      if (signal.aborted) break;
      cb?.onToolStart?.(call);
      const t0 = Date.now();
      const tool = byName.get(call.name);
      let result: ToolResult;
      if (!tool) {
        result = {
          id: call.id,
          name: call.name,
          ok: false,
          content: `Unknown tool '${call.name}'. Available: ${[...byName.keys()].join(", ")}`,
          ms: 0,
        };
      } else {
        try {
          const out = await tool.run(call.arguments, ctx);
          result = { id: call.id, name: call.name, ok: true, content: out, ms: Date.now() - t0 };
        } catch (e: any) {
          result = { id: call.id, name: call.name, ok: false, content: String(e?.message || e), ms: Date.now() - t0 };
        }
      }
      cb?.onToolResult?.(result);
      wire.push({ role: "tool", content: toolResponseWire(result) });
    }
  }

  // budget spent → one final no-tools answer pass
  if (!signal.aborted) {
    wire.push({
      role: "user",
      content:
        "You've reached the tool-call budget for this turn. Give your best final answer now using what you already have. Do NOT call any tools.",
    });
    const { text } = await engine.rawComplete(buildPrompt(wire), {
      temperature: params.temperature,
      maxTokens: params.maxTokens,
      signal,
      stopStrings: [],
      onToken: (full) => {
        const v = visibleText(full);
        cb?.onStream?.(params.maxIterations, v.visible, v.think);
      },
    });
    const v = visibleText(text);
    finalVisible = v.visible;
    cb?.onAssistant?.(params.maxIterations, v.visible, v.think, []);
  }
  return finalVisible;
}
