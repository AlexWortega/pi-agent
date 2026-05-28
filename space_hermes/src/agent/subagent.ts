import { runAgent } from "./loop";
import { enabledTools, toolsSystemBlock } from "./registry";
import type { ToolContext } from "./registry";
import { buildSystemPrompt, DEFAULT_PARAMS } from "../config";

/** delegate_task implementation: run an isolated sub-agent (fresh context,
 *  no delegation/clarify tools, capped depth) and return only its summary. */
export async function runSubagent(task: string, parentCtx: ToolContext): Promise<string> {
  if (parentCtx.depth >= 2) return "Delegation depth limit reached — handle this directly.";

  const subTools = enabledTools(parentCtx.settings, true);
  const ctx: ToolContext = { ...parentCtx, depth: parentCtx.depth + 1 };
  const system =
    buildSystemPrompt(toolsSystemBlock(subTools)) +
    "\n\nYou are a focused sub-agent with an isolated context. Complete the task end-to-end, then finish with a concise summary of the result for your parent agent.";

  const summary = await runAgent({
    wire: [
      { role: "system", content: system },
      { role: "user", content: task },
    ],
    tools: subTools,
    ctx,
    params: { temperature: DEFAULT_PARAMS.temperature, maxTokens: DEFAULT_PARAMS.maxTokens, maxIterations: 4 },
    signal: parentCtx.signal,
  });
  return summary || "(sub-agent produced no output)";
}
