import type { Tool } from "../registry";

export const delegateTaskTool: Tool = {
  name: "delegate_task",
  description:
    "Spawn an isolated sub-agent to handle a self-contained subtask. It has its own fresh context and the same tools (minus delegation), runs to completion, and returns only a summary. Use for parallelisable or context-heavy side quests.",
  orchestratorOnly: true,
  parameters: {
    type: "object",
    properties: { task: { type: "string", description: "a clear, self-contained instruction for the sub-agent" } },
    required: ["task"],
  },
  async run(args, ctx) {
    const task = String(args?.task ?? "").trim();
    if (!task) throw new Error("task is empty");
    const summary = await ctx.runSubagent(task, ctx);
    return { summary };
  },
};
