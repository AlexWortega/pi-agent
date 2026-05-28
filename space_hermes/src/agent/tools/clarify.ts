import type { Tool } from "../registry";

export const clarifyTool: Tool = {
  name: "clarify",
  description:
    "Ask the user a clarifying question before proceeding. Provide options for a multiple-choice prompt, or omit them for a free-text answer. Blocks until the user responds.",
  orchestratorOnly: true,
  parameters: {
    type: "object",
    properties: {
      question: { type: "string" },
      options: { type: "array", items: { type: "string" }, description: "optional multiple-choice options" },
    },
    required: ["question"],
  },
  async run(args, ctx) {
    const question = String(args?.question ?? "");
    const options = Array.isArray(args?.options) ? args.options.map(String) : undefined;
    const answer = await ctx.requestClarify(question, options);
    return { answer };
  },
};
