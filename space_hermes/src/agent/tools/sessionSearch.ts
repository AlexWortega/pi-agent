import type { Tool } from "../registry";

export const sessionSearchTool: Tool = {
  name: "session_search",
  description: "Search your own past conversations (all saved sessions) for relevant earlier messages.",
  parameters: {
    type: "object",
    properties: { query: { type: "string" }, limit: { type: "number", description: "max hits, default 6" } },
    required: ["query"],
  },
  async run(args, ctx) {
    const hits = ctx.searchSessions(String(args?.query ?? ""), Number(args?.limit) || 6);
    return {
      hits: hits.map((h) => ({ session: h.projectName, role: h.role, snippet: h.snippet })),
    };
  },
};
