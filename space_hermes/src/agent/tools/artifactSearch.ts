import type { Tool } from "../registry";

export const artifactSearchTool: Tool = {
  name: "artifact_search",
  description:
    "Search the HTML artifacts you've built (across every session) by title + content. Use this to find an earlier app to reference, reuse or remix before editing it with artifact_update.",
  parameters: {
    type: "object",
    properties: { query: { type: "string" }, limit: { type: "number", description: "max hits, default 6" } },
    required: ["query"],
  },
  async run(args, ctx) {
    const hits = ctx.searchArtifacts(String(args?.query ?? ""), Number(args?.limit) || 6);
    return {
      hits: hits.map((h) => ({ session: h.projectName, id: h.artifactId, title: h.title, snippet: h.snippet })),
    };
  },
};
