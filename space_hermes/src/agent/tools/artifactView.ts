import type { Tool } from "../registry";

export const artifactViewTool: Tool = {
  name: "artifact_view",
  description: "Load the full HTML of one artifact by id so you can read or quote it before editing.",
  parameters: {
    type: "object",
    properties: { artifactId: { type: "string" } },
    required: ["artifactId"],
  },
  async run(args, ctx) {
    const id = String(args?.artifactId ?? "");
    const a = ctx.getArtifact(id);
    if (!a) throw new Error(`no artifact with id ${id}`);
    return { id, title: a.title, html: a.html, bytes: a.html.length };
  },
};
