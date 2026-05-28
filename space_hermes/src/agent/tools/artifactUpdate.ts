import type { Tool } from "../registry";

export const artifactUpdateTool: Tool = {
  name: "artifact_update",
  description:
    "Replace an existing artifact's HTML in place and re-focus it in the live canvas. Use for iterative edits — call artifact_view first to read the current source, then submit the FULL new HTML.",
  parameters: {
    type: "object",
    properties: {
      artifactId: { type: "string" },
      html: { type: "string", description: "complete HTML document — replaces the previous one" },
      title: { type: "string", description: "optional new title" },
    },
    required: ["artifactId", "html"],
  },
  async run(args, ctx) {
    const html = String(args?.html ?? "").trim();
    if (!/<\w+/.test(html)) throw new Error("html doesn't look like markup");
    const id = String(args?.artifactId ?? "");
    const ok = ctx.updateArtifact(id, html, args?.title ? String(args.title) : undefined);
    if (!ok) throw new Error(`no artifact with id ${id}`);
    return { updated: true, artifactId: id, bytes: html.length };
  },
};
