import type { Tool } from "../registry";

export const renderHtmlTool: Tool = {
  name: "render_html",
  description:
    "Render a complete, self-contained HTML page in the live canvas on the right (inline CSS + JS, no external files). Use for any app / page / game / visual UI the user asks to build.",
  parameters: {
    type: "object",
    properties: {
      title: { type: "string", description: "short title for the artifact" },
      html: { type: "string", description: "full HTML document" },
    },
    required: ["html"],
  },
  async run(args, ctx) {
    const html = String(args?.html ?? "").trim();
    if (!/<\w+/.test(html)) throw new Error("html argument doesn't look like markup");
    const title = String(args?.title ?? "").trim() || html.match(/<title>([^<]+)<\/title>/i)?.[1]?.trim() || "App";
    const id = ctx.emitArtifact(title, html);
    return { rendered: true, title, artifactId: id, bytes: html.length };
  },
};
