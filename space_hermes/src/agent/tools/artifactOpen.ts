import type { Tool } from "../registry";

export const artifactOpenTool: Tool = {
  name: "artifact_open",
  description: "Bring an artifact to focus in the live canvas on the right (UI side-effect — switches the active session if it lives in another).",
  parameters: {
    type: "object",
    properties: { artifactId: { type: "string" } },
    required: ["artifactId"],
  },
  async run(args, ctx) {
    const id = String(args?.artifactId ?? "");
    const ok = ctx.focusArtifact(id);
    if (!ok) throw new Error(`no artifact with id ${id}`);
    return { focused: id };
  },
};
