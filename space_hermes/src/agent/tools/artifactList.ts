import type { Tool } from "../registry";

export const artifactListTool: Tool = {
  name: "artifact_list",
  description: "List every HTML artifact you have access to (id + title + session, newest first). Use before artifact_view / artifact_update.",
  parameters: { type: "object", properties: {} },
  async run(_args, ctx) {
    return {
      artifacts: ctx.listArtifacts().map((a) => ({
        id: a.artifactId,
        title: a.title,
        session: a.projectName,
        ts: a.ts,
      })),
    };
  },
};
