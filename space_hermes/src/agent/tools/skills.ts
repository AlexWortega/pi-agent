import type { Tool } from "../registry";
import { listSkills, getSkill, upsertSkill, deleteSkill } from "../../lib/skillsStore";

export const skillsListTool: Tool = {
  name: "skills_list",
  description: "List the names + descriptions of all available skills.",
  parameters: { type: "object", properties: {} },
  async run() {
    return { skills: listSkills().map((s) => ({ name: s.name, description: s.description, by: s.createdBy })) };
  },
};

export const skillViewTool: Tool = {
  name: "skill_view",
  description: "Load the full markdown body of one skill by name.",
  parameters: {
    type: "object",
    properties: { name: { type: "string" } },
    required: ["name"],
  },
  async run(args) {
    const s = getSkill(String(args?.name ?? ""));
    if (!s) throw new Error(`no skill named '${args?.name}'`);
    return { name: s.name, description: s.description, body: s.body };
  },
};

export const skillManageTool: Tool = {
  name: "skill_manage",
  description:
    "Create, update or delete a skill (reusable markdown instructions). action='save' upserts {name, description, body}; action='delete' removes by name.",
  parameters: {
    type: "object",
    properties: {
      action: { type: "string", enum: ["save", "delete"] },
      name: { type: "string" },
      description: { type: "string" },
      body: { type: "string", description: "markdown instructions (action=save)" },
    },
    required: ["action", "name"],
  },
  async run(args) {
    if (args?.action === "delete") {
      const ok = deleteSkill(String(args.name));
      return { deleted: ok, name: args.name };
    }
    if (args?.action === "save") {
      if (!args.body) throw new Error("body is required to save a skill");
      const s = upsertSkill({
        name: String(args.name),
        description: args.description ? String(args.description) : undefined,
        body: String(args.body),
        createdBy: "agent",
      });
      return { saved: s.name };
    }
    throw new Error("unknown action; use save|delete");
  },
};
