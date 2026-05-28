import type { Tool } from "../registry";
import { saveMemory, searchMemory, listMemory, deleteMemory } from "../../lib/memoryStore";

export const memoryTool: Tool = {
  name: "memory",
  description:
    "Persist or recall facts about the user across sessions. action='save' stores text (optional key); action='recall' searches; action='list' returns all; action='forget' deletes by id.",
  parameters: {
    type: "object",
    properties: {
      action: { type: "string", enum: ["save", "recall", "list", "forget"] },
      text: { type: "string", description: "fact to save (action=save)" },
      key: { type: "string", description: "optional label to overwrite (action=save)" },
      query: { type: "string", description: "search query (action=recall)" },
      id: { type: "string", description: "memory id to delete (action=forget)" },
    },
    required: ["action"],
  },
  async run(args) {
    switch (args?.action) {
      case "save": {
        if (!args.text) throw new Error("text is required to save a memory");
        const item = saveMemory(String(args.text), args.key ? String(args.key) : undefined);
        return { saved: item.id, text: item.text };
      }
      case "recall":
        return { results: searchMemory(String(args.query ?? "")).map((m) => ({ id: m.id, key: m.key, text: m.text })) };
      case "list":
        return { memories: listMemory().map((m) => ({ id: m.id, key: m.key, text: m.text })) };
      case "forget":
        if (!args.id) throw new Error("id is required to forget a memory");
        deleteMemory(String(args.id));
        return { forgot: args.id };
      default:
        throw new Error("unknown action; use save|recall|list|forget");
    }
  },
};
