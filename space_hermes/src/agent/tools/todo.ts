import type { Tool } from "../registry";
import { addTodo, listTodos, toggleTodo, removeTodo, clearDone } from "../../lib/todoStore";

export const todoTool: Tool = {
  name: "todo",
  description:
    "Manage the task board. action='add' creates a task; 'list' returns tasks; 'done' marks one complete by id; 'remove' deletes by id; 'clear_done' removes completed tasks.",
  parameters: {
    type: "object",
    properties: {
      action: { type: "string", enum: ["add", "list", "done", "remove", "clear_done"] },
      text: { type: "string", description: "task text (action=add)" },
      id: { type: "string", description: "task id (action=done|remove)" },
    },
    required: ["action"],
  },
  async run(args) {
    switch (args?.action) {
      case "add": {
        if (!args.text) throw new Error("text is required");
        const t = addTodo(String(args.text));
        return { added: t.id, text: t.text };
      }
      case "list":
        return { tasks: listTodos().map((t) => ({ id: t.id, text: t.text, done: t.done })) };
      case "done":
        if (!args.id) throw new Error("id is required");
        toggleTodo(String(args.id), true);
        return { done: args.id };
      case "remove":
        if (!args.id) throw new Error("id is required");
        removeTodo(String(args.id));
        return { removed: args.id };
      case "clear_done":
        clearDone();
        return { cleared: true };
      default:
        throw new Error("unknown action");
    }
  },
};
