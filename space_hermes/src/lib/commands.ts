export type PanelId = "chat" | "skills" | "memory" | "tasks" | "tools" | "schedules";

export interface Command {
  name: string; // includes leading slash
  description: string;
  /** does it take free-text args after the command? */
  arg?: string;
}

/** Unified slash-command registry — drives the autocomplete menu and /help,
 *  mirroring Hermes' shared COMMAND_REGISTRY across surfaces. */
export const COMMAND_REGISTRY: Command[] = [
  { name: "/help", description: "List commands and available tools" },
  { name: "/tools", description: "Open the tools inspector" },
  { name: "/skills", description: "Manage skills (agentskills.io style)" },
  { name: "/memory", description: "View what the agent remembers" },
  { name: "/tasks", description: "Open the task board (todos)" },
  { name: "/search", description: "Search your past sessions", arg: "query" },
  { name: "/schedule", description: "Re-run a prompt on an interval", arg: "every 5m <prompt>" },
  { name: "/model", description: "Choose / reload the model" },
  { name: "/new", description: "Start a new session" },
  { name: "/clear", description: "Clear this session's messages" },
];

export function matchCommands(input: string): Command[] {
  if (!input.startsWith("/")) return [];
  const head = input.split(/\s/)[0].toLowerCase();
  return COMMAND_REGISTRY.filter((c) => c.name.startsWith(head));
}

/** Split "/schedule every 5m do X" into { name:"/schedule", rest:"every 5m do X" }. */
export function parseCommand(input: string): { name: string; rest: string } | null {
  if (!input.startsWith("/")) return null;
  const sp = input.indexOf(" ");
  if (sp < 0) return { name: input.toLowerCase(), rest: "" };
  return { name: input.slice(0, sp).toLowerCase(), rest: input.slice(sp + 1).trim() };
}
