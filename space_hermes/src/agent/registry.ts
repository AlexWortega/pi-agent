import type { AgentSettings } from "../lib/settings";
import type { SessionHit } from "../lib/sessionIndex";
import type { ArtifactHit, ArtifactRef } from "../lib/artifactIndex";

export interface JSONSchema {
  type: "object";
  properties: Record<string, any>;
  required?: string[];
}

/** Everything a tool needs from the host that can't be a module singleton. */
export interface ToolContext {
  signal: AbortSignal;
  depth: number;
  settings: AgentSettings;
  /** ask the user a question; resolves with their answer */
  requestClarify(question: string, options?: string[]): Promise<string>;
  /** push an HTML artifact to the canvas; returns its id */
  emitArtifact(title: string, html: string): string;
  /** run an isolated sub-agent and get its summary back */
  runSubagent(task: string, ctx: ToolContext): Promise<string>;
  /** full-text search across saved sessions */
  searchSessions(query: string, limit?: number): SessionHit[];
  /** flat list of every artifact across every session, newest first */
  listArtifacts(): ArtifactRef[];
  /** full-text search across saved HTML artifacts */
  searchArtifacts(query: string, limit?: number): ArtifactHit[];
  /** load one artifact's HTML by id (null if not found) */
  getArtifact(id: string): { title: string; html: string } | null;
  /** replace an artifact in place and focus it; returns false if id is unknown */
  updateArtifact(id: string, html: string, title?: string): boolean;
  /** bring an artifact to focus in the canvas (may switch sessions) */
  focusArtifact(id: string): boolean;
}

export interface Tool {
  name: string;
  description: string;
  parameters: JSONSchema;
  /** withheld from sub-agents (only orchestrators can delegate / clarify) */
  orchestratorOnly?: boolean;
  /** hits the network — gated by settings.webToolsEnabled */
  web?: boolean;
  run(args: any, ctx: ToolContext): Promise<any>;
}

import { calculatorTool } from "./tools/calculator";
import { datetimeTool } from "./tools/datetime";
import { memoryTool } from "./tools/memory";
import { sessionSearchTool } from "./tools/sessionSearch";
import { todoTool } from "./tools/todo";
import { clarifyTool } from "./tools/clarify";
import { skillsListTool, skillViewTool, skillManageTool } from "./tools/skills";
import { renderHtmlTool } from "./tools/renderHtml";
import { executeCodeTool } from "./tools/executeCode";
import { webSearchTool } from "./tools/webSearch";
import { webExtractTool } from "./tools/webExtract";
import { delegateTaskTool } from "./tools/delegateTask";
import { artifactSearchTool } from "./tools/artifactSearch";
import { artifactListTool } from "./tools/artifactList";
import { artifactViewTool } from "./tools/artifactView";
import { artifactUpdateTool } from "./tools/artifactUpdate";
import { artifactOpenTool } from "./tools/artifactOpen";

/** Every tool the in-browser Hermes agent ships with. */
export const ALL_TOOLS: Tool[] = [
  calculatorTool,
  datetimeTool,
  memoryTool,
  sessionSearchTool,
  todoTool,
  clarifyTool,
  skillsListTool,
  skillViewTool,
  skillManageTool,
  renderHtmlTool,
  artifactSearchTool,
  artifactListTool,
  artifactViewTool,
  artifactUpdateTool,
  artifactOpenTool,
  executeCodeTool,
  webSearchTool,
  webExtractTool,
  delegateTaskTool,
];

/** Tools active for a run, honouring settings + sub-agent restrictions. */
export function enabledTools(settings: AgentSettings, forSubagent = false): Tool[] {
  return ALL_TOOLS.filter((t) => {
    if (settings.disabledTools.includes(t.name)) return false;
    if (t.web && !settings.webToolsEnabled) return false;
    if (forSubagent && t.orchestratorOnly) return false;
    return true;
  });
}

/** The <tools> JSON payload injected into the system prompt. */
export function toolsSystemBlock(tools: Tool[]): string {
  return JSON.stringify(
    tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
  );
}
