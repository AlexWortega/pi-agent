export type Role = "system" | "user" | "assistant" | "tool";

/** A function call the model emitted inside <tool_call>…</tool_call>. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

/** The result of running a ToolCall, fed back as a <tool_response>. */
export interface ToolResult {
  id: string; // matches the ToolCall.id
  name: string;
  ok: boolean;
  content: any; // JSON-serialisable
  ms?: number;
}

export interface ChatMessage {
  id: string;
  role: Role;
  /** visible answer text (think + tool_call tags already stripped out) */
  content: string;
  /** chain-of-thought captured from <think>/<scratch_pad>, if any */
  think?: string;
  /** tool calls this assistant turn emitted */
  toolCalls?: ToolCall[];
  /** results for the tool calls above (filled in as they complete) */
  toolResults?: ToolResult[];
  /** which agent iteration produced this message */
  iteration?: number;
  ts: number;
  /** still streaming */
  pending?: boolean;
}

export interface Artifact {
  id: string;
  title: string;
  html: string;
  ts: number;
}

export interface Project {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  modelId: string;
  messages: ChatMessage[];
  artifacts: Artifact[];
}

export interface ModelPreset {
  id: string;
  label: string;
  repo: string;
  file: string;
  sizeLabel: string;
  note: string;
  /** confirmed to run on the WebGPU backend */
  verified: boolean;
  accent?: string;
}

export interface GenParams {
  temperature: number;
  /** max tokens per single model call (one agent iteration) */
  maxTokens: number;
  contextLength: number;
  /** how many tool-calling rounds the agent may take per user message */
  maxIterations: number;
}

// ---- agent-side records (localStorage) ------------------------------------

export interface MemoryItem {
  id: string;
  key?: string;
  text: string;
  ts: number;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  /** markdown instructions, agentskills.io style */
  body: string;
  createdBy: "user" | "agent";
  ts: number;
}

export interface TodoItem {
  id: string;
  text: string;
  done: boolean;
  ts: number;
}

export interface Schedule {
  id: string;
  prompt: string;
  everyMs: number;
  nextRun: number;
  projectId: string;
  lastRun?: number;
}
