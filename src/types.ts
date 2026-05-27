export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  /** visible answer text (think block already stripped out) */
  content: string;
  /** chain-of-thought captured from <think>…</think>, if any */
  think?: string;
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
  maxTokens: number;
  contextLength: number;
}
