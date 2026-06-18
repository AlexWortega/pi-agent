/**
 * Agent runtime: wires the real pi agent loop (pi-agent-core) to our in-browser
 * pieces — the WebGPU stream function, the OPFS-backed tools, and a system
 * prompt — and exposes it as an AgentEvent stream the React layer subscribes to.
 *
 * The pi loop is used unmodified: we only inject `streamFn` (local model) and
 * the tool set, and provide an identity `convertToLlm` because our transcript
 * is already plain pi-ai Messages (no custom AgentMessage types).
 */
import { agentLoop } from "@earendil-works/pi-agent-core";
import type {
  AgentContext,
  AgentEvent,
  AgentLoopConfig,
  AgentMessage,
} from "@earendil-works/pi-agent-core";
import type { EventStream } from "@earendil-works/pi-ai";
import type { Message, Model } from "@earendil-works/pi-ai";
import { getFsBackend } from "./fs/backend";
import { buildLocalModel } from "./model";
import { localStream } from "./stream/localStream";
import { buildTools } from "./tools";
import type { GenParams } from "../types";

export const WORKSPACE_ROOT = "/workspace";

/** Safety cap so a confused small model can't loop forever on tool calls. */
const MAX_TURNS = 6;

export const AGENT_SYSTEM_PROMPT = `You are Soyuz, the Pi Agent — a fast in-browser coding agent. You build a self-contained web app at ${WORKSPACE_ROOT}/index.html, shown live in the preview pane.

First PLAN inside <think> … </think>: the key features, layout, and main logic (game loop, controls, scoring, sound, …). Keep it a short bullet plan. Then act immediately.

To create or rewrite the app, call the \`write\` tool ONCE:
- path: "${WORKSPACE_ROOT}/index.html"
- content: the COMPLETE file — a full <!doctype html> document with inline CSS and JavaScript, no external files.
Write the whole file in that single call and finish it.

For a small change to an existing file, use \`edit\`. Do NOT explore with shell commands — there is no shell, and you do not need to inspect anything before the first write.`;

export interface AgentRun {
  stream: EventStream<AgentEvent, AgentMessage[]>;
  abort: () => void;
}

export interface StartRunOptions {
  prompt: string;
  /** Prior transcript (accumulated pi Messages from earlier turns this session). */
  history: Message[];
  model: Model<any>;
  systemPrompt?: string;
  /** Sampling temperature (default 0.1). */
  temperature?: number;
}

export function startAgentRun(opts: StartRunOptions): AgentRun {
  const fs = getFsBackend();
  const tools = buildTools(fs, WORKSPACE_ROOT);

  const context: AgentContext = {
    systemPrompt: opts.systemPrompt ?? AGENT_SYSTEM_PROMPT,
    messages: [...opts.history],
    // Tools are always available — Soyuz reliably calls `write` with the full
    // file at temp 0.1; we let it, and the HTML bridge covers the rare case it
    // emits a raw ```html block instead.
    tools,
  };

  let turns = 0;
  const seenToolCalls = new Set<string>();
  const config: AgentLoopConfig = {
    model: opts.model,
    convertToLlm: (messages) => messages as Message[],
    toolExecution: "sequential",
    temperature: opts.temperature ?? 0.1,
    shouldStopAfterTurn: (ctx) => {
      turns += 1;
      if (turns >= MAX_TURNS) return true;

      // The app is built once index.html has been written successfully — stop
      // instead of letting the model write it again next turn.
      const wroteIndex = ctx.toolResults?.some(
        (r) => !r.isError && r.toolName === "write" && /index\.html/.test(JSON.stringify(r.content)),
      );
      if (wroteIndex) {
        console.debug(`[pi] stop: index.html written (turn ${turns})`);
        return true;
      }

      const blocks = ctx.message.content;
      const text = blocks
        .filter((b): b is Extract<typeof b, { type: "text" }> => b.type === "text")
        .map((b) => b.text)
        .join("\n");

      // The build is done once a complete HTML document has been emitted — stop
      // instead of letting the model keep churning turns.
      if (/```html[\s\S]*?```/i.test(text) || /<\/html\s*>/i.test(text)) {
        console.debug(`[pi] stop: HTML document emitted (turn ${turns})`);
        return true;
      }

      // No-progress guard: if every tool call this turn repeats one we've
      // already run (same name+args), the agent is looping — stop.
      const calls = blocks.filter((b): b is Extract<typeof b, { type: "toolCall" }> => b.type === "toolCall");
      if (calls.length > 0) {
        let allSeen = true;
        for (const c of calls) {
          const sig = `${c.name}:${JSON.stringify(c.arguments)}`;
          if (!seenToolCalls.has(sig)) {
            allSeen = false;
            seenToolCalls.add(sig);
          }
        }
        if (allSeen) {
          console.debug(`[pi] stop: repeated tool call(s), no progress (turn ${turns})`);
          return true;
        }
      }
      return false;
    },
  };

  const controller = new AbortController();
  const prompt: Message = { role: "user", content: opts.prompt, timestamp: 0 };
  const stream = agentLoop([prompt], context, config, controller.signal, localStream);

  return { stream, abort: () => controller.abort() };
}

/** Build the local model descriptor for the agent loop. */
export function makeAgentModel(descriptor: { id: string; label: string }, params: GenParams): Model<any> {
  return buildLocalModel(descriptor, params);
}
