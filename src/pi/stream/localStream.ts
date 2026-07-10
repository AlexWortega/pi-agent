/**
 * Local in-browser StreamFn for the pi agent loop.
 *
 * Drives the WebGPU llama.cpp engine and adapts its single cumulative-text
 * stream into pi-ai's AssistantMessageEvent protocol (start → text/thinking
 * deltas → toolcall blocks → done). The agent loop reads the final message's
 * `toolCall` content blocks to decide what to execute, so the load-bearing
 * output is the final AssistantMessage; the deltas exist for live UI.
 *
 * Contract (from pi-ai StreamFunction): never throw — encode request/runtime
 * failures as a final AssistantMessage with stopReason "error"/"aborted".
 */
import type {
  AssistantMessage,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  TextContent,
  ThinkingContent,
  ToolCall,
  Usage,
} from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { engine } from "../../engine/llama";
import { LOCAL_API, LOCAL_PROVIDER, REASONING_OPTS, type ReasoningOpts } from "../model";
import { setStats } from "../stats";
import { parseOutput } from "./parse";
import { buildChatMessages } from "./serialize";

type AssistantBlock = TextContent | ThinkingContent | ToolCall;

const ZERO_USAGE: Usage = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

let toolCallSeq = 0;
function nextToolCallId(): string {
  toolCallSeq += 1;
  return `local-tool-${toolCallSeq}`;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Tracks which structured blocks have been opened/streamed so we can translate
 * the cumulative parse into incremental pi-ai events without double-emitting.
 */
class EventEmitter {
  private content: AssistantBlock[] = [];
  private thinkingIndex = -1;
  private textIndex = -1;
  private emittedThinking = 0;
  private emittedText = 0;

  constructor(
    private stream: AssistantMessageEventStream,
    private model: Model<any>,
  ) {}

  private partial(): AssistantMessage {
    return this.buildMessage("stop");
  }

  private buildMessage(stopReason: AssistantMessage["stopReason"], errorMessage?: string): AssistantMessage {
    return {
      role: "assistant",
      content: this.content.map((b) => ({ ...b })),
      api: LOCAL_API,
      provider: LOCAL_PROVIDER,
      model: this.model.id,
      usage: ZERO_USAGE,
      stopReason,
      errorMessage,
      timestamp: 0,
    };
  }

  start(): void {
    this.stream.push({ type: "start", partial: this.partial() });
  }

  /** Reconcile against a fresh cumulative parse, emitting any new deltas. */
  syncStreaming(thinking: string, text: string): void {
    if (thinking.length > this.emittedThinking) {
      if (this.thinkingIndex === -1) {
        this.thinkingIndex = this.content.length;
        this.content.push({ type: "thinking", thinking: "" });
        this.stream.push({ type: "thinking_start", contentIndex: this.thinkingIndex, partial: this.partial() });
      }
      const delta = thinking.slice(this.emittedThinking);
      (this.content[this.thinkingIndex] as ThinkingContent).thinking = thinking;
      this.emittedThinking = thinking.length;
      this.stream.push({
        type: "thinking_delta",
        contentIndex: this.thinkingIndex,
        delta,
        partial: this.partial(),
      });
    }
    if (text.length > this.emittedText) {
      if (this.textIndex === -1) {
        this.textIndex = this.content.length;
        this.content.push({ type: "text", text: "" });
        this.stream.push({ type: "text_start", contentIndex: this.textIndex, partial: this.partial() });
      }
      const delta = text.slice(this.emittedText);
      (this.content[this.textIndex] as TextContent).text = text;
      this.emittedText = text.length;
      this.stream.push({ type: "text_delta", contentIndex: this.textIndex, delta, partial: this.partial() });
    }
  }

  /** Close streamed blocks, append tool calls, and push the terminal event. */
  finish(
    thinking: string,
    text: string,
    toolCalls: ToolCall[],
    outputText: string,
    promptTokens: number,
  ): AssistantMessage {
    this.syncStreaming(thinking, text);

    if (this.thinkingIndex !== -1) {
      this.stream.push({
        type: "thinking_end",
        contentIndex: this.thinkingIndex,
        content: (this.content[this.thinkingIndex] as ThinkingContent).thinking,
        partial: this.partial(),
      });
    }
    if (this.textIndex !== -1) {
      this.stream.push({
        type: "text_end",
        contentIndex: this.textIndex,
        content: (this.content[this.textIndex] as TextContent).text,
        partial: this.partial(),
      });
    }

    for (const toolCall of toolCalls) {
      const contentIndex = this.content.length;
      this.content.push(toolCall);
      this.stream.push({ type: "toolcall_start", contentIndex, partial: this.partial() });
      this.stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: this.partial() });
    }

    const stopReason = toolCalls.length > 0 ? "toolUse" : "stop";
    const message = this.buildMessage(stopReason);
    const output = estimateTokens(outputText);
    message.usage = {
      ...ZERO_USAGE,
      input: promptTokens,
      output,
      totalTokens: promptTokens + output,
    };
    this.stream.push({ type: "done", reason: stopReason, message });
    this.stream.end(message);
    return message;
  }

  error(reason: "error" | "aborted", errorMessage: string): AssistantMessage {
    const message = this.buildMessage(reason, errorMessage);
    this.stream.push({ type: "error", reason, error: message });
    this.stream.end(message);
    return message;
  }
}

/**
 * The pi-ai StreamFunction backed by the local WebGPU model.
 * Signature matches `streamSimple`, so it can be passed directly as the agent
 * loop's `streamFn`.
 */
export function localStream(
  model: Model<any>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const emitter = new EventEmitter(stream, model);

  // Run asynchronously; the contract requires errors to surface through the
  // stream, never as a thrown/rejected value from this function.
  queueMicrotask(async () => {
    if (options?.signal?.aborted) {
      emitter.error("aborted", "Request was aborted");
      return;
    }
    emitter.start();

    const messages = buildChatMessages(context);
    const promptTokens = estimateTokens(messages.map((m) => m.content).join("\n"));
    let lastFull = "";
    let sawFirstToken = false;
    let lastStat = 0;
    const t0 = performance.now();
    // Remote (cloud) runs go through a non-streaming proxy: show "connecting"
    // until the worker reports status, so it never looks like a frozen "thinking".
    setStats({ generating: true, liveTokens: 0, phase: engine.serverMode ? "connecting…" : "thinking" });

    try {
      console.debug(`[pi] chat start — ${messages.length} messages, ~${promptTokens} prompt tokens`);
      // Full context dump (dev builds only — serializing the whole transcript
      // to the console every turn is not free in production).
      if (import.meta.env.DEV) {
        // eslint-disable-next-line no-console
        console.groupCollapsed(`[pi] context → ${messages.length} msgs, ~${promptTokens} tokens (click to expand)`);
        for (const m of messages) {
          // eslint-disable-next-line no-console
          console.log(`%c──[${m.role}]──`, "color:#7c5cff;font-weight:bold");
          // eslint-disable-next-line no-console
          console.log(m.content);
        }
        // eslint-disable-next-line no-console
        console.groupEnd();
        (globalThis as any).__piContext = messages;
      }
      const reasoning = (model as unknown as Record<string, unknown>)[REASONING_OPTS] as ReasoningOpts | undefined;
      const finalText = await engine.chat(
        messages.map((m) => ({ id: "", role: m.role, content: m.content, ts: 0 })),
        {
          temperature: options?.temperature ?? 0.1,
          // Per-turn output cap (from config). Kept modest on purpose — a huge
          // cap lets the model ramble for tens of thousands of tokens.
          maxTokens: model.maxTokens || 4096,
          // Remote-only (SIQ-1) reasoning controls; ignored by the in-browser path.
          thinking: reasoning?.thinking,
          effort: reasoning?.effort,
          // Native OpenAI tool_calls on direct-API endpoints (OpenRouter). The
          // engine only forwards these when remote.apiKey is set; the text
          // <tool_call> protocol stays the source of truth everywhere else.
          tools: (context.tools ?? []).map((t) => ({
            name: t.name,
            description: t.description,
            parameters: t.parameters as unknown as Record<string, unknown>,
          })),
          // Remote-only: cloud worker lifecycle → a live, honest status (no frozen spinner).
          onStatus: (status) => {
            const phase =
              status === "IN_QUEUE"
                ? "starting cloud GPU…"
                : status === "IN_PROGRESS"
                  ? "generating on cloud GPU…"
                  : status.toLowerCase();
            setStats({ generating: true, phase, liveTokens: 0 });
          },
          signal: options?.signal,
          onToken: (full) => {
            if (!sawFirstToken) {
              sawFirstToken = true;
              console.debug(`[pi] first token after ${((performance.now() - t0) / 1000).toFixed(1)}s`);
            }
            lastFull = full;
            // The parser already withholds an in-progress <tool_call> block from
            // `text`, so a half-written JSON object never flashes as prose.
            const parsed = parseOutput(full);
            emitter.syncStreaming(parsed.thinking, parsed.text);
            // Live readout (throttled) so the UI visibly ticks while generating.
            const now = performance.now();
            if (now - lastStat > 120) {
              lastStat = now;
              const toks = estimateTokens(full);
              const secs = (now - t0) / 1000;
              const phase =
                parsed.open === "tool_call" || /<tool_call/.test(full)
                  ? "calling tool"
                  : parsed.open === "think" || (parsed.thinking && !parsed.text)
                    ? "thinking"
                    : /```html/i.test(full) || /<html/i.test(full)
                      ? "writing index.html"
                      : "writing";
              setStats({
                generating: true,
                liveTokens: toks,
                tps: secs > 0.3 ? Math.round(toks / secs) : null,
                phase,
              });
            }
          },
        },
      );

      const full = finalText || lastFull;
      try {
        (globalThis as any).__lastRaw = full;
      } catch {
        /* debug only */
      }
      const outputTokens = estimateTokens(full);
      const seconds = (performance.now() - t0) / 1000;
      // Debug aid while tuning Soyuz: shows whether a turn was cut off (length
      // near the token budget) vs. an early stop.
      console.debug(`[pi] generated ${full.length} chars (~${outputTokens} tokens) in ${seconds.toFixed(1)}s`);
      setStats({
        tps: seconds > 0.1 ? Math.round(outputTokens / seconds) : null,
        contextUsed: promptTokens + outputTokens,
        contextWindow: model.contextWindow,
        generating: false,
        liveTokens: outputTokens,
        phase: null,
      });
      const parsed = parseOutput(full);
      const toolCalls: ToolCall[] = parsed.toolCalls
        .filter((tc) => !tc.malformed)
        .map((tc) => ({ type: "toolCall", id: nextToolCallId(), name: tc.name, arguments: tc.arguments }));
      emitter.finish(parsed.thinking, parsed.text, toolCalls, full, promptTokens);
    } catch (err) {
      setStats({ generating: false, phase: null });
      const aborted = options?.signal?.aborted || (err instanceof Error && /abort/i.test(err.message));
      emitter.error(aborted ? "aborted" : "error", err instanceof Error ? err.message : String(err));
    }
  });

  return stream;
}
