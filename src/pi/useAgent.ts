/**
 * React controller for the in-browser pi agent.
 *
 * Subscribes to the agent loop's AgentEvent stream and projects it into UI
 * state: a transcript of user/assistant/tool messages, a "running" flag, and an
 * `fsVersion` counter that bumps whenever a tool mutates the filesystem (so the
 * file explorer / preview can refresh). The raw pi Message transcript is kept in
 * a ref and fed back as history on the next turn for multi-turn continuity.
 */
import { useCallback, useRef, useState } from "react";
import type { AssistantMessage, Message, Model } from "@earendil-works/pi-ai";
import { startAgentRun, WORKSPACE_ROOT } from "./runtime";
import { getFsBackend } from "./fs/backend";
import { loadRepoMeta } from "../lib/github";
import { extractHtmlArtifact } from "../lib/parse";

/** Remove fenced ```html``` blocks from chat text (they live in the file now). */
function stripHtmlBlocks(text: string): string {
  return text.replace(/```(?:html|HTML)?\s*[\s\S]*?```/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

export type UiMessage =
  | { id: string; kind: "user"; text: string }
  | {
      id: string;
      kind: "assistant";
      text: string;
      thinking: string;
      streaming: boolean;
      error?: string;
    }
  | {
      id: string;
      kind: "tool";
      toolCallId: string;
      name: string;
      args: unknown;
      status: "running" | "ok" | "error";
      resultText: string;
      diff?: string;
    };

interface AssistantBlocks {
  text: string;
  thinking: string;
}

function extractAssistant(content: AssistantMessage["content"]): AssistantBlocks {
  let text = "";
  let thinking = "";
  for (const block of content) {
    if (block.type === "text") text += block.text;
    else if (block.type === "thinking") thinking += block.thinking;
  }
  return { text, thinking };
}

function resultToText(result: { content?: Array<{ type: string; text?: string }> } | undefined): string {
  if (!result?.content) return "";
  return result.content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("\n");
}

let idSeq = 0;
const nextId = () => `ui-${++idSeq}`;

/** Result of preparing a run: a ready model (cloud = via OpenRouter) or a user-facing error. */
export type PreparedRun = { model: Model<any>; cloud?: boolean } | { error: string };

/** A function that ensures the model is ready and returns it (or an error to show in chat). */
export type PrepareModel = () => Promise<PreparedRun>;

export interface UseAgentResult {
  messages: UiMessage[];
  running: boolean;
  fsVersion: number;
  /** HTML being written live this turn (partial), for a streaming canvas preview. */
  liveHtml: string | null;
  send: (text: string) => Promise<void>;
  stop: () => void;
  reset: () => void;
  clearWorkspace: () => Promise<void>;
}

export function useAgent(prepare: PrepareModel): UseAgentResult {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [running, setRunning] = useState(false);
  const [fsVersion, setFsVersion] = useState(0);
  const [liveHtml, setLiveHtml] = useState<string | null>(null);

  const transcriptRef = useRef<Message[]>([]);
  const abortRef = useRef<(() => void) | null>(null);
  const assistantIdRef = useRef<string | null>(null);
  // HTML-bridge bookkeeping: the latest HTML document the model emitted in chat
  // during this run (across all turns) and which bubble held it, so we can
  // salvage it into index.html if the model never wrote the file via tools.
  const lastHtmlRef = useRef<string | null>(null);
  const lastHtmlMsgIdRef = useRef<string | null>(null);

  const updateMessage = useCallback((id: string, patch: (m: UiMessage) => UiMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));
  }, []);

  const ensureAssistant = useCallback((): string => {
    if (assistantIdRef.current) return assistantIdRef.current;
    const id = nextId();
    assistantIdRef.current = id;
    setMessages((prev) => [
      ...prev,
      { id, kind: "assistant", text: "", thinking: "", streaming: true },
    ]);
    return id;
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || running) return;

      setMessages((prev) => [...prev, { id: nextId(), kind: "user", text: trimmed }]);
      setRunning(true);
      assistantIdRef.current = null;
      lastHtmlRef.current = null;
      lastHtmlMsgIdRef.current = null;
      setLiveHtml(null);
      console.debug("[pi] ▶ run:", trimmed);

      const prepared = await prepare();
      if ("error" in prepared) {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            kind: "assistant",
            text: "",
            thinking: "",
            streaming: false,
            error: prepared.error,
          },
        ]);
        setRunning(false);
        return;
      }

      // One mode: tools available, prompt directs Soyuz to a single `write` of
      // the full file (which it does cleanly at temp 0.1). The HTML bridge
      // covers the rare case it emits a raw ```html block instead.
      const repoMeta = await loadRepoMeta(getFsBackend()).catch(() => null);
      const run = startAgentRun({
        prompt: trimmed,
        history: transcriptRef.current,
        model: prepared.model,
        cloud: prepared.cloud,
        repo: repoMeta ? `${repoMeta.owner}/${repoMeta.repo}@${repoMeta.branch}` : undefined,
      });
      abortRef.current = run.abort;

      try {
        for await (const event of run.stream) {
          switch (event.type) {
            case "message_update": {
              const msg = event.message as AssistantMessage;
              if (msg.role !== "assistant") break;
              const id = ensureAssistant();
              const { text: t, thinking } = extractAssistant(msg.content);
              updateMessage(id, (m) =>
                m.kind === "assistant" ? { ...m, text: t, thinking, streaming: true } : m,
              );
              // Stream the in-progress HTML into the canvas as it's typed.
              const partial = extractHtmlArtifact(t);
              if (partial) setLiveHtml(partial.html);
              break;
            }
            case "message_end": {
              const msg = event.message as Message;
              if (msg.role === "assistant") {
                const id = ensureAssistant();
                const { text: t, thinking } = extractAssistant(msg.content);
                const error =
                  msg.stopReason === "error" || msg.stopReason === "aborted" ? msg.errorMessage : undefined;
                updateMessage(id, (m) =>
                  m.kind === "assistant" ? { ...m, text: t, thinking, streaming: false, error } : m,
                );
                // Capture any complete HTML document the model dropped in chat.
                const html = extractHtmlArtifact(t);
                if (html) {
                  lastHtmlRef.current = html.html;
                  lastHtmlMsgIdRef.current = id;
                }
                assistantIdRef.current = null;
              }
              break;
            }
            case "tool_execution_start": {
              console.debug(`[pi] 🔧 ${event.toolName}(`, event.args, ")");
              setMessages((prev) => [
                ...prev,
                {
                  id: nextId(),
                  kind: "tool",
                  toolCallId: event.toolCallId,
                  name: event.toolName,
                  args: event.args,
                  status: "running",
                  resultText: "",
                },
              ]);
              // A fresh assistant message will follow the tool batch.
              assistantIdRef.current = null;
              break;
            }
            case "tool_execution_end": {
              const diff = event.result?.details?.diff as string | undefined;
              const resultText = resultToText(event.result);
              console.debug(
                `[pi] ${event.isError ? "❌" : "✅"} ${event.toolName} →`,
                event.isError ? resultText : diff || resultText || "(ok)",
              );
              setMessages((prev) =>
                prev.map((m) =>
                  m.kind === "tool" && m.toolCallId === event.toolCallId
                    ? { ...m, status: event.isError ? "error" : "ok", resultText, diff }
                    : m,
                ),
              );
              setFsVersion((v) => v + 1);
              break;
            }
            case "agent_end": {
              transcriptRef.current = [...transcriptRef.current, ...(event.messages as Message[])];
              break;
            }
            default:
              break;
          }
        }

        // HTML bridge: Soyuz tends to emit a whole HTML document in chat (it
        // can't reliably stuff one through a JSON tool arg). If the run finished
        // without an index.html on disk but the model dropped an HTML doc in any
        // turn, salvage it into the file so the preview works — and trim the wall
        // of code out of that chat bubble.
        const indexPath = `${WORKSPACE_ROOT}/index.html`;
        const hasIndex = await getFsBackend().exists(indexPath);
        console.debug(`[pi] ⏹ run done — index.html exists: ${hasIndex}, captured html: ${!!lastHtmlRef.current}`);
        // Never salvage chat-HTML into an imported repo — an index.html the
        // repo doesn't have would show up in the push diff as a new file.
        if (!hasIndex && lastHtmlRef.current && !repoMeta) {
          try {
            await getFsBackend().writeText(indexPath, lastHtmlRef.current);
            setFsVersion((v) => v + 1);
            const id = lastHtmlMsgIdRef.current;
            if (id) {
              const note = "✅ Saved to `index.html` — see the live preview →";
              updateMessage(id, (m) => {
                if (m.kind !== "assistant") return m;
                const stripped = stripHtmlBlocks(m.text);
                return { ...m, text: stripped ? `${stripped}\n\n${note}` : note };
              });
            }
          } catch {
            /* ignore — preview just won't update */
          }
        }
      } catch (err) {
        // Surface otherwise-silent failures (stream iteration, engine crash)
        // into the chat instead of swallowing them.
        console.error("[pi] agent run failed", err);
        const message = err instanceof Error ? err.message : String(err);
        setMessages((prev) => [
          ...prev,
          { id: nextId(), kind: "assistant", text: "", thinking: "", streaming: false, error: message },
        ]);
      } finally {
        setRunning(false);
        abortRef.current = null;
        assistantIdRef.current = null;
        // Saved file (if any) now drives the canvas via fsVersion.
        setLiveHtml(null);
      }
    },
    [prepare, running, ensureAssistant, updateMessage],
  );

  const stop = useCallback(() => abortRef.current?.(), []);

  const clearWorkspace = useCallback(async () => {
    try {
      await getFsBackend().remove(WORKSPACE_ROOT);
    } catch {
      /* nothing to remove */
    }
    setFsVersion((v) => v + 1);
  }, []);

  const reset = useCallback(() => {
    transcriptRef.current = [];
    assistantIdRef.current = null;
    setMessages([]);
    setLiveHtml(null);
    void clearWorkspace();
  }, [clearWorkspace]);

  return { messages, running, fsVersion, liveHtml, send, stop, reset, clearWorkspace };
}
