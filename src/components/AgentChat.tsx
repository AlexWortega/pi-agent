import { useEffect, useRef, useState } from "react";
import type { UiMessage } from "../pi/useAgent";
import type { EngineState } from "../App";
import { renderMarkdown } from "../lib/parse";
import { useStats } from "../pi/stats";
import { Send, Stop, Satellite, Check, X, Code } from "./Icons";

/**
 * Pre-content indicator: while the model is working but hasn't emitted any text
 * yet (e.g. the cloud model is starting a GPU or generating without token
 * streaming), show the live phase + a ticking elapsed counter so it's clearly
 * alive — not a frozen "thinking".
 */
function StreamingIndicator() {
  const { phase } = useStats();
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setSecs(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="flex items-center gap-2 pl-1 text-[12px] text-[var(--color-ink-faint)]">
      <span className="flex items-center gap-1">
        <span className="typing-dot" />
        <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
        <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
      </span>
      <span>{phase ?? "working"}</span>
      {secs >= 2 && <span className="font-[var(--font-mono)] opacity-70">{secs}s</span>}
    </div>
  );
}

interface Props {
  messages: UiMessage[];
  running: boolean;
  eng: EngineState;
  onSend: (text: string) => void;
  onStop: () => void;
}

const SUGGESTIONS = [
  "Build a neon synthwave pomodoro timer in index.html",
  "Create a single-file markdown notes app with localStorage",
  "Make a particle constellation that follows the cursor",
  "Code a playable breakout game with sound",
];

export function AgentChat({ messages, running, eng, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, running]);

  const submit = () => {
    if (!text.trim() || running) return;
    onSend(text);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const grow = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  return (
    <section className="flex-1 min-w-0 flex flex-col border-r border-[var(--color-edge)]">
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto px-5 py-6">
        <div className="max-w-2xl mx-auto">
          {messages.length === 0 ? (
            <Welcome onPick={onSend} />
          ) : (
            <div className="space-y-4">
              {messages.map((m) => (
                <MessageView key={m.id} m={m} />
              ))}
              {running && eng.phase === "loading" && <ModelLoading eng={eng} />}
            </div>
          )}
        </div>
      </div>

      <div className="px-5 pb-5 pt-2">
        <div className="max-w-2xl mx-auto">
          <div className="glass rounded-2xl border border-[var(--color-edge)] focus-within:border-[var(--color-pi)] transition p-2 flex items-end gap-2">
            <textarea
              ref={taRef}
              value={text}
              rows={1}
              placeholder="Ask Pi Agent to build something…"
              className="flex-1 bg-transparent resize-none outline-none text-[14px] px-2 py-1.5 max-h-[200px] placeholder:text-[var(--color-ink-faint)]"
              onChange={(e) => {
                setText(e.target.value);
                grow(e.target);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            {running ? (
              <button className="btn btn-ghost" onClick={onStop} title="Stop">
                <Stop className="w-4 h-4" /> Stop
              </button>
            ) : (
              <button className="btn btn-primary" onClick={submit} disabled={!text.trim()}>
                <Send className="w-4 h-4" />
              </button>
            )}
          </div>
          <div className="text-center text-[10.5px] text-[var(--color-ink-faint)] mt-2">
            Real pi agent loop · runs 100% locally on your GPU · edits files with tools · ⏎ send · ⇧⏎ newline
          </div>
        </div>
      </div>
    </section>
  );
}

function MessageView({ m }: { m: UiMessage }) {
  if (m.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-[var(--color-pi)] to-[#6244e0] px-4 py-2.5 text-[14px] text-white whitespace-pre-wrap shadow-lg shadow-[var(--color-pi)]/20">
          {m.text}
        </div>
      </div>
    );
  }
  if (m.kind === "tool") return <ToolCard m={m} />;
  return <AssistantView m={m} />;
}

function AssistantView({ m }: { m: Extract<UiMessage, { kind: "assistant" }> }) {
  // Reasoning is shown live while streaming, then auto-collapses; a click pins it.
  const [manual, setManual] = useState<boolean | null>(null);
  const showThink = manual ?? m.streaming;
  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 rounded-lg grid place-items-center bg-gradient-to-br from-[var(--color-soyuz)] to-[var(--color-pi)] mt-0.5">
        <Satellite className="w-4 h-4 text-white" />
      </div>
      <div className="min-w-0 flex-1">
        {m.thinking && (
          <button
            onClick={() => setManual(!showThink)}
            className="mb-1.5 text-[11px] text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)] inline-flex items-center gap-1"
          >
            <span className={`transition ${showThink ? "rotate-90" : ""}`}>▸</span>
            {showThink ? "Hide" : "Show"} reasoning
            {m.streaming && m.thinking && !m.text ? " · thinking…" : ""}
          </button>
        )}
        {m.thinking && showThink && (
          <div className="mb-2 text-[12px] text-[var(--color-ink-faint)] border-l-2 border-[var(--color-edge-2)] pl-3 whitespace-pre-wrap italic">
            {m.thinking}
          </div>
        )}
        {m.text && (
          <div
            className="msg-md text-[14px] text-[var(--color-ink)] break-words"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(m.text) }}
          />
        )}
        {m.streaming && !m.text && !m.thinking && <StreamingIndicator />}
        {m.error && (
          <div className="mt-1 text-[13px] text-[#ff8a8a] whitespace-pre-wrap">⚠️ {m.error}</div>
        )}
      </div>
    </div>
  );
}

function argSummary(args: unknown): string {
  const a = (args ?? {}) as Record<string, unknown>;
  if (typeof a.path === "string") return a.path;
  if (typeof a.command === "string") return a.command;
  return "";
}

function prettyArgs(args: unknown): string {
  try {
    const s = JSON.stringify(args ?? {}, null, 2);
    return s.length > 2000 ? s.slice(0, 2000) + "\n… (truncated)" : s;
  } catch {
    return String(args);
  }
}

function ToolCard({ m }: { m: Extract<UiMessage, { kind: "tool" }> }) {
  // Auto-open while running or on error so the user always sees what's happening.
  const [manual, setManual] = useState<boolean | null>(null);
  const open = manual ?? (m.status !== "ok");

  const tone =
    m.status === "error"
      ? "border-[#ff8a8a]/40 bg-[#ff8a8a]/5"
      : m.status === "running"
        ? "border-[var(--color-pi)]/40 bg-[var(--color-pi)]/5"
        : "border-[var(--color-edge)] bg-[var(--color-panel-2)]/40";

  const badge =
    m.status === "running" ? (
      <span className="flex items-center gap-1.5 text-[10.5px] text-[var(--color-pi-2)]">
        <span className="typing-dot" /> running
      </span>
    ) : m.status === "ok" ? (
      <Check className="w-3.5 h-3.5 text-[#5fd39a]" />
    ) : (
      <span className="flex items-center gap-1 text-[10.5px] text-[#ff8a8a]">
        <X className="w-3.5 h-3.5" /> error
      </span>
    );

  const summary = argSummary(m.args);
  const body = m.status === "error" ? m.resultText : m.diff || m.resultText;

  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 rounded-lg grid place-items-center bg-[var(--color-pi)]/15 text-[var(--color-pi-2)] mt-0.5">
        <Code className="w-4 h-4" />
      </div>
      <div className="min-w-0 flex-1">
        <button
          onClick={() => setManual(!open)}
          className={`w-full text-left rounded-t-xl border ${tone} ${open ? "" : "rounded-b-xl"} px-3 py-2 hover:border-[var(--color-pi)] transition flex items-center gap-2`}
        >
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)]">tool</span>
          <span className="text-[12.5px] font-semibold text-[var(--color-ink)] font-mono">{m.name}</span>
          {summary && (
            <span className="text-[11.5px] text-[var(--color-ink-faint)] font-mono truncate">{summary}</span>
          )}
          <span className="ml-auto flex items-center gap-2">{badge}</span>
        </button>
        {open && (
          <div className={`border border-t-0 ${tone} rounded-b-xl px-3 py-2 space-y-2`}>
            <div>
              <div className="text-[9.5px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">arguments</div>
              <pre className="text-[11px] leading-relaxed font-mono text-[var(--color-ink-dim)] overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
                {prettyArgs(m.args)}
              </pre>
            </div>
            {(body || m.status === "running") && (
              <div>
                <div className="text-[9.5px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">
                  {m.status === "error" ? "error" : m.diff ? "diff" : "result"}
                </div>
                <pre
                  className={`text-[11px] leading-relaxed font-mono overflow-x-auto max-h-72 overflow-y-auto whitespace-pre-wrap ${
                    m.status === "error" ? "text-[#ff8a8a]" : "text-[var(--color-ink-dim)]"
                  }`}
                >
                  {body || "…"}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Welcome({ onPick }: { onPick: (t: string) => void }) {
  return (
    <div className="pt-10 text-center">
      <div className="w-14 h-14 mx-auto rounded-2xl grid place-items-center bg-gradient-to-br from-[var(--color-pi)] to-[var(--color-soyuz)] shadow-xl shadow-[var(--color-pi)]/30">
        <Satellite className="w-7 h-7 text-white" />
      </div>
      <h1 className="mt-4 text-xl font-semibold">The real Pi Agent, in your browser</h1>
      <p className="mt-1.5 text-[13.5px] text-[var(--color-ink-dim)] max-w-md mx-auto">
        Soyuz drives the actual earendil-works/pi agent loop — it reads, writes and edits files
        with tools in a virtual project, all 100% local on your GPU.
      </p>
      <div className="mt-6 grid sm:grid-cols-2 gap-2.5 max-w-lg mx-auto text-left">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-xl border border-[var(--color-edge)] hover:border-[var(--color-pi)] bg-[var(--color-panel-2)]/40 px-3.5 py-3 text-[13px] text-[var(--color-ink-dim)] hover:text-[var(--color-ink)] transition"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function ModelLoading({ eng }: { eng: EngineState }) {
  const pct = Math.round(eng.progress * 100);
  const mb = (b: number) => (b > 0 ? `${(b / 1024 / 1024).toFixed(0)} MB` : "");
  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 rounded-lg grid place-items-center bg-gradient-to-br from-[var(--color-soyuz)] to-[var(--color-pi)] mt-0.5">
        <Satellite className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0 max-w-sm">
        <div className="text-[12.5px] text-[var(--color-ink-dim)] mb-1.5">
          Loading the Soyuz model… first run downloads it (~2.5 GB), then it's cached in your browser.
        </div>
        <div className="h-1.5 rounded-full bg-[var(--color-panel-2)] overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[var(--color-pi)] to-[var(--color-soyuz)] transition-[width] duration-200"
            style={{ width: `${Math.max(3, pct)}%` }}
          />
        </div>
        <div className="text-[10.5px] text-[var(--color-ink-faint)] mt-1">
          {pct}% {mb(eng.loaded)}
          {eng.total ? ` / ${mb(eng.total)}` : ""}
        </div>
      </div>
    </div>
  );
}
