import { useEffect, useRef, useState } from "react";
import type { Project, ChatMessage } from "../types";
import type { EngineState, AgentStatus } from "../App";
import { renderMarkdown } from "../lib/parse";
import { matchCommands, COMMAND_REGISTRY } from "../lib/commands";
import { Send, Stop, Code, Bot } from "./Icons";
import { ToolCallCard } from "./ToolCallCard";
import { ClarifyPrompt } from "./ClarifyPrompt";
import type { ClarifyRequest } from "./ClarifyPrompt";
import { SlashMenu } from "./SlashMenu";

interface Props {
  project: Project;
  generating: boolean;
  eng: EngineState;
  status: AgentStatus | null;
  clarify: ClarifyRequest | null;
  onSend: (text: string) => void;
  onStop: () => void;
  onOpenArtifact: (id: string) => void;
  /** Hide on mobile when the user is viewing the canvas. */
  mobileHidden?: boolean;
}

const SUGGESTIONS = [
  "What's 18% of 2,340, then remember my budget is the result",
  "Build a neon synthwave pomodoro timer",
  "Search the web for what Hermes 4 is, then summarize",
  "Run JS to find the 10th Fibonacci number",
];

export function ChatPanel({ project, generating, eng, status, clarify, onSend, onStop, onOpenArtifact, mobileHidden }: Props) {
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const msgs = project?.messages ?? [];
  const slash = matchCommands(text.trim());

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs, generating, clarify]);

  const submit = () => {
    if (!text.trim() || generating) return;
    onSend(text);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const pickCommand = (name: string) => {
    const cmd = COMMAND_REGISTRY.find((c) => c.name === name);
    if (cmd?.arg) {
      setText(name + " ");
      taRef.current?.focus();
    } else {
      onSend(name);
      setText("");
    }
  };

  const grow = (el: HTMLTextAreaElement) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  return (
    <section className={`${mobileHidden ? "hidden md:flex" : "flex"} flex-1 min-w-0 flex-col md:border-r md:border-[var(--color-edge)]`}>
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto px-5 py-6">
        <div className="max-w-2xl mx-auto">
          {msgs.length === 0 ? (
            <Welcome onPick={onSend} />
          ) : (
            <div className="space-y-5">
              {msgs.map((m) =>
                m.role === "user" ? (
                  <UserBubble key={m.id} m={m} />
                ) : (
                  <AssistantBubble key={m.id} m={m} project={project} onOpenArtifact={onOpenArtifact} />
                ),
              )}
              {clarify && <ClarifyPrompt req={clarify} />}
              {generating && eng.phase === "loading" && <ModelLoading eng={eng} />}
              {generating && eng.phase === "ready" && !clarify && <AgentStatusLine status={status} />}
            </div>
          )}
        </div>
      </div>

      <div className="px-5 pb-5 pt-2">
        <div className="max-w-2xl mx-auto">
          <div className="relative">
            <SlashMenu commands={slash} onPick={pickCommand} />
            <div className="glass rounded-2xl border border-[var(--color-edge)] focus-within:border-[var(--color-pi)] transition p-2 flex items-end gap-2">
              <textarea
                ref={taRef}
                value={text}
                rows={1}
                placeholder="Ask Hermes anything — or type / for commands…"
                className="flex-1 bg-transparent resize-none outline-none text-[14px] px-2 py-1.5 max-h-[200px] placeholder:text-[var(--color-ink-faint)]"
                onChange={(e) => {
                  setText(e.target.value);
                  grow(e.target);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    // commands and prompts both submit on Enter; send() routes slash commands
                    e.preventDefault();
                    submit();
                  }
                }}
              />
              {generating ? (
                <button className="btn btn-ghost" onClick={onStop} title="Stop">
                  <Stop className="w-4 h-4" /> Stop
                </button>
              ) : (
                <button className="btn btn-primary" onClick={submit} disabled={!text.trim()}>
                  <Send className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
          <div className="text-center text-[10.5px] text-[var(--color-ink-faint)] mt-2 normal-text">
            Tool-calling agent · runs locally on your GPU · ⏎ send · ⇧⏎ newline · <code>/help</code> for commands
            <br />
            <span className="opacity-70">
              chat turns logged to Railway for analytics (
              <a href="https://github.com/AlexWortega/pi-agent/blob/main/space_hermes/src/lib/logger.ts" target="_blank" rel="noopener" className="underline hover:text-[var(--color-pi-2)]">
                what + why
              </a>
              ) · localStorage data never leaves your browser
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

function Welcome({ onPick }: { onPick: (t: string) => void }) {
  return (
    <div className="pt-12 text-center">
      <div className="label-faint mb-4">Open Source · MIT · WebGPU</div>
      <h1 className="font-sans font-bold tracking-[0.02em] text-[34px] leading-[1.04] text-[var(--color-ink)]">
        THE AGENT THAT
        <br />
        GROWS WITH YOU.
      </h1>
      <p className="mt-4 text-[13.5px] text-[var(--color-ink-dim)] max-w-md mx-auto leading-relaxed normal-text">
        Not a coding copilot tethered to an IDE or a chatbot wrapper around a single API. A
        tool-calling agent that runs on your GPU, remembers what it learns, and gets more capable the
        longer you use it.
      </p>

      <div className="mt-7 max-w-md mx-auto text-left">
        <div className="flex items-center justify-between mb-1">
          <span className="label-faint">1. Try</span>
        </div>
        <div className="frame border-double border-[var(--color-edge-2)] divide-y divide-[var(--color-edge)]">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onPick(s)}
              className="w-full px-3.5 py-2.5 text-left text-[12.5px] text-[var(--color-ink-dim)] hover:text-[var(--color-pi-2)] hover:bg-[color-mix(in_oklab,var(--color-ink)_5%,transparent)] transition normal-text"
            >
              <span className="text-[var(--color-pi-2)] font-mono mr-2">›</span>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 max-w-md mx-auto text-left">
        <div className="label-faint mb-1">2. Or type a command</div>
        <div className="frame border border-[var(--color-edge)] px-3 py-2 font-mono text-[12px] text-[var(--color-ink-dim)] normal-text">
          /help · /tools · /skills · /memory · /tasks · /search
        </div>
      </div>
    </div>
  );
}

function ModelLoading({ eng }: { eng: EngineState }) {
  const pct = Math.round(eng.progress * 100);
  const mb = (b: number) => (b > 0 ? `${(b / 1024 / 1024).toFixed(0)} MB` : "");
  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 grid place-items-center border border-[var(--color-edge-2)] bg-[var(--color-panel-2)] text-[var(--color-pi-2)] mt-0.5">
        <Bot className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0 max-w-sm">
        <div className="text-[12.5px] text-[var(--color-ink-dim)] mb-1.5">
          Loading the Soyuz model… first run downloads (~2.5 GB), then it's cached in your browser.
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

function AgentStatusLine({ status }: { status: AgentStatus | null }) {
  const label =
    status?.phase === "tools" ? "running tools" : status?.phase === "done" ? "wrapping up" : "thinking";
  return (
    <div className="flex items-center gap-2 pl-1 text-[12px] text-[var(--color-ink-faint)]">
      <span className="flex items-center gap-1.5">
        <span className="typing-dot" />
        <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
        <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
      </span>
      {status ? `${label} · step ${status.iteration}/${status.maxIterations}` : "thinking"}
    </div>
  );
}

function UserBubble({ m }: { m: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] border border-[var(--color-pi-2)]/60 bg-[color-mix(in_oklab,var(--color-pi-2)_10%,transparent)] px-3.5 py-2 text-[13.5px] text-[var(--color-ink)] whitespace-pre-wrap normal-text">
        <span className="label-faint mr-2 text-[var(--color-pi-2)]">YOU</span>
        {m.content}
      </div>
    </div>
  );
}

function AssistantBubble({
  m,
  project,
  onOpenArtifact,
}: {
  m: ChatMessage;
  project: Project;
  onOpenArtifact: (id: string) => void;
}) {
  const [showThink, setShowThink] = useState(false);
  const calls = m.toolCalls ?? [];
  const results = m.toolResults ?? [];
  const artId = "art-" + m.id;
  const art = project.artifacts.find((a) => a.id === artId);

  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 grid place-items-center border border-[var(--color-edge-2)] bg-[var(--color-panel-2)] text-[var(--color-pi-2)] mt-0.5">
        <Bot className="w-4 h-4" />
      </div>
      <div className="min-w-0 flex-1">
        {m.think && (
          <button
            onClick={() => setShowThink((v) => !v)}
            className="mb-1.5 text-[11px] text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)] inline-flex items-center gap-1"
          >
            <span className={`transition ${showThink ? "rotate-90" : ""}`}>▸</span>
            {showThink ? "Hide" : "Show"} reasoning
          </button>
        )}
        {m.think && showThink && (
          <div className="mb-2 text-[12px] text-[var(--color-ink-faint)] border-l-2 border-[var(--color-edge-2)] pl-3 whitespace-pre-wrap italic">
            {m.think}
          </div>
        )}
        {m.content && (
          <div
            className="msg-md text-[14px] text-[var(--color-ink)] break-words"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
          />
        )}
        {calls.map((c) => (
          <ToolCallCard key={c.id} call={c} result={results.find((r) => r.id === c.id)} />
        ))}
        {m.pending && !m.content && !m.think && calls.length === 0 && (
          <span className="text-[13px] text-[var(--color-ink-faint)] blink" />
        )}
        {art && (
          <button
            onClick={() => onOpenArtifact(artId)}
            className="mt-2.5 inline-flex items-center gap-2 rounded-xl border border-[var(--color-edge)] hover:border-[var(--color-pi)] bg-[var(--color-panel-2)]/50 pl-2 pr-3 py-2 transition group"
          >
            <span className="w-7 h-7 rounded-lg grid place-items-center bg-[var(--color-pi)]/15 text-[var(--color-pi-2)]">
              <Code className="w-4 h-4" />
            </span>
            <span className="text-left leading-tight">
              <span className="block text-[12.5px] font-medium group-hover:text-[var(--color-pi-2)]">{art.title}</span>
              <span className="block text-[10.5px] text-[var(--color-ink-faint)]">Open in canvas →</span>
            </span>
          </button>
        )}
      </div>
    </div>
  );
}
