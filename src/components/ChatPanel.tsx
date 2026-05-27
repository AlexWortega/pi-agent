import { useEffect, useRef, useState } from "react";
import type { Project, ChatMessage } from "../types";
import type { EngineState } from "../App";
import { renderMarkdown } from "../lib/parse";
import { Send, Stop, Code, Satellite } from "./Icons";

interface Props {
  project: Project;
  generating: boolean;
  eng: EngineState;
  onSend: (text: string) => void;
  onStop: () => void;
  onOpenArtifact: (id: string) => void;
}

const SUGGESTIONS = [
  "Build a neon synthwave pomodoro timer",
  "Make a single-file markdown notes app with localStorage",
  "Create a particle constellation that follows the cursor",
  "Code a playable breakout game with sound",
];

export function ChatPanel({ project, generating, eng, onSend, onStop, onOpenArtifact }: Props) {
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const msgs = project?.messages ?? [];

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs, generating]);

  const submit = () => {
    if (!text.trim() || generating) return;
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
          {msgs.length === 0 ? (
            <Welcome onPick={onSend} />
          ) : (
            <div className="space-y-5">
              {msgs.map((m) => (
                <Bubble key={m.id} m={m} project={project} onOpenArtifact={onOpenArtifact} />
              ))}
              {generating && eng.phase === "loading" && <ModelLoading eng={eng} />}
              {generating && eng.phase === "ready" && <Typing />}
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
          <div className="text-center text-[10.5px] text-[var(--color-ink-faint)] mt-2">
            Runs entirely on your machine · ⏎ send · ⇧⏎ newline
          </div>
        </div>
      </div>
    </section>
  );
}

function Welcome({ onPick }: { onPick: (t: string) => void }) {
  return (
    <div className="pt-10 text-center">
      <div className="w-14 h-14 mx-auto rounded-2xl grid place-items-center bg-gradient-to-br from-[var(--color-pi)] to-[var(--color-soyuz)] shadow-xl shadow-[var(--color-pi)]/30">
        <Satellite className="w-7 h-7 text-white" />
      </div>
      <h1 className="mt-4 text-xl font-semibold">Build apps with Pi Agent</h1>
      <p className="mt-1.5 text-[13.5px] text-[var(--color-ink-dim)] max-w-md mx-auto">
        Soyuz writes complete, self-contained web apps — rendered live on the right, ready to
        download. 100% local, powered by llama.cpp on your GPU.
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
          Загружаю модель Soyuz… первый раз качается (~2.5 GB), потом кэш в браузере.
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

function Typing() {
  return (
    <div className="flex items-center gap-1.5 pl-1">
      <span className="typing-dot" />
      <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
      <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
    </div>
  );
}

function Bubble({
  m,
  project,
  onOpenArtifact,
}: {
  m: ChatMessage;
  project: Project;
  onOpenArtifact: (id: string) => void;
}) {
  const [showThink, setShowThink] = useState(false);
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-[var(--color-pi)] to-[#6244e0] px-4 py-2.5 text-[14px] text-white whitespace-pre-wrap shadow-lg shadow-[var(--color-pi)]/20">
          {m.content}
        </div>
      </div>
    );
  }

  const artId = "art-" + m.id;
  const art = project.artifacts.find((a) => a.id === artId);

  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 rounded-lg grid place-items-center bg-gradient-to-br from-[var(--color-soyuz)] to-[var(--color-pi)] mt-0.5">
        <Satellite className="w-4 h-4 text-white" />
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
        <div
          className="msg-md text-[14px] text-[var(--color-ink)] break-words"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content || "") }}
        />
        {m.pending && !m.content && !m.think && (
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
              <span className="block text-[12.5px] font-medium group-hover:text-[var(--color-pi-2)]">
                {art.title}
              </span>
              <span className="block text-[10.5px] text-[var(--color-ink-faint)]">
                Open in canvas →
              </span>
            </span>
          </button>
        )}
      </div>
    </div>
  );
}
