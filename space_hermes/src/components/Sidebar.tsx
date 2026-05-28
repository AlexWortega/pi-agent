import { useState } from "react";
import type { Project } from "../types";
import type { PanelId } from "../lib/commands";
import { Plus, Trash, Bot, Sparkles, Brain, ListTodo, Wrench, Clock, X } from "./Icons";

interface Props {
  projects: Project[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
  webgpu: boolean;
  onOpenPanel: (p: PanelId) => void;
  /** Mobile drawer mode — show a close button + auto-close on interactions. */
  onClose?: () => void;
}

const NAV: { id: PanelId; label: string; icon: React.ReactNode }[] = [
  { id: "skills", label: "Skills", icon: <Sparkles className="w-4 h-4" /> },
  { id: "memory", label: "Memory", icon: <Brain className="w-4 h-4" /> },
  { id: "tasks", label: "Tasks", icon: <ListTodo className="w-4 h-4" /> },
  { id: "tools", label: "Tools", icon: <Wrench className="w-4 h-4" /> },
  { id: "schedules", label: "Schedules", icon: <Clock className="w-4 h-4" /> },
];

export function Sidebar({ projects, activeId, onSelect, onCreate, onDelete, onRename, webgpu, onOpenPanel, onClose }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const pickPanel = (p: PanelId) => { onOpenPanel(p); onClose?.(); };
  const pickSession = (id: string) => { onSelect(id); onClose?.(); };
  const newSession = () => { onCreate(); onClose?.(); };

  return (
    <aside className="w-full h-full glass border-r border-[var(--color-edge)] flex flex-col">
      <div className="px-4 pt-5 pb-4 border-b border-[var(--color-edge)] relative">
        {onClose && (
          <button className="absolute top-3 right-3 btn btn-ghost p-2" onClick={onClose} aria-label="Close menu">
            <X className="w-4 h-4" />
          </button>
        )}
        <h2 className="font-sans font-bold leading-[0.95] tracking-[0.04em] text-[28px] text-[var(--color-ink)]">
          HERMES
          <br />
          AGENT
        </h2>
        <div className="mt-2 label-faint">v0.1 · WebGPU · on-device</div>
        <div
          className="mt-3 inline-flex items-center gap-1.5 text-[10px] tracking-[0.18rem] uppercase font-mono px-2 py-1 border"
          style={{
            color: webgpu ? "var(--color-pi-2)" : "var(--color-soyuz)",
            borderColor: webgpu ? "color-mix(in oklab, var(--color-pi-2) 35%, transparent)" : "color-mix(in oklab, var(--color-soyuz) 35%, transparent)",
          }}
          title="llama.cpp runs locally on your GPU via WebGPU"
        >
          <span className="w-1.5 h-1.5" style={{ background: "currentColor" }} />
          {webgpu ? "WebGPU ready" : "WebGPU off / CPU"}
        </div>
      </div>

      <div className="px-3 mt-3">
        <button className="btn btn-primary w-full justify-center" onClick={newSession}>
          <Plus className="w-4 h-4" /> New session
        </button>
      </div>

      {/* agent panels */}
      <div className="px-3 mt-4">
        <div className="label-faint px-1 mb-1.5">Panels</div>
        <div className="grid grid-cols-1 gap-0">
          {NAV.map((n) => (
            <button key={n.id} className="nav-btn" onClick={() => pickPanel(n.id)}>
              <span className="text-[var(--color-pi-2)]">{n.icon}</span>
              {n.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 px-2 flex-1 min-h-0 overflow-y-auto space-y-0 border-t border-[var(--color-edge)] pt-3">
        <div className="px-2.5 pb-1.5 label-faint">Sessions</div>
        {projects.map((p) => {
          const isActive = p.id === activeId;
          return (
            <div
              key={p.id}
              onClick={() => pickSession(p.id)}
              className={`group rounded-lg px-2.5 py-2 cursor-pointer transition border ${
                isActive ? "bg-[var(--color-panel-2)] border-[var(--color-edge-2)]" : "border-transparent hover:bg-[var(--color-panel-2)]/60"
              }`}
            >
              <div className="flex items-center gap-2">
                {editing === p.id ? (
                  <input
                    autoFocus
                    defaultValue={p.name}
                    className="field py-1 text-[13px]"
                    onClick={(e) => e.stopPropagation()}
                    onBlur={(e) => {
                      onRename(p.id, e.target.value.trim() || p.name);
                      setEditing(null);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                ) : (
                  <div
                    className="flex-1 min-w-0"
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      setEditing(p.id);
                    }}
                  >
                    <div className="text-[13px] truncate font-medium">{p.name}</div>
                    <div className="text-[10.5px] text-[var(--color-ink-faint)]">
                      {p.messages.length} msg · {p.artifacts.length} app{p.artifacts.length === 1 ? "" : "s"}
                    </div>
                  </div>
                )}
                <button
                  className="opacity-0 group-hover:opacity-100 text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)] transition"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(p.id);
                  }}
                  title="Delete session"
                >
                  <Trash className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="px-4 py-3 border-t border-[var(--color-edge)] label-faint leading-relaxed">
        Inspired by{" "}
        <a className="text-[var(--color-ink-dim)] hover:text-[var(--color-pi-2)]" href="https://github.com/nousresearch/hermes-agent" target="_blank" rel="noopener">
          nousresearch/hermes-agent
        </a>
        <br />
        Engine: llama.cpp · WebGPU
      </div>
    </aside>
  );
}
