import { useState } from "react";
import type { Project } from "../types";
import { Plus, Trash, Satellite } from "./Icons";

interface Props {
  projects: Project[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
  webgpu: boolean;
}

export function Sidebar({ projects, activeId, onSelect, onCreate, onDelete, onRename, webgpu }: Props) {
  const [editing, setEditing] = useState<string | null>(null);

  return (
    <aside className="w-64 shrink-0 h-full glass border-r border-[var(--color-edge)] flex flex-col">
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl grid place-items-center bg-gradient-to-br from-[var(--color-pi)] to-[var(--color-soyuz)] shadow-lg shadow-[var(--color-pi)]/30">
            <Satellite className="w-5 h-5 text-white" />
          </div>
          <div className="leading-tight">
            <div className="font-semibold text-[15px]">Pi Agent</div>
            <div className="text-[11px] text-[var(--color-ink-faint)]">Soyuz · in-browser</div>
          </div>
        </div>
        <div
          className="mt-3 inline-flex items-center gap-1.5 text-[10.5px] px-2 py-1 rounded-full border"
          style={{
            color: webgpu ? "var(--color-mint)" : "#ffb454",
            borderColor: webgpu ? "color-mix(in oklab, var(--color-mint) 40%, transparent)" : "#5a4420",
            background: webgpu ? "color-mix(in oklab, var(--color-mint) 8%, transparent)" : "rgba(90,68,32,.18)",
          }}
          title="llama.cpp runs locally on your GPU via WebGPU"
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: "currentColor" }} />
          {webgpu ? "WebGPU ready" : "WebGPU off — CPU fallback"}
        </div>
      </div>

      <div className="px-3">
        <button className="btn btn-primary w-full justify-center" onClick={onCreate}>
          <Plus className="w-4 h-4" /> New project
        </button>
      </div>

      <div className="mt-3 px-2 flex-1 min-h-0 overflow-y-auto space-y-0.5">
        {projects.map((p) => {
          const isActive = p.id === activeId;
          return (
            <div
              key={p.id}
              onClick={() => onSelect(p.id)}
              className={`group rounded-lg px-2.5 py-2 cursor-pointer transition border ${
                isActive
                  ? "bg-[var(--color-panel-2)] border-[var(--color-edge-2)]"
                  : "border-transparent hover:bg-[var(--color-panel-2)]/60"
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
                      {p.messages.length} msg · {p.artifacts.length} app
                      {p.artifacts.length === 1 ? "" : "s"}
                    </div>
                  </div>
                )}
                <button
                  className="opacity-0 group-hover:opacity-100 text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)] transition"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(p.id);
                  }}
                  title="Delete project"
                >
                  <Trash className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="px-4 py-3 border-t border-[var(--color-edge)] text-[10.5px] text-[var(--color-ink-faint)] leading-relaxed">
        Inference: llama.cpp WebGPU · or your OpenRouter key
        <br />
        <a
          className="text-[var(--color-ink-dim)] hover:text-[var(--color-pi-2)]"
          href="https://github.com/ggml-org/llama.cpp"
          target="_blank"
          rel="noopener"
        >
          ggml-org/llama.cpp
        </a>
        {" · "}
        <a
          className="text-[var(--color-ink-dim)] hover:text-[var(--color-pi-2)]"
          href="https://reeselevine.github.io/llamas-on-the-web/"
          target="_blank"
          rel="noopener"
        >
          llamas-on-the-web
        </a>
      </div>
    </aside>
  );
}
