import type { EngineState } from "../App";
import type { ResolvedModel } from "../lib/models";
import { Chip, Gear, Refresh, Bolt, Eye, Code } from "./Icons";

interface Props {
  model: ResolvedModel;
  eng: EngineState;
  webgpu: boolean;
  onOpenPicker: () => void;
  onReload: () => void;
  /** mobile: hamburger to open the sidebar drawer */
  onOpenSidebar?: () => void;
  /** mobile: switch the main view between chat & canvas */
  mobileView?: "chat" | "canvas";
  onMobileView?: (v: "chat" | "canvas") => void;
  /** show the canvas toggle (only when there's something to show) */
  hasCanvas?: boolean;
}

const fmtMB = (b: number) => (b > 0 ? `${(b / 1024 / 1024).toFixed(0)} MB` : "");

export function TopBar({ model, eng, webgpu, onOpenPicker, onReload, onOpenSidebar, mobileView, onMobileView, hasCanvas }: Props) {
  const isThis = eng.modelId === model.id;
  const loading = isThis && eng.phase === "loading";
  const ready = isThis && eng.phase === "ready";
  const errored = isThis && eng.phase === "error";

  return (
    <header className="h-14 shrink-0 px-3 sm:px-4 flex items-center gap-2 sm:gap-3 border-b border-[var(--color-edge)] glass">
      {onOpenSidebar && (
        <button
          className="md:hidden btn btn-ghost p-2 shrink-0"
          onClick={onOpenSidebar}
          aria-label="Open menu"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
      )}
      <button
        onClick={onOpenPicker}
        className="flex items-center gap-2 sm:gap-2.5 rounded pl-2 pr-2 sm:pr-3 py-1.5 border border-[var(--color-edge)] hover:border-[var(--color-edge-2)] bg-[var(--color-panel-2)]/50 transition min-w-0"
        title="Choose model"
      >
        <span
          className="w-7 h-7 rounded-lg grid place-items-center shrink-0"
          style={{ background: `color-mix(in oklab, ${model.accent} 22%, transparent)` }}
        >
          <Chip className="w-4 h-4" />
        </span>
        <span className="text-left leading-tight min-w-0">
          <span className="flex items-center gap-1.5 text-[12px] sm:text-[13px] font-medium truncate max-w-[140px] sm:max-w-none">
            {model.label}
            {model.verified && (
              <span className="hidden sm:inline text-[9px] px-1 py-px rounded bg-[var(--color-mint)]/15 text-[var(--color-mint)] border border-[var(--color-mint)]/30">
                verified
              </span>
            )}
          </span>
          <span className="hidden sm:block text-[10.5px] text-[var(--color-ink-faint)]">{model.sizeLabel}</span>
        </span>
        <Gear className="w-3.5 h-3.5 text-[var(--color-ink-faint)] ml-1 hidden sm:block" />
      </button>

      {/* status pill */}
      <div className="flex-1 min-w-0 hidden sm:block">
        {loading && (
          <div className="max-w-md">
            <div className="flex items-center justify-between text-[11px] text-[var(--color-ink-dim)] mb-1">
              <span className="flex items-center gap-1.5">
                <Bolt className="w-3 h-3 text-[var(--color-pi-2)]" /> Downloading model…
              </span>
              <span>
                {Math.round(eng.progress * 100)}% {fmtMB(eng.loaded)}
                {eng.total ? ` / ${fmtMB(eng.total)}` : ""}
              </span>
            </div>
            <div className="h-1.5 rounded-full bg-[var(--color-panel-2)] overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[var(--color-pi)] to-[var(--color-soyuz)] transition-[width] duration-200"
                style={{ width: `${Math.max(3, eng.progress * 100)}%` }}
              />
            </div>
            <div className="text-[10px] text-[var(--color-ink-faint)] mt-1">
              First load downloads the GGUF; it's cached in your browser (OPFS) for next time.
            </div>
          </div>
        )}
        {ready && (
          <span className="inline-flex items-center gap-1.5 text-[11.5px] text-[var(--color-mint)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-mint)]" />
            Loaded · running on {webgpu ? "WebGPU" : "CPU"}
          </span>
        )}
        {errored && (
          <span className="text-[11.5px] text-[var(--color-soyuz)]">
            ⚠️ Failed to load — {eng.error?.slice(0, 120)}
          </span>
        )}
        {!isThis && eng.phase === "idle" && (
          <span className="text-[11.5px] text-[var(--color-ink-faint)]">
            Model loads on your first message.
          </span>
        )}
      </div>

      {/* mobile-only spacer to push controls right */}
      <div className="flex-1 sm:hidden" />

      {/* mobile-only progress dot when loading */}
      {loading && (
        <span className="sm:hidden inline-flex items-center gap-1 text-[11px] text-[var(--color-pi-2)] shrink-0">
          <Bolt className="w-3 h-3 spin" />
          {Math.round(eng.progress * 100)}%
        </span>
      )}

      {/* mobile-only chat <-> canvas toggle */}
      {onMobileView && hasCanvas && (
        <div className="md:hidden inline-flex border border-[var(--color-edge)] p-0.5 shrink-0">
          <button
            className={`px-2 py-1 ${mobileView === "chat" ? "bg-[var(--color-ink)] text-[var(--color-void)]" : "text-[var(--color-ink-dim)]"}`}
            onClick={() => onMobileView("chat")}
            aria-label="Show chat"
          >
            <Code className="w-3.5 h-3.5" />
          </button>
          <button
            className={`px-2 py-1 ${mobileView === "canvas" ? "bg-[var(--color-ink)] text-[var(--color-void)]" : "text-[var(--color-ink-dim)]"}`}
            onClick={() => onMobileView("canvas")}
            aria-label="Show canvas"
          >
            <Eye className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <button
        className="btn btn-ghost shrink-0"
        onClick={onReload}
        disabled={loading}
        title="Reload / download model now"
      >
        <Refresh className="w-3.5 h-3.5" />
        <span className="hidden sm:inline">{ready ? "Reload" : "Load now"}</span>
      </button>
    </header>
  );
}
