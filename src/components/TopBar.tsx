import type { EngineState } from "../App";
import type { ResolvedModel } from "../lib/models";
import { Chip, Gear, Refresh, Bolt } from "./Icons";
import { useStats, estimateModelBytes, formatBytes } from "../pi/stats";

interface Props {
  model: ResolvedModel;
  eng: EngineState;
  webgpu: boolean;
  onOpenPicker: () => void;
  onReload: () => void;
}

const fmtMB = (b: number) => (b > 0 ? `${(b / 1024 / 1024).toFixed(0)} MB` : "");

function fmtTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

/** Live "model is working" readout: phase + ticking tokens + tok/s, then idle stats. */
function ModelStats() {
  const { tps, contextUsed, contextWindow, modelBytes, generating, liveTokens, phase } = useStats();
  const mem = estimateModelBytes(modelBytes, contextWindow);
  const pct = contextUsed && contextWindow ? Math.min(100, (contextUsed / contextWindow) * 100) : 0;

  if (generating) {
    // Animated, always-moving while the model works.
    return (
      <div className="hidden md:flex items-center gap-2 text-[10.5px] font-mono whitespace-nowrap text-[var(--color-pi-2)]">
        <span className="flex items-center gap-1">
          <span className="typing-dot" />
          <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
          <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
        </span>
        <span className="text-[var(--color-ink)]">{phase ?? "working"}…</span>
        <span className="text-[var(--color-ink-faint)]">
          {fmtTokens(liveTokens)} tok{tps ? ` · ${tps} tok/s` : ""}
        </span>
      </div>
    );
  }

  const parts: string[] = [];
  if (tps) parts.push(`${tps} tok/s`);
  if (mem) parts.push(`~${formatBytes(mem)}`);
  if (parts.length === 0 && !contextWindow) return null;

  return (
    <div className="hidden md:flex items-center gap-2 text-[10.5px] text-[var(--color-ink-faint)] font-mono whitespace-nowrap">
      {parts.length > 0 && <span>{parts.join(" · ")}</span>}
      {contextWindow ? (
        <span className="flex items-center gap-1.5" title="Context window usage (estimated tokens)">
          <span>
            ctx {contextUsed ? fmtTokens(contextUsed) : "0"}/{fmtTokens(contextWindow)}
          </span>
          <span className="w-12 h-1 rounded-full bg-[var(--color-panel-2)] overflow-hidden">
            <span
              className="block h-full rounded-full bg-[var(--color-pi)]"
              style={{ width: `${Math.max(2, pct)}%` }}
            />
          </span>
        </span>
      ) : null}
    </div>
  );
}

export function TopBar({ model, eng, webgpu, onOpenPicker, onReload }: Props) {
  const isThis = eng.modelId === model.id;
  const loading = isThis && eng.phase === "loading";
  const ready = isThis && eng.phase === "ready";
  const errored = isThis && eng.phase === "error";

  return (
    <header className="h-14 shrink-0 px-4 flex items-center gap-3 border-b border-[var(--color-edge)] glass">
      <button
        onClick={onOpenPicker}
        className="flex items-center gap-2.5 rounded-xl pl-2 pr-3 py-1.5 border border-[var(--color-edge)] hover:border-[var(--color-edge-2)] bg-[var(--color-panel-2)]/50 transition"
        title="Choose model"
      >
        <span
          className="w-7 h-7 rounded-lg grid place-items-center shrink-0"
          style={{ background: `color-mix(in oklab, ${model.accent} 22%, transparent)` }}
        >
          <Chip className="w-4 h-4" />
        </span>
        <span className="text-left leading-tight">
          <span className="flex items-center gap-1.5 text-[13px] font-medium">
            {model.label}
            {model.verified && (
              <span className="text-[9px] px-1 py-px rounded bg-[var(--color-mint)]/15 text-[var(--color-mint)] border border-[var(--color-mint)]/30">
                verified
              </span>
            )}
          </span>
          <span className="block text-[10.5px] text-[var(--color-ink-faint)]">{model.sizeLabel}</span>
        </span>
        <Gear className="w-3.5 h-3.5 text-[var(--color-ink-faint)] ml-1" />
      </button>

      <ModelStats />

      {/* status pill */}
      <div className="flex-1 min-w-0">
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
                className="h-full rounded-full bg-[var(--color-pi)] transition-[width] duration-200"
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
            {model.id.startsWith("or:")
              ? "Cloud · via OpenRouter with your key"
              : model.remote
                ? "Cloud · remote endpoint"
                : `Loaded · running on ${webgpu ? "WebGPU" : "CPU"}`}
          </span>
        )}
        {errored && (
          <span className="text-[11.5px] text-[var(--color-soyuz)]">
            ⚠️ Failed to load — {eng.error?.slice(0, 120)}
          </span>
        )}
        {!isThis && eng.phase === "idle" && (
          <span className="text-[11.5px] text-[var(--color-ink-faint)]">
            {model.id.startsWith("or:")
              ? "Ready — runs via OpenRouter with your key."
              : "Model loads on your first message."}
          </span>
        )}
      </div>

      <button
        className="btn btn-ghost"
        onClick={onReload}
        disabled={loading}
        title="Reload / download model now"
      >
        <Refresh className="w-3.5 h-3.5" />
        {ready ? "Reload" : "Load now"}
      </button>
    </header>
  );
}
