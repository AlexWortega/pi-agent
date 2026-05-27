import { useState } from "react";
import type { GenParams } from "../types";
import { MODEL_PRESETS } from "../config";
import { customModelId } from "../lib/models";
import { Chip, X, Check, Bolt } from "./Icons";

interface Props {
  currentId: string;
  onClose: () => void;
  onPick: (id: string) => void;
  params: GenParams;
  onParams: (p: GenParams) => void;
}

export function ModelPicker({ currentId, onClose, onPick, params, onParams }: Props) {
  const [customUrl, setCustomUrl] = useState(
    currentId.startsWith("url:") ? currentId.slice(4) : "",
  );

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="glass rounded-2xl border border-[var(--color-edge-2)] w-full max-w-lg max-h-[88vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-4 flex items-center justify-between border-b border-[var(--color-edge)]">
          <div>
            <h2 className="font-semibold text-[15px]">Model</h2>
            <p className="text-[11.5px] text-[var(--color-ink-faint)]">
              Downloaded once, cached in your browser, runs on your GPU.
            </p>
          </div>
          <button className="btn btn-ghost p-2" onClick={onClose}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="overflow-y-auto p-4 space-y-2">
          {MODEL_PRESETS.map((m) => {
            const selected = currentId === m.id;
            return (
              <button
                key={m.id}
                onClick={() => onPick(m.id)}
                className={`w-full text-left rounded-xl border p-3 flex items-start gap-3 transition ${
                  selected
                    ? "border-[var(--color-pi)] bg-[var(--color-pi)]/8"
                    : "border-[var(--color-edge)] hover:border-[var(--color-edge-2)] bg-[var(--color-panel-2)]/30"
                }`}
              >
                <span
                  className="w-9 h-9 shrink-0 rounded-lg grid place-items-center"
                  style={{ background: `color-mix(in oklab, ${m.accent} 22%, transparent)` }}
                >
                  <Chip className="w-4.5 h-4.5" />
                </span>
                <span className="flex-1 min-w-0">
                  <span className="flex items-center gap-2">
                    <span className="text-[13.5px] font-medium">{m.label}</span>
                    {m.verified ? (
                      <span className="text-[9px] px-1.5 py-px rounded-full bg-[var(--color-mint)]/15 text-[var(--color-mint)] border border-[var(--color-mint)]/30">
                        verified
                      </span>
                    ) : (
                      <span className="text-[9px] px-1.5 py-px rounded-full bg-[var(--color-soyuz)]/15 text-[var(--color-soyuz)] border border-[var(--color-soyuz)]/30">
                        experimental
                      </span>
                    )}
                    <span className="ml-auto text-[10.5px] text-[var(--color-ink-faint)]">{m.sizeLabel}</span>
                  </span>
                  <span className="block text-[11.5px] text-[var(--color-ink-faint)] mt-0.5 leading-snug">
                    {m.note}
                  </span>
                </span>
                {selected && <Check className="w-4 h-4 text-[var(--color-pi-2)] mt-1 shrink-0" />}
              </button>
            );
          })}

          {/* custom url */}
          <div className="rounded-xl border border-[var(--color-edge)] p-3 bg-[var(--color-panel-2)]/30">
            <div className="text-[12px] font-medium mb-1.5">Custom GGUF URL</div>
            <div className="flex gap-2">
              <input
                className="field text-[12px]"
                placeholder="https://huggingface.co/…/model.gguf"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
              />
              <button
                className="btn btn-primary shrink-0"
                disabled={!customUrl.trim().endsWith(".gguf")}
                onClick={() => onPick(customModelId(customUrl))}
              >
                <Bolt className="w-3.5 h-3.5" /> Use
              </button>
            </div>
            <p className="text-[10.5px] text-[var(--color-ink-faint)] mt-1.5">
              Any GGUF a recent llama.cpp WebGPU build supports. Hybrid linear-attention archs
              (incl. Soyuz) may not load yet.
            </p>
          </div>
        </div>

        {/* params */}
        <div className="px-5 py-4 border-t border-[var(--color-edge)] space-y-3">
          <Slider
            label="Temperature"
            value={params.temperature}
            min={0}
            max={1.5}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => onParams({ ...params, temperature: v })}
          />
          <Slider
            label="Max output tokens"
            value={params.maxTokens}
            min={256}
            max={8192}
            step={256}
            fmt={(v) => String(v)}
            onChange={(v) => onParams({ ...params, maxTokens: v })}
          />
          <Slider
            label="Context length"
            value={params.contextLength}
            min={2048}
            max={16384}
            step={1024}
            fmt={(v) => String(v)}
            hint="Applied on next model load"
            onChange={(v) => onParams({ ...params, contextLength: v })}
          />
        </div>
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  fmt,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-[12px] mb-1">
        <span className="text-[var(--color-ink-dim)]">
          {label}
          {hint && <span className="text-[var(--color-ink-faint)]"> · {hint}</span>}
        </span>
        <span className="font-[var(--font-mono)] text-[var(--color-pi-2)]">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--color-pi)]"
      />
    </div>
  );
}
