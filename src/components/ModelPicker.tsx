import { useState } from "react";
import type { GenParams, ReasoningEffort } from "../types";
import { CLOUD_PRESETS, MODEL_PRESETS, OPENROUTER_KEY_STORAGE } from "../config";
import { cloudModelId, customModelId, resolveModel } from "../lib/models";
import { Chip, X, Check, Bolt, Satellite } from "./Icons";

const EFFORTS: ReasoningEffort[] = ["low", "medium", "high"];

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
  const [apiKey, setApiKey] = useState(() => {
    try {
      return localStorage.getItem(OPENROUTER_KEY_STORAGE) ?? "";
    } catch {
      return "";
    }
  });
  const [customCloud, setCustomCloud] = useState(
    currentId.startsWith("or:") && !CLOUD_PRESETS.some((c) => "or:" + c.id === currentId)
      ? currentId.slice(3)
      : "",
  );
  const hasKey = apiKey.trim().length > 0;
  const saveKey = (value: string) => {
    setApiKey(value);
    const trimmed = value.trim();
    try {
      if (trimmed) localStorage.setItem(OPENROUTER_KEY_STORAGE, trimmed);
      else localStorage.removeItem(OPENROUTER_KEY_STORAGE);
    } catch {
      /* private mode — key just won't persist */
    }
  };

  const current = resolveModel(currentId);
  const isOpenRouterModel = currentId.startsWith("or:");
  const showReasoning = !!current.remote?.reasoning;
  const isRemote = !!current.remote;
  // The RunPod/local-server toggle is SIQ-specific; OpenRouter always talks to openrouter.ai.
  const showEndpointToggle = isRemote && !isOpenRouterModel;
  const remoteCtx = current.remote?.contextWindow ?? 0;

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
              Local models download once and run on your GPU; cloud models (RunPod or OpenRouter with your key) have nothing to download.
            </p>
          </div>
          <button className="btn btn-ghost p-2" onClick={onClose}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="overflow-y-auto p-4 space-y-2">
          {MODEL_PRESETS.map((m) => {
            const selected = currentId === m.id;
            const remote = !!m.remote;
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
                  {remote ? <Satellite className="w-4.5 h-4.5" /> : <Chip className="w-4.5 h-4.5" />}
                </span>
                <span className="flex-1 min-w-0">
                  <span className="flex items-center gap-2">
                    <span className="text-[13.5px] font-medium">{m.label}</span>
                    {remote && (
                      <span className="text-[9px] px-1.5 py-px rounded-full bg-[var(--color-pi)]/15 text-[var(--color-pi-2)] border border-[var(--color-pi)]/30">
                        cloud
                      </span>
                    )}
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
              Optional — load any other GGUF a recent llama.cpp WebGPU build supports. The default
              is the Soyuz Pi Agent above.
            </p>
          </div>

          {/* frontier models via OpenRouter (BYO key) */}
          <div className="pt-2">
            <div className="text-[12px] font-semibold text-[var(--color-ink-dim)] mb-1.5 px-1">
              Frontier models · OpenRouter (your key)
            </div>
            <div className="rounded-xl border border-[var(--color-edge)] p-3 bg-[var(--color-panel-2)]/30 mb-2">
              <div className="text-[12px] font-medium mb-1.5">OpenRouter API key</div>
              <input
                className="field text-[12px]"
                type="password"
                placeholder="sk-or-v1-…"
                value={apiKey}
                data-testid="openrouter-key"
                onChange={(e) => saveKey(e.target.value)}
              />
              <p className="text-[10.5px] text-[var(--color-ink-faint)] mt-1.5">
                Stored only in this browser (localStorage) and sent straight to openrouter.ai — no
                middleman server.
              </p>
            </div>

            {CLOUD_PRESETS.map((c) => {
              const id = cloudModelId(c.id);
              const selected = currentId === id;
              return (
                <button
                  key={c.id}
                  onClick={() => hasKey && onPick(id)}
                  disabled={!hasKey}
                  title={hasKey ? c.id : "Paste your OpenRouter API key first"}
                  className={`w-full text-left rounded-xl border p-3 flex items-start gap-3 transition mb-2 ${
                    selected
                      ? "border-[var(--color-pi)] bg-[var(--color-pi)]/8"
                      : "border-[var(--color-edge)] hover:border-[var(--color-edge-2)] bg-[var(--color-panel-2)]/30"
                  } ${hasKey ? "" : "opacity-50 cursor-not-allowed"}`}
                >
                  <span
                    className="w-9 h-9 shrink-0 rounded-lg grid place-items-center"
                    style={{ background: `color-mix(in oklab, ${c.accent} 22%, transparent)` }}
                  >
                    <Satellite className="w-4.5 h-4.5" />
                  </span>
                  <span className="flex-1 min-w-0">
                    <span className="flex items-center gap-2">
                      <span className="text-[13.5px] font-medium">{c.label}</span>
                      <span className="text-[9px] px-1.5 py-px rounded-full bg-[var(--color-pi)]/15 text-[var(--color-pi-2)] border border-[var(--color-pi)]/30">
                        cloud
                      </span>
                      <span className="ml-auto text-[10.5px] text-[var(--color-ink-faint)] font-mono truncate max-w-[45%]">{c.id}</span>
                    </span>
                    <span className="block text-[11.5px] text-[var(--color-ink-faint)] mt-0.5 leading-snug">
                      {c.note}
                    </span>
                  </span>
                  {selected && <Check className="w-4 h-4 text-[var(--color-pi-2)] mt-1 shrink-0" />}
                </button>
              );
            })}

            {/* custom openrouter model id */}
            <div className="rounded-xl border border-[var(--color-edge)] p-3 bg-[var(--color-panel-2)]/30">
              <div className="text-[12px] font-medium mb-1.5">Custom OpenRouter model</div>
              <div className="flex gap-2">
                <input
                  className="field text-[12px]"
                  placeholder="vendor/model-id (see openrouter.ai/models)"
                  value={customCloud}
                  onChange={(e) => setCustomCloud(e.target.value)}
                />
                <button
                  className="btn btn-primary shrink-0"
                  disabled={!hasKey || !customCloud.trim().includes("/")}
                  onClick={() => onPick(cloudModelId(customCloud))}
                >
                  <Bolt className="w-3.5 h-3.5" /> Use
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* params */}
        <div className="px-5 py-4 border-t border-[var(--color-edge)] space-y-3">
          {showEndpointToggle && (
            <div className="space-y-1.5">
              <div className="text-[12px] text-[var(--color-ink-dim)]">
                Run on
                <span className="text-[var(--color-ink-faint)]"> · where SIQ-1 executes</span>
              </div>
              <div className="flex gap-1.5">
                {(["remote", "local"] as const).map((mode) => {
                  const on = (params.endpointMode ?? "remote") === mode;
                  return (
                    <button
                      key={mode}
                      onClick={() => onParams({ ...params, endpointMode: mode })}
                      className={`flex-1 rounded-lg border py-1.5 text-[12px] transition ${
                        on
                          ? "border-[var(--color-pi)] bg-[var(--color-pi)]/10 text-[var(--color-pi-2)]"
                          : "border-[var(--color-edge)] hover:border-[var(--color-edge-2)] text-[var(--color-ink-dim)]"
                      }`}
                    >
                      {mode === "remote" ? "Cloud · RunPod" : "Local server"}
                    </button>
                  );
                })}
              </div>
              {(params.endpointMode ?? "remote") === "local" && (
                <p className="text-[10.5px] text-[var(--color-ink-faint)] leading-snug">
                  Expects an OpenAI-compatible llama-server with the SIQ-1 GGUF at <code>localhost:8080</code>.
                </p>
              )}
            </div>
          )}
          {showReasoning && (
            <div className="space-y-3 pb-1">
              <div className="flex items-center justify-between">
                <span className="text-[12px] text-[var(--color-ink-dim)]">
                  Thinking
                  <span className="text-[var(--color-ink-faint)]"> · emit &lt;think&gt; before answering</span>
                </span>
                <button
                  role="switch"
                  aria-checked={params.thinking !== false}
                  onClick={() => onParams({ ...params, thinking: !(params.thinking !== false) })}
                  className={`relative w-9 h-5 rounded-full transition ${
                    params.thinking !== false ? "bg-[var(--color-pi)]" : "bg-[var(--color-panel-2)]"
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${
                      params.thinking !== false ? "left-[18px]" : "left-0.5"
                    }`}
                  />
                </button>
              </div>
              <div className={params.thinking !== false ? "" : "opacity-40 pointer-events-none"}>
                <div className="text-[12px] text-[var(--color-ink-dim)] mb-1.5">Reasoning effort</div>
                <div className="flex gap-1.5">
                  {EFFORTS.map((e) => {
                    const on = (params.effort ?? "medium") === e;
                    return (
                      <button
                        key={e}
                        onClick={() => onParams({ ...params, effort: e })}
                        className={`flex-1 rounded-lg border py-1.5 text-[12px] capitalize transition ${
                          on
                            ? "border-[var(--color-pi)] bg-[var(--color-pi)]/10 text-[var(--color-pi-2)]"
                            : "border-[var(--color-edge)] hover:border-[var(--color-edge-2)] text-[var(--color-ink-dim)]"
                        }`}
                      >
                        {e}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
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
            // the cloud 35B has plenty of room for long reasoning + a full file
            max={isRemote ? 32768 : 8192}
            step={256}
            fmt={(v) => String(v)}
            onChange={(v) => onParams({ ...params, maxTokens: v })}
          />
          {isRemote ? (
            // Context is fixed server-side for remote models — the slider would be
            // a no-op, so show the real window the SIQ-1 endpoint serves.
            <div className="flex items-center justify-between text-[12px]">
              <span className="text-[var(--color-ink-dim)]">
                Context window
                <span className="text-[var(--color-ink-faint)]"> · served by the endpoint</span>
              </span>
              <span className="font-[var(--font-mono)] text-[var(--color-pi-2)]">
                {remoteCtx >= 1024 ? `${Math.round(remoteCtx / 1024)}K` : remoteCtx}
              </span>
            </div>
          ) : (
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
          )}
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
