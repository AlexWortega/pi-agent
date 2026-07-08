import { CLOUD_PRESETS, MODEL_PRESETS, OPENROUTER_API, OPENROUTER_KEY_STORAGE, ggufUrl } from "../config";
import type { RemoteModel } from "../types";

export interface ResolvedModel {
  id: string;
  label: string;
  /** GGUF download url — empty for remote models. */
  url: string;
  /** When set, inference runs against this OpenAI-compatible endpoint. */
  remote?: RemoteModel;
  accent: string;
  note: string;
  verified: boolean;
  sizeLabel: string;
}

/** The user's OpenRouter key, if they pasted one in the model picker. */
export function getOpenRouterKey(): string {
  try {
    return (localStorage.getItem(OPENROUTER_KEY_STORAGE) ?? "").trim();
  } catch {
    return "";
  }
}

/**
 * A model id is a preset id, "url:<full gguf url>" for a custom local GGUF, or
 * "or:<openrouter model id>" for a frontier model via OpenRouter (BYO key).
 */
export function resolveModel(id: string): ResolvedModel {
  if (id.startsWith("or:")) {
    const cloudId = id.slice(3);
    const preset = CLOUD_PRESETS.find((c) => c.id === cloudId);
    return {
      id,
      label: preset?.label ?? cloudId,
      url: "",
      remote: {
        endpoint: OPENROUTER_API,
        model: cloudId,
        // Read fresh on every resolve so pasting a key takes effect immediately.
        apiKey: getOpenRouterKey() || undefined,
        contextWindow: preset?.contextWindow ?? 128_000,
      },
      accent: preset?.accent ?? "#4aa8ff",
      note: preset?.note ?? "Custom model via OpenRouter.",
      verified: !!preset,
      sizeLabel: "cloud · your key",
    };
  }
  if (id.startsWith("url:")) {
    const url = id.slice(4);
    const file = url.split("/").pop() || url;
    return {
      id,
      label: file.replace(/\.gguf$/i, ""),
      url,
      accent: "#2fe6b0",
      note: "Custom GGUF from a URL.",
      verified: false,
      sizeLabel: "custom",
    };
  }
  const p = MODEL_PRESETS.find((m) => m.id === id) ?? MODEL_PRESETS[0];
  return {
    id: p.id,
    label: p.label,
    url: p.remote ? "" : ggufUrl(p.repo!, p.file!),
    remote: p.remote,
    accent: p.accent || "#7c5cff",
    note: p.note,
    verified: p.verified,
    sizeLabel: p.sizeLabel,
  };
}

/** True when the model runs server-side (no WebGPU / download needed). */
export function isRemote(m: ResolvedModel): boolean {
  return !!m.remote;
}

/** True when the model is a frontier model reached through OpenRouter (BYO key). */
export function isOpenRouter(m: ResolvedModel): boolean {
  return m.id.startsWith("or:");
}

export function customModelId(url: string): string {
  return "url:" + url.trim();
}

export function cloudModelId(openrouterId: string): string {
  return "or:" + openrouterId.trim();
}
