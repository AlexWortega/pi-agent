import { MODEL_PRESETS, ggufUrl } from "../config";

export interface ResolvedModel {
  id: string;
  label: string;
  url: string;
  accent: string;
  note: string;
  verified: boolean;
  sizeLabel: string;
}

/** A model id is either a preset id, or "url:<full gguf url>". */
export function resolveModel(id: string): ResolvedModel {
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
    url: ggufUrl(p.repo, p.file),
    accent: p.accent || "#7c5cff",
    note: p.note,
    verified: p.verified,
    sizeLabel: p.sizeLabel,
  };
}

export function customModelId(url: string): string {
  return "url:" + url.trim();
}
