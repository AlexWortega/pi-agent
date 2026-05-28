import { readJSON, writeJSON } from "./store";
import { READER_PROXY } from "../config";

const KEY = "hermes.settings.v1";
const listeners = new Set<() => void>();

export interface AgentSettings {
  /** allow tools that hit the network (web_search, web_extract) */
  webToolsEnabled: boolean;
  /** CORS reader-proxy prefix for web_extract */
  readerProxy: string;
  /** disabled tool names (everything else is on) */
  disabledTools: string[];
  /** keenable.ai API key (runtime override; build-time fallback in env) */
  keenableApiKey?: string;
}

const DEFAULTS: AgentSettings = {
  webToolsEnabled: true,
  readerProxy: READER_PROXY,
  disabledTools: [],
};

/** Resolve the keenable key: user-pasted (localStorage) first, then build-time env. */
export function getKeenableKey(): string {
  return (
    getSettings().keenableApiKey?.trim() ||
    (import.meta.env.VITE_KEENABLE_API_KEY as string | undefined)?.trim() ||
    ""
  );
}

function emit() {
  listeners.forEach((l) => l());
}
export function subscribeSettings(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function getSettings(): AgentSettings {
  return { ...DEFAULTS, ...readJSON<Partial<AgentSettings>>(KEY, {}) };
}

export function setSettings(patch: Partial<AgentSettings>): void {
  writeJSON(KEY, { ...getSettings(), ...patch });
  emit();
}

export function toggleTool(name: string, enabled: boolean): void {
  const s = getSettings();
  const set = new Set(s.disabledTools);
  if (enabled) set.delete(name);
  else set.add(name);
  setSettings({ disabledTools: [...set] });
}
