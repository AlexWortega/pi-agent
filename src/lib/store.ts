import type { Project } from "../types";
import { DEFAULT_MODEL_ID } from "../config";

const KEY = "piagent.projects.v1";
// One-time migration flag: older sessions saved projects pinned to the in-browser
// "soyuz-4b" model (the previous default, needs a 2.5GB download + WebGPU). The
// default is now the cloud SIQ-1-35B, so move existing projects onto it once.
const MIGRATED_KEY = "piagent.migrated.cloud-default.v1";

export function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Project[];
    if (!Array.isArray(parsed)) return [];
    if (!localStorage.getItem(MIGRATED_KEY)) {
      let changed = false;
      for (const p of parsed) {
        if (p && p.modelId === "soyuz-4b") {
          p.modelId = DEFAULT_MODEL_ID;
          changed = true;
        }
      }
      localStorage.setItem(MIGRATED_KEY, "1");
      if (changed) saveProjects(parsed);
    }
    return parsed;
  } catch {
    return [];
  }
}

export function saveProjects(projects: Project[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(projects));
  } catch {
    /* quota / private mode — ignore */
  }
}

export function newProject(name?: string): Project {
  const now = Date.now();
  return {
    id: uid(),
    name: name?.trim() || "Untitled project",
    createdAt: now,
    updatedAt: now,
    modelId: DEFAULT_MODEL_ID,
    messages: [],
    artifacts: [],
  };
}
