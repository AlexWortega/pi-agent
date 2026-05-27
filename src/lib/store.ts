import type { Project } from "../types";
import { DEFAULT_MODEL_ID } from "../config";

const KEY = "piagent.projects.v1";

export function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

export function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Project[];
    return Array.isArray(parsed) ? parsed : [];
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
