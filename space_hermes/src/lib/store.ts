import type { Project } from "../types";
import { DEFAULT_MODEL_ID } from "../config";

const KEY = "hermes.projects.v1";

export function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

/** Tiny typed localStorage helper used by all the agent stores. */
export function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota / private mode — ignore */
  }
}

export function loadProjects(): Project[] {
  const parsed = readJSON<Project[]>(KEY, []);
  return Array.isArray(parsed) ? parsed : [];
}

export function saveProjects(projects: Project[]): void {
  writeJSON(KEY, projects);
}

export function newProject(name?: string): Project {
  const now = Date.now();
  return {
    id: uid(),
    name: name?.trim() || "Untitled session",
    createdAt: now,
    updatedAt: now,
    modelId: DEFAULT_MODEL_ID,
    messages: [],
    artifacts: [],
  };
}
