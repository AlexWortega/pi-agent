import type { Skill } from "../types";
import { readJSON, writeJSON, uid } from "./store";

const KEY = "hermes.skills.v1";
const SEEDED = "hermes.skills.seeded.v1";
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}
export function subscribeSkills(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

const DEFAULT_SKILLS: Omit<Skill, "id" | "ts">[] = [
  {
    name: "concise-answers",
    description: "Answer briefly and directly, with bullet points when listing.",
    body: "When responding:\n- Lead with the answer, then the why.\n- Prefer short paragraphs and bullet lists.\n- Skip filler like \"Certainly!\" or \"As an AI\".",
    createdBy: "agent",
  },
  {
    name: "web-app-builder",
    description: "Build polished single-file HTML apps with the render_html tool.",
    body: "When asked for an app, page, game or visual tool:\n1. Plan the UI briefly.\n2. Call render_html with ONE self-contained HTML file (inline CSS+JS, responsive, dark-friendly).\n3. Avoid external files; use a CDN only if strictly necessary.",
    createdBy: "agent",
  },
];

function seedIfNeeded(): void {
  try {
    if (localStorage.getItem(SEEDED)) return;
    const now = Date.now();
    const skills: Skill[] = DEFAULT_SKILLS.map((s, i) => ({ ...s, id: uid(), ts: now + i }));
    writeJSON(KEY, skills);
    localStorage.setItem(SEEDED, "1");
  } catch {
    /* ignore */
  }
}

export function listSkills(): Skill[] {
  seedIfNeeded();
  return readJSON<Skill[]>(KEY, []);
}

export function getSkill(nameOrId: string): Skill | undefined {
  const q = nameOrId.toLowerCase();
  return listSkills().find((s) => s.id === nameOrId || s.name.toLowerCase() === q);
}

export function upsertSkill(input: { name: string; description?: string; body: string; createdBy?: "user" | "agent" }): Skill {
  const skills = listSkills();
  const existing = skills.find((s) => s.name.toLowerCase() === input.name.toLowerCase());
  if (existing) {
    existing.description = input.description ?? existing.description;
    existing.body = input.body;
    existing.ts = Date.now();
    writeJSON(KEY, skills);
    emit();
    return existing;
  }
  const skill: Skill = {
    id: uid(),
    name: input.name,
    description: input.description ?? "",
    body: input.body,
    createdBy: input.createdBy ?? "user",
    ts: Date.now(),
  };
  skills.push(skill);
  writeJSON(KEY, skills);
  emit();
  return skill;
}

export function deleteSkill(nameOrId: string): boolean {
  const skills = listSkills();
  const next = skills.filter((s) => s.id !== nameOrId && s.name.toLowerCase() !== nameOrId.toLowerCase());
  const changed = next.length !== skills.length;
  if (changed) {
    writeJSON(KEY, next);
    emit();
  }
  return changed;
}

/** Skills relevant to the user's message, injected as user messages at loop start. */
export function relevantSkills(query: string, limit = 3): Skill[] {
  const skills = listSkills();
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
  const scored = skills.map((s) => {
    const hay = `${s.name} ${s.description} ${s.body}`.toLowerCase();
    const score = terms.reduce((acc, t) => (hay.includes(t) ? acc + 1 : acc), 0);
    return { s, score };
  });
  // always include the always-on basics, then top-scored
  const picked = scored.sort((a, b) => b.score - a.score).slice(0, limit).map((x) => x.s);
  return picked;
}
