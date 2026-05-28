import type { Project } from "../types";

export interface SessionHit {
  projectId: string;
  projectName: string;
  role: string;
  snippet: string;
  ts: number;
  score: number;
}

/** Lightweight full-text search over every message in every session
 *  (the browser stand-in for Hermes' session_search FTS). */
export function searchSessions(projects: Project[], query: string, limit = 8): SessionHit[] {
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
  if (terms.length === 0) return [];

  const hits: SessionHit[] = [];
  for (const p of projects) {
    for (const m of p.messages) {
      const text = (m.content || "").toLowerCase();
      if (!text) continue;
      const score = terms.reduce((s, t) => (text.includes(t) ? s + 1 : s), 0);
      if (score === 0) continue;
      const idx = text.indexOf(terms[0]);
      const start = Math.max(0, idx - 40);
      const snippet =
        (start > 0 ? "…" : "") + (m.content || "").slice(start, start + 160).trim() + "…";
      hits.push({ projectId: p.id, projectName: p.name, role: m.role, snippet, ts: m.ts, score });
    }
  }
  return hits.sort((a, b) => b.score - a.score || b.ts - a.ts).slice(0, limit);
}
