import type { Project } from "../types";

export interface ArtifactRef {
  projectId: string;
  projectName: string;
  artifactId: string;
  title: string;
  ts: number;
}

export interface ArtifactHit extends ArtifactRef {
  snippet: string;
  score: number;
}

/** Flat list of every artifact across every session, newest first. */
export function listAllArtifacts(projects: Project[]): ArtifactRef[] {
  const out: ArtifactRef[] = [];
  for (const p of projects) {
    for (const a of p.artifacts) {
      out.push({ projectId: p.id, projectName: p.name, artifactId: a.id, title: a.title, ts: a.ts });
    }
  }
  return out.sort((a, b) => b.ts - a.ts);
}

/** Find an artifact by id along with its owning project, anywhere. */
export function findArtifact(projects: Project[], id: string) {
  for (const p of projects) {
    const a = p.artifacts.find((x) => x.id === id);
    if (a) return { projectId: p.id, projectName: p.name, artifact: a };
  }
  return null;
}

/** Token-overlap search over titles + HTML bodies. */
export function searchArtifacts(projects: Project[], query: string, limit = 8): ArtifactHit[] {
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
  if (terms.length === 0) return [];

  const hits: ArtifactHit[] = [];
  for (const p of projects) {
    for (const a of p.artifacts) {
      const hay = `${a.title}\n${a.html}`.toLowerCase();
      let score = 0;
      let firstIdx = -1;
      for (const t of terms) {
        const i = hay.indexOf(t);
        if (i >= 0) {
          score++;
          if (firstIdx < 0 || i < firstIdx) firstIdx = i;
        }
      }
      if (score === 0) continue;
      const body = a.html.replace(/\s+/g, " ");
      const idx = body.toLowerCase().indexOf(terms[0]);
      const start = Math.max(0, idx - 40);
      const snippet =
        `${a.title} — ` +
        (start > 0 ? "…" : "") +
        body.slice(start, start + 160).trim() +
        (start + 160 < body.length ? "…" : "");
      hits.push({
        projectId: p.id,
        projectName: p.name,
        artifactId: a.id,
        title: a.title,
        ts: a.ts,
        score,
        snippet,
      });
    }
  }
  return hits.sort((a, b) => b.score - a.score || b.ts - a.ts).slice(0, limit);
}
