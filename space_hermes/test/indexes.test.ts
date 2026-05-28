import { describe, it, expect } from "vitest";
import type { Project } from "../src/types";
import { searchSessions } from "../src/lib/sessionIndex";
import { searchArtifacts, listAllArtifacts, findArtifact } from "../src/lib/artifactIndex";

const projects: Project[] = [
  {
    id: "p1",
    name: "alpha session",
    createdAt: 0,
    updatedAt: 0,
    modelId: "soyuz-4b",
    messages: [
      { id: "m1", role: "user", content: "tell me about hermes agent", ts: 1 },
      { id: "m2", role: "assistant", content: "Hermes is an AI agent built by Nous Research", ts: 2 },
      { id: "m3", role: "user", content: "ok now build me a synthwave pomodoro timer", ts: 3 },
    ],
    artifacts: [
      { id: "a1", title: "Pomodoro Timer", html: "<html><body>tomato 25 minute focus timer with neon</body></html>", ts: 10 },
      { id: "a2", title: "Notes App", html: "<html><body>markdown notes saved to localStorage</body></html>", ts: 20 },
    ],
  },
  {
    id: "p2",
    name: "beta session",
    createdAt: 0,
    updatedAt: 0,
    modelId: "soyuz-4b",
    messages: [{ id: "m4", role: "user", content: "what is markdown", ts: 5 }],
    artifacts: [],
  },
];

describe("sessionIndex — FTS over saved chat messages", () => {
  it("ranks by term overlap", () => {
    const hits = searchSessions(projects, "hermes agent");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].score).toBeGreaterThanOrEqual(2);
  });

  it("returns [] for empty query", () => {
    expect(searchSessions(projects, "")).toEqual([]);
  });

  it("finds across multiple sessions", () => {
    const hits = searchSessions(projects, "markdown");
    // appears in p1 (artifact-related message) and p2 — but our search is on
    // messages only, so p2 should be present
    expect(hits.some((h) => h.projectId === "p2")).toBe(true);
  });

  it("returns role + snippet for each hit", () => {
    const [first] = searchSessions(projects, "pomodoro");
    expect(first.role).toBe("user");
    expect(first.snippet).toContain("pomodoro");
  });
});

describe("artifactIndex — FTS over generated HTML apps", () => {
  it("listAllArtifacts returns newest first", () => {
    const list = listAllArtifacts(projects);
    expect(list).toHaveLength(2);
    expect(list[0].artifactId).toBe("a2"); // ts=20 newer
    expect(list[1].artifactId).toBe("a1");
  });

  it("findArtifact by id", () => {
    const f = findArtifact(projects, "a1");
    expect(f?.artifact.title).toBe("Pomodoro Timer");
    expect(f?.projectName).toBe("alpha session");
  });

  it("findArtifact returns null when missing", () => {
    expect(findArtifact(projects, "does-not-exist")).toBeNull();
  });

  it("searchArtifacts matches title", () => {
    const hits = searchArtifacts(projects, "Pomodoro");
    expect(hits).toHaveLength(1);
    expect(hits[0].artifactId).toBe("a1");
  });

  it("searchArtifacts matches HTML body content", () => {
    const hits = searchArtifacts(projects, "neon tomato");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].artifactId).toBe("a1");
  });

  it("snippet includes the title prefix", () => {
    const [h] = searchArtifacts(projects, "markdown");
    expect(h.snippet).toContain("Notes App");
  });

  it("returns [] for empty query", () => {
    expect(searchArtifacts(projects, "")).toEqual([]);
  });
});
