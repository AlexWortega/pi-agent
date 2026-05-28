import { describe, it, expect } from "vitest";
import { saveMemory, listMemory, deleteMemory, searchMemory, memoryBlock } from "../src/lib/memoryStore";
import { listSkills, getSkill, upsertSkill, deleteSkill, relevantSkills } from "../src/lib/skillsStore";
import { listTodos, addTodo, toggleTodo, removeTodo, clearDone } from "../src/lib/todoStore";
import { listSchedules, addSchedule, removeSchedule, dueSchedules, parseEvery } from "../src/lib/scheduler";
import { getSettings, setSettings, toggleTool, getKeenableKey } from "../src/lib/settings";

describe("memoryStore — Honcho-style persistent user model", () => {
  it("saves and lists a memory", () => {
    const m = saveMemory("budget is 500");
    expect(m.text).toBe("budget is 500");
    expect(listMemory()).toHaveLength(1);
    expect(listMemory()[0].id).toBe(m.id);
  });

  it("dedupes by key (last write wins)", () => {
    saveMemory("first", "name");
    saveMemory("second", "name");
    const all = listMemory();
    expect(all).toHaveLength(1);
    expect(all[0].text).toBe("second");
    expect(all[0].key).toBe("name");
  });

  it("searches by content", () => {
    saveMemory("loves carrots and beets");
    saveMemory("hates pickles");
    const hits = searchMemory("carrots");
    expect(hits).toHaveLength(1);
    expect(hits[0].text).toContain("carrots");
  });

  it("searches by key too", () => {
    saveMemory("alpha", "secret-label");
    const hits = searchMemory("secret-label");
    expect(hits).toHaveLength(1);
  });

  it("deletes by id", () => {
    const m = saveMemory("temporary");
    deleteMemory(m.id);
    expect(listMemory()).toHaveLength(0);
  });

  it("memoryBlock renders most-recent N as bullets", () => {
    saveMemory("alpha", "k1");
    saveMemory("beta");
    const block = memoryBlock();
    expect(block).toContain("k1: alpha");
    expect(block).toContain("- beta");
  });
});

describe("skillsStore — agentskills.io style", () => {
  it("seeds bundled skills on first read", () => {
    const skills = listSkills();
    expect(skills.length).toBeGreaterThanOrEqual(2);
    expect(skills.some((s) => s.name === "concise-answers")).toBe(true);
    expect(skills.some((s) => s.name === "web-app-builder")).toBe(true);
  });

  it("seeds idempotently — second read doesn't double them", () => {
    const a = listSkills().length;
    const b = listSkills().length;
    expect(a).toBe(b);
  });

  it("upserts a new skill", () => {
    const s = upsertSkill({ name: "my-test", description: "t", body: "do X" });
    expect(getSkill("my-test")?.id).toBe(s.id);
  });

  it("upsert overwrites existing by name", () => {
    upsertSkill({ name: "x", body: "v1" });
    upsertSkill({ name: "x", body: "v2", description: "newer" });
    const s = getSkill("x");
    expect(s?.body).toBe("v2");
    expect(s?.description).toBe("newer");
    // ensure no duplicates
    expect(listSkills().filter((y) => y.name === "x")).toHaveLength(1);
  });

  it("deletes by name or id", () => {
    upsertSkill({ name: "tmp", body: "y" });
    expect(deleteSkill("tmp")).toBe(true);
    expect(getSkill("tmp")).toBeUndefined();
  });

  it("relevantSkills surfaces matches by token overlap", () => {
    upsertSkill({ name: "purple-elephant", body: "carry crates of bananas across the savannah" });
    upsertSkill({ name: "orange-zebra", body: "swim quietly through lakes" });
    const top = relevantSkills("bananas savannah", 5);
    expect(top.some((s) => s.name === "purple-elephant")).toBe(true);
  });
});

describe("todoStore — kanban", () => {
  it("adds and lists tasks", () => {
    addTodo("buy milk");
    expect(listTodos()).toHaveLength(1);
  });

  it("toggles done both directions", () => {
    const t = addTodo("ship code");
    toggleTodo(t.id);
    expect(listTodos()[0].done).toBe(true);
    toggleTodo(t.id);
    expect(listTodos()[0].done).toBe(false);
  });

  it("force-sets done state", () => {
    const t = addTodo("x");
    toggleTodo(t.id, true);
    toggleTodo(t.id, true); // idempotent
    expect(listTodos()[0].done).toBe(true);
  });

  it("removes by id", () => {
    const t = addTodo("x");
    removeTodo(t.id);
    expect(listTodos()).toHaveLength(0);
  });

  it("clearDone removes only completed", () => {
    const a = addTodo("a");
    addTodo("b");
    toggleTodo(a.id, true);
    clearDone();
    const after = listTodos();
    expect(after).toHaveLength(1);
    expect(after[0].text).toBe("b");
  });
});

describe("scheduler", () => {
  it("parseEvery handles several formats", () => {
    expect(parseEvery("30s")).toBe(30_000);
    expect(parseEvery("5m")).toBe(300_000);
    expect(parseEvery("1h")).toBe(3_600_000);
    expect(parseEvery("every 2m")).toBe(120_000);
    expect(parseEvery("every 90 seconds")).toBe(90_000);
    expect(parseEvery("30")).toBe(30 * 60_000); // default minutes
    expect(parseEvery("garbage")).toBeNull();
  });

  it("addSchedule + listSchedules", () => {
    addSchedule("check news", 60_000, "proj1");
    addSchedule("backup db", 600_000, "proj2");
    expect(listSchedules()).toHaveLength(2);
  });

  it("dueSchedules respects nextRun and advances it", () => {
    const s = addSchedule("ping", 60_000, "p1");
    const dueEarly = dueSchedules(s.nextRun - 1000);
    expect(dueEarly).toHaveLength(0);
    const dueOn = dueSchedules(s.nextRun + 1);
    expect(dueOn).toHaveLength(1);
    const after = listSchedules()[0];
    expect(after.nextRun).toBeGreaterThan(s.nextRun);
    expect(after.lastRun).toBeDefined();
  });

  it("removeSchedule", () => {
    const s = addSchedule("x", 1000, "p");
    removeSchedule(s.id);
    expect(listSchedules()).toHaveLength(0);
  });
});

describe("settings + getKeenableKey precedence", () => {
  it("returns defaults if nothing saved", () => {
    const s = getSettings();
    expect(s.webToolsEnabled).toBe(true);
    expect(s.disabledTools).toEqual([]);
    expect(s.readerProxy).toMatch(/^https:\/\//);
  });

  it("setSettings merges, doesn't replace", () => {
    setSettings({ webToolsEnabled: false });
    const s = getSettings();
    expect(s.webToolsEnabled).toBe(false);
    expect(s.readerProxy).toMatch(/^https:\/\//); // default preserved
  });

  it("toggleTool adds/removes from disabledTools", () => {
    toggleTool("calculator", false);
    expect(getSettings().disabledTools).toContain("calculator");
    toggleTool("calculator", true);
    expect(getSettings().disabledTools).not.toContain("calculator");
  });

  it("getKeenableKey: BYO wins over env (env is undefined in tests)", () => {
    expect(getKeenableKey()).toBe("");
    setSettings({ keenableApiKey: "keen_user_paste_abc" });
    expect(getKeenableKey()).toBe("keen_user_paste_abc");
  });
});
