import { describe, it, expect } from "vitest";
import type { ToolContext } from "../src/agent/registry";
import { getSettings } from "../src/lib/settings";

import { calculatorTool } from "../src/agent/tools/calculator";
import { datetimeTool } from "../src/agent/tools/datetime";
import { memoryTool } from "../src/agent/tools/memory";
import { todoTool } from "../src/agent/tools/todo";
import { skillsListTool, skillViewTool, skillManageTool } from "../src/agent/tools/skills";
import { renderHtmlTool } from "../src/agent/tools/renderHtml";
import { sessionSearchTool } from "../src/agent/tools/sessionSearch";
import { artifactSearchTool } from "../src/agent/tools/artifactSearch";
import { artifactListTool } from "../src/agent/tools/artifactList";
import { artifactViewTool } from "../src/agent/tools/artifactView";
import { artifactUpdateTool } from "../src/agent/tools/artifactUpdate";
import { artifactOpenTool } from "../src/agent/tools/artifactOpen";
import { clarifyTool } from "../src/agent/tools/clarify";
import { delegateTaskTool } from "../src/agent/tools/delegateTask";

import { listMemory } from "../src/lib/memoryStore";
import { listTodos } from "../src/lib/todoStore";
import { listSkills, getSkill } from "../src/lib/skillsStore";

function ctx(overrides: Partial<ToolContext> = {}): ToolContext {
  const ac = new AbortController();
  const seenArtifacts: { title: string; html: string; id: string }[] = [];
  return {
    signal: ac.signal,
    depth: 0,
    settings: getSettings(),
    requestClarify: async () => "yes",
    emitArtifact: (title, html) => {
      const id = "art-" + Math.random().toString(36).slice(2, 10);
      seenArtifacts.push({ title, html, id });
      return id;
    },
    runSubagent: async () => "subagent summary",
    searchSessions: () => [],
    listArtifacts: () => [],
    searchArtifacts: () => [],
    getArtifact: () => null,
    updateArtifact: () => false,
    focusArtifact: () => false,
    ...overrides,
  };
}

describe("calculator tool", () => {
  it("evaluates a simple expression", async () => {
    const r: any = await calculatorTool.run({ expression: "840 * 0.125" }, ctx());
    expect(r.result).toBe(105);
  });

  it("respects precedence + parens", async () => {
    const r: any = await calculatorTool.run({ expression: "(1+2)*3" }, ctx());
    expect(r.result).toBe(9);
  });

  it("supports unary minus + ^", async () => {
    const r: any = await calculatorTool.run({ expression: "-3 + 2^3" }, ctx());
    expect(r.result).toBe(5);
  });

  it("knows functions and constants", async () => {
    const r: any = await calculatorTool.run({ expression: "sqrt(2) * pi" }, ctx());
    expect(r.result).toBeCloseTo(Math.SQRT2 * Math.PI);
  });

  it("throws on garbage", async () => {
    await expect(calculatorTool.run({ expression: "((" }, ctx())).rejects.toThrow();
  });
});

describe("datetime tool", () => {
  it("returns iso + local + tz", async () => {
    const r: any = await datetimeTool.run({}, ctx());
    expect(typeof r.iso).toBe("string");
    expect(typeof r.local).toBe("string");
    expect(typeof r.timezone).toBe("string");
    expect(typeof r.weekday).toBe("string");
  });

  it("honours a passed timezone", async () => {
    const r: any = await datetimeTool.run({ timezone: "Asia/Tokyo" }, ctx());
    expect(r.timezone).toBe("Asia/Tokyo");
  });
});

describe("memory tool", () => {
  it("save → list returns it", async () => {
    const s: any = await memoryTool.run({ action: "save", text: "budget 500" }, ctx());
    expect(s.saved).toBeDefined();
    const l: any = await memoryTool.run({ action: "list" }, ctx());
    expect(l.memories).toHaveLength(1);
    expect(l.memories[0].text).toBe("budget 500");
  });

  it("save with key + recall finds it", async () => {
    await memoryTool.run({ action: "save", text: "carrot lover", key: "diet" }, ctx());
    const r: any = await memoryTool.run({ action: "recall", query: "carrot" }, ctx());
    expect(r.results).toHaveLength(1);
    expect(r.results[0].key).toBe("diet");
  });

  it("forget removes by id", async () => {
    const s: any = await memoryTool.run({ action: "save", text: "tmp" }, ctx());
    await memoryTool.run({ action: "forget", id: s.saved }, ctx());
    expect(listMemory()).toHaveLength(0);
  });

  it("rejects unknown action", async () => {
    await expect(memoryTool.run({ action: "explode" }, ctx())).rejects.toThrow(/unknown action/);
  });
});

describe("todo tool", () => {
  it("add + list", async () => {
    await todoTool.run({ action: "add", text: "ship it" }, ctx());
    const r: any = await todoTool.run({ action: "list" }, ctx());
    expect(r.tasks).toHaveLength(1);
    expect(r.tasks[0].text).toBe("ship it");
  });

  it("done flips state", async () => {
    const s: any = await todoTool.run({ action: "add", text: "x" }, ctx());
    await todoTool.run({ action: "done", id: s.added }, ctx());
    expect(listTodos()[0].done).toBe(true);
  });

  it("remove deletes", async () => {
    const s: any = await todoTool.run({ action: "add", text: "x" }, ctx());
    await todoTool.run({ action: "remove", id: s.added }, ctx());
    expect(listTodos()).toHaveLength(0);
  });

  it("clear_done removes completed", async () => {
    const a: any = await todoTool.run({ action: "add", text: "a" }, ctx());
    await todoTool.run({ action: "add", text: "b" }, ctx());
    await todoTool.run({ action: "done", id: a.added }, ctx());
    await todoTool.run({ action: "clear_done" }, ctx());
    expect(listTodos()).toHaveLength(1);
    expect(listTodos()[0].text).toBe("b");
  });
});

describe("skills tools", () => {
  it("skills_list shows seeded + custom", async () => {
    const r: any = await skillsListTool.run({}, ctx());
    expect(r.skills.length).toBeGreaterThanOrEqual(2);
  });

  it("skill_manage save creates new", async () => {
    const s: any = await skillManageTool.run({ action: "save", name: "css-tips", description: "d", body: "use grid" }, ctx());
    expect(s.saved).toBe("css-tips");
    expect(getSkill("css-tips")?.body).toBe("use grid");
  });

  it("skill_view returns body", async () => {
    await skillManageTool.run({ action: "save", name: "ruby-tips", body: "use blocks" }, ctx());
    const v: any = await skillViewTool.run({ name: "ruby-tips" }, ctx());
    expect(v.body).toBe("use blocks");
  });

  it("skill_view throws on missing", async () => {
    await expect(skillViewTool.run({ name: "does-not-exist" }, ctx())).rejects.toThrow(/no skill/);
  });

  it("skill_manage delete removes", async () => {
    await skillManageTool.run({ action: "save", name: "tmp-skill", body: "x" }, ctx());
    await skillManageTool.run({ action: "delete", name: "tmp-skill" }, ctx());
    expect(getSkill("tmp-skill")).toBeUndefined();
  });
});

describe("renderHtml tool", () => {
  it("calls emitArtifact with the html and returns the id", async () => {
    let captured: { title: string; html: string } | null = null;
    const c = ctx({
      emitArtifact: (title, html) => {
        captured = { title, html };
        return "art-stub-1";
      },
    });
    const r: any = await renderHtmlTool.run({ title: "App", html: "<html><body>hi</body></html>" }, c);
    expect(r.rendered).toBe(true);
    expect(r.artifactId).toBe("art-stub-1");
    expect(captured).toBeTruthy();
    expect(captured!.title).toBe("App");
  });

  it("rejects non-markup html", async () => {
    await expect(renderHtmlTool.run({ html: "just text" }, ctx())).rejects.toThrow(/markup/);
  });

  it("derives a title from <title> tag if not provided", async () => {
    let title = "";
    const c = ctx({
      emitArtifact: (t) => {
        title = t;
        return "x";
      },
    });
    await renderHtmlTool.run({ html: "<html><title>Inferred</title><body>x</body></html>" }, c);
    expect(title).toBe("Inferred");
  });
});

describe("session_search tool wraps ctx.searchSessions", () => {
  it("passes through to ctx.searchSessions", async () => {
    let called: { q: string; limit: number | undefined } | null = null;
    const c = ctx({
      searchSessions: (q, limit) => {
        called = { q, limit };
        return [{ projectId: "p", projectName: "P", role: "user", snippet: "...x...", ts: 0, score: 1 }];
      },
    });
    const r: any = await sessionSearchTool.run({ query: "x", limit: 3 }, c);
    expect(called).toEqual({ q: "x", limit: 3 });
    expect(r.hits).toHaveLength(1);
  });
});

describe("artifact_* tools", () => {
  it("artifact_search wraps ctx.searchArtifacts", async () => {
    const c = ctx({
      searchArtifacts: () => [
        { projectId: "p", projectName: "P", artifactId: "a1", title: "T", snippet: "...", ts: 0, score: 2 },
      ],
    });
    const r: any = await artifactSearchTool.run({ query: "x" }, c);
    expect(r.hits[0].id).toBe("a1");
    expect(r.hits[0].title).toBe("T");
  });

  it("artifact_list wraps ctx.listArtifacts", async () => {
    const c = ctx({
      listArtifacts: () => [{ projectId: "p", projectName: "P", artifactId: "a", title: "T", ts: 9 }],
    });
    const r: any = await artifactListTool.run({}, c);
    expect(r.artifacts).toHaveLength(1);
    expect(r.artifacts[0].id).toBe("a");
  });

  it("artifact_view returns html", async () => {
    const c = ctx({ getArtifact: () => ({ title: "T", html: "<html></html>" }) });
    const r: any = await artifactViewTool.run({ artifactId: "a1" }, c);
    expect(r.html).toBe("<html></html>");
  });

  it("artifact_view throws on unknown id", async () => {
    const c = ctx({ getArtifact: () => null });
    await expect(artifactViewTool.run({ artifactId: "missing" }, c)).rejects.toThrow(/no artifact/);
  });

  it("artifact_update succeeds when ctx.updateArtifact returns true", async () => {
    let called = false;
    const c = ctx({
      updateArtifact: (_id, _html, _title) => {
        called = true;
        return true;
      },
    });
    const r: any = await artifactUpdateTool.run({ artifactId: "a1", html: "<div>new</div>" }, c);
    expect(called).toBe(true);
    expect(r.updated).toBe(true);
  });

  it("artifact_update throws on unknown id", async () => {
    const c = ctx({ updateArtifact: () => false });
    await expect(artifactUpdateTool.run({ artifactId: "x", html: "<p>y</p>" }, c)).rejects.toThrow(/no artifact/);
  });

  it("artifact_open wraps ctx.focusArtifact", async () => {
    let focused = "";
    const c = ctx({
      focusArtifact: (id) => {
        focused = id;
        return true;
      },
    });
    const r: any = await artifactOpenTool.run({ artifactId: "a1" }, c);
    expect(focused).toBe("a1");
    expect(r.focused).toBe("a1");
  });
});

describe("clarify tool", () => {
  it("resolves with the user's answer via ctx.requestClarify", async () => {
    const c = ctx({ requestClarify: async (_q, opts) => opts?.[0] ?? "free-form" });
    const r: any = await clarifyTool.run({ question: "Which?", options: ["A", "B"] }, c);
    expect(r.answer).toBe("A");
  });
});

describe("delegate_task tool", () => {
  it("delegates via ctx.runSubagent", async () => {
    const c = ctx({ runSubagent: async (task) => `did: ${task}` });
    const r: any = await delegateTaskTool.run({ task: "scrape feeds" }, c);
    expect(r.summary).toBe("did: scrape feeds");
  });
});

describe("tool schema sanity (Hermes <tools> payload requires {name, description, parameters})", () => {
  it("every tool exposes the required fields", async () => {
    const tools = [
      calculatorTool,
      datetimeTool,
      memoryTool,
      todoTool,
      skillsListTool,
      skillViewTool,
      skillManageTool,
      renderHtmlTool,
      sessionSearchTool,
      artifactSearchTool,
      artifactListTool,
      artifactViewTool,
      artifactUpdateTool,
      artifactOpenTool,
      clarifyTool,
      delegateTaskTool,
    ];
    for (const t of tools) {
      expect(typeof t.name).toBe("string");
      expect(t.name.length).toBeGreaterThan(0);
      expect(typeof t.description).toBe("string");
      expect(t.parameters.type).toBe("object");
      expect(typeof t.parameters.properties).toBe("object");
    }
  });
});
