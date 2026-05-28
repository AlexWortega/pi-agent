import { describe, it, expect } from "vitest";
import { buildPrompt, parseToolCalls, visibleText, assistantWire, toolResponseWire } from "../src/agent/chatml";
import { splitThink, stripToolCalls, extractHtmlArtifact } from "../src/lib/parse";

describe("buildPrompt — ChatML serialisation", () => {
  it("formats turns and appends assistant prefix", () => {
    const p = buildPrompt([
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ]);
    expect(p).toBe(
      "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    );
  });

  it("supports a 'tool' role turn (Hermes-specific)", () => {
    const p = buildPrompt([
      { role: "assistant", content: '<tool_call>{"name":"x"}</tool_call>' },
      { role: "tool", content: '<tool_response>{"name":"x","content":"ok"}</tool_response>' },
    ]);
    expect(p).toContain("<|im_start|>tool\n");
    expect(p).toContain("<tool_response>");
  });
});

describe("parseToolCalls — Hermes function-calling parser", () => {
  it("parses one well-formed call", () => {
    const c = parseToolCalls('<tool_call>{"name":"calculator","arguments":{"x":1}}</tool_call>');
    expect(c).toHaveLength(1);
    expect(c[0].name).toBe("calculator");
    expect(c[0].arguments).toEqual({ x: 1 });
    expect(c[0].id).toBeDefined();
  });

  it("recovers from trailing comma", () => {
    const c = parseToolCalls('<tool_call>{"name":"a","arguments":{},}</tool_call>');
    expect(c).toHaveLength(1);
    expect(c[0].name).toBe("a");
  });

  it("handles streaming: missing close tag", () => {
    const c = parseToolCalls('<tool_call>{"name":"todo","arguments":{"action":"list"}}');
    expect(c).toHaveLength(1);
    expect(c[0].name).toBe("todo");
  });

  it("parses multiple sequential calls", () => {
    const c = parseToolCalls(
      '<tool_call>{"name":"a","arguments":{}}</tool_call> then <tool_call>{"name":"b","arguments":{"x":2}}</tool_call>',
    );
    expect(c).toHaveLength(2);
    expect(c.map((x) => x.name)).toEqual(["a", "b"]);
    expect(c[1].arguments).toEqual({ x: 2 });
  });

  it("returns [] for plain prose", () => {
    expect(parseToolCalls("just a normal answer with no call")).toEqual([]);
  });

  it("ignores garbage inside the tags", () => {
    expect(parseToolCalls("<tool_call>this is not json</tool_call>")).toEqual([]);
  });

  it("defaults arguments to {} when missing", () => {
    const c = parseToolCalls('<tool_call>{"name":"now"}</tool_call>');
    expect(c).toHaveLength(1);
    expect(c[0].arguments).toEqual({});
  });
});

describe("visibleText", () => {
  it("strips both think and tool_call, leaving prose", () => {
    const t = visibleText('hi <think>secret</think> answer <tool_call>{"name":"x"}</tool_call>');
    expect(t.think).toBe("secret");
    expect(t.visible).toBe("hi  answer");
  });

  it("captures streaming <think> with no close", () => {
    const t = visibleText("ok <think>thinking out loud right now");
    expect(t.think).toContain("thinking out loud");
  });

  it("also strips <scratch_pad>", () => {
    const t = visibleText("<scratch_pad>plan</scratch_pad>final answer");
    expect(t.think).toBe("plan");
    expect(t.visible).toBe("final answer");
  });
});

describe("assistantWire — replay format", () => {
  it("includes prose then each tool_call as a JSON block", () => {
    const w = assistantWire("hello world", [
      { id: "tc1", name: "calc", arguments: { e: "2+2" } },
      { id: "tc2", name: "now", arguments: {} },
    ]);
    expect(w).toContain("hello world");
    expect(w).toContain("<tool_call>");
    expect(w).toContain('"name":"calc"');
    expect(w).toContain('"name":"now"');
    expect((w.match(/<\/tool_call>/g) || []).length).toBe(2);
  });

  it("works with no prose (calls only)", () => {
    const w = assistantWire("", [{ id: "1", name: "x", arguments: {} }]);
    expect(w).toContain("<tool_call>");
  });
});

describe("toolResponseWire", () => {
  it("wraps ok result", () => {
    const w = toolResponseWire({ id: "1", name: "x", ok: true, content: { v: 42 } });
    expect(w).toContain("<tool_response>");
    expect(w).toContain('"name":"x"');
    expect(w).toContain('"content":{"v":42}');
  });

  it("wraps an error as {error: msg}", () => {
    const w = toolResponseWire({ id: "1", name: "x", ok: false, content: "boom" });
    expect(w).toContain('"error":"boom"');
  });
});

describe("parse.ts — reasoning + artifact extraction", () => {
  it("splitThink extracts closed <think>", () => {
    const r = splitThink("<think>plan</think>answer");
    expect(r.think).toBe("plan");
    expect(r.answer).toBe("answer");
  });

  it("splitThink handles streaming open", () => {
    const r = splitThink("ok <think>still in progress");
    expect(r.think).toContain("still in progress");
  });

  it("stripToolCalls removes complete and unterminated blocks", () => {
    expect(stripToolCalls('a<tool_call>{}</tool_call>b')).toBe("ab");
    expect(stripToolCalls('a <tool_call>{stream...').trim()).toBe("a");
  });

  it("extractHtmlArtifact pulls last fenced html block", () => {
    const md = "```html\n<html><title>Foo</title><body>hi</body></html>\n```";
    const r = extractHtmlArtifact(md);
    expect(r?.title).toBe("Foo");
    expect(r?.html).toContain("<title>Foo</title>");
  });

  it("extractHtmlArtifact handles streaming open block", () => {
    const r = extractHtmlArtifact("intro\n```html\n<html><h1>Bar</h1>");
    expect(r?.title).toBe("Bar");
  });

  it("extractHtmlArtifact returns null on plain prose", () => {
    expect(extractHtmlArtifact("text only — no fence here")).toBeNull();
  });
});
