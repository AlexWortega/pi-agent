import { describe, it, expect } from "vitest";
import { parseOutput } from "./parse";

describe("parseOutput", () => {
  it("returns plain prose untouched", () => {
    const r = parseOutput("Hello there, here is your answer.");
    expect(r.text).toBe("Hello there, here is your answer.");
    expect(r.thinking).toBe("");
    expect(r.toolCalls).toEqual([]);
    expect(r.open).toBeNull();
  });

  it("separates a closed think block from prose", () => {
    const r = parseOutput("<think>plan the steps</think>Now I will do it.");
    expect(r.thinking).toBe("plan the steps");
    expect(r.text).toBe("Now I will do it.");
    expect(r.open).toBeNull();
  });

  it("reports an unterminated think block as open (streaming)", () => {
    const r = parseOutput("<think>still reasoning");
    expect(r.thinking).toBe("still reasoning");
    expect(r.open).toBe("think");
  });

  it("parses a single tool call", () => {
    const r = parseOutput(
      '<tool_call>\n{"name": "write", "arguments": {"path": "/x.txt", "content": "hi"}}\n</tool_call>',
    );
    expect(r.toolCalls).toHaveLength(1);
    expect(r.toolCalls[0].name).toBe("write");
    expect(r.toolCalls[0].arguments).toEqual({ path: "/x.txt", content: "hi" });
    expect(r.toolCalls[0].malformed).toBe(false);
  });

  it("parses multiple tool calls with surrounding prose", () => {
    const r = parseOutput(
      'Let me set up the files.\n' +
        '<tool_call>{"name":"write","arguments":{"path":"/a"}}</tool_call>\n' +
        '<tool_call>{"name":"read","arguments":{"path":"/b"}}</tool_call>',
    );
    expect(r.toolCalls.map((t) => t.name)).toEqual(["write", "read"]);
    expect(r.text).toContain("Let me set up the files.");
  });

  it("withholds an in-progress tool_call from text and flags it open", () => {
    const r = parseOutput('Working...\n<tool_call>{"name":"write","argum');
    expect(r.open).toBe("tool_call");
    expect(r.text).toContain("Working...");
    expect(r.text).not.toContain("tool_call");
    expect(r.toolCalls).toHaveLength(0);
  });

  it("tolerates code-fenced JSON inside a tool call", () => {
    const r = parseOutput('<tool_call>\n```json\n{"name":"read","arguments":{"path":"/p"}}\n```\n</tool_call>');
    expect(r.toolCalls[0].name).toBe("read");
    expect(r.toolCalls[0].arguments).toEqual({ path: "/p" });
    expect(r.toolCalls[0].malformed).toBe(false);
  });

  it("tolerates trailing commas", () => {
    const r = parseOutput('<tool_call>{"name":"read","arguments":{"path":"/p",},}</tool_call>');
    expect(r.toolCalls[0].malformed).toBe(false);
    expect(r.toolCalls[0].arguments).toEqual({ path: "/p" });
  });

  it("flags malformed JSON without throwing", () => {
    const r = parseOutput("<tool_call>not json at all</tool_call>");
    expect(r.toolCalls[0].malformed).toBe(true);
  });

  it("handles think + prose + tool call together", () => {
    const r = parseOutput(
      '<think>need to write a file</think>I will create it now.\n<tool_call>{"name":"write","arguments":{"path":"/i"}}</tool_call>',
    );
    expect(r.thinking).toBe("need to write a file");
    expect(r.text).toContain("I will create it now.");
    expect(r.toolCalls[0].name).toBe("write");
  });
});
