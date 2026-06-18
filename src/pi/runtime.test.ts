import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the WebGPU engine with a scripted two-turn response: first a write
// tool call, then a final prose answer. This exercises the whole chain —
// localStream → agent loop → tool execution → OPFS-backed write — without a
// browser or the real model.
const chatMock = vi.fn();
vi.mock("../engine/llama", () => ({
  engine: { chat: (...args: any[]) => chatMock(...args) },
  hasWebGPU: () => false,
}));

import type { AgentEvent } from "@earendil-works/pi-agent-core";
import { startAgentRun, makeAgentModel, WORKSPACE_ROOT } from "./runtime";
import { getFsBackend } from "./fs/backend";

const params = { temperature: 0.6, maxTokens: 256, contextLength: 4096 };

function scriptResponse(text: string) {
  // Mimic engine.chat: stream the text then resolve with it.
  return async (_messages: any, opts: any) => {
    opts.onToken?.(text);
    return text;
  };
}

describe("agent runtime end-to-end (mocked engine)", () => {
  beforeEach(() => {
    chatMock.mockReset();
  });

  it("executes a write tool call and finishes with prose", async () => {
    chatMock
      .mockImplementationOnce(
        scriptResponse(
          '<think>create the file</think>\n<tool_call>{"name":"write","arguments":{"path":"hello.txt","content":"hi from agent"}}</tool_call>',
        ),
      )
      .mockImplementationOnce(scriptResponse("Done! I created hello.txt for you."));

    const model = makeAgentModel({ id: "soyuz-4b", label: "Soyuz" }, params);
    const run = startAgentRun({ prompt: "create hello.txt with 'hi from agent'", history: [], model });

    const events: AgentEvent[] = [];
    for await (const ev of run.stream) events.push(ev);

    const types = events.map((e) => e.type);
    expect(types).toContain("tool_execution_start");
    expect(types).toContain("tool_execution_end");
    expect(types).toContain("agent_end");

    const toolStart = events.find((e) => e.type === "tool_execution_start") as Extract<
      AgentEvent,
      { type: "tool_execution_start" }
    >;
    expect(toolStart.toolName).toBe("write");

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as Extract<
      AgentEvent,
      { type: "tool_execution_end" }
    >;
    expect(toolEnd.isError).toBe(false);

    // The file is actually on the virtual filesystem.
    const fs = getFsBackend();
    const written = await fs.readText(`${WORKSPACE_ROOT}/hello.txt`);
    expect(written).toBe("hi from agent");

    // The model was called twice: once to act, once to wrap up after the tool result.
    expect(chatMock).toHaveBeenCalledTimes(2);
  });

  it("surfaces an edit diff in the tool result details", async () => {
    // Seed a file, then script an edit.
    await getFsBackend().writeText(`${WORKSPACE_ROOT}/app.js`, "const x = 1;\n");
    chatMock
      .mockImplementationOnce(
        scriptResponse(
          '<tool_call>{"name":"edit","arguments":{"path":"app.js","edits":[{"oldText":"const x = 1;","newText":"const x = 99;"}]}}</tool_call>',
        ),
      )
      .mockImplementationOnce(scriptResponse("Updated."));

    const model = makeAgentModel({ id: "soyuz-4b", label: "Soyuz" }, params);
    const run = startAgentRun({ prompt: "set x to 99", history: [], model });

    const events: AgentEvent[] = [];
    for await (const ev of run.stream) events.push(ev);

    const toolEnd = events.find((e) => e.type === "tool_execution_end") as Extract<
      AgentEvent,
      { type: "tool_execution_end" }
    >;
    expect(toolEnd.isError).toBe(false);
    expect(toolEnd.result.details.diff).toContain("const x = 99;");
    expect(await getFsBackend().readText(`${WORKSPACE_ROOT}/app.js`)).toContain("const x = 99;");
  });

  it("breaks out of a loop when the model repeats the same tool call", async () => {
    // The model keeps emitting the identical ls call every turn (classic loop).
    chatMock.mockImplementation(
      scriptResponse('<tool_call>{"name":"ls","arguments":{"path":"/workspace"}}</tool_call>'),
    );

    const model = makeAgentModel({ id: "soyuz-4b", label: "Soyuz" }, params);
    const run = startAgentRun({ prompt: "look around", history: [], model });
    for await (const _ev of run.stream) void _ev;

    // Without the no-progress guard this would run to MAX_TURNS (6); the repeat
    // detector stops it on the 2nd identical call.
    expect(chatMock.mock.calls.length).toBeLessThanOrEqual(2);
  });

  it("stops once a complete HTML document is emitted, even mid tool use", async () => {
    chatMock
      .mockImplementationOnce(scriptResponse('<tool_call>{"name":"ls","arguments":{"path":"/workspace"}}</tool_call>'))
      .mockImplementationOnce(
        // a different tool call (so the repeat guard doesn't fire) + a finished HTML doc
        scriptResponse(
          'Here you go:\n```html\n<!doctype html><html><body>hi</body></html>\n```\n<tool_call>{"name":"ls","arguments":{"path":"/workspace/sub"}}</tool_call>',
        ),
      )
      .mockImplementation(scriptResponse("still going…"));

    const model = makeAgentModel({ id: "soyuz-4b", label: "Soyuz" }, params);
    const run = startAgentRun({ prompt: "make a page", history: [], model });
    for await (const _ev of run.stream) void _ev;

    // Turn 2 emits a complete </html> → run stops there, not at MAX_TURNS.
    expect(chatMock.mock.calls.length).toBe(2);
  });
});
