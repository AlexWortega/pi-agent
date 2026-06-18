import { describe, it, expect } from "vitest";
import { cutAtStop } from "./stops";

describe("cutAtStop", () => {
  it("returns null when no stop sequence is present", () => {
    expect(cutAtStop("hello <h1>hi</h1>")).toBeNull();
  });

  it("cuts at a fabricated <tool_response>", () => {
    expect(cutAtStop("done.\n<tool_response>\nfake result")).toBe("done.\n");
  });

  it("cuts at the earliest of several stops", () => {
    expect(cutAtStop("a<|im_end|>b<tool_response>c")).toBe("a");
  });

  it("cuts at a role marker the model hallucinated", () => {
    expect(cutAtStop("answer<|im_start|>user")).toBe("answer");
  });

  it("keeps a complete html doc intact (no stop inside)", () => {
    const html = "```html\n<!doctype html><html></html>\n```";
    expect(cutAtStop(html)).toBeNull();
  });
});
