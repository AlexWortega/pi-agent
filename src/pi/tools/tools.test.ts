import { describe, it, expect, beforeEach } from "vitest";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { getFsBackend } from "../fs/backend";
import { buildTools } from "./index";

const CWD = "/workspace";

function byName(tools: AgentTool<any>[], name: string): AgentTool<any> {
  const t = tools.find((x) => x.name === name);
  if (!t) throw new Error(`tool ${name} not found`);
  return t;
}

async function runText(tool: AgentTool<any>, args: unknown): Promise<string> {
  const prepared = tool.prepareArguments ? tool.prepareArguments(args) : args;
  const result = await tool.execute("t1", prepared as any);
  return result.content.map((c: any) => c.text ?? "").join("\n");
}

describe("browser tools over the virtual fs", () => {
  let tools: AgentTool<any>[];

  beforeEach(() => {
    tools = buildTools(getFsBackend(), CWD);
  });

  it("write then read round-trips a file (relative path resolves under cwd)", async () => {
    const w = await runText(byName(tools, "write"), { path: "index.html", content: "<h1>hi</h1>\n" });
    expect(w).toContain("Successfully wrote");
    const r = await runText(byName(tools, "read"), { path: "index.html" });
    expect(r).toContain("<h1>hi</h1>");
  });

  it("edit applies an exact replacement and reports it", async () => {
    await runText(byName(tools, "write"), { path: "app.js", content: "const x = 1;\nconst y = 2;\n" });
    const edit = byName(tools, "edit");
    const prepared = edit.prepareArguments!({ path: "app.js", edits: [{ oldText: "const x = 1;", newText: "const x = 42;" }] });
    const result = await edit.execute("t1", prepared as any);
    expect(result.content[0].text).toContain("Successfully replaced 1 block(s)");
    expect(result.details.diff).toContain("const x = 42;");
    expect(result.details.patch).toContain("@@");
    const r = await runText(byName(tools, "read"), { path: "app.js" });
    expect(r).toContain("const x = 42;");
    expect(r).toContain("const y = 2;");
  });

  it("edit errors clearly when oldText is not unique", async () => {
    await runText(byName(tools, "write"), { path: "dup.txt", content: "foo\nfoo\n" });
    const edit = byName(tools, "edit");
    await expect(
      edit.execute("t1", { path: "dup.txt", edits: [{ oldText: "foo", newText: "bar" }] } as any),
    ).rejects.toThrow(/unique/i);
  });

  it("edit errors when oldText is not found", async () => {
    await runText(byName(tools, "write"), { path: "n.txt", content: "alpha\n" });
    const edit = byName(tools, "edit");
    await expect(
      edit.execute("t1", { path: "n.txt", edits: [{ oldText: "omega", newText: "x" }] } as any),
    ).rejects.toThrow(/could not find/i);
  });

  it("edit prepareArguments accepts a JSON-string edits payload", async () => {
    await runText(byName(tools, "write"), { path: "s.txt", content: "one\n" });
    const edit = byName(tools, "edit");
    const prepared = edit.prepareArguments!({
      path: "s.txt",
      edits: JSON.stringify([{ oldText: "one", newText: "two" }]),
    });
    const result = await edit.execute("t1", prepared as any);
    expect(result.content[0].text).toContain("Successfully replaced");
  });

  it("ls lists written files", async () => {
    await runText(byName(tools, "write"), { path: "a.txt", content: "1" });
    await runText(byName(tools, "write"), { path: "sub/b.txt", content: "2" });
    const out = await runText(byName(tools, "ls"), {});
    expect(out).toContain("a.txt");
    expect(out).toContain("sub/");
  });

  it("read fails on a missing file", async () => {
    await expect(byName(tools, "read").execute("t1", { path: "nope.txt" } as any)).rejects.toThrow(/not found/i);
  });

  it("bash is a stub that redirects to the file tools", async () => {
    const bash = tools.find((t) => t.name === "bash");
    expect(bash).toBeDefined();
    await expect(bash!.execute("t1", { command: "ls" } as any)).rejects.toThrow(/no shell|file tools/i);
  });
});
