import type { Tool } from "../registry";

const WORKER_SRC = `
self.onmessage = (e) => {
  const logs = [];
  const cap = (level) => (...a) => logs.push(a.map(x => {
    try { return typeof x === 'object' ? JSON.stringify(x) : String(x); } catch { return String(x); }
  }).join(' '));
  const sandbox = { log: cap('log'), info: cap('info'), warn: cap('warn'), error: cap('error') };
  const done = (payload) => self.postMessage(payload);
  try {
    const fn = new Function('console', '"use strict";\\n' + e.data.code);
    Promise.resolve(fn(sandbox)).then((result) => {
      let serial;
      try { serial = result === undefined ? undefined : JSON.parse(JSON.stringify(result)); }
      catch { serial = String(result); }
      done({ ok: true, logs, result: serial });
    }).catch((err) => done({ ok: false, logs, error: String(err && err.message || err) }));
  } catch (err) {
    done({ ok: false, logs, error: String(err && err.message || err) });
  }
};
`;

function runInWorker(code: string, timeoutMs: number, signal: AbortSignal): Promise<any> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    const worker = new Worker(url, { type: "classic" });
    let settled = false;
    const cleanup = () => {
      if (settled) return;
      settled = true;
      worker.terminate();
      URL.revokeObjectURL(url);
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`execution timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    const onAbort = () => {
      cleanup();
      reject(new Error("aborted"));
    };
    signal.addEventListener("abort", onAbort);
    worker.onmessage = (e) => {
      const data = e.data;
      cleanup();
      resolve(data);
    };
    worker.onerror = (e) => {
      cleanup();
      reject(new Error(e.message || "worker error"));
    };
    worker.postMessage({ code });
  });
}

export const executeCodeTool: Tool = {
  name: "execute_code",
  description:
    "Run a snippet of JavaScript in a sandboxed Web Worker (no DOM/network). `return` a value or use console.log; you get back the logs and result. Good for data wrangling, parsing, quick computation.",
  parameters: {
    type: "object",
    properties: { code: { type: "string", description: "JavaScript; may `return` a JSON-serialisable value" } },
    required: ["code"],
  },
  async run(args, ctx) {
    const code = String(args?.code ?? "");
    if (!code.trim()) throw new Error("code is empty");
    const out = await runInWorker(code, 5000, ctx.signal);
    if (!out.ok) return { ok: false, logs: out.logs, error: out.error };
    return { ok: true, logs: out.logs, result: out.result };
  },
};
