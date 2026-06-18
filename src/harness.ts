/**
 * Headless test harness: runs the REAL agent pipeline (engine.chat → localStream
 * → agent loop → tools → parse → stop/loop guards) against a tiny Qwen2.5-0.5B
 * model in the browser, so we can observe actual generation + errors without the
 * 2.5 GB Soyuz model. Driven by scripts/harness-run.mjs via Playwright.
 *
 * NOT part of the app — harness.html + this file exist only for debugging.
 */
import { engine, hasWebGPU } from "./engine/llama";
import { startAgentRun, makeAgentModel, WORKSPACE_ROOT } from "./pi/runtime";
import { getFsBackend } from "./pi/fs/backend";
import { extractHtmlArtifact } from "./lib/parse";

// Served same-origin from public/ — dodges HF Xet CDN + COEP cross-origin block.
// Absolute URL so wllama's blob worker can parse it.
const MODEL_URL = `${location.origin}/${(window as any).__model || "soyuz.gguf"}`;

const log: string[] = [];
const out = document.getElementById("out")!;
function emit(line: string) {
  log.push(line);
  (window as any).__log = log;
  out.textContent = log.join("\n");
  // eslint-disable-next-line no-console
  console.log(line);
}

(async () => {
  try {
    emit(`backend: ${hasWebGPU() ? "webgpu" : "cpu"} (navigator.gpu=${!!(navigator as any).gpu})`);
    emit(`loading ${MODEL_URL}…`);
    const ctx = 8192;
    const t0 = performance.now();
    await engine.load({ url: MODEL_URL }, {
      contextLength: ctx,
      onProgress: (f) => {
        if (Math.round(f * 100) % 25 === 0) emit(`  download ${Math.round(f * 100)}%`);
      },
    });
    emit(`model ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

    const prompt = (window as any).__prompt || "make a page with a big centered Hello and a button that alerts hi";
    emit(`prompt: ${prompt}`);
    const temp = (window as any).__temp ?? 0.1;
    emit(`temperature: ${temp}`);
    const model = makeAgentModel({ id: "soyuz", label: "Soyuz" }, { temperature: temp, maxTokens: 1024, contextLength: ctx });
    const run = startAgentRun({ prompt, history: [], model, temperature: temp });

    let turn = 0;
    for await (const ev of run.stream) {
      if (ev.type === "message_end" && (ev.message as any).role === "assistant") {
        turn++;
        const m: any = ev.message;
        const text = m.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("");
        const think = m.content.filter((b: any) => b.type === "thinking").map((b: any) => b.thinking).join("");
        const calls = m.content.filter((b: any) => b.type === "toolCall").map((b: any) => b.name);
        emit(`── turn ${turn} (${m.stopReason}) think=${think.length}ch text=${text.length}ch tools=[${calls}]`);
        emit(`   has \`\`\`html: ${/```html/i.test(text)} | has <html: ${/<html/i.test(text)} | has <tool_call: ${/<tool_call/i.test(text)}`);
        (window as any).__fullText =
          ((window as any).__fullText || "") +
          `\n\n===== turn ${turn} =====\n<think>${think}</think>\n--- answer ---\n${text}`;
        (window as any).__lastAnswer = text;
      } else if (ev.type === "tool_execution_end") {
        emit(`   tool ${ev.toolName} → ${ev.isError ? "ERROR" : "ok"}`);
      }
    }

    const idx = `${WORKSPACE_ROOT}/index.html`;
    // Replicate the useAgent HTML bridge: if no file on disk but the model
    // emitted an HTML doc, salvage it (this is what the real app does).
    if (!(await getFsBackend().exists(idx))) {
      const parsed = extractHtmlArtifact((window as any).__lastAnswer || "");
      if (parsed) {
        await getFsBackend().writeText(idx, parsed.html);
        emit(`bridge: saved ${parsed.html.length} chars to index.html`);
      } else {
        emit("bridge: no HTML document found in the answer");
      }
    }
    const exists = await getFsBackend().exists(idx);
    emit(`done. index.html exists: ${exists}`);
    if (exists) {
      const html = await getFsBackend().readText(idx);
      emit(`index.html length: ${html.length}, has <html>: ${/<html/i.test(html)}`);
    }
  } catch (e: any) {
    emit(`ERROR: ${e?.stack || e?.message || String(e)}`);
  } finally {
    (window as any).__done = true;
  }
})();
