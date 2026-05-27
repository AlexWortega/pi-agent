# Pi Agent · Soyuz — in-browser coding agent

A polished web app where the **Soyuz** Pi Agent builds complete, self-contained web apps for you —
rendered live, ready to download. Inference runs **100% in your browser on your GPU** via
[`@reeselevine/wllama-webgpu`](https://www.npmjs.com/package/@reeselevine/wllama-webgpu) — i.e.
real **llama.cpp compiled to WebGPU** ([llamas-on-the-web](https://reeselevine.github.io/llamas-on-the-web/)).
No server, no API key, nothing leaves your machine.

## What you get

- **Projects** — each project is a chat workspace with its own model, history and generated apps,
  saved in `localStorage`. Create/rename/delete in the sidebar.
- **Live canvas** — when Soyuz emits a `\`\`\`html` block it renders instantly in a sandboxed iframe.
  Toggle **Preview / Code**, copy, open in a new tab, or **Download** it as a single `.html` file —
  that's your "скачать на локалку".
- **Reasoning panel** — the model's `<think>…</think>` chain is collapsed by default, expandable.
- **Model picker** — switch models, paste a custom GGUF URL, tune temperature / max tokens / context.
  First load downloads the GGUF and **caches it in the browser (OPFS)** so it's instant next time.

## Model

The Pi Agent runs the **real Soyuz brain**: `AlexWortega/qwen35-4b-soyuz-merged-gguf`
(`qwen35-4b-soyuz-merged.nomtp.Q4_K_M.gguf`, ~2.5 GB) — Qwen3.5-4B fine-tuned to emit
self-contained web apps. It's a hybrid linear-attention model (`qwen3next` gguf arch), and the
`@reeselevine/wllama-webgpu` build **ships that arch + its ops** (`gated_delta`, `linear_attn`,
`ssm_scan`), so it loads and runs in-browser on your GPU.

You can still paste any other GGUF URL in the model picker, but Soyuz is the default and only preset.

## Requirements

- A **WebGPU** browser: Chrome (best), Edge, or Safari (macOS / iOS 17+). The sidebar badge shows
  `WebGPU ready` or falls back to CPU.
- Enough RAM/VRAM for the chosen quant (Q4_K_M): ~0.2 GB (Gemma-270M) → ~2.5 GB (Soyuz-4B).

## Run

```bash
npm install
npm run dev          # http://localhost:5050
```

Build a static bundle (host it on GitHub Pages, an HF static Space, Netlify, anywhere):

```bash
npm run build        # → dist/
npm run preview      # serve dist/ locally
```

Deployed in two places (same static build):
- **GitHub Pages:** https://alexwortega.github.io/pi-agent/ (auto on push to `main`)
- **HF static Space:** https://huggingface.co/spaces/AlexWortega/pi-agent — redeploy with
  `npm run deploy:hf` (builds + uploads `dist/` via `scripts/deploy_hf.py`; needs `hf auth login`).

> **Cross-origin headers:** the dev/preview servers set `Cross-Origin-Opener-Policy: same-origin`
> and `Cross-Origin-Embedder-Policy: require-corp` (needed for the multi-threaded WASM fallback /
> `SharedArrayBuffer`). When you deploy the static `dist/`, set the **same two headers** on your
> host, or the CPU-fallback path won't get threads. The WebGPU path works without them.

## How it works

```
You ──▶ ChatPanel ──▶ engine.chat()  ──▶ @reeselevine/wllama-webgpu (llama.cpp + WebGPU)
                                              │  streams tokens
        parse <think> / ```html ◀────────────┘
              │
              ├─▶ reasoning panel
              └─▶ Artifact ──▶ CanvasPanel (iframe preview + Download .html)
```

Stack: Vite + React + TypeScript + Tailwind v4. Engine wrapper in `src/engine/llama.ts`; response
parsing (`<think>` + html extraction + tiny markdown renderer) in `src/lib/parse.ts`; project
persistence in `src/lib/store.ts`.
