# SIQ-1-35B in the Pi Agent web app

How the "SIQ-1-35B (cloud)" model in the model picker is wired and deployed. The
in-browser Soyuz-4B model is unaffected — SIQ-1 is an additional, selectable
**remote** model. See `siq1.md` for the model/benchmark notes.

## Live architecture (deployed)

```
browser (alexwortega.github.io/pi-agent)
  └─ engine.chatViaServer → POST ${LOG_API}/api/siq/v1/chat/completions   (OpenAI SSE)
       └─ Railway server  server/index.js  (api-production-bd22.up.railway.app)  ← holds RUNPOD_API_KEY, CORS, SIQ shaping
            └─ RunPod serverless  /run + poll /status   (openai_input envelope)
                 └─ endpoint SIQ_EID=leufrm6iskrs3v  (siq1-gguf, the warm workhorse — 18k+ jobs)
```

**Primary host = the Railway server** (same host as `LOG_API`, already in the bundle).
A byte-identical mirror also runs as the HF Space `AlexWortega/siq-proxy`
(`siq_proxy_space/`) at `https://alexwortega-siq-proxy.hf.space/api/siq` — set
`VITE_SIQ_API` there as a fallback if Railway is down. Both are live and verified.

Why a proxy: the browser can't call RunPod directly — the API key would leak into
the public bundle, RunPod returns no CORS headers, and its serverless API is async
`/run`+poll. The proxy holds the key server-side, adds CORS, applies SIQ-1 shaping,
and re-emits the completed result as SSE deltas (`reasoning_content` → folded into
`<think>…</think>`, then `content`) — exactly what the client engine accumulates.

Why `SIQ_EID=leufrm6iskrs3v` and not a fresh vLLM deploy: see below.

## Pieces in this repo

- **`src/`** — `engine/llama.ts` routes inference per-model; `config.ts` has the
  SIQ-1 preset + `SIQ_API` default (the Space URL); `ModelPicker.tsx` adds the
  thinking toggle + low/medium/high effort selector.
- **`siq_proxy_space/`** — the live proxy (zero-dep Node, Docker Space). `app.mjs`,
  `Dockerfile`, `README.md`. Deployed to `https://huggingface.co/spaces/AlexWortega/siq-proxy`.
- **`server/index.js`** — the same `/api/siq` route for the Railway server.
- **`scripts/runpod-siq-serverless.mjs`** — `npm run siq:deploy up|list|down`,
  creates a fresh vLLM serverless endpoint (see caveat).

## Deploying / re-deploying the proxy (HF Space)

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
# secrets (Space → Settings → Variables and secrets, or via API):
#   RUNPOD_API_KEY=<key>   SIQ_EID=leufrm6iskrs3v   SIQ_MODEL=siq
cd siq_proxy_space
git push   # to the Space remote; Docker rebuilds automatically
curl -s https://alexwortega-siq-proxy.hf.space/api/health   # → {"ok":true,"siq":true,"model":"siq"}
```

## Use it in the app

Model picker (top bar) → **SIQ-1-35B (cloud)**. No download, no WebGPU. The picker
shows a **Thinking** toggle and **low/medium/high** reasoning effort, sent per request.
First request after the workers scale to zero has a cold-start queue (tens of seconds);
warm requests are fast. Everything else — agent loop, tools, live preview — is identical.

## Caveat: the fresh vLLM serverless deploy

`scripts/runpod-siq-serverless.mjs` builds a fresh endpoint on RunPod's `worker-v1-vllm`
image. In practice that attempt did **not** serve: the worker's vLLM didn't pick up jobs
on SIQ-1's custom Qwen3.6-35B-A3B MoE arch, and A100-80GB capacity was throttled — so the
fresh endpoint was torn down and the proxy points at the **existing, proven** serverless
endpoint `leufrm6iskrs3v` (GGUF Q4, driven via `/run` + `openai_input`, which is the path
all the `siq1.md` benchmarks used). The script is kept for when a newer vLLM image supports
the arch; bump `SIQ_VLLM_IMAGE` and re-run, then set `SIQ_EID` to the new id.
