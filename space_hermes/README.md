# Hermes Agent · WebGPU

A self-improving **tool-calling agent**, modelled on
[NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent), running **100% in your
browser on your GPU**. The brain is the same **Soyuz Qwen3.5-4B** GGUF as the Pi Agent demo,
executed by llama.cpp compiled to **WebGPU** (no server, no API key).

It implements the Hermes function-calling loop in ChatML: the model declares intent inside
`<tool_call>` tags, the browser runs the tool, feeds back a `<tool_response>`, and the loop repeats
until a final answer — exactly the synchronous loop of `run_agent.py`, ported to the browser.

## Features (all local)

- **Tool-calling agent loop** with 14 in-browser tools: `calculator`, `datetime`, `web_search`,
  `web_extract`, `execute_code` (sandboxed Web Worker), `memory`, `session_search`, `todo`,
  `skills_list` / `skill_view` / `skill_manage`, `render_html` (live canvas), `clarify`,
  `delegate_task` (sub-agents).
- **Persistent memory** (Honcho-style user model), **skills** (agentskills.io markdown), and a
  **task board** — all stored in `localStorage`.
- **Slash commands** (`/help`, `/tools`, `/skills`, `/memory`, `/tasks`, `/search`, `/schedule`, …).
- **Reasoning** shown as collapsible `<think>` blocks.
- **Scheduled prompts** (tab-open only — no real background cron without a server).

## Run

```bash
npm install
npm run dev        # http://localhost:5050
```

First message downloads the GGUF (~2.5 GB) and caches it in your browser (OPFS); afterwards it runs
offline. Needs a WebGPU browser (Chrome / Edge / Safari).

## Deploy (static HF Space)

```bash
npm run deploy:hf   # builds dist/ then uploads to AlexWortega/hermes-webgpu
# override target: HERMES_SPACE_ID=<owner>/<name> npm run deploy:hf
```

Requires `hf auth login` (or `HF_TOKEN`).

## Notes

- The Soyuz model is **not** Hermes-tuned, so `<tool_call>` emission isn't perfectly reliable. The
  prompt carries a few-shot example, the parser is tolerant of malformed JSON / missing close tags,
  and a turn with no tool call is treated as a direct answer. You can toggle tools in the Tools panel.
- `web_search` / `web_extract` are limited by browser CORS — they use CORS-open endpoints
  (DuckDuckGo Instant Answer, Wikipedia) and a configurable reader proxy fallback.
