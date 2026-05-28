#!/usr/bin/env python3
"""Deploy the built `dist/` to the static HF Space for the Hermes WebGPU demo.

Usage:
    npm run build          # produce dist/
    python scripts/deploy_hf.py   (or: npm run deploy:hf)

Requires being logged in: `hf auth login` (or HF_TOKEN env).
Override the target with HERMES_SPACE_ID=<owner>/<name>.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ.get("HERMES_SPACE_ID", "AlexWortega/hermes-webgpu")
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

SPACE_README = f"""---
title: Hermes Agent · WebGPU
emoji: ⚡
colorFrom: indigo
colorTo: green
sdk: static
pinned: true
license: apache-2.0
short_description: In-browser tool-calling agent (à la hermes-agent) on Soyuz 4B via llama.cpp WebGPU
---

# Hermes Agent · WebGPU

A self-improving **tool-calling agent** — modelled on
[NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent) — that runs
**100% in your browser on your GPU**. The brain is the **Soyuz Qwen3.5-4B** GGUF executed
by llama.cpp compiled to **WebGPU** (no server, no API key).

It implements the Hermes function-calling loop in ChatML: the model declares intent inside
`<tool_call>` tags, the browser runs the tool, feeds back a `<tool_response>`, and the loop
continues until a final answer.

**Features (all local, in-browser):**
- Tool-calling agent loop with: calculator, datetime, web_search, web_extract, execute_code
  (sandboxed Web Worker), memory, session_search, todo, skills (list/view/manage), render_html
  (live canvas), clarify, and delegate_task (sub-agents).
- Persistent **memory** + **skills** (agentskills.io style) + **task board**, stored in your browser.
- Slash commands (`/help`, `/tools`, `/skills`, `/memory`, `/tasks`, `/search`, `/schedule`, …).
- Reasoning shown as collapsible `<think>` blocks.

First load fetches the GGUF (~2.5 GB) and caches it in your browser (OPFS); afterwards it runs offline.
Needs a WebGPU browser (Chrome / Edge / Safari).
"""


def main() -> int:
    if not (DIST / "index.html").exists():
        print("dist/ not built — run `npm run build` first", file=sys.stderr)
        return 1
    (DIST / "README.md").write_text(SPACE_README, encoding="utf-8")
    api = HfApi()
    create_repo(REPO_ID, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(DIST),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Hermes Agent · WebGPU (tool-calling agent on Soyuz, llama.cpp WebGPU)",
    )
    print(f"deployed: https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
