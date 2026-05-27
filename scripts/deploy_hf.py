#!/usr/bin/env python3
"""Deploy the built `dist/` to the static HF Space AlexWortega/pi-agent.

Usage:
    npm run build          # produce dist/
    python scripts/deploy_hf.py   (or: npm run deploy:hf)

Requires being logged in: `hf auth login` (or HF_TOKEN env).
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "AlexWortega/pi-agent"
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

SPACE_README = """---
title: Pi Agent · Soyuz (llama.cpp WebGPU)
emoji: 🛰️
colorFrom: purple
colorTo: indigo
sdk: static
pinned: true
license: apache-2.0
short_description: In-browser coding agent — Soyuz 4B via llama.cpp WebGPU
---

# Pi Agent · Soyuz

A coding agent that runs the **Soyuz Qwen3.5-4B** model **100% in your browser on your GPU**
via llama.cpp compiled to **WebGPU** (no server, no API key). Describe an app -> it writes a
self-contained HTML file -> live preview -> download.

First load fetches the GGUF (~2.5 GB) and caches it in your browser; afterwards it runs offline.
Needs a WebGPU browser (Chrome / Edge / Safari). Source: https://github.com/AlexWortega/pi-agent
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
        commit_message="Deploy Pi Agent · Soyuz (llama.cpp WebGPU) static build",
    )
    print(f"deployed: https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
