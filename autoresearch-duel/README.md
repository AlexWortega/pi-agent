# Autoresearch duel — SIQ-1 vs GLM-5.2 vs Opus 4.8 on `openai/parameter-golf`

A head-to-head **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)**-style loop:
each "researcher" LLM autonomously drives `edit train_gpt.py → train (fixed 300 s) →
eval val_bpb → keep / revert`, as the **same Pi-Agent** (forked from
`vibetest/runners/vibeapps.mjs`), on its **own RunPod A6000**, for **2 hours**.
Task = `openai/parameter-golf`: train the smallest LM ≤16 MB, minimize bits-per-byte
on the FineWeb val split (lower = better).

Identical hardware / time / harness / baseline — only the model behind `chatCompletion`
differs. Absolute BPB is *not* leaderboard-valid (1×A6000 ≠ the official 8×H100/10 min);
the comparison is fair because everything except the model is held constant.

## Pieces
- `vibetest/runners/autoresearch.mjs` — the driver (real `bash`, deadline loop, EXPERIMENTS.md
  ledger + `best/` champion, OpenRouter auth + reasoning handling). `…/autoresearch.dryrun.mjs`
  is the offline, GPU-free smoke test.
- `program.md` — the researcher spec (the only human-edited file; the model edits `train_gpt.py`).
- `run_experiment.sh` — fixed harness: `torchrun train_gpt.py` (`MAX_WALLCLOCK_SECONDS=300`) →
  eval → prints `AUTORESEARCH_RESULT val_bpb=.. artifact_bytes=.. seconds=.. exp_id=..`.
- `../scripts/runpod-duel-provision.mjs` · `duel-setup.sh` · `duel-run.sh` · `serve-siq.sh` ·
  `siq_proxy.py` — provision A6000 pods, set up parameter-golf, serve SIQ-1 (llama.cpp GGUF), run.
- `draw_steps.py` → `duel_steps.png` — per-step visualization.
- `../results-duel/{glm,siq,opus}/` — each run's `summary.json`, `EXPERIMENTS.md`, transcript,
  and winning `best_train_gpt.py`.

## Result (champion val_bpb, lower = better)
| model | champion | experiments | kept | notes |
|---|---|---|---|---|
| **Opus 4.8** | **1.760*** | 4 | 4/4 | sniped `matrix_lr` / LR schedule, monotonic; *still running |
| **SIQ-1-35B** | 1.767 | 12 | 4 | broad search; wins: warmdown 1200→800, matrix_lr 0.04→0.05, ↑capacity; full 2 h |
| GLM-5.2 | 2.078 | 3 | 0 | only tried "add depth" (9→12, 9→10), both worse; idled after 65 min |

Models reached as the Pi-Agent: SIQ-1 via a dedicated A6000 (llama.cpp GGUF `--jinja`),
GLM-5.2 and Opus 4.8 via OpenRouter (`z-ai/glm-5.2`, `anthropic/claude-opus-4-8`).
