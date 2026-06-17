# program.md — parameter-golf autoresearch duel

> karpathy/autoresearch style: **you program train_gpt.py, not this file.** The harness
> (`run_experiment.sh`, data prep, eval) is fixed; each experiment is a small diff to
> `train_gpt.py`. Keep ONE thing changed per experiment so results are comparable.

## Task (experiment 0 = baseline)
- goal: **minimize `val_bpb`** (validation bits-per-byte) on the fixed FineWeb val split.
- hard rule: the submission artifact (int8+zlib model bytes + code bytes) must be **≤ 16,000,000 bytes**.
  An experiment over 16 MB is INVALID regardless of its val_bpb — it will be rejected.
- file under experiment: `train_gpt.py` (only).
- dataset: parameter-golf `sp1024` variant (`DATA_PATH=./data/datasets/fineweb10B_sp1024/`,
  `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`, `VOCAB_SIZE=1024`) — already prepared.
- run an experiment with: `bash ./run_experiment.sh <exp_id>` → prints
  `AUTORESEARCH_RESULT val_bpb=<x> artifact_bytes=<n> seconds=<s> exp_id=<id>`.
- per-experiment training budget: **300s** (train_gpt.py's `MAX_WALLCLOCK_SECONDS`, compile excluded).

## Rules of the loop
- One change vs the current best per experiment. Compare every result to the best so far.
- Lower `val_bpb` is better. Keep the change iff `val_bpb` improved AND `artifact_bytes ≤ 16,000,000`;
  otherwise revert with bash command `cp best/train_gpt.py train_gpt.py`.
- A low number is not proof: if a result looks too good, re-run it once to confirm it is not a fluke.
- Establish exp-0 (unmodified) FIRST and snapshot it with bash command `cp train_gpt.py best/train_gpt.py`.
- The bash tool runs commands directly in the shell — pass plain commands ("cp ...", "./run_experiment.sh exp-2"); never prefix them with the word "bash".

## Key knobs in train_gpt.py (read the file to confirm exact names/lines)
Architecture: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS` (GQA), `MLP_MULT`, `VOCAB_SIZE`.
Schedule/optim: `ITERATIONS`, `WARMUP_STEPS`, `WARMDOWN_ITERS`, learning rate, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`.
The 16 MB budget couples model size ↔ quality: a bigger/deeper model lowers val_bpb only until int8+zlib
pushes the artifact past 16 MB. Find the best quality-per-byte, not just the best quality.

## Open ideas (turn each into ONE experiment; mine /records READMEs for more)
- depth↔width at ~constant params: more `NUM_LAYERS`, smaller `MODEL_DIM` (and vice versa).
- GQA: lower `NUM_KV_HEADS` to free bytes for depth/width within 16 MB.
- `MLP_MULT` sweep (e.g. 4 → 3 or 2.67) — cheaper FFN, reallocate the saved bytes.
- weight tying (embeddings ↔ output head) to shrink the artifact, then grow the model.
- learning-rate / `WARMDOWN_ITERS` tuning to extract more from the fixed 300s.
- `TRAIN_SEQ_LEN` / `TRAIN_BATCH_TOKENS` for better tokens-per-second under the time cap.
- quantization tweaks (already int8+zlib): per-channel scales, smaller code footprint.

## Ledger (the driver maintains EXPERIMENTS.md automatically from the sentinel line)
| exp | change (one line) | val_bpb | Δ vs best | artifact_bytes | verdict |
|-----|-------------------|---------|-----------|----------------|---------|
| 0   | baseline          |         | 0         |                | base    |
