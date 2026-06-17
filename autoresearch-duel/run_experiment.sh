#!/usr/bin/env bash
# run_experiment.sh — FIXED harness for the parameter-golf autoresearch duel.
# The researcher agent calls this and must NOT edit it. It trains the current
# train_gpt.py for a fixed wall-clock budget (compile excluded — train_gpt.py's
# own MAX_WALLCLOCK_SECONDS handles that), evaluates, and prints ONE sentinel
# line the driver parses:
#
#   AUTORESEARCH_RESULT val_bpb=<x> artifact_bytes=<n> seconds=<s> exp_id=<id>
#
# Usage: bash ./run_experiment.sh <exp_id>
# Env (with parameter-golf sp1024 defaults):
#   EXP_BUDGET   fixed training seconds (default 300)
#   DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE   dataset/tokenizer (sp1024 by default)
set -uo pipefail

EXP_ID="${1:-exp-$(date +%s)}"
EXP_BUDGET="${EXP_BUDGET:-300}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
NPROC="${NPROC:-1}"

mkdir -p logs
LOG="logs/${EXP_ID}.log"

emit() { echo "AUTORESEARCH_RESULT val_bpb=$1 artifact_bytes=$2 seconds=$3 exp_id=${EXP_ID}"; }

start=$(date +%s)
# MAX_WALLCLOCK_SECONDS is train_gpt.py's own fixed budget (excludes warmup/compile).
RUN_ID="${EXP_ID}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
VOCAB_SIZE="${VOCAB_SIZE}" \
MAX_WALLCLOCK_SECONDS="${EXP_BUDGET}" \
  torchrun --standalone --nproc_per_node="${NPROC}" train_gpt.py >"${LOG}" 2>&1
rc=$?
end=$(date +%s)
secs=$((end - start))

if [ $rc -ne 0 ]; then
  echo "[run_experiment] train_gpt.py exited $rc — tail of ${LOG}:"
  tail -n 25 "${LOG}"
  emit 999.0 0 "${secs}"
  exit 0   # exit 0 so the agent sees the sentinel and reverts; the driver records val_bpb=999
fi

# Final quantized eval line: "final_int8_zlib_roundtrip val_loss:.. val_bpb:.."
val_bpb=$(grep -oE 'final_int8_zlib_roundtrip[^\n]*val_bpb:[0-9.]+' "${LOG}" | tail -n1 | grep -oE 'val_bpb:[0-9.]+' | cut -d: -f2)
# Fallback: last val_bpb printed during training.
[ -z "${val_bpb}" ] && val_bpb=$(grep -oE 'val_bpb:[0-9.]+' "${LOG}" | tail -n1 | cut -d: -f2)
# Artifact size: "Total submission size int8+zlib: <n> bytes" — capture ONLY the digits
# before " bytes" (a bare [0-9]+ also matches the 8 in "int8" → corrupts the value).
bytes=$(grep -oE 'Total submission size int8\+zlib: [0-9]+ bytes' "${LOG}" | tail -n1 | sed -E 's/.*: ([0-9]+) bytes/\1/')
[ -z "${bytes}" ] && bytes=$(stat -c%s final_model.int8.ptz 2>/dev/null || echo 0)

[ -z "${val_bpb}" ] && val_bpb=999.0
[ -z "${bytes}" ] && bytes=0

echo "[run_experiment] ${EXP_ID}: val_bpb=${val_bpb} artifact_bytes=${bytes} wall=${secs}s rc=${rc}"
emit "${val_bpb}" "${bytes}" "${secs}"
