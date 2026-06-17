#!/usr/bin/env bash
# duel-setup.sh — provision a single RunPod pod for the autoresearch duel.
# Copies the driver + harness up, installs deps, clones parameter-golf, prepares
# the sp1024 FineWeb data, and stages program.md + run_experiment.sh in the repo.
#
# Usage: scripts/duel-setup.sh <ip> <sshPort>
# Run once per pod (pod-SIQ and pod-GLM). Idempotent-ish: safe to re-run.
set -euo pipefail

IP="${1:?usage: duel-setup.sh <ip> <sshPort>}"
PORT="${2:?usage: duel-setup.sh <ip> <sshPort>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -o StrictHostKeyChecking=no -p ${PORT} root@${IP}"
SCP="scp -o StrictHostKeyChecking=no -P ${PORT}"

echo "[setup] copying driver + harness to ${IP}:${PORT}"
$SSH 'mkdir -p /workspace/runner /workspace/stage'
$SCP "${HERE}/vibetest/runners/autoresearch.mjs" "${HERE}/vibetest/runners/vibeapps.mjs" "root@${IP}:/workspace/runner/"
$SCP "${HERE}/autoresearch-duel/program.md" "${HERE}/autoresearch-duel/run_experiment.sh" "root@${IP}:/workspace/stage/"
$SCP "${HERE}/scripts/siq_proxy.py" "root@${IP}:/workspace/"

echo "[setup] installing node + uv + cloning parameter-golf + preparing data (this takes a few minutes)"
$SSH 'bash -s' <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace

# Node 20 (driver runtime)
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
  apt-get install -y nodejs >/dev/null 2>&1 || apt-get install -y nodejs npm >/dev/null 2>&1
fi
node -v

# clone parameter-golf
if [ ! -d /workspace/parameter-golf ]; then
  git clone --depth 1 https://github.com/openai/parameter-golf.git /workspace/parameter-golf
fi
cd /workspace/parameter-golf

# deps via requirements.txt (the image already ships torch+CUDA; bare `torch` in
# requirements is satisfied by the preinstalled build and won't be reinstalled).
python3 -m pip install -q -r requirements.txt

# stage harness into the repo
cp /workspace/stage/program.md ./program.md
cp /workspace/stage/run_experiment.sh ./run_experiment.sh
chmod +x ./run_experiment.sh
mkdir -p best logs

# one-time data prep (sp1024, 10 train shards = ~1B train tokens + full val split)
if [ ! -d ./data/datasets/fineweb10B_sp1024 ] || [ -z "$(ls -A ./data/datasets/fineweb10B_sp1024 2>/dev/null)" ]; then
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi
echo "[setup] done on $(hostname). GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
REMOTE
echo "[setup] pod ${IP}:${PORT} ready."
