#!/usr/bin/env bash
# serve-siq.sh — stand up SIQ-1-35B as an OpenAI-compatible server on a dedicated
# RunPod A6000, via llama.cpp + GGUF Q4_K_M (siq1.md §B). The experiments pod
# points its autoresearch driver at this endpoint (SIQ_MODE=1).
#
# Usage: HF_TOKEN=$(cat ~/.cache/huggingface/token) scripts/serve-siq.sh <ip> <sshPort>
# Prints the in-pod endpoint; pair with `runpod-duel-provision.mjs list` to get the
# public host:port (the pod must expose 8080/tcp — RUNPOD_PORTS=22/tcp,8080/tcp).
set -euo pipefail

IP="${1:?usage: serve-siq.sh <ip> <sshPort>}"
PORT="${2:?usage: serve-siq.sh <ip> <sshPort>}"
: "${HF_TOKEN:?HF_TOKEN required (private repo AlexWortega/SIQ-1-35B)}"
GGUF="${SIQ_GGUF:-gguf/SIQ-1-35B.Q4_K_M.gguf}"
CTX="${SIQ_CTX:-32768}"
SSH="ssh -o StrictHostKeyChecking=no -p ${PORT} root@${IP}"

echo "[serve] building llama.cpp + downloading SIQ GGUF on ${IP}:${PORT} (several minutes)"
$SSH "HF_TOKEN='${HF_TOKEN}' GGUF='${GGUF}' CTX='${CTX}' bash -s" <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace

# CUDA toolchain: nvcc exists but isn't on PATH by default → cmake can't find the
# CUDA compiler. Put it on PATH and point cmake at it explicitly.
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="/usr/local/cuda/bin/nvcc"

# build deps (devel image already has nvcc/gcc)
command -v cmake >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq cmake build-essential git libcurl4-openssl-dev)

# download the GGUF (private repo) to the PERSISTENT volume (/workspace survives a
# container restart; /models on the container disk does not). `huggingface-cli` is
# deprecated/no-op now → use `hf`.
python3 -m pip install -q -U huggingface_hub
mkdir -p /workspace/models
( hf download AlexWortega/SIQ-1-35B "${GGUF}" --local-dir /workspace/models --token "${HF_TOKEN}" \
    > /workspace/gguf_download.log 2>&1 ; echo DL_DONE >> /workspace/gguf_download.log ) &
DLPID=$!

# build llama-server with CUDA (A6000 = sm_86)
if [ ! -x /workspace/llama.cpp/build/bin/llama-server ]; then
  [ -d /workspace/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cd /workspace/llama.cpp
  rm -rf build
  cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 >/workspace/llama_build.log 2>&1
  cmake --build build --config Release -j"$(nproc)" --target llama-server >>/workspace/llama_build.log 2>&1
fi
[ -x /workspace/llama.cpp/build/bin/llama-server ] || { echo "[serve] ERROR: build failed; tail:"; tail -15 /workspace/llama_build.log; exit 2; }
echo "[serve] llama-server built: $(ls -la /workspace/llama.cpp/build/bin/llama-server)"

# wait for the GGUF download to finish
wait $DLPID || true
GGUF_PATH="/workspace/models/${GGUF}"
[ -f "$GGUF_PATH" ] || { echo "[serve] ERROR: GGUF missing at $GGUF_PATH"; tail -8 /workspace/gguf_download.log; exit 3; }
echo "[serve] GGUF: $(ls -la "$GGUF_PATH")"

# launch llama-server durably: a respawn loop in its own session (setsid) so it
# survives SSH disconnect AND auto-restarts if it dies mid-run.
pkill -f 'llama-server' 2>/dev/null || true; pkill -f 'siq_keepalive' 2>/dev/null || true
cat > /workspace/siq_keepalive.sh <<KA
#!/usr/bin/env bash
# siq_keepalive
while true; do
  /workspace/llama.cpp/build/bin/llama-server -m "$GGUF_PATH" --alias siq \
    -ngl 99 -c ${CTX} -np 1 --jinja --host 0.0.0.0 --port 8080 >> /workspace/llama-server.log 2>&1
  echo "[keepalive] llama-server exited \$? — restarting in 5s" >> /workspace/llama-server.log
  sleep 5
done
KA
chmod +x /workspace/siq_keepalive.sh
setsid bash /workspace/siq_keepalive.sh >/dev/null 2>&1 < /dev/null &
echo "[serve] keepalive launched — loading weights…"

# health check (weight load on A6000 ~30-90s)
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "[serve] READY after ${i}s — /v1/models responds"; break
  fi
  sleep 2
done
curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 || { echo "[serve] ERROR: server not healthy; tail log:"; tail -20 /workspace/llama-server.log; exit 4; }

# quick generation smoke
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"siq","max_tokens":24,"messages":[{"role":"system","content":"Reasoning effort: low"},{"role":"user","content":"Reply with the single word: ready"}],"chat_template_kwargs":{"enable_thinking":false}}' \
  | head -c 400; echo
echo "[serve] SIQ-1 served on :8080 of $(hostname)"
REMOTE
echo "[serve] done. In-pod endpoint: http://127.0.0.1:8080/v1 (model tag: siq)"
