#!/usr/bin/env bash
# duel-run.sh — launch the autoresearch driver on one prepared pod, with the
# right model wiring, and (optionally) wait for it to finish and pull results.
#
# Usage:
#   scripts/duel-run.sh <ip> <sshPort> siq  [deadline] [expBudget]
#   scripts/duel-run.sh <ip> <sshPort> glm  [deadline] [expBudget]
#
# Env (read locally, forwarded to the pod):
#   siq → RUNPOD_API_KEY (required; the SIQ proxy needs it), SIQ_EID (optional)
#   glm → OPENROUTER_API_KEY (required)
#   WAIT=1 → block until summary.json appears, then scp the run dir back to
#            ./results-duel/<model>/.  DEADLINE/EXPBUDGET default to 7200/300.
set -euo pipefail

IP="${1:?usage: duel-run.sh <ip> <sshPort> <siq|glm> [deadline] [expBudget]}"
PORT="${2:?}"; MODEL="${3:?need siq|glm}"
DEADLINE="${4:-7200}"; EXPBUDGET="${5:-300}"
SSH="ssh -o StrictHostKeyChecking=no -p ${PORT} root@${IP}"
SCP="scp -o StrictHostKeyChecking=no -P ${PORT}"

if [ "$MODEL" = "siq" ]; then
  MODEL_TAG="siq"
  EXTRA="SIQ_MODE=1 VIBETEST_JSON_REPAIR=1 GEN_MAXTOK='${GEN_MAXTOK:-16000}'"
  if [ -n "${SIQ_BASE_URL:-}" ]; then
    # SIQ served on a dedicated pod (llama.cpp). Point straight at it, no proxy.
    REMOTE_ENV=""
    BASE_URL="${SIQ_BASE_URL}"
    START_PROXY=0
  else
    # Fallback: SIQ serverless endpoint via the local async proxy (siq1.md §3.A).
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY required for the SIQ proxy}"
    REMOTE_ENV="RUNPOD_API_KEY='${RUNPOD_API_KEY}' SIQ_EID='${SIQ_EID:-leufrm6iskrs3v}' SIQ_MINTOK=8192 SIQ_EFFORT=high"
    BASE_URL="http://127.0.0.1:8080/v1"
    START_PROXY=1
  fi
elif [ "$MODEL" = "glm" ]; then
  : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY required}"
  # Prefer providers that are actually serving glm-5.2; allow_fallbacks routes around 429s.
  PROV_ORDER="${OPENROUTER_PROVIDER_ORDER:-DeepInfra,Z.AI,Novita,Cloudflare,Friendli}"
  REMOTE_ENV="LLM_API_KEY='${OPENROUTER_API_KEY}' OPENROUTER_PROVIDER_ORDER='${PROV_ORDER}' OPENROUTER_REASONING_MAXTOK='${OPENROUTER_REASONING_MAXTOK:-6000}' GEN_MAXTOK='${GEN_MAXTOK:-16000}' HTTP_REFERER='https://piagent.local' X_TITLE='autoresearch-duel'"
  MODEL_TAG="z-ai/glm-5.2"
  BASE_URL="https://openrouter.ai/api/v1"
  EXTRA=""
  START_PROXY=0
elif [ "$MODEL" = "opus" ]; then
  : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY required}"
  REMOTE_ENV="LLM_API_KEY='${OPENROUTER_API_KEY}' OPENROUTER_REASONING_MAXTOK='${OPENROUTER_REASONING_MAXTOK:-8000}' GEN_MAXTOK='${GEN_MAXTOK:-16000}' HTTP_REFERER='https://piagent.local' X_TITLE='autoresearch-duel'"
  MODEL_TAG="anthropic/claude-opus-4-8"
  BASE_URL="https://openrouter.ai/api/v1"
  EXTRA=""
  START_PROXY=0
else
  echo "model must be siq, glm, or opus"; exit 2
fi

echo "[run] launching ${MODEL} driver on ${IP}:${PORT} (deadline=${DEADLINE}s exp=${EXPBUDGET}s)"
$SSH "bash -s" <<REMOTE
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd /workspace/parameter-golf
rm -rf /workspace/run && mkdir -p /workspace/run
if [ "${START_PROXY}" = "1" ]; then
  pkill -f siq_proxy.py 2>/dev/null || true
  ${REMOTE_ENV} nohup python3 /workspace/siq_proxy.py >/workspace/run/proxy.log 2>&1 &
  for i in \$(seq 1 30); do curl -sf ${BASE_URL%/v1}/v1/models >/dev/null 2>&1 && break || sleep 1; done
fi
${REMOTE_ENV} ${EXTRA} \
  VIBETEST_BASE_URL='${BASE_URL}' VIBETEST_MODEL_TAG='${MODEL_TAG}' EXP_BUDGET='${EXPBUDGET}' \
  nohup node /workspace/runner/autoresearch.mjs \
    --workspace /workspace/parameter-golf --program /workspace/parameter-golf/program.md \
    --out /workspace/run --deadline ${DEADLINE} --exp-budget ${EXPBUDGET} \
    >/workspace/run/driver.log 2>&1 &
echo "[run] driver PID \$!  — tail /workspace/run/driver.log"
REMOTE

if [ "${WAIT:-0}" = "1" ]; then
  echo "[run] waiting for /workspace/run/summary.json (deadline ${DEADLINE}s + slack)…"
  endt=$(( $(date +%s) + DEADLINE + 1800 ))
  while [ "$(date +%s)" -lt "$endt" ]; do
    if $SSH 'test -f /workspace/run/summary.json' 2>/dev/null; then break; fi
    sleep 30
  done
  mkdir -p "results-duel/${MODEL}"
  $SCP -r "root@${IP}:/workspace/run/*" "results-duel/${MODEL}/" || true
  echo "[run] pulled results to results-duel/${MODEL}/"
  cat "results-duel/${MODEL}/summary.json" 2>/dev/null || echo "(no summary.json yet)"
fi
