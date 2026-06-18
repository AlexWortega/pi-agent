#!/usr/bin/env node
/**
 * runpod-siq-serverless.mjs — create / list / delete a RunPod *serverless*
 * endpoint that serves AlexWortega/SIQ-1-35B over an OpenAI-compatible API,
 * using RunPod's official vLLM worker image. The Railway proxy
 * (server/index.js, /api/siq) bridges the browser to this endpoint.
 *
 *   node scripts/runpod-siq-serverless.mjs up      # create template + endpoint, print SIQ_EID
 *   node scripts/runpod-siq-serverless.mjs list     # list your serverless endpoints
 *   node scripts/runpod-siq-serverless.mjs down <endpointId>
 *
 * Required env:
 *   RUNPOD_API_KEY   RunPod API key (Settings → API Keys)
 *   HF_TOKEN         Hugging Face token — SIQ-1-35B is a PRIVATE repo
 *
 * Tunable env (sane defaults for a 35B-A3B MoE in bf16):
 *   SIQ_MODEL_REPO   default AlexWortega/SIQ-1-35B
 *   SIQ_SERVED_NAME  default "siq"   (model tag the proxy/clients send)
 *   SIQ_GPU          default HOPPER_141,ADA_80_PRO  (H100/L40-class 80GB+; the cards the
 *                                       proven siq1-vllm-serverless endpoint runs on. A100
 *                                       AMPERE_80 was capacity-throttled in practice.)
 *   SIQ_WORKERS_MAX  default 3
 *   SIQ_WORKERS_MIN  default 0         (0 = scale-to-zero; cold starts on first hit)
 *   SIQ_IDLE         default 30        (seconds before idle workers scale down)
 *   SIQ_MAX_LEN      default 16384     (vLLM max_model_len; matches proven config)
 *   SIQ_GPU_UTIL     default 0.95
 *   SIQ_VLLM_IMAGE   default runpod/worker-v1-vllm:v2.22.4  (the version that serves the
 *                                       SIQ-1 MoE arch — the old v2.7.0 did NOT)
 *   SIQ_DTYPE        default bfloat16  (set to fp8 to fit a 48GB card)
 *   SIQ_CONTAINER_GB default 120       (container disk; holds the ~70GB bf16 download)
 *   SIQ_NETVOL_GB    default 0         (>0 = create a network volume to cache weights across
 *                                       cold starts; needs a datacenter with the GPU)
 *
 * After `up`, set the printed endpoint id on the Railway proxy:
 *   railway variables -s api --set RUNPOD_API_KEY=… --set SIQ_EID=<printed id> --set SIQ_MODEL=siq
 */
import process from "node:process";

const API = "https://api.runpod.io/graphql";
const KEY = process.env.RUNPOD_API_KEY;
const HF_TOKEN = process.env.HF_TOKEN || "";

const MODEL_REPO = process.env.SIQ_MODEL_REPO || "AlexWortega/SIQ-1-35B";
const SERVED_NAME = process.env.SIQ_SERVED_NAME || "siq";
const GPU = process.env.SIQ_GPU || "HOPPER_141,ADA_80_PRO";
const WORKERS_MAX = Number(process.env.SIQ_WORKERS_MAX || 3);
const WORKERS_MIN = Number(process.env.SIQ_WORKERS_MIN || 0);
const IDLE = Number(process.env.SIQ_IDLE || 30);
const MAX_LEN = Number(process.env.SIQ_MAX_LEN || 16384);
const GPU_UTIL = process.env.SIQ_GPU_UTIL || "0.95";
const VLLM_IMAGE = process.env.SIQ_VLLM_IMAGE || "runpod/worker-v1-vllm:v2.22.4";
const DTYPE = process.env.SIQ_DTYPE || "bfloat16";
const CONTAINER_GB = Number(process.env.SIQ_CONTAINER_GB || 120);
const NETVOL_GB = Number(process.env.SIQ_NETVOL_GB || 0);

if (!KEY) {
  console.error("RUNPOD_API_KEY is required");
  process.exit(2);
}

async function gql(query, variables = {}) {
  const res = await fetch(`${API}?api_key=${KEY}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, variables }),
  });
  const json = await res.json();
  if (json.errors) throw new Error("GraphQL: " + JSON.stringify(json.errors));
  return json.data;
}

/**
 * vLLM worker config is passed entirely as container env. These knobs are read
 * by runpod/worker-v1-vllm to launch the embedded vLLM OpenAI server.
 */
function workerEnv() {
  // Matches the proven `siq1-vllm-serverless` template (image v2.22.4) — the
  // minimal set that actually loads the SIQ-1 MoE arch — plus a served-name
  // override so the proxy can keep sending model:"siq".
  const env = [
    { key: "MODEL_NAME", value: MODEL_REPO },
    { key: "OPENAI_SERVED_MODEL_NAME_OVERRIDE", value: SERVED_NAME },
    { key: "TRUST_REMOTE_CODE", value: "true" },
    { key: "DTYPE", value: DTYPE },
    { key: "MAX_MODEL_LEN", value: String(MAX_LEN) },
    { key: "GPU_MEMORY_UTILIZATION", value: String(GPU_UTIL) },
  ];
  if (HF_TOKEN) env.push({ key: "HF_TOKEN", value: HF_TOKEN });
  return env;
}

async function saveTemplate(name) {
  const q = `mutation($input: SaveTemplateInput!) {
    saveTemplate(input: $input) { id name imageName }
  }`;
  const input = {
    name,
    imageName: VLLM_IMAGE,
    dockerArgs: "",
    containerDiskInGb: CONTAINER_GB,
    volumeInGb: 0, // serverless uses a network volume, set on the endpoint
    volumeMountPath: "/runpod-volume",
    ports: "",
    env: workerEnv(),
    isServerless: true,
    readme: `SIQ-1-35B (vLLM serverless) — ${MODEL_REPO}, served as "${SERVED_NAME}".`,
  };
  const data = await gql(q, { input });
  return data.saveTemplate;
}

async function createNetworkVolume(name) {
  // A network volume caches the ~70GB of weights across cold starts. Picks the
  // first datacenter that has the requested GPU; override with SIQ_DATACENTER.
  const dc = process.env.SIQ_DATACENTER || "EU-RO-1";
  const q = `mutation($input: CreateNetworkVolumeInput!) {
    createNetworkVolume(input: $input) { id name size dataCenterId }
  }`;
  try {
    const data = await gql(q, { input: { name, size: NETVOL_GB, dataCenterId: dc } });
    return data.createNetworkVolume;
  } catch (e) {
    process.stderr.write(`[warn] could not create network volume (${e.message}); endpoint will run without a weight cache (slower cold starts)\n`);
    return null;
  }
}

async function saveEndpoint(name, templateId, networkVolumeId) {
  const q = `mutation($input: EndpointInput!) {
    saveEndpoint(input: $input) { id name templateId workersMin workersMax }
  }`;
  const input = {
    name,
    templateId,
    gpuIds: GPU,
    gpuCount: 1,
    workersMin: WORKERS_MIN,
    workersMax: WORKERS_MAX,
    idleTimeout: IDLE,
    scalerType: "QUEUE_DELAY",
    scalerValue: 4,
    ...(networkVolumeId ? { networkVolumeId } : {}),
  };
  const data = await gql(q, { input });
  return data.saveEndpoint;
}

async function listEndpoints() {
  const q = `query { myself { endpoints {
    id name templateId workersMin workersMax idleTimeout gpuIds
  } } }`;
  const data = await gql(q);
  return data.myself.endpoints || [];
}

async function deleteEndpoint(id) {
  // workersMax must be 0 before delete is accepted.
  await gql(`mutation($input: EndpointInput!) { saveEndpoint(input: $input) { id } }`, {
    input: { id, workersMin: 0, workersMax: 0 },
  }).catch(() => {});
  await gql(`mutation($id: String!) { deleteEndpoint(id: $id) }`, { id });
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === "up") {
    if (!HF_TOKEN) {
      process.stderr.write("[warn] HF_TOKEN not set — SIQ-1-35B is a private repo; the worker will fail to pull weights without it.\n");
    }
    const tag = process.env.SIQ_NAME || "siq1-35b";
    process.stderr.write(`[siq] creating template (${VLLM_IMAGE}, model=${MODEL_REPO})…\n`);
    const tpl = await saveTemplate(`${tag}-tpl`);
    let vol = null;
    if (NETVOL_GB > 0) {
      process.stderr.write(`[siq] template ${tpl.id}\n[siq] creating ${NETVOL_GB}GB network volume…\n`);
      vol = await createNetworkVolume(`${tag}-cache`);
    } else {
      process.stderr.write(`[siq] template ${tpl.id}\n[siq] no network volume (SIQ_NETVOL_GB=0) — weights download to container disk on cold start\n`);
    }
    process.stderr.write(`[siq] creating serverless endpoint (gpu=${GPU}, workers ${WORKERS_MIN}-${WORKERS_MAX})…\n`);
    const ep = await saveEndpoint(tag, tpl.id, vol?.id);
    console.log(JSON.stringify({ endpointId: ep.id, name: ep.name, templateId: tpl.id, networkVolumeId: vol?.id || null }));
    process.stderr.write(
      `\n[siq] DONE. endpoint id = ${ep.id}\n` +
        `Set it on the Railway proxy:\n` +
        `  railway variables -s api --set RUNPOD_API_KEY=<key> --set SIQ_EID=${ep.id} --set SIQ_MODEL=${SERVED_NAME}\n` +
        `First request triggers a cold start (weights download to the volume) — can take several minutes.\n`,
    );
  } else if (cmd === "list") {
    for (const e of await listEndpoints()) console.log(JSON.stringify(e));
  } else if (cmd === "down") {
    const id = rest[0];
    if (!id) { console.error("down needs <endpointId>"); process.exit(2); }
    await deleteEndpoint(id);
    process.stderr.write(`[siq] deleted endpoint ${id}\n`);
  } else {
    console.error("usage: runpod-siq-serverless.mjs up | list | down <endpointId>");
    process.exit(2);
  }
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
