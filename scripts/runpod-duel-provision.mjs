#!/usr/bin/env node
/**
 * runpod-duel-provision.mjs — provision / list / terminate RunPod A6000 pods for
 * the autoresearch duel via the RunPod GraphQL API.
 *
 *   node scripts/runpod-duel-provision.mjs up   [--name <n>] [--count 2]
 *   node scripts/runpod-duel-provision.mjs list
 *   node scripts/runpod-duel-provision.mjs ssh   <podId>
 *   node scripts/runpod-duel-provision.mjs down  <podId|--all>
 *
 * Env: RUNPOD_API_KEY (required). Your RunPod account must have an SSH public key
 * configured (Settings → SSH Public Keys) so the pod accepts `ssh root@<ip> -p <port>`.
 *
 * Prints a JSON line per pod: {name, id, ip, sshPort, ssh}. The duel scripts read
 * these to connect. GPU: "NVIDIA RTX A6000". Image: a CUDA+PyTorch devel image.
 */
import process from "node:process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const API = "https://api.runpod.io/graphql";
const KEY = process.env.RUNPOD_API_KEY;
const GPU = process.env.RUNPOD_GPU || "NVIDIA RTX A6000";
// parameter-golf's baseline uses scaled_dot_product_attention(enable_gqa=…) → needs torch>=2.5.
const IMAGE = process.env.RUNPOD_IMAGE || "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04";
const CLOUD = process.env.RUNPOD_CLOUD || "ALL"; // ALL | SECURE | COMMUNITY
const DISK_GB = Number(process.env.RUNPOD_DISK_GB || 80);
const VOL_GB = Number(process.env.RUNPOD_VOL_GB || 60);

if (!KEY) { console.error("RUNPOD_API_KEY is required"); process.exit(2); }

// SSH access is granted by injecting our public key as PUBLIC_KEY/SSH_KEY env
// (the runpod/pytorch image's entrypoint installs it + starts sshd). Without
// this the pod has no authorized key and ssh fails. (Mirrors create_pod.py.)
function readPubKey() {
  for (const f of ["id_ed25519.pub", "id_rsa.pub"]) {
    const p = path.join(os.homedir(), ".ssh", f);
    if (fs.existsSync(p)) return fs.readFileSync(p, "utf8").trim();
  }
  throw new Error("no SSH public key at ~/.ssh/id_ed25519.pub or id_rsa.pub");
}
const PUBKEY = readPubKey();

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

async function deploy(name) {
  // podFindAndDeployOnDemand: find an available host with the GPU and start a pod.
  const q = `mutation($in: PodFindAndDeployOnDemandInput!) {
    podFindAndDeployOnDemand(input: $in) { id imageName machineId }
  }`;
  const input = {
    cloudType: CLOUD,
    gpuCount: 1,
    gpuTypeId: GPU,
    minMemoryInGb: 32,
    minVcpuCount: 8,
    name,
    imageName: IMAGE,
    containerDiskInGb: DISK_GB,
    volumeInGb: VOL_GB,
    volumeMountPath: "/workspace",
    ports: process.env.RUNPOD_PORTS || "22/tcp,8080/http",
    startSsh: true,
    dockerArgs: "",
    env: [
      { key: "PUBLIC_KEY", value: PUBKEY },
      { key: "SSH_KEY", value: PUBKEY },
    ],
  };
  const data = await gql(q, { in: input });
  return data.podFindAndDeployOnDemand;
}

async function podInfo(id) {
  const q = `query pod($input: PodFilter!) { pod(input: $input) {
    id name desiredStatus
    runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
  } }`;
  const data = await gql(q, { input: { podId: id } });
  return data.pod;
}

function sshFromPod(pod) {
  const p = pod?.runtime?.ports?.find((x) => x.privatePort === 22 && x.isIpPublic);
  if (!p) return null;
  return { ip: p.ip, sshPort: p.publicPort, ssh: `ssh -o StrictHostKeyChecking=no root@${p.ip} -p ${p.publicPort}` };
}

async function waitForSsh(id, timeoutMs = 240000) {
  const t0 = Date.now();
  for (;;) {
    const pod = await podInfo(id);
    const ssh = sshFromPod(pod);
    if (ssh) return { pod, ...ssh };
    if (Date.now() - t0 > timeoutMs) throw new Error(`pod ${id} did not expose SSH within ${timeoutMs / 1000}s`);
    await new Promise((r) => setTimeout(r, 5000));
  }
}

async function listPods() {
  const q = `query { myself { pods { id name desiredStatus gpuCount
    runtime { ports { ip isIpPublic privatePort publicPort type } } } } }`;
  const data = await gql(q);
  return data.myself.pods;
}

async function terminate(id) {
  await gql(`mutation($id: String!) { podTerminate(input: {podId: $id}) }`, { id });
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === "up") {
    const nameArg = rest.includes("--name") ? rest[rest.indexOf("--name") + 1] : "duel";
    const count = rest.includes("--count") ? Number(rest[rest.indexOf("--count") + 1]) : 2;
    const names = count === 2 ? [`${nameArg}-siq`, `${nameArg}-glm`] : Array.from({ length: count }, (_, i) => `${nameArg}-${i}`);
    const out = [];
    for (const name of names) {
      const dep = await deploy(name);
      process.stderr.write(`deployed ${name} → ${dep.id}, waiting for SSH…\n`);
      const ready = await waitForSsh(dep.id);
      const rec = { name, id: dep.id, ip: ready.ip, sshPort: ready.sshPort, ssh: ready.ssh };
      out.push(rec);
      console.log(JSON.stringify(rec));
    }
    process.stderr.write(`\nProvisioned ${out.length} pod(s). Next: scripts/duel-setup.sh + scripts/duel-run.sh\n`);
  } else if (cmd === "list") {
    for (const p of await listPods()) {
      const ssh = sshFromPod(p);
      console.log(JSON.stringify({ id: p.id, name: p.name, status: p.desiredStatus, ...(ssh || {}) }));
    }
  } else if (cmd === "ssh") {
    const pod = await podInfo(rest[0]);
    const ssh = sshFromPod(pod);
    if (!ssh) { console.error("no public SSH port"); process.exit(1); }
    console.log(ssh.ssh);
  } else if (cmd === "down") {
    if (rest[0] === "--all") {
      for (const p of await listPods()) { await terminate(p.id); process.stderr.write(`terminated ${p.id}\n`); }
    } else if (rest[0]) {
      await terminate(rest[0]); process.stderr.write(`terminated ${rest[0]}\n`);
    } else { console.error("down needs <podId> or --all"); process.exit(2); }
  } else {
    console.error("usage: runpod-duel-provision.mjs up|list|ssh <id>|down <id|--all>");
    process.exit(2);
  }
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
