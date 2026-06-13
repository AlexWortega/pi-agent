// Forward completed Pi Agent traces to tracehouse (https://tracehouse.ai).
//
// SERVER-SIDE ONLY. The API key lives in the TRACEHOUSE_API_KEY env var (set it on
// Railway, never in the frontend bundle or git). The browser keeps logging to
// /api/log as before; this backend mirrors each completed trace into tracehouse so
// the key is never exposed to the client.
//
// Minimal port of the tracehouse REST API used by the Python SDK:
//   POST  /v1/runs                      {name, project, config}        -> {run_id}
//   POST  /v1/runs/:id/metrics          {points:[{key,step,value,wall_time}]}
//   PATCH /v1/runs/:id                  {status, ended_at}
// Auth: Authorization: Bearer <key>.

const BASE = (process.env.TRACEHOUSE_API_BASE || "https://tracehouse.ai").replace(/\/+$/, "");
const KEY = process.env.TRACEHOUSE_API_KEY || "";

export function tracehouseEnabled() {
  return !!KEY;
}

async function req(method, path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`tracehouse ${method} ${path} -> ${r.status} ${t}`.slice(0, 300));
  }
  return r.json().catch(() => ({}));
}

// Log one completed trace as a tracehouse run. Best-effort: never throws into the
// caller, so a tracehouse outage can't break request logging.
export async function logTrace({ project, name, config, metrics, clientRunId }) {
  if (!KEY) return;
  try {
    // client_run_id is the idempotency key the server dedupes on — give each trace a
    // unique one (the request id) so traces land as distinct runs, not one shared run.
    const run = await req("POST", "/v1/runs", {
      name,
      project: project || "pi-agent",
      client_run_id: clientRunId || name || (globalThis.crypto?.randomUUID?.() ?? String(Date.now())),
      started_at: new Date().toISOString(),
      config: config || {},
    });
    const id = run && (run.id || run.run_id);
    if (!id) return;
    const wall = new Date().toISOString();
    const points = Object.entries(metrics || {})
      .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
      .map(([key, value]) => ({ key, step: 0, value, wall_time: wall }));
    if (points.length) {
      await req("POST", `/v1/runs/${id}/metrics`, { points });
    }
    await req("PATCH", `/v1/runs/${id}`, { status: "finished", ended_at: new Date().toISOString() });
  } catch (e) {
    console.error("[tracehouse] logTrace failed:", String(e).slice(0, 200));
  }
}
