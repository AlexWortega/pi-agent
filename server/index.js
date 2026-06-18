import express from "express";
import pg from "pg";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { logTrace, tracehouseEnabled } from "./tracehouse.js";

const { Pool } = pg;

const PORT = process.env.PORT || 3000;
const DATABASE_URL = process.env.DATABASE_URL;
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || "";
const KEENABLE_API_KEY = process.env.KEENABLE_API_KEY || "";
const SEARCH_LIMIT_HOUR_CLIENT = parseInt(process.env.SEARCH_LIMIT_HOUR_CLIENT || "50", 10);
const SEARCH_LIMIT_HOUR_IP = parseInt(process.env.SEARCH_LIMIT_HOUR_IP || "200", 10);

// ---- SIQ-1 (RunPod serverless) proxy config -------------------------------
// The browser can't call RunPod directly (key + CORS + the async /run API), so
// this server bridges an OpenAI-compatible streaming route to the RunPod
// serverless vLLM endpoint, holding the key server-side. See server/README.md.
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || "";
const SIQ_EID = process.env.SIQ_EID || ""; // RunPod serverless endpoint id
const SIQ_MODEL = process.env.SIQ_MODEL || "siq"; // served-model-name on the worker
const SIQ_MINTOK = parseInt(process.env.SIQ_MINTOK || "2048", 10);
const SIQ_LIMIT_HOUR_CLIENT = parseInt(process.env.SIQ_LIMIT_HOUR_CLIENT || "60", 10);
const SIQ_LIMIT_HOUR_IP = parseInt(process.env.SIQ_LIMIT_HOUR_IP || "120", 10);
const siqEnabled = !!(RUNPOD_API_KEY && SIQ_EID);
const publicDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");

if (!DATABASE_URL) {
  console.error("FATAL: DATABASE_URL is not set");
  process.exit(1);
}

// Railway's private network (…​.railway.internal) needs no SSL; the public
// proxy URL does. Toggle with PGSSL=true if you point at the public host.
const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: process.env.PGSSL === "true" ? { rejectUnauthorized: false } : false,
});

async function initDb() {
  await pool.query(`
    create table if not exists requests (
      id            uuid primary key default gen_random_uuid(),
      client_id     text,
      project_id    text,
      project_name  text,
      model_id      text,
      prompt        text not null,
      response      text,
      has_artifact  boolean not null default false,
      user_agent    text,
      created_at    timestamptz not null default now(),
      updated_at    timestamptz not null default now()
    );
    create index if not exists requests_created_at_idx on requests (created_at desc);
    create index if not exists requests_client_idx on requests (client_id);

    create table if not exists searches (
      id          uuid primary key default gen_random_uuid(),
      client_id   text,
      ip          text,
      query       text not null,
      n_results   int,
      source      text,
      status      int,
      created_at  timestamptz not null default now()
    );
    create index if not exists searches_created_at_idx on searches (created_at desc);
    create index if not exists searches_client_idx on searches (client_id);
  `);
  console.log("db ready");
}

// ---- in-memory rate limiter (single-instance Railway service) -------------
const searchCounters = new Map(); // key -> [timestamps]
function rateLimit(key, limit, windowMs = 3_600_000) {
  const now = Date.now();
  const arr = (searchCounters.get(key) || []).filter((t) => now - t < windowMs);
  if (arr.length >= limit) return { ok: false, retryInMs: windowMs - (now - arr[0]) };
  arr.push(now);
  searchCounters.set(key, arr);
  return { ok: true };
}
setInterval(() => {
  const now = Date.now();
  for (const [k, arr] of searchCounters) {
    const fresh = arr.filter((t) => now - t < 3_600_000);
    if (fresh.length === 0) searchCounters.delete(k);
    else searchCounters.set(k, fresh);
  }
}, 600_000);

const app = express();
app.use(express.json({ limit: "4mb" }));

// CORS — public logging endpoint. Allow the known frontends; fall back to echo.
const ALLOWED = new Set([
  "https://alexwortega.github.io",
  "http://localhost:5050",
  "http://localhost:4173",
]);
app.use((req, res, next) => {
  const origin = req.headers.origin;
  res.setHeader("Access-Control-Allow-Origin", origin && ALLOWED.has(origin) ? origin : "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,PATCH,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

app.get("/api/health", (_req, res) =>
  res.json({ ok: true, search: !!KEENABLE_API_KEY, siq: siqEnabled }),
);

// ---- SIQ-1 inference proxy -------------------------------------------------
// OpenAI-compatible. `GET /api/siq/v1/models` + streaming `POST
// /api/siq/v1/chat/completions`. Bridges to the RunPod serverless vLLM worker's
// OpenAI passthrough (api.runpod.ai/v2/<eid>/openai/v1/...), which supports SSE.
app.get("/api/siq/v1/models", (_req, res) => {
  res.json({ object: "list", data: [{ id: SIQ_MODEL, object: "model", owned_by: "runpod-serverless" }] });
});

function clientIp(req) {
  return (
    (req.headers["x-forwarded-for"] || "").toString().split(",")[0].trim() ||
    req.socket.remoteAddress ||
    "unknown"
  ).slice(0, 64);
}

app.post("/api/siq/v1/chat/completions", async (req, res) => {
  if (!siqEnabled) {
    return res.status(503).json({ error: { message: "SIQ proxy not configured (RUNPOD_API_KEY / SIQ_EID)", type: "config" } });
  }
  const body = req.body || {};
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({ error: { message: "messages required", type: "invalid_request" } });
  }

  const cid = str(body.client_id, 64);
  const ip = clientIp(req);
  // Per-client limit only when a real id is present (else all anon users would
  // share one bucket); the per-IP limit always applies as the real backstop.
  if (cid && cid !== "anon") {
    const c = rateLimit("siqc:" + cid, SIQ_LIMIT_HOUR_CLIENT);
    if (!c.ok) {
      return res
        .status(429)
        .json({ error: { message: `rate limit: ${SIQ_LIMIT_HOUR_CLIENT}/hour per client`, type: "rate_limit", retry_in_s: Math.ceil(c.retryInMs / 1000) } });
    }
  }
  const i = rateLimit("siqi:" + ip, SIQ_LIMIT_HOUR_IP);
  if (!i.ok) {
    return res
      .status(429)
      .json({ error: { message: `rate limit: ${SIQ_LIMIT_HOUR_IP}/hour per IP`, type: "rate_limit", retry_in_s: Math.ceil(i.retryInMs / 1000) } });
  }

  // ---- shape the request (siq1.md §3): thinking toggle, effort, min tokens.
  const ctk = { ...(body.chat_template_kwargs || {}) };
  const thinking = ctk.enable_thinking !== false; // default on
  const messages = body.messages.map((m) => ({ role: m.role, content: m.content }));
  const effort = typeof body.effort === "string" ? body.effort : "";
  if (thinking && effort && !messages.some((m) => m.role === "system" && /Reasoning effort:/i.test(String(m.content || "")))) {
    // SIQ-1 obeys a trained "Reasoning effort: low|medium|high" system directive.
    messages.unshift({ role: "system", content: `Reasoning effort: ${effort}` });
  }

  const openaiInput = {
    model: SIQ_MODEL,
    messages,
    temperature: typeof body.temperature === "number" ? body.temperature : 0.7,
    max_tokens: Math.max(parseInt(body.max_tokens, 10) || 0, SIQ_MINTOK),
    top_p: typeof body.top_p === "number" ? body.top_p : 0.95,
    chat_template_kwargs: { enable_thinking: thinking },
  };
  if (typeof body.top_k === "number") openaiInput.top_k = body.top_k;
  if (typeof body.presence_penalty === "number") openaiInput.presence_penalty = body.presence_penalty;

  // SSE framing. RunPod serverless is async (/run + poll /status — NOT /runsync,
  // siq1.md §3.A), so we emit the completed result as one reasoning delta + one
  // content delta; the client engine accumulates delta.reasoning_content (folded
  // into <think>…</think>) and delta.content exactly as for a token stream.
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();
  const sse = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  const ka = setInterval(() => res.write(": keepalive\n\n"), 15_000);
  // Detect a real client disconnect via res 'close' BEFORE we finish — req
  // 'close' fires early under Express (body already buffered) and is NOT a
  // disconnect signal.
  const aborted = { v: false };
  let finished = false;
  res.on("close", () => { if (!finished) aborted.v = true; });
  try {
    const out = await siqRunAndPoll(openaiInput, aborted);
    const msg = out?.choices?.[0]?.message || out?.choices?.[0]?.delta || {};
    const content = msg.content ?? (typeof out === "string" ? out : "");
    const reasoning = msg.reasoning_content ?? "";
    if (reasoning) sse({ choices: [{ index: 0, delta: { reasoning_content: reasoning } }] });
    sse({ choices: [{ index: 0, delta: { content }, finish_reason: "stop" }] });
    res.write("data: [DONE]\n\n");
  } catch (e) {
    console.error("siq run/poll", e);
    sse({ error: { message: "siq upstream: " + (e?.message || e), type: "upstream" } });
  } finally {
    finished = true;
    clearInterval(ka);
    res.end();
  }
});

// Submit via RunPod async /run and poll /status to completion. The SIQ-1 GGUF
// worker takes the OpenAI chat request DIRECTLY as `input` — NOT the
// openai_route/openai_input envelope (that's for the vLLM worker; this worker
// ignores it and answers an empty prompt).
async function siqRunAndPoll(openaiInput, aborted, pollMs = 280_000) {
  const base = `https://api.runpod.ai/v2/${SIQ_EID}`;
  const h = { "Content-Type": "application/json", Authorization: `Bearer ${RUNPOD_API_KEY}` };
  const r = await fetch(`${base}/run`, {
    method: "POST",
    headers: h,
    body: JSON.stringify({ input: openaiInput }),
  });
  const job = await r.json();
  if (job.output) return job.output;
  const id = job.id;
  if (!id) throw new Error("no job id: " + JSON.stringify(job).slice(0, 200));
  const deadline = Date.now() + pollMs;
  while (Date.now() < deadline) {
    if (aborted.v) throw new Error("client aborted");
    await new Promise((res) => setTimeout(res, 1200));
    const st = await fetch(`${base}/status/${id}`, { headers: h }).then((x) => x.json());
    if (st.status === "COMPLETED") {
      const out = st.output;
      return Array.isArray(out) && out.length ? out[out.length - 1] : out;
    }
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(st.status)) {
      throw new Error(`job ${st.status}: ` + JSON.stringify(st).slice(0, 200));
    }
  }
  throw new Error("poll timeout");
}

// ---- search proxy: forwards to keenable.ai with server-side key -----------
app.post("/api/search", async (req, res) => {
  if (!KEENABLE_API_KEY) {
    return res.status(503).json({ error: "search proxy not configured" });
  }
  const { query, client_id } = req.body || {};
  if (!query || typeof query !== "string" || !query.trim()) {
    return res.status(400).json({ error: "query required" });
  }
  const q = query.trim().slice(0, 500);
  const cid = str(client_id, 64) || "anon";
  const ip = (
    (req.headers["x-forwarded-for"] || "").toString().split(",")[0].trim() ||
    req.socket.remoteAddress ||
    "unknown"
  ).slice(0, 64);

  const c = rateLimit("c:" + cid, SEARCH_LIMIT_HOUR_CLIENT);
  if (!c.ok) {
    return res
      .status(429)
      .json({ error: `rate limit: ${SEARCH_LIMIT_HOUR_CLIENT}/hour per client`, retry_in_s: Math.ceil(c.retryInMs / 1000) });
  }
  const i = rateLimit("i:" + ip, SEARCH_LIMIT_HOUR_IP);
  if (!i.ok) {
    return res
      .status(429)
      .json({ error: `rate limit: ${SEARCH_LIMIT_HOUR_IP}/hour per IP`, retry_in_s: Math.ceil(i.retryInMs / 1000) });
  }

  try {
    const r = await fetch("https://api.keenable.ai/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": KEENABLE_API_KEY },
      body: JSON.stringify({ query: q }),
    });
    let payload = null;
    if (r.ok) {
      const data = await r.json();
      const results = (data.results || []).slice(0, 8).map((x) => ({
        title: x.title,
        url: x.url,
        snippet: x.snippet || x.description || "",
      }));
      payload = { source: "keenable", query: q, results };
    } else {
      const txt = await r.text().catch(() => "");
      console.error("keenable upstream", r.status, txt.slice(0, 200));
    }
    // best-effort audit log
    pool
      .query(
        `insert into searches (client_id, ip, query, n_results, source, status) values ($1,$2,$3,$4,$5,$6)`,
        [cid, ip, q, payload ? payload.results.length : 0, payload ? "keenable" : "error", r.status],
      )
      .catch((e) => console.error("searches insert", e.message));
    if (!payload) return res.status(502).json({ error: "upstream error", status: r.status });
    res.json(payload);
  } catch (e) {
    console.error("POST /api/search", e);
    res.status(500).json({ error: "proxy error" });
  }
});

// Insert a new user request, return its id.
app.post("/api/log", async (req, res) => {
  try {
    const { client_id, project_id, project_name, model_id, prompt } = req.body || {};
    if (!prompt || typeof prompt !== "string") {
      return res.status(400).json({ error: "prompt required" });
    }
    const ua = (req.headers["user-agent"] || "").slice(0, 512);
    const { rows } = await pool.query(
      `insert into requests (client_id, project_id, project_name, model_id, prompt, user_agent)
       values ($1,$2,$3,$4,$5,$6) returning id`,
      [
        str(client_id, 64),
        str(project_id, 64),
        str(project_name, 200),
        str(model_id, 120),
        prompt.slice(0, 20000),
        ua,
      ],
    );
    res.json({ id: rows[0].id });
  } catch (e) {
    console.error("POST /api/log", e);
    res.status(500).json({ error: "db error" });
  }
});

// Attach the model's final answer to an existing request.
app.patch("/api/log/:id", async (req, res) => {
  try {
    const { response, has_artifact } = req.body || {};
    const { rows } = await pool.query(
      `update requests set response = $1, has_artifact = $2, updated_at = now() where id = $3
       returning client_id, project_name, model_id, prompt`,
      [typeof response === "string" ? response.slice(0, 200000) : null, !!has_artifact, req.params.id],
    );
    if (!rows.length) return res.status(404).json({ error: "not found" });
    res.json({ ok: true });

    // best-effort: mirror the completed trace to tracehouse (key is server-side only).
    // fire-and-forget — the response above already returned, so logging can't slow or
    // break the request even if tracehouse is down.
    if (tracehouseEnabled() && typeof response === "string") {
      const row = rows[0];
      const prompt = row.prompt || "";
      logTrace({
        project: row.project_name || "pi-agent",
        name: `trace-${req.params.id}`,
        clientRunId: `pi-agent-${req.params.id}`,
        config: {
          model_id: row.model_id,
          client_id: row.client_id,
          has_artifact: !!has_artifact,
          prompt: prompt.slice(0, 4000),
          response: response.slice(0, 8000),
        },
        metrics: { prompt_chars: prompt.length, response_chars: response.length },
      }).catch(() => {});
    }
  } catch (e) {
    console.error("PATCH /api/log", e);
    res.status(500).json({ error: "db error" });
  }
});

// ---- admin dashboard -------------------------------------------------------
function requireAdmin(req, res, next) {
  if (!ADMIN_TOKEN) {
    return res.status(503).json({ error: "dashboard disabled: ADMIN_TOKEN not set" });
  }
  const auth = req.headers.authorization || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : String(req.query.token || "");
  if (token !== ADMIN_TOKEN) return res.status(401).json({ error: "unauthorized" });
  next();
}

app.get("/api/recent", requireAdmin, async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit, 10) || 100, 500);
    const offset = parseInt(req.query.offset, 10) || 0;
    const q = String(req.query.q || "").slice(0, 200);
    const where = q ? "where prompt ilike $3 or response ilike $3" : "";
    const params = q ? [limit, offset, `%${q}%`] : [limit, offset];
    const { rows } = await pool.query(
      `select id, client_id, project_name, model_id, prompt, response, has_artifact, created_at, updated_at
         from requests ${where}
        order by created_at desc limit $1 offset $2`,
      params,
    );
    const { rows: s } = await pool.query(`
      select count(*)::int total,
             count(*) filter (where created_at > now() - interval '24 hours')::int today,
             count(distinct client_id)::int users,
             count(*) filter (where has_artifact)::int artifacts
        from requests`);
    res.json({ rows, stats: s[0] });
  } catch (e) {
    console.error("GET /api/recent", e);
    res.status(500).json({ error: "db error" });
  }
});

// Serve the dashboard (static). Data still requires the admin token above.
app.use(express.static(publicDir));

function str(v, n) {
  return typeof v === "string" ? v.slice(0, n) : null;
}

initDb()
  .then(() => app.listen(PORT, () => console.log(`pi-agent-api listening on :${PORT}`)))
  .catch((e) => {
    console.error("init failed", e);
    process.exit(1);
  });
