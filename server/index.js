import express from "express";
import pg from "pg";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { Pool } = pg;

const PORT = process.env.PORT || 3000;
const DATABASE_URL = process.env.DATABASE_URL;
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || "";
const KEENABLE_API_KEY = process.env.KEENABLE_API_KEY || "";
const SEARCH_LIMIT_HOUR_CLIENT = parseInt(process.env.SEARCH_LIMIT_HOUR_CLIENT || "50", 10);
const SEARCH_LIMIT_HOUR_IP = parseInt(process.env.SEARCH_LIMIT_HOUR_IP || "200", 10);
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
  res.json({ ok: true, search: !!KEENABLE_API_KEY }),
);

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
    const { rowCount } = await pool.query(
      `update requests set response = $1, has_artifact = $2, updated_at = now() where id = $3`,
      [typeof response === "string" ? response.slice(0, 200000) : null, !!has_artifact, req.params.id],
    );
    if (!rowCount) return res.status(404).json({ error: "not found" });
    res.json({ ok: true });
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
