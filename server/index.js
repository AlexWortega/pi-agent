import express from "express";
import pg from "pg";

const { Pool } = pg;

const PORT = process.env.PORT || 3000;
const DATABASE_URL = process.env.DATABASE_URL;

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
  `);
  console.log("db ready");
}

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

app.get("/api/health", (_req, res) => res.json({ ok: true }));

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

function str(v, n) {
  return typeof v === "string" ? v.slice(0, n) : null;
}

initDb()
  .then(() => app.listen(PORT, () => console.log(`pi-agent-api listening on :${PORT}`)))
  .catch((e) => {
    console.error("init failed", e);
    process.exit(1);
  });
