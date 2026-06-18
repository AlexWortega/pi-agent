# pi-agent-api — request logger

Tiny Express + Postgres service that logs Pi Agent user requests. Deployed on Railway
(project **pi-agent-api**, service **api** + a **Postgres** plugin), live at
`https://api-production-bd22.up.railway.app`.

## Endpoints

- `GET  /api/health` → `{ ok, search, siq }`
- `POST /api/log` `{ client_id, project_id, project_name, model_id, prompt }` → `{ id }`
- `PATCH /api/log/:id` `{ response, has_artifact }` → `{ ok: true }`
- `POST /api/search` `{ query, client_id }` → keenable proxy (key server-side)
- `GET  /api/siq/v1/models` · `POST /api/siq/v1/chat/completions` → **SIQ-1 inference proxy** (below)

The frontend calls `POST` on send and `PATCH` when generation finishes
(`src/lib/logger.ts`). CORS is open to `https://alexwortega.github.io` and localhost.

## SIQ-1 inference proxy

The web app's "SIQ-1-35B (cloud)" model streams through `POST /api/siq/v1/chat/completions`.
The browser can't call RunPod directly (the API key would leak, RunPod sends no CORS
headers, and its serverless API is async `/run`+poll). This route holds the key
server-side, submits via RunPod async **`/run` + poll `/status`** (the `openai_input`
envelope the SIQ-1 serverless worker accepts), and re-emits the completed result as an
**OpenAI-compatible SSE stream** (`reasoning_content` delta + `content` delta + `[DONE]`).

> Note: the same proxy currently runs as an HF Space (`siq_proxy_space/`,
> `AlexWortega/siq-proxy`) because the Railway trial is expired. This route is the
> Railway-hosted equivalent for when that's available again. See `../siq1-web.md`.

It applies the SIQ-1 request shaping from `siq1.md`: a `Reasoning effort: <effort>`
system line when thinking is on, an `enable_thinking` chat-template toggle, and a
`max_tokens` floor. It is rate-limited per client/IP. Configure on the `api` service:

```bash
railway variables -s api \
  --set RUNPOD_API_KEY=<your runpod key> \
  --set SIQ_EID=<serverless endpoint id from scripts/runpod-siq-serverless.mjs>
# optional:
#   SIQ_MODEL    served-model-name on the worker (default "siq")
#   SIQ_MINTOK   max_tokens floor (default 2048)
#   SIQ_LIMIT_HOUR_CLIENT / SIQ_LIMIT_HOUR_IP   rate limits (default 60 / 120)
```

Leave `RUNPOD_API_KEY`/`SIQ_EID` unset to disable the route (returns 503; the cloud
model in the picker just won't work, the in-browser Soyuz model is unaffected).
The frontend points at `${VITE_LOG_API}/api/siq` by default — override with `VITE_SIQ_API`.

## Schema

`requests(id uuid, client_id, project_id, project_name, model_id, prompt, response,
has_artifact bool, user_agent, created_at, updated_at)` — auto-created on boot.

## Looking at the data

```bash
# open a psql shell against the DB (Railway injects the connection):
railway connect Postgres        # from this dir, project must be linked

# or with a local psql + the public URL:
psql "$(railway variables -s Postgres --json | jq -r .DATABASE_PUBLIC_URL)" \
  -c "select created_at, model_id, left(prompt,60), has_artifact from requests order by created_at desc limit 20;"
```

## Redeploy after changes

```bash
cd server
railway up -s api -c          # uploads this dir, builds via Nixpacks, deploys
```

`DATABASE_URL` on the `api` service references `${{Postgres.DATABASE_URL}}` (private
network, no SSL). Set `PGSSL=true` only if you point at the public host.

### tracehouse logging (optional)

Each completed trace (`PATCH /api/log/:id`) is mirrored to
[tracehouse](https://tracehouse.ai) as a run — prompt/response/model in the run config,
char counts as metrics. Best-effort and non-blocking: if tracehouse is down or the key
is unset, request logging is unaffected. Set on the `api` service only (the key stays
server-side, never in the frontend bundle):

```bash
railway variables -s api --set TRACEHOUSE_API_KEY=<your key>
# optional, defaults to https://tracehouse.ai
railway variables -s api --set TRACEHOUSE_API_BASE=https://tracehouse.ai
```

Leave `TRACEHOUSE_API_KEY` unset to disable.
