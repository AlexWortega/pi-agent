# pi-agent-api — request logger

Tiny Express + Postgres service that logs Pi Agent user requests. Deployed on Railway
(project **pi-agent-api**, service **api** + a **Postgres** plugin), live at
`https://api-production-bd22.up.railway.app`.

## Endpoints

- `GET  /api/health` → `{ ok: true }`
- `POST /api/log` `{ client_id, project_id, project_name, model_id, prompt }` → `{ id }`
- `PATCH /api/log/:id` `{ response, has_artifact }` → `{ ok: true }`

The frontend calls `POST` on send and `PATCH` when generation finishes
(`src/lib/logger.ts`). CORS is open to `https://alexwortega.github.io` and localhost.

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
