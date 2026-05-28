import { LOG_API } from "../config";
import { uid } from "./store";

/* ─────────────────────────────────────────────────────────────────────────────
 * TRANSPARENT LOGGING NOTICE
 *
 * This file ships in the public JS bundle on purpose — you (the user) should
 * be able to read exactly what we record. Every chat turn (your prompt + the
 * model's final answer) is POSTed to our Railway API (LOG_API) for analytics
 * and to catch interesting / broken behaviour. The same client_id below is
 * also sent with /api/search so we can rate-limit per browser.
 *
 * What we store, server-side, in Postgres:
 *   requests(client_id, project_id, project_name, model_id, prompt, response,
 *            has_artifact, user_agent, created_at, updated_at)
 *   searches(client_id, ip, query, n_results, source, status, created_at)
 *
 * What we do NOT store: anything from your localStorage (memories, skills,
 * todos, schedules) — those never leave your browser.
 *
 * Disable everything by setting VITE_LOG_API="" at build time, or just block
 * the LOG_API origin in your browser.
 * ─────────────────────────────────────────────────────────────────────────── */

const CLIENT_KEY = "piagent.clientId.v1";

/** Stable anonymous id for this browser (not tied to any identity).
 *  Shared with the search proxy for per-client rate-limiting. */
export function clientId(): string {
  let id = "";
  try {
    id = localStorage.getItem(CLIENT_KEY) || "";
    if (!id) {
      id = uid();
      localStorage.setItem(CLIENT_KEY, id);
    }
  } catch {
    id = "anon";
  }
  return id;
}

export interface LogMeta {
  projectId: string;
  projectName: string;
  modelId: string;
}

/**
 * Fire-and-forget: record a user request, resolve with its row id (or null on
 * failure). Never throws — logging must not affect the chat.
 */
export async function logRequest(prompt: string, meta: LogMeta): Promise<string | null> {
  if (!LOG_API) return null;
  try {
    const r = await fetch(`${LOG_API}/api/log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: clientId(),
        project_id: meta.projectId,
        project_name: meta.projectName,
        model_id: meta.modelId,
        prompt,
      }),
      keepalive: true,
    });
    if (!r.ok) return null;
    const d = await r.json();
    return d.id ?? null;
  } catch {
    return null;
  }
}

/** Attach the model's final answer to a previously logged request. */
export async function logResponse(
  id: string | null,
  response: string,
  hasArtifact: boolean,
): Promise<void> {
  if (!LOG_API || !id) return;
  try {
    await fetch(`${LOG_API}/api/log/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ response, has_artifact: hasArtifact }),
      keepalive: true,
    });
  } catch {
    /* ignore */
  }
}
