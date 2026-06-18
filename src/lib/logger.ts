import { LOG_API } from "../config";
import { uid } from "./store";

const CLIENT_KEY = "piagent.clientId.v1";

/** Stable anonymous id for this browser (not tied to any identity). */
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
