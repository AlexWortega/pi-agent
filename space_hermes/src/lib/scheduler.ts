import type { Schedule } from "../types";
import { readJSON, writeJSON, uid } from "./store";

const KEY = "hermes.schedules.v1";
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}
export function subscribeSchedules(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function listSchedules(): Schedule[] {
  return readJSON<Schedule[]>(KEY, []);
}

export function addSchedule(prompt: string, everyMs: number, projectId: string): Schedule {
  const items = listSchedules();
  const s: Schedule = { id: uid(), prompt: prompt.trim(), everyMs, nextRun: Date.now() + everyMs, projectId };
  items.push(s);
  writeJSON(KEY, items);
  emit();
  return s;
}

export function removeSchedule(id: string): void {
  writeJSON(
    KEY,
    listSchedules().filter((s) => s.id !== id),
  );
  emit();
}

/** Schedules whose nextRun has passed; also advances their nextRun. */
export function dueSchedules(now = Date.now()): Schedule[] {
  const items = listSchedules();
  const due: Schedule[] = [];
  let changed = false;
  for (const s of items) {
    if (s.nextRun <= now) {
      due.push(s);
      s.lastRun = now;
      s.nextRun = now + s.everyMs;
      changed = true;
    }
  }
  if (changed) {
    writeJSON(KEY, items);
    emit();
  }
  return due;
}

/** Parse "every 30s" / "every 5m" / "30s" / "5m" / "1h" → milliseconds. */
export function parseEvery(spec: string): number | null {
  const m = spec.trim().match(/(?:every\s+)?(\d+)\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)?/i);
  if (!m) return null;
  const n = Number(m[1]);
  const unit = (m[2] || "m").toLowerCase();
  if (unit.startsWith("s")) return n * 1000;
  if (unit.startsWith("h")) return n * 3600_000;
  return n * 60_000; // minutes default
}
