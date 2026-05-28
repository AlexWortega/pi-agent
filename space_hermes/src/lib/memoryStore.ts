import type { MemoryItem } from "../types";
import { readJSON, writeJSON, uid } from "./store";

const KEY = "hermes.memory.v1";
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}
export function subscribeMemory(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function listMemory(): MemoryItem[] {
  return readJSON<MemoryItem[]>(KEY, []);
}

export function saveMemory(text: string, key?: string): MemoryItem {
  const items = listMemory();
  const item: MemoryItem = { id: uid(), text: text.trim(), key: key?.trim() || undefined, ts: Date.now() };
  // de-dupe by key
  const next = key ? items.filter((i) => i.key !== key) : items;
  next.push(item);
  writeJSON(KEY, next);
  emit();
  return item;
}

export function deleteMemory(id: string): void {
  writeJSON(
    KEY,
    listMemory().filter((i) => i.id !== id),
  );
  emit();
}

function score(item: MemoryItem, q: string): number {
  const hay = `${item.key ?? ""} ${item.text}`.toLowerCase();
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  return terms.reduce((s, t) => (hay.includes(t) ? s + 1 : s), 0);
}

export function searchMemory(query: string, limit = 5): MemoryItem[] {
  if (!query.trim()) return listMemory().slice(-limit).reverse();
  return listMemory()
    .map((i) => ({ i, s: score(i, query) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, limit)
    .map((x) => x.i);
}

/** Compact block of the most recent memories for injection into the system prompt. */
export function memoryBlock(limit = 8): string {
  const items = listMemory().slice(-limit);
  return items.map((i) => `- ${i.key ? i.key + ": " : ""}${i.text}`).join("\n");
}
