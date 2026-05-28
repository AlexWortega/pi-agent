import type { TodoItem } from "../types";
import { readJSON, writeJSON, uid } from "./store";

const KEY = "hermes.todos.v1";
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}
export function subscribeTodos(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function listTodos(): TodoItem[] {
  return readJSON<TodoItem[]>(KEY, []);
}

export function addTodo(text: string): TodoItem {
  const items = listTodos();
  const item: TodoItem = { id: uid(), text: text.trim(), done: false, ts: Date.now() };
  items.push(item);
  writeJSON(KEY, items);
  emit();
  return item;
}

export function toggleTodo(id: string, done?: boolean): void {
  const items = listTodos().map((t) => (t.id === id ? { ...t, done: done ?? !t.done } : t));
  writeJSON(KEY, items);
  emit();
}

export function removeTodo(id: string): void {
  writeJSON(
    KEY,
    listTodos().filter((t) => t.id !== id),
  );
  emit();
}

export function clearDone(): void {
  writeJSON(
    KEY,
    listTodos().filter((t) => !t.done),
  );
  emit();
}
