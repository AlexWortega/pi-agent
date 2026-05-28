import { useEffect, useState } from "react";
import type { Skill, MemoryItem, TodoItem, Schedule } from "../types";
import { listSkills, upsertSkill, deleteSkill, subscribeSkills } from "../lib/skillsStore";
import { listMemory, saveMemory, deleteMemory, subscribeMemory } from "../lib/memoryStore";
import { listTodos, addTodo, toggleTodo, removeTodo, clearDone, subscribeTodos } from "../lib/todoStore";
import { listSchedules, removeSchedule, subscribeSchedules } from "../lib/scheduler";
import { ALL_TOOLS } from "../agent/registry";
import { getSettings, setSettings, toggleTool, subscribeSettings } from "../lib/settings";
import type { AgentSettings } from "../lib/settings";
import { X, Trash, Plus, Sparkles, Brain, ListTodo, Wrench, Clock, Check } from "./Icons";

/** subscribe to a store and re-read its snapshot on change */
function useStore<T>(read: () => T, subscribe: (cb: () => void) => () => void): T {
  const [v, setV] = useState(read);
  useEffect(() => subscribe(() => setV(read())), []); // eslint-disable-line
  return v;
}

function Shell({ title, icon, onClose, children }: { title: string; icon: React.ReactNode; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        className="glass rounded-2xl border border-[var(--color-edge-2)] w-full max-w-xl max-h-[86vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3.5 flex items-center gap-2.5 border-b border-[var(--color-edge)]">
          <span className="w-7 h-7 rounded-lg grid place-items-center bg-[var(--color-pi)]/15 text-[var(--color-pi-2)]">{icon}</span>
          <h2 className="font-semibold text-[15px] flex-1">{title}</h2>
          <button className="btn btn-ghost p-2" onClick={onClose}>
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="overflow-y-auto p-4 space-y-2.5">{children}</div>
      </div>
    </div>
  );
}

export function SkillsPanel({ onClose }: { onClose: () => void }) {
  const skills = useStore<Skill[]>(listSkills, subscribeSkills);
  const [editing, setEditing] = useState<Skill | null>(null);
  const [draft, setDraft] = useState({ name: "", description: "", body: "" });

  const startNew = () => {
    setEditing({ id: "", name: "", description: "", body: "", createdBy: "user", ts: 0 });
    setDraft({ name: "", description: "", body: "" });
  };

  return (
    <Shell title="Skills" icon={<Sparkles className="w-4 h-4" />} onClose={onClose}>
      <p className="text-[11.5px] text-[var(--color-ink-faint)]">
        Reusable markdown instructions (agentskills.io style). Relevant skills are injected when you chat; the agent can also create its own via the
        <code className="mx-1">skill_manage</code> tool.
      </p>
      {editing ? (
        <div className="rounded-xl border border-[var(--color-edge)] p-3 space-y-2">
          <input className="field text-[13px]" placeholder="skill-name" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          <input className="field text-[13px]" placeholder="short description" value={draft.description} onChange={(e) => setDraft({ ...draft, description: e.target.value })} />
          <textarea className="field text-[13px] min-h-[120px] font-[var(--font-mono)]" placeholder="markdown instructions…" value={draft.body} onChange={(e) => setDraft({ ...draft, body: e.target.value })} />
          <div className="flex gap-2 justify-end">
            <button className="btn btn-ghost" onClick={() => setEditing(null)}>Cancel</button>
            <button
              className="btn btn-primary"
              disabled={!draft.name.trim() || !draft.body.trim()}
              onClick={() => {
                upsertSkill({ name: draft.name.trim(), description: draft.description.trim(), body: draft.body, createdBy: "user" });
                setEditing(null);
              }}
            >
              Save skill
            </button>
          </div>
        </div>
      ) : (
        <button className="btn btn-ghost w-full justify-center" onClick={startNew}>
          <Plus className="w-4 h-4" /> New skill
        </button>
      )}
      {skills.map((s) => (
        <div key={s.id} className="rounded-xl border border-[var(--color-edge)] p-3">
          <div className="flex items-center gap-2">
            <span className="font-[var(--font-mono)] text-[13px] text-[var(--color-ink)]">{s.name}</span>
            <span className="text-[9px] px-1.5 py-px rounded-full border border-[var(--color-edge-2)] text-[var(--color-ink-faint)]">{s.createdBy}</span>
            <button
              className="ml-auto text-[var(--color-ink-faint)] hover:text-[var(--color-pi-2)]"
              onClick={() => {
                setEditing(s);
                setDraft({ name: s.name, description: s.description, body: s.body });
              }}
            >
              edit
            </button>
            <button className="text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)]" onClick={() => deleteSkill(s.id)}>
              <Trash className="w-3.5 h-3.5" />
            </button>
          </div>
          {s.description && <div className="text-[12px] text-[var(--color-ink-dim)] mt-1">{s.description}</div>}
        </div>
      ))}
    </Shell>
  );
}

export function MemoryPanel({ onClose }: { onClose: () => void }) {
  const memories = useStore<MemoryItem[]>(listMemory, subscribeMemory);
  const [text, setText] = useState("");
  return (
    <Shell title="Memory" icon={<Brain className="w-4 h-4" />} onClose={onClose}>
      <p className="text-[11.5px] text-[var(--color-ink-faint)]">Facts the agent persists across sessions (its Honcho-style user model), stored locally in your browser.</p>
      <div className="flex gap-2">
        <input className="field text-[13px]" placeholder="Remember that…" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && text.trim() && (saveMemory(text.trim()), setText(""))} />
        <button className="btn btn-primary shrink-0" disabled={!text.trim()} onClick={() => { saveMemory(text.trim()); setText(""); }}>
          <Plus className="w-4 h-4" />
        </button>
      </div>
      {memories.length === 0 && <div className="text-[12px] text-[var(--color-ink-faint)] text-center py-4">No memories yet.</div>}
      {memories.map((m) => (
        <div key={m.id} className="rounded-xl border border-[var(--color-edge)] p-2.5 flex items-start gap-2">
          <div className="flex-1 min-w-0 text-[13px] text-[var(--color-ink-dim)]">
            {m.key && <span className="text-[var(--color-pi-2)] font-medium">{m.key}: </span>}
            {m.text}
          </div>
          <button className="text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)]" onClick={() => deleteMemory(m.id)}>
            <Trash className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </Shell>
  );
}

export function KanbanPanel({ onClose }: { onClose: () => void }) {
  const todos = useStore<TodoItem[]>(listTodos, subscribeTodos);
  const [text, setText] = useState("");
  return (
    <Shell title="Tasks" icon={<ListTodo className="w-4 h-4" />} onClose={onClose}>
      <div className="flex gap-2">
        <input className="field text-[13px]" placeholder="Add a task…" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && text.trim() && (addTodo(text.trim()), setText(""))} />
        <button className="btn btn-primary shrink-0" disabled={!text.trim()} onClick={() => { addTodo(text.trim()); setText(""); }}>
          <Plus className="w-4 h-4" />
        </button>
      </div>
      {todos.length === 0 && <div className="text-[12px] text-[var(--color-ink-faint)] text-center py-4">No tasks. The agent can add some with the <code>todo</code> tool.</div>}
      {todos.map((t) => (
        <div key={t.id} className="rounded-xl border border-[var(--color-edge)] p-2.5 flex items-center gap-2.5">
          <button
            className={`w-5 h-5 rounded-md border grid place-items-center shrink-0 ${t.done ? "bg-[var(--color-mint)]/20 border-[var(--color-mint)] text-[var(--color-mint)]" : "border-[var(--color-edge-2)] text-transparent"}`}
            onClick={() => toggleTodo(t.id)}
          >
            <Check className="w-3 h-3" />
          </button>
          <span className={`flex-1 text-[13px] ${t.done ? "line-through text-[var(--color-ink-faint)]" : "text-[var(--color-ink-dim)]"}`}>{t.text}</span>
          <button className="text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)]" onClick={() => removeTodo(t.id)}>
            <Trash className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      {todos.some((t) => t.done) && (
        <button className="btn btn-ghost w-full justify-center" onClick={clearDone}>
          Clear completed
        </button>
      )}
    </Shell>
  );
}

export function ToolsInspector({ onClose }: { onClose: () => void }) {
  const settings = useStore<AgentSettings>(getSettings, subscribeSettings);
  return (
    <Shell title="Tools" icon={<Wrench className="w-4 h-4" />} onClose={onClose}>
      <p className="text-[11.5px] text-[var(--color-ink-faint)]">
        {ALL_TOOLS.length} tools the in-browser agent can call. Toggle any off, or disable all network tools.
      </p>
      <label className="flex items-center gap-2 border border-[var(--color-edge)] p-2.5 cursor-pointer">
        <input type="checkbox" className="accent-[var(--color-pi)]" checked={settings.webToolsEnabled} onChange={(e) => setSettings({ webToolsEnabled: e.target.checked })} />
        <span className="text-[13px] text-[var(--color-ink-dim)] normal-text">Allow network tools (web_search / web_extract)</span>
      </label>
      <div className="border border-[var(--color-edge)] p-2.5">
        <div className="label-faint mb-1">web_search · keenable.ai api key</div>
        <input
          type="password"
          className="field text-[12px]"
          placeholder={(import.meta.env.VITE_KEENABLE_API_KEY as string | undefined) ? "(baked-in default in use — override here)" : "keen_..."}
          value={settings.keenableApiKey ?? ""}
          onChange={(e) => setSettings({ keenableApiKey: e.target.value })}
        />
        <div className="text-[10.5px] text-[var(--color-ink-faint)] mt-1.5 leading-snug normal-text">
          Stored only in your browser (localStorage). Without a key, web_search falls back to DuckDuckGo Instant Answer + Wikipedia.
        </div>
      </div>
      {ALL_TOOLS.map((t) => {
        const off = settings.disabledTools.includes(t.name) || (t.web && !settings.webToolsEnabled);
        return (
          <div key={t.name} className={`rounded-xl border border-[var(--color-edge)] p-2.5 ${off ? "opacity-50" : ""}`}>
            <div className="flex items-center gap-2">
              <span className="font-[var(--font-mono)] text-[12.5px] text-[var(--color-ink)]">{t.name}</span>
              {t.web && <span className="text-[9px] px-1.5 py-px rounded-full border border-[var(--color-soyuz)]/40 text-[var(--color-soyuz)]">net</span>}
              {t.orchestratorOnly && <span className="text-[9px] px-1.5 py-px rounded-full border border-[var(--color-edge-2)] text-[var(--color-ink-faint)]">orchestrator</span>}
              <label className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-[var(--color-ink-faint)] cursor-pointer">
                <input
                  type="checkbox"
                  className="accent-[var(--color-pi)]"
                  checked={!settings.disabledTools.includes(t.name)}
                  onChange={(e) => toggleTool(t.name, e.target.checked)}
                />
                on
              </label>
            </div>
            <div className="text-[11.5px] text-[var(--color-ink-faint)] mt-1 leading-snug">{t.description}</div>
          </div>
        );
      })}
    </Shell>
  );
}

export function SchedulesPanel({ onClose }: { onClose: () => void }) {
  const schedules = useStore<Schedule[]>(listSchedules, subscribeSchedules);
  return (
    <Shell title="Scheduled prompts" icon={<Clock className="w-4 h-4" />} onClose={onClose}>
      <p className="text-[11.5px] text-[var(--color-ink-faint)]">
        Re-run a prompt on an interval with <code>/schedule every 5m &lt;prompt&gt;</code>. Browser-only: these fire only while this tab is open (no real
        background cron without a server).
      </p>
      {schedules.length === 0 && <div className="text-[12px] text-[var(--color-ink-faint)] text-center py-4">No schedules.</div>}
      {schedules.map((s) => (
        <div key={s.id} className="rounded-xl border border-[var(--color-edge)] p-2.5 flex items-center gap-2.5">
          <Clock className="w-4 h-4 text-[var(--color-pi-2)] shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-[13px] text-[var(--color-ink-dim)] truncate">{s.prompt}</div>
            <div className="text-[10.5px] text-[var(--color-ink-faint)]">every {Math.round(s.everyMs / 1000)}s · next {new Date(s.nextRun).toLocaleTimeString()}</div>
          </div>
          <button className="text-[var(--color-ink-faint)] hover:text-[var(--color-soyuz)]" onClick={() => removeSchedule(s.id)}>
            <Trash className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </Shell>
  );
}
