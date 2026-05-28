import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import type { Project, ChatMessage, Artifact, GenParams, ToolResult } from "./types";
import { DEFAULT_PARAMS, buildSystemPrompt, FEW_SHOT_WIRE } from "./config";
import { loadProjects, saveProjects, newProject, uid } from "./lib/store";
import { resolveModel } from "./lib/models";
import { extractHtmlArtifact } from "./lib/parse";
import { engine, hasWebGPU } from "./engine/llama";
import { runAgent } from "./agent/loop";
import type { AgentCallbacks } from "./agent/loop";
import { enabledTools, toolsSystemBlock } from "./agent/registry";
import type { ToolContext } from "./agent/registry";
import { runSubagent } from "./agent/subagent";
import { assistantWire, toolResponseWire } from "./agent/chatml";
import type { WireMessage } from "./agent/chatml";
import { memoryBlock } from "./lib/memoryStore";
import { relevantSkills } from "./lib/skillsStore";
import { getSettings } from "./lib/settings";
import { searchSessions } from "./lib/sessionIndex";
import { listAllArtifacts, findArtifact, searchArtifacts } from "./lib/artifactIndex";
import { parseCommand, COMMAND_REGISTRY } from "./lib/commands";
import type { PanelId } from "./lib/commands";
import { dueSchedules, addSchedule, parseEvery } from "./lib/scheduler";
import { ALL_TOOLS } from "./agent/registry";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { ChatPanel } from "./components/ChatPanel";
import { CanvasPanel } from "./components/CanvasPanel";
import { ModelPicker } from "./components/ModelPicker";
import { SkillsPanel, MemoryPanel, KanbanPanel, ToolsInspector, SchedulesPanel } from "./components/Panels";
import type { ClarifyRequest } from "./components/ClarifyPrompt";

export type EnginePhase = "idle" | "loading" | "ready" | "error";
export interface EngineState {
  phase: EnginePhase;
  modelId: string | null;
  progress: number;
  loaded: number;
  total: number;
  error?: string;
}
export interface AgentStatus {
  iteration: number;
  maxIterations: number;
  phase: "thinking" | "tools" | "done";
}

/** Replay stored messages into ChatML wire turns for the next prompt. */
function replayMessages(msgs: ChatMessage[]): WireMessage[] {
  const wire: WireMessage[] = [];
  for (const m of msgs) {
    if (m.role === "user") {
      wire.push({ role: "user", content: m.content });
    } else if (m.role === "assistant") {
      const calls = m.toolCalls ?? [];
      const content = assistantWire(m.content, calls);
      if (content.trim()) wire.push({ role: "assistant", content });
      if (m.toolResults?.length) {
        wire.push({ role: "tool", content: m.toolResults.map(toolResponseWire).join("\n") });
      }
    }
  }
  return wire;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(() => {
    const ps = loadProjects();
    return ps.length ? ps : [newProject("My first session")];
  });
  const [activeId, setActiveId] = useState<string>(() => projects[0]?.id ?? "");
  const [params, setParams] = useState<GenParams>(DEFAULT_PARAMS);
  const [eng, setEng] = useState<EngineState>({ phase: "idle", modelId: null, progress: 0, loaded: 0, total: 0 });
  const [generating, setGenerating] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [panel, setPanel] = useState<PanelId | null>(null);
  const [canvasView, setCanvasView] = useState<"preview" | "code">("preview");
  const [activeArtifactId, setActiveArtifactId] = useState<string | null>(null);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [clarify, setClarify] = useState<ClarifyRequest | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const webgpu = useMemo(() => hasWebGPU(), []);
  const active = projects.find((p) => p.id === activeId) ?? projects[0];
  const activeModel = resolveModel(active?.modelId ?? "");

  // refs for the scheduler (avoids stale closures)
  const projectsRef = useRef(projects);
  const generatingRef = useRef(generating);
  const activeIdRef = useRef(activeId);
  projectsRef.current = projects;
  generatingRef.current = generating;
  activeIdRef.current = activeId;

  useEffect(() => {
    saveProjects(projects);
  }, [projects]);

  // ---- project mutation helpers -------------------------------------------
  const patchActive = useCallback(
    (fn: (p: Project) => Project) => {
      setProjects((prev) => prev.map((p) => (p.id === activeIdRef.current ? { ...fn(p), updatedAt: Date.now() } : p)));
    },
    [],
  );

  const createProject = () => {
    const p = newProject(`Session ${projects.length + 1}`);
    p.modelId = active?.modelId ?? p.modelId;
    setProjects((prev) => [p, ...prev]);
    setActiveId(p.id);
    setActiveArtifactId(null);
  };
  const deleteProject = (id: string) => {
    setProjects((prev) => {
      const next = prev.filter((p) => p.id !== id);
      const ensured = next.length ? next : [newProject("My first session")];
      if (id === activeId) setActiveId(ensured[0].id);
      return ensured;
    });
  };
  const renameProject = (id: string, name: string) => setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, name } : p)));

  const addInfo = (content: string) =>
    patchActive((p) => ({ ...p, messages: [...p.messages, { id: uid(), role: "assistant" as const, content, ts: Date.now() }] }));

  // ---- model loading ------------------------------------------------------
  const loadModel = useCallback(
    async (modelId: string) => {
      const m = resolveModel(modelId);
      patchActive((p) => ({ ...p, modelId }));
      setEng({ phase: "loading", modelId, progress: 0, loaded: 0, total: 0 });
      try {
        await engine.load(m.url, {
          contextLength: params.contextLength,
          onProgress: (frac, loaded, total) => setEng((s) => ({ ...s, phase: "loading", progress: frac, loaded, total })),
        });
        setEng({ phase: "ready", modelId, progress: 1, loaded: 0, total: 0 });
      } catch (e: any) {
        setEng({ phase: "error", modelId, progress: 0, loaded: 0, total: 0, error: String(e?.message || e) });
      }
    },
    [params.contextLength, patchActive],
  );

  // ---- artifact emission + ops (used by render_html / artifact_* tools) ---
  const emitArtifact = useCallback(
    (title: string, html: string): string => {
      const id = "art-" + uid();
      patchActive((p) => ({ ...p, artifacts: [...p.artifacts, { id, title, html, ts: Date.now() }] }));
      setActiveArtifactId(id);
      setCanvasView("preview");
      return id;
    },
    [patchActive],
  );

  const updateArtifactById = useCallback((id: string, html: string, title?: string): boolean => {
    const exists = projectsRef.current.some((p) => p.artifacts.some((a) => a.id === id));
    if (!exists) return false;
    setProjects((prev) =>
      prev.map((p) =>
        p.artifacts.some((a) => a.id === id)
          ? {
              ...p,
              artifacts: p.artifacts.map((a) => (a.id === id ? { ...a, html, title: title ?? a.title, ts: Date.now() } : a)),
              updatedAt: Date.now(),
            }
          : p,
      ),
    );
    setActiveArtifactId(id);
    setCanvasView("preview");
    return true;
  }, []);

  const focusArtifactById = useCallback((id: string): boolean => {
    const owner = projectsRef.current.find((p) => p.artifacts.some((a) => a.id === id));
    if (!owner) return false;
    if (owner.id !== activeIdRef.current) setActiveId(owner.id);
    setActiveArtifactId(id);
    setCanvasView("preview");
    return true;
  }, []);

  // ---- slash commands -----------------------------------------------------
  const handleCommand = (name: string, rest: string) => {
    switch (name) {
      case "/help": {
        const cmds = COMMAND_REGISTRY.map((c) => `- \`${c.name}${c.arg ? " " + c.arg : ""}\` — ${c.description}`).join("\n");
        const tools = ALL_TOOLS.map((t) => `\`${t.name}\``).join(", ");
        addInfo(`**Commands**\n${cmds}\n\n**Tools the agent can call**\n${tools}`);
        return;
      }
      case "/tools":
        setPanel("tools");
        return;
      case "/skills":
        setPanel("skills");
        return;
      case "/memory":
        setPanel("memory");
        return;
      case "/tasks":
        setPanel("tasks");
        return;
      case "/model":
        setPickerOpen(true);
        return;
      case "/new":
        createProject();
        return;
      case "/clear":
        patchActive((p) => ({ ...p, messages: [] }));
        return;
      case "/schedule": {
        if (!rest) {
          setPanel("schedules");
          return;
        }
        const m = rest.match(/^(every\s+)?(\d+\s*\w+)\s+([\s\S]+)$/i);
        if (!m) {
          addInfo("Usage: `/schedule every 5m <prompt>`");
          return;
        }
        const ms = parseEvery(m[2]);
        if (!ms) {
          addInfo("Could not parse the interval. Try `/schedule every 5m <prompt>`.");
          return;
        }
        addSchedule(m[3].trim(), ms, activeIdRef.current);
        addInfo(`⏰ Scheduled — I'll re-run “${m[3].trim()}” every ${Math.round(ms / 1000)}s **while this tab is open**.`);
        return;
      }
      case "/search": {
        if (!rest) {
          addInfo("Usage: `/search <query>`");
          return;
        }
        const sHits = searchSessions(projectsRef.current, rest, 6);
        const aHits = searchArtifacts(projectsRef.current, rest, 6);
        const parts = [`**Search: “${rest}”**`];
        if (sHits.length) parts.push(`**Sessions** (${sHits.length})\n` + sHits.map((h) => `- *${h.projectName}* (${h.role}): ${h.snippet}`).join("\n"));
        if (aHits.length) parts.push(`**Artifacts** (${aHits.length})\n` + aHits.map((h) => `- *${h.projectName}* — \`${h.title}\` (\`${h.artifactId}\`): ${h.snippet}`).join("\n"));
        if (sHits.length === 0 && aHits.length === 0) parts.push("No matches.");
        addInfo(parts.join("\n\n"));
        return;
      }
      default:
        addInfo(`Unknown command \`${name}\`. Try \`/help\`.`);
    }
  };

  // ---- generation (the agent loop) ----------------------------------------
  const stop = () => abortRef.current?.abort();

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || generating || !active) return;

      // slash commands run locally, no model call
      const cmd = parseCommand(trimmed);
      if (cmd) {
        handleCommand(cmd.name, cmd.rest);
        return;
      }

      const userMsg: ChatMessage = { id: uid(), role: "user", content: trimmed, ts: Date.now() };
      const history = active.messages;
      patchActive((p) => ({ ...p, messages: [...p.messages, userMsg] }));
      setGenerating(true);
      setStatus(null);

      // ensure the model is loaded (first use downloads ~2.5 GB)
      if (eng.phase !== "ready" || eng.modelId !== active.modelId) {
        await loadModel(active.modelId).catch(() => {});
        if (!engine.ready) {
          addInfo(
            "⚠️ Could not load the Soyuz model. Make sure your browser supports WebGPU (Chrome / Edge / Safari) and hit “Load now” in the top bar.",
          );
          setGenerating(false);
          return;
        }
      }

      const ac = new AbortController();
      abortRef.current = ac;

      const tools = enabledTools(getSettings());
      const sys = buildSystemPrompt(toolsSystemBlock(tools), memoryBlock());
      const skillMsgs: WireMessage[] = relevantSkills(trimmed).map((s) => ({
        role: "user",
        content: `# Skill: ${s.name}\n${s.body}`,
      }));
      const wire: WireMessage[] = [
        { role: "system", content: sys },
        ...FEW_SHOT_WIRE,
        ...skillMsgs,
        ...replayMessages([...history, userMsg]),
      ];

      const ctx: ToolContext = {
        signal: ac.signal,
        depth: 0,
        settings: getSettings(),
        requestClarify: (question, options) =>
          new Promise<string>((resolve) => {
            const done = (ans: string) => {
              setClarify(null);
              ac.signal.removeEventListener("abort", onAbort);
              resolve(ans);
            };
            const onAbort = () => done("(user cancelled)");
            ac.signal.addEventListener("abort", onAbort);
            setClarify({ question, options, resolve: done });
          }),
        emitArtifact,
        runSubagent: (task, c) => runSubagent(task, c),
        searchSessions: (q, limit) => searchSessions(projectsRef.current, q, limit),
        listArtifacts: () => listAllArtifacts(projectsRef.current),
        searchArtifacts: (q, limit) => searchArtifacts(projectsRef.current, q, limit),
        getArtifact: (id) => {
          const f = findArtifact(projectsRef.current, id);
          return f ? { title: f.artifact.title, html: f.artifact.html } : null;
        },
        updateArtifact: updateArtifactById,
        focusArtifact: focusArtifactById,
      };

      // turn-local message bookkeeping
      const base = "m" + uid();
      const created = new Set<number>();
      const msgId = (iter: number) => `${base}-i${iter}`;
      const ensure = (iter: number) => {
        const id = msgId(iter);
        if (!created.has(iter)) {
          created.add(iter);
          patchActive((p) => ({
            ...p,
            messages: [...p.messages, { id, role: "assistant" as const, content: "", ts: Date.now(), pending: true, iteration: iter }],
          }));
        }
        return id;
      };
      const setMsg = (id: string, patch: Partial<ChatMessage>) =>
        patchActive((p) => ({ ...p, messages: p.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)) }));

      const cb: AgentCallbacks = {
        onStatus: (s) => setStatus(s),
        onStream: (iter, visible, think) => {
          const id = ensure(iter);
          setMsg(id, { content: visible, think, pending: true });
        },
        onAssistant: (iter, visible, think, calls) => {
          const id = ensure(iter);
          // also surface a ```html block as an artifact (model didn't use render_html)
          const parsed = extractHtmlArtifact(visible);
          if (parsed) {
            const artId = "art-" + id;
            patchActive((p) => {
              const art: Artifact = { id: artId, title: parsed.title, html: parsed.html, ts: Date.now() };
              const artifacts = p.artifacts.some((a) => a.id === artId)
                ? p.artifacts.map((a) => (a.id === artId ? art : a))
                : [...p.artifacts, art];
              return { ...p, artifacts };
            });
            setActiveArtifactId(artId);
          }
          setMsg(id, {
            content: visible,
            think,
            toolCalls: calls,
            pending: calls.length > 0,
          });
        },
        onToolResult: (result: ToolResult) => {
          patchActive((p) => ({
            ...p,
            messages: p.messages.map((m) =>
              m.toolCalls?.some((c) => c.id === result.id) ? { ...m, toolResults: [...(m.toolResults ?? []), result] } : m,
            ),
          }));
        },
      };

      try {
        await runAgent({
          wire,
          tools,
          ctx,
          params: { temperature: params.temperature, maxTokens: params.maxTokens, maxIterations: params.maxIterations },
          signal: ac.signal,
          cb,
        });
      } catch (e: any) {
        if (!ac.signal.aborted) addInfo(`⚠️ ${String(e?.message || e)}`);
      } finally {
        patchActive((p) => ({ ...p, messages: p.messages.map((m) => (m.id.startsWith(base) ? { ...m, pending: false } : m)) }));
        setGenerating(false);
        setStatus(null);
        setClarify(null);
        abortRef.current = null;
      }
    },
    [active, eng.phase, eng.modelId, generating, loadModel, params, patchActive, emitArtifact],
  );

  // ---- scheduler (tab-open only) ------------------------------------------
  const sendRef = useRef(send);
  sendRef.current = send;
  useEffect(() => {
    const t = setInterval(() => {
      if (generatingRef.current) return;
      const due = dueSchedules();
      const mine = due.find((s) => s.projectId === activeIdRef.current);
      if (mine) sendRef.current(mine.prompt);
    }, 20000);
    return () => clearInterval(t);
  }, []);

  const artifacts = active?.artifacts ?? [];
  const shownArtifact = artifacts.find((a) => a.id === activeArtifactId) ?? artifacts[artifacts.length - 1] ?? null;

  return (
    <>
      <div className="aurora" />
      <div className="grain" />
      <div className="relative z-10 h-full flex">
        <Sidebar
          projects={projects}
          activeId={activeId}
          onSelect={(id) => {
            setActiveId(id);
            setActiveArtifactId(null);
          }}
          onCreate={createProject}
          onDelete={deleteProject}
          onRename={renameProject}
          webgpu={webgpu}
          onOpenPanel={setPanel}
        />

        <main className="flex-1 min-w-0 flex flex-col">
          <TopBar
            model={activeModel}
            eng={eng}
            webgpu={webgpu}
            onOpenPicker={() => setPickerOpen(true)}
            onReload={() => loadModel(active.modelId)}
          />

          <div className="flex-1 min-h-0 flex">
            <ChatPanel
              key={active?.id}
              project={active}
              generating={generating}
              eng={eng}
              status={status}
              clarify={clarify}
              onSend={send}
              onStop={stop}
              onOpenArtifact={(id) => {
                setActiveArtifactId(id);
                setCanvasView("preview");
              }}
            />
            <CanvasPanel
              artifact={shownArtifact}
              artifacts={artifacts}
              view={canvasView}
              generating={generating}
              onView={setCanvasView}
              onSelect={setActiveArtifactId}
            />
          </div>
        </main>
      </div>

      {pickerOpen && (
        <ModelPicker
          currentId={active.modelId}
          onClose={() => setPickerOpen(false)}
          onPick={(id) => {
            setPickerOpen(false);
            loadModel(id);
          }}
          params={params}
          onParams={setParams}
        />
      )}

      {panel === "skills" && <SkillsPanel onClose={() => setPanel(null)} />}
      {panel === "memory" && <MemoryPanel onClose={() => setPanel(null)} />}
      {panel === "tasks" && <KanbanPanel onClose={() => setPanel(null)} />}
      {panel === "tools" && <ToolsInspector onClose={() => setPanel(null)} />}
      {panel === "schedules" && <SchedulesPanel onClose={() => setPanel(null)} />}
    </>
  );
}
