import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import type { Project, ChatMessage, Artifact, GenParams } from "./types";
import { DEFAULT_PARAMS, SYSTEM_PROMPT } from "./config";
import { loadProjects, saveProjects, newProject, uid } from "./lib/store";
import { resolveModel } from "./lib/models";
import { splitThink, extractHtmlArtifact } from "./lib/parse";
import { engine, hasWebGPU } from "./engine/llama";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { ChatPanel } from "./components/ChatPanel";
import { CanvasPanel } from "./components/CanvasPanel";
import { ModelPicker } from "./components/ModelPicker";

export type EnginePhase = "idle" | "loading" | "ready" | "error";
export interface EngineState {
  phase: EnginePhase;
  modelId: string | null;
  progress: number;
  loaded: number;
  total: number;
  error?: string;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(() => {
    const ps = loadProjects();
    return ps.length ? ps : [newProject("My first project")];
  });
  const [activeId, setActiveId] = useState<string>(() => projects[0]?.id ?? "");
  const [params, setParams] = useState<GenParams>(DEFAULT_PARAMS);
  const [eng, setEng] = useState<EngineState>({
    phase: "idle",
    modelId: null,
    progress: 0,
    loaded: 0,
    total: 0,
  });
  const [generating, setGenerating] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [canvasView, setCanvasView] = useState<"preview" | "code">("preview");
  const [activeArtifactId, setActiveArtifactId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const webgpu = useMemo(() => hasWebGPU(), []);

  useEffect(() => {
    saveProjects(projects);
  }, [projects]);

  const active = projects.find((p) => p.id === activeId) ?? projects[0];
  const activeModel = resolveModel(active?.modelId ?? "");

  // ---- project mutation helpers -------------------------------------------
  const patchActive = useCallback(
    (fn: (p: Project) => Project) => {
      setProjects((prev) =>
        prev.map((p) => (p.id === activeId ? { ...fn(p), updatedAt: Date.now() } : p)),
      );
    },
    [activeId],
  );

  const createProject = () => {
    const p = newProject(`Project ${projects.length + 1}`);
    p.modelId = active?.modelId ?? p.modelId;
    setProjects((prev) => [p, ...prev]);
    setActiveId(p.id);
    setActiveArtifactId(null);
  };

  const deleteProject = (id: string) => {
    setProjects((prev) => {
      const next = prev.filter((p) => p.id !== id);
      const ensured = next.length ? next : [newProject("My first project")];
      if (id === activeId) setActiveId(ensured[0].id);
      return ensured;
    });
  };

  const renameProject = (id: string, name: string) =>
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, name } : p)));

  // ---- model loading ------------------------------------------------------
  const loadModel = useCallback(
    async (modelId: string) => {
      const m = resolveModel(modelId);
      patchActive((p) => ({ ...p, modelId }));
      setEng({ phase: "loading", modelId, progress: 0, loaded: 0, total: 0 });
      try {
        await engine.load(m.url, {
          contextLength: params.contextLength,
          onProgress: (frac, loaded, total) =>
            setEng((s) => ({ ...s, phase: "loading", progress: frac, loaded, total })),
        });
        setEng({ phase: "ready", modelId, progress: 1, loaded: 0, total: 0 });
      } catch (e: any) {
        setEng({
          phase: "error",
          modelId,
          progress: 0,
          loaded: 0,
          total: 0,
          error: String(e?.message || e),
        });
      }
    },
    [params.contextLength, patchActive],
  );

  // ---- generation ---------------------------------------------------------
  const stop = () => abortRef.current?.abort();

  const send = useCallback(
    async (text: string) => {
      if (!text.trim() || generating || !active) return;

      if (eng.phase !== "ready" || eng.modelId !== active.modelId) {
        await loadModel(active.modelId);
        // if load failed, bail
        if (!engine.ready) return;
      }

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: text.trim(),
        ts: Date.now(),
      };
      const asstId = uid();
      const asstMsg: ChatMessage = {
        id: asstId,
        role: "assistant",
        content: "",
        ts: Date.now(),
        pending: true,
      };

      const history = [...active.messages, userMsg];
      patchActive((p) => ({ ...p, messages: [...p.messages, userMsg, asstMsg] }));

      const wire = [
        { role: "system" as const, content: SYSTEM_PROMPT },
        ...history.map((m) => ({ role: m.role, content: m.content })),
      ];

      const ac = new AbortController();
      abortRef.current = ac;
      setGenerating(true);

      const artId = "art-" + asstId;
      const applyChunk = (raw: string) => {
        const { think, answer } = splitThink(raw);
        const parsed = extractHtmlArtifact(answer);
        patchActive((p) => {
          const messages = p.messages.map((m) =>
            m.id === asstId ? { ...m, content: answer, think } : m,
          );
          let artifacts = p.artifacts;
          if (parsed) {
            const art: Artifact = {
              id: artId,
              title: parsed.title,
              html: parsed.html,
              ts: Date.now(),
            };
            artifacts = p.artifacts.some((a) => a.id === artId)
              ? p.artifacts.map((a) => (a.id === artId ? art : a))
              : [...p.artifacts, art];
          }
          return { ...p, messages, artifacts };
        });
        if (parsed) setActiveArtifactId(artId);
      };

      try {
        await engine.chat(wire, {
          temperature: params.temperature,
          maxTokens: params.maxTokens,
          signal: ac.signal,
          onToken: applyChunk,
        });
      } catch (e: any) {
        if (!ac.signal.aborted) {
          patchActive((p) => ({
            ...p,
            messages: p.messages.map((m) =>
              m.id === asstId
                ? { ...m, content: (m.content || "") + `\n\n⚠️ ${String(e?.message || e)}` }
                : m,
            ),
          }));
        }
      } finally {
        patchActive((p) => ({
          ...p,
          messages: p.messages.map((m) => (m.id === asstId ? { ...m, pending: false } : m)),
        }));
        setGenerating(false);
        abortRef.current = null;
      }
    },
    [active, eng.phase, eng.modelId, generating, loadModel, params, patchActive],
  );

  const artifacts = active?.artifacts ?? [];
  const shownArtifact =
    artifacts.find((a) => a.id === activeArtifactId) ?? artifacts[artifacts.length - 1] ?? null;

  return (
    <>
      <div className="aurora" />
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
    </>
  );
}
