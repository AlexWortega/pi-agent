import { useCallback, useEffect, useMemo, useState } from "react";
import type { Model } from "@earendil-works/pi-ai";
import type { Project, GenParams } from "./types";
import { DEFAULT_PARAMS, SIQ_LOCAL } from "./config";
import { loadProjects, saveProjects, newProject } from "./lib/store";
import { resolveModel } from "./lib/models";
import { engine, hasWebGPU } from "./engine/llama";
import { useAgent } from "./pi/useAgent";
import { makeAgentModel } from "./pi/runtime";
import { setStats } from "./pi/stats";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { AgentChat } from "./components/AgentChat";
import { WorkspacePanel } from "./components/WorkspacePanel";
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
  const [pickerOpen, setPickerOpen] = useState(false);

  const webgpu = useMemo(() => hasWebGPU(), []);

  useEffect(() => {
    saveProjects(projects);
  }, [projects]);

  const active = projects.find((p) => p.id === activeId) ?? projects[0];
  const activeModel = resolveModel(active?.modelId ?? "");

  const patchActive = useCallback(
    (fn: (p: Project) => Project) => {
      setProjects((prev) => prev.map((p) => (p.id === activeId ? { ...fn(p), updatedAt: Date.now() } : p)));
    },
    [activeId],
  );

  // ---- model loading ------------------------------------------------------
  const loadModel = useCallback(
    async (modelId: string) => {
      const m = resolveModel(modelId);
      // local/remote toggle for the cloud model: swap the endpoint to a local
      // llama-server when the user picks "local" (default is the RunPod proxy).
      const remote =
        m.remote && params.endpointMode === "local"
          ? { ...m.remote, endpoint: SIQ_LOCAL }
          : m.remote;
      patchActive((p) => ({ ...p, modelId }));
      setEng({ phase: "loading", modelId, progress: 0, loaded: 0, total: 0 });
      // remote models report their server-side context window; local uses the slider.
      setStats({ contextWindow: m.remote?.contextWindow ?? params.contextLength });
      try {
        await engine.load(
          { url: m.url, remote },
          {
            contextLength: params.contextLength,
            onProgress: (frac, loaded, total) => {
              setEng((s) => ({ ...s, phase: "loading", progress: frac, loaded, total }));
              if (total > 0) setStats({ modelBytes: total });
            },
          },
        );
        setEng({ phase: "ready", modelId, progress: 1, loaded: 0, total: 0 });
      } catch (e: any) {
        setEng({ phase: "error", modelId, progress: 0, loaded: 0, total: 0, error: String(e?.message || e) });
      }
    },
    [params.contextLength, params.endpointMode, patchActive],
  );

  // ---- agent --------------------------------------------------------------
  // `prepare` ensures the Soyuz model is downloaded/loaded, then hands the
  // agent loop a model descriptor. Returns null on failure so useAgent can
  // surface a friendly error in the chat.
  const prepare = useCallback(async (): Promise<Model<any> | null> => {
    if (eng.phase !== "ready" || eng.modelId !== active.modelId) {
      await loadModel(active.modelId);
    }
    if (!engine.ready) return null;
    return makeAgentModel({ id: activeModel.id, label: activeModel.label }, params);
  }, [eng.phase, eng.modelId, active?.modelId, activeModel.id, activeModel.label, loadModel, params]);

  const { messages, running, fsVersion, liveHtml, send, stop, reset, clearWorkspace } = useAgent(prepare);

  // Cloud model only: when the local/remote toggle flips, re-point the engine.
  useEffect(() => {
    if (activeModel.remote && eng.phase === "ready" && eng.modelId === active?.modelId) {
      loadModel(active.modelId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.endpointMode]);

  const createProject = () => {
    const p = newProject(`Project ${projects.length + 1}`);
    p.modelId = active?.modelId ?? p.modelId;
    setProjects((prev) => [p, ...prev]);
    setActiveId(p.id);
    reset();
  };

  const deleteProject = (id: string) => {
    setProjects((prev) => {
      const next = prev.filter((p) => p.id !== id);
      const ensured = next.length ? next : [newProject("My first project")];
      if (id === activeId) {
        setActiveId(ensured[0].id);
        reset();
      }
      return ensured;
    });
  };

  const renameProject = (id: string, name: string) =>
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, name } : p)));

  return (
    <>
      <div className="aurora" />
      <div className="relative z-10 h-full flex">
        <Sidebar
          projects={projects}
          activeId={activeId}
          onSelect={(id) => {
            setActiveId(id);
            reset();
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
            <AgentChat messages={messages} running={running} eng={eng} onSend={send} onStop={stop} />
            <WorkspacePanel fsVersion={fsVersion} running={running} liveHtml={liveHtml} onClear={clearWorkspace} />
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
