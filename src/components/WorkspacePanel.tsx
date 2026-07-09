import { useEffect, useState } from "react";
import { getFsBackend } from "../pi/fs/backend";
import { WORKSPACE_ROOT } from "../pi/runtime";
import { inlineAssets } from "../lib/inlineAssets";
import { RepoBar } from "./RepoBar";
import { Code, Eye, Download, External, Refresh, Satellite, Trash } from "./Icons";

interface Props {
  /** Bumps whenever the agent mutates the filesystem; triggers a refresh. */
  fsVersion: number;
  running: boolean;
  /** HTML being written live this turn (partial) — streamed into the canvas. */
  liveHtml: string | null;
  /** Wipe the virtual workspace (clears stale files / preview). */
  onClear: () => void | Promise<void>;
}

const PREVIEW_CANDIDATES = ["/workspace/index.html", "/workspace/main.html"];

function rel(path: string): string {
  return path.startsWith(WORKSPACE_ROOT + "/") ? path.slice(WORKSPACE_ROOT.length + 1) : path;
}

function isHtml(path: string): boolean {
  return /\.html?$/i.test(path);
}

export function WorkspacePanel({ fsVersion, running, liveHtml, onClear }: Props) {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [previewHtml, setPreviewHtml] = useState<string>("");
  const [view, setView] = useState<"preview" | "code">("preview");
  // Bumped by the RepoBar (import/detach) — mutations that happen outside the agent.
  const [localVersion, setLocalVersion] = useState(0);

  // Re-read the virtual filesystem whenever a tool mutates it.
  useEffect(() => {
    let cancelled = false;
    const fs = getFsBackend();
    (async () => {
      const all = await fs.walk(WORKSPACE_ROOT);
      if (cancelled) return;
      setFiles(all);

      // Resolve a preview target: explicit candidates, else first .html.
      const previewPath =
        PREVIEW_CANDIDATES.find((p) => all.includes(p)) ?? all.find(isHtml) ?? null;
      if (previewPath) {
        try {
          const raw = await fs.readText(previewPath);
          // srcDoc can't resolve relative css/js against the virtual FS —
          // inline workspace-local assets so multi-file apps render correctly.
          const baseDir = previewPath.slice(0, previewPath.lastIndexOf("/")) || "/";
          const inlined = await inlineAssets(raw, baseDir, async (p) => {
            try {
              return await fs.readText(p);
            } catch {
              return null;
            }
          });
          if (!cancelled) setPreviewHtml(inlined);
        } catch {
          setPreviewHtml("");
        }
      } else {
        setPreviewHtml("");
      }

      // Default the code view to the preview target until the user picks a file.
      const target = selected && all.includes(selected) ? selected : previewPath;
      if (target) {
        try {
          const text = await fs.readText(target);
          if (!cancelled) {
            setSelected(target);
            setContent(text);
          }
        } catch {
          /* ignore */
        }
      } else {
        setSelected(null);
        setContent("");
      }
    })();
    return () => {
      cancelled = true;
    };
    // selected intentionally omitted: re-reads are driven by fsVersion only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fsVersion, localVersion]);

  const pickFile = async (path: string) => {
    setSelected(path);
    setView("code");
    try {
      setContent(await getFsBackend().readText(path));
    } catch {
      setContent("");
    }
  };

  const download = async () => {
    // Multi-file workspace → zip everything; single file → download it directly.
    if (files.length > 1) {
      const { default: JSZip } = await import("jszip");
      const zip = new JSZip();
      const fs = getFsBackend();
      for (const f of files) {
        try {
          zip.file(rel(f), await fs.readText(f));
        } catch {
          /* skip unreadable */
        }
      }
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "workspace.zip";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      return;
    }
    const html = previewHtml || content;
    if (!html) return;
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (selected ? rel(selected).split("/").pop() : "index") || "index.html";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const openExternal = () => {
    if (!previewHtml) return;
    const blob = new Blob([previewHtml], { type: "text/html" });
    window.open(URL.createObjectURL(blob), "_blank");
  };

  const hasFiles = files.length > 0;

  // While the model is streaming an HTML doc, show it live (and watch it type
  // in the Code pane); otherwise show the saved file.
  const live = liveHtml !== null;
  const effView = live ? "code" : view;
  const previewSrc = liveHtml ?? previewHtml;
  const codeSrc = liveHtml ?? content;

  return (
    <section className="w-[46%] min-w-[380px] flex flex-col bg-[var(--color-panel)]/30">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-[var(--color-edge)]">
        <div className="flex rounded-lg border border-[var(--color-edge)] overflow-hidden text-[12px]">
          <button
            className={`px-3 py-1.5 inline-flex items-center gap-1.5 ${view === "preview" ? "bg-[var(--color-pi)]/15 text-[var(--color-pi-2)]" : "text-[var(--color-ink-dim)]"}`}
            onClick={() => setView("preview")}
          >
            <Eye className="w-3.5 h-3.5" /> Preview
          </button>
          <button
            className={`px-3 py-1.5 inline-flex items-center gap-1.5 ${view === "code" ? "bg-[var(--color-pi)]/15 text-[var(--color-pi-2)]" : "text-[var(--color-ink-dim)]"}`}
            onClick={() => setView("code")}
          >
            <Code className="w-3.5 h-3.5" /> Code
          </button>
        </div>
        {running && (
          <span className="text-[11px] text-[var(--color-pi-2)] inline-flex items-center gap-1.5">
            <Refresh className="w-3.5 h-3.5 animate-spin" />
            {live ? "✍️ writing index.html…" : "working…"}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          <button
            className="btn btn-ghost py-1.5"
            onClick={() => onClear()}
            disabled={running || !hasFiles}
            title="Clear the workspace (remove all files)"
          >
            <Trash className="w-4 h-4" />
          </button>
          <button className="btn btn-ghost py-1.5" onClick={openExternal} disabled={!previewHtml} title="Open preview in a new tab">
            <External className="w-4 h-4" />
          </button>
          <button className="btn btn-primary py-1.5" onClick={download} disabled={!hasFiles && !previewHtml && !content}>
            <Download className="w-4 h-4" /> {files.length > 1 ? "Zip" : "Download"}
          </button>
        </div>
      </div>

      <RepoBar fsVersion={fsVersion + localVersion} running={running} onMutated={() => setLocalVersion((v) => v + 1)} />

      <div className="flex-1 min-h-0 flex">
        {/* file tree */}
        <div className="w-44 shrink-0 border-r border-[var(--color-edge)] overflow-y-auto py-2">
          {hasFiles ? (
            files.map((f) => (
              <button
                key={f}
                onClick={() => pickFile(f)}
                className={`block w-full text-left px-3 py-1 text-[11.5px] font-mono truncate transition ${
                  selected === f
                    ? "text-[var(--color-pi-2)] bg-[var(--color-pi)]/10"
                    : "text-[var(--color-ink-dim)] hover:text-[var(--color-ink)]"
                }`}
                title={rel(f)}
              >
                {rel(f)}
              </button>
            ))
          ) : (
            <div className="px-3 py-2 text-[11px] text-[var(--color-ink-faint)]">No files yet</div>
          )}
        </div>

        {/* main view */}
        <div className="flex-1 min-w-0 min-h-0">
          {effView === "preview" ? (
            previewSrc ? (
              <iframe
                title="preview"
                className="w-full h-full bg-white"
                srcDoc={previewSrc}
                sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
              />
            ) : (
              <EmptyPreview />
            )
          ) : (
            <pre
              ref={(el) => {
                // Auto-scroll to the bottom while the file is being written.
                if (el && live) el.scrollTop = el.scrollHeight;
              }}
              className="w-full h-full overflow-auto text-[11.5px] leading-relaxed font-mono text-[var(--color-ink-dim)] px-4 py-3 whitespace-pre"
            >
              {codeSrc || "// select a file"}
            </pre>
          )}
        </div>
      </div>
    </section>
  );
}

function EmptyPreview() {
  return (
    <div className="h-full grid place-items-center text-center px-8">
      <div>
        <div className="w-12 h-12 mx-auto rounded-2xl grid place-items-center bg-[var(--color-panel-2)] text-[var(--color-ink-faint)]">
          <Satellite className="w-6 h-6" />
        </div>
        <p className="mt-3 text-[12.5px] text-[var(--color-ink-faint)] max-w-xs mx-auto">
          When the agent writes <code className="font-mono">index.html</code> into the project, it renders
          here live.
        </p>
      </div>
    </div>
  );
}
