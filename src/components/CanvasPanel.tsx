import { useMemo, useState } from "react";
import type { Artifact } from "../types";
import { Eye, Code, Download, External, Refresh, Copy, Check } from "./Icons";

interface Props {
  artifact: Artifact | null;
  artifacts: Artifact[];
  view: "preview" | "code";
  onView: (v: "preview" | "code") => void;
  onSelect: (id: string) => void;
}

function slug(s: string) {
  return (
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "app"
  );
}

export function CanvasPanel({ artifact, artifacts, view, onView, onSelect }: Props) {
  const [reloadKey, setReloadKey] = useState(0);
  const [copied, setCopied] = useState(false);

  const blobUrl = useMemo(() => {
    if (!artifact) return null;
    return URL.createObjectURL(new Blob([artifact.html], { type: "text/html" }));
  }, [artifact?.html, reloadKey]);

  const download = () => {
    if (!artifact) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([artifact.html], { type: "text/html" }));
    a.download = `${slug(artifact.title)}.html`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const copy = async () => {
    if (!artifact) return;
    await navigator.clipboard.writeText(artifact.html);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <section className="w-[46%] min-w-[360px] max-w-[760px] shrink-0 flex flex-col bg-[var(--color-void)]">
      <div className="h-12 shrink-0 px-3 flex items-center gap-2 border-b border-[var(--color-edge)]">
        <div className="flex items-center rounded-lg border border-[var(--color-edge)] p-0.5 bg-[var(--color-panel-2)]/50">
          <Tab active={view === "preview"} onClick={() => onView("preview")}>
            <Eye className="w-3.5 h-3.5" /> Preview
          </Tab>
          <Tab active={view === "code"} onClick={() => onView("code")}>
            <Code className="w-3.5 h-3.5" /> Code
          </Tab>
        </div>

        <div className="flex-1 min-w-0 text-center">
          {artifact && (
            <span className="text-[12px] text-[var(--color-ink-dim)] truncate inline-block max-w-full px-2">
              {artifact.title}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          <IconBtn onClick={() => setReloadKey((k) => k + 1)} disabled={!artifact} title="Reload preview">
            <Refresh className="w-3.5 h-3.5" />
          </IconBtn>
          <IconBtn onClick={copy} disabled={!artifact} title="Copy HTML">
            {copied ? <Check className="w-3.5 h-3.5 text-[var(--color-mint)]" /> : <Copy className="w-3.5 h-3.5" />}
          </IconBtn>
          <IconBtn
            onClick={() => blobUrl && window.open(blobUrl, "_blank")}
            disabled={!artifact}
            title="Open in new tab"
          >
            <External className="w-3.5 h-3.5" />
          </IconBtn>
          <button className="btn btn-primary py-1.5" onClick={download} disabled={!artifact}>
            <Download className="w-3.5 h-3.5" /> Download
          </button>
        </div>
      </div>

      {/* artifact tabs */}
      {artifacts.length > 1 && (
        <div className="flex gap-1 px-3 py-2 overflow-x-auto border-b border-[var(--color-edge)]">
          {artifacts.map((a) => (
            <button
              key={a.id}
              onClick={() => onSelect(a.id)}
              className={`shrink-0 text-[11px] px-2.5 py-1 rounded-full border transition ${
                a.id === artifact?.id
                  ? "border-[var(--color-pi)] text-[var(--color-pi-2)] bg-[var(--color-pi)]/10"
                  : "border-[var(--color-edge)] text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
              }`}
            >
              {a.title}
            </button>
          ))}
        </div>
      )}

      <div className="flex-1 min-h-0 relative">
        {!artifact ? (
          <Empty />
        ) : view === "preview" ? (
          <iframe
            key={reloadKey}
            title="preview"
            srcDoc={artifact.html}
            sandbox="allow-scripts allow-modals allow-forms allow-popups allow-pointer-lock"
            className="w-full h-full bg-white"
          />
        ) : (
          <pre className="w-full h-full overflow-auto m-0 p-4 text-[12px] leading-relaxed font-[var(--font-mono)] text-[var(--color-ink-dim)] bg-[#05060a]">
            <code>{artifact.html}</code>
          </pre>
        )}
      </div>
    </section>
  );
}

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 text-[12px] px-2.5 py-1 rounded-md transition ${
        active ? "bg-[var(--color-edge)] text-[var(--color-ink)]" : "text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
      }`}
    >
      {children}
    </button>
  );
}

function IconBtn({
  children,
  onClick,
  disabled,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="w-8 h-8 grid place-items-center rounded-lg text-[var(--color-ink-dim)] hover:text-[var(--color-ink)] hover:bg-[var(--color-panel-2)] disabled:opacity-30 disabled:hover:bg-transparent transition"
    >
      {children}
    </button>
  );
}

function Empty() {
  return (
    <div className="absolute inset-0 grid place-items-center p-8 text-center">
      <div className="max-w-xs">
        <div className="w-12 h-12 mx-auto rounded-2xl grid place-items-center border border-dashed border-[var(--color-edge-2)] text-[var(--color-ink-faint)]">
          <Eye className="w-5 h-5" />
        </div>
        <h3 className="mt-4 text-[14px] font-medium text-[var(--color-ink-dim)]">Live canvas</h3>
        <p className="mt-1.5 text-[12px] text-[var(--color-ink-faint)] leading-relaxed">
          When Pi Agent generates an HTML app it renders here instantly. Then download it as a single
          file to run anywhere.
        </p>
      </div>
    </div>
  );
}
