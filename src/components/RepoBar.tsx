/**
 * GitHub repo bar for the workspace: import a repo into the virtual FS, show
 * what's imported, and push the agent's changes back as a real commit — all
 * client-side with the user's PAT (localStorage, straight to api.github.com).
 * Also surfaces the browser-VM status since bash runs against this workspace.
 */
import { useEffect, useState } from "react";
import { getFsBackend } from "../pi/fs/backend";
import { WORKSPACE_ROOT } from "../pi/runtime";
import {
  GITHUB_TOKEN_STORAGE,
  clearRepoMeta,
  commitAndPush,
  getGithubToken,
  importRepo,
  loadRepoMeta,
  parseRepoRef,
  type RepoMeta,
} from "../lib/github";
import { browserVm, type VmPhase } from "../vm/vm";

interface Props {
  /** Bumps when tools mutate the FS — refreshes the meta/diff hints. */
  fsVersion: number;
  running: boolean;
  /** Notify the parent that the workspace changed outside the agent (import/detach). */
  onMutated: () => void;
}

function useVmPhase(): { phase: VmPhase; error: string } {
  const [state, setState] = useState({ phase: browserVm.phase, error: browserVm.error });
  useEffect(() => browserVm.onChange(() => setState({ phase: browserVm.phase, error: browserVm.error })), []);
  return state;
}

export function RepoBar({ fsVersion, running, onMutated }: Props) {
  const [meta, setMeta] = useState<RepoMeta | null>(null);
  const [open, setOpen] = useState(false);
  const [refInput, setRefInput] = useState("");
  const [token, setToken] = useState(() => getGithubToken());
  const [commitMsg, setCommitMsg] = useState("");
  const [busy, setBusy] = useState<"" | "import" | "push">("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const vm = useVmPhase();

  useEffect(() => {
    let cancelled = false;
    loadRepoMeta(getFsBackend()).then((m) => !cancelled && setMeta(m));
    return () => {
      cancelled = true;
    };
  }, [fsVersion]);

  const saveToken = (value: string) => {
    setToken(value);
    try {
      const t = value.trim();
      if (t) localStorage.setItem(GITHUB_TOKEN_STORAGE, t);
      else localStorage.removeItem(GITHUB_TOKEN_STORAGE);
    } catch {
      /* private mode */
    }
  };

  const doImport = async () => {
    const ref = parseRepoRef(refInput);
    if (!ref) {
      setError("Enter owner/repo, owner/repo@branch or a github.com URL");
      return;
    }
    setBusy("import");
    setError("");
    try {
      const result = await importRepo(ref, getFsBackend(), WORKSPACE_ROOT, token.trim(), (p) => {
        setStatus(
          p.phase === "downloading" ? `downloading ${p.done}/${p.total} files…` : `${p.phase}…`,
        );
      });
      setMeta(result.meta);
      setStatus(
        `imported ${result.imported} files` +
          (result.skipped.length ? ` (${result.skipped.length} binary/large skipped)` : ""),
      );
      onMutated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    } finally {
      setBusy("");
    }
  };

  const doPush = async () => {
    if (!meta) return;
    setBusy("push");
    setError("");
    try {
      const message = commitMsg.trim() || "Edits by Pi Agent (in-browser)";
      const res = await commitAndPush(meta, getFsBackend(), WORKSPACE_ROOT, message, token.trim(), setStatus);
      setStatus(
        `pushed ${res.commitSha.slice(0, 7)} (+${res.added} ~${res.changed} -${res.deleted}) → ${res.commitUrl}`,
      );
      setCommitMsg("");
      setMeta(await loadRepoMeta(getFsBackend()));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    } finally {
      setBusy("");
    }
  };

  const doDetach = async () => {
    await clearRepoMeta(getFsBackend());
    setMeta(null);
    setStatus("");
    setError("");
    onMutated();
  };

  return (
    <div className="border-b border-[var(--color-edge)] px-4 py-2 text-[11.5px] space-y-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <button
          className="btn btn-ghost py-1 px-2 text-[11.5px]"
          onClick={() => setOpen((v) => !v)}
          data-testid="repo-toggle"
        >
          {meta ? (
            <span className="font-mono text-[var(--color-pi-2)]">
              ⎇ {meta.owner}/{meta.repo}@{meta.branch}
            </span>
          ) : (
            <span>⎇ GitHub repo…</span>
          )}
        </button>

        {vm.phase !== "off" && (
          <span
            className={`inline-flex items-center gap-1 text-[10.5px] px-1.5 py-px rounded-full border ${
              vm.phase === "ready"
                ? "text-[var(--color-mint)] border-[var(--color-mint)]/30 bg-[var(--color-mint)]/10"
                : vm.phase === "error"
                  ? "text-[var(--color-soyuz)] border-[var(--color-soyuz)]/30"
                  : "text-[var(--color-pi-2)] border-[var(--color-pi)]/30"
            }`}
            title={vm.error || "Linux VM (v86) backing the bash tool"}
          >
            VM {vm.phase === "booting" ? "booting…" : vm.phase}
          </span>
        )}

        {status && <span className="text-[var(--color-ink-faint)] truncate max-w-[50%]" data-testid="repo-status">{status}</span>}
        {error && <span className="text-[var(--color-soyuz)] truncate max-w-[60%]" data-testid="repo-error">{error}</span>}
      </div>

      {open && (
        <div className="space-y-1.5" data-testid="repo-panel">
          {!meta ? (
            <div className="flex gap-1.5 flex-wrap">
              <input
                className="field text-[11.5px] flex-1 min-w-[180px]"
                placeholder="owner/repo, owner/repo@branch or GitHub URL"
                value={refInput}
                data-testid="repo-ref"
                onChange={(e) => setRefInput(e.target.value)}
              />
              <button
                className="btn btn-primary py-1 shrink-0"
                disabled={busy !== "" || running || !refInput.trim()}
                data-testid="repo-import"
                onClick={doImport}
              >
                {busy === "import" ? "Importing…" : "Import"}
              </button>
            </div>
          ) : (
            <div className="flex gap-1.5 flex-wrap">
              <input
                className="field text-[11.5px] flex-1 min-w-[180px]"
                placeholder="Commit message (optional)"
                value={commitMsg}
                data-testid="repo-commit-msg"
                onChange={(e) => setCommitMsg(e.target.value)}
              />
              <button
                className="btn btn-primary py-1 shrink-0"
                disabled={busy !== "" || running || !token.trim()}
                title={token.trim() ? "Create a commit on GitHub from the workspace changes" : "Paste a GitHub token first"}
                data-testid="repo-push"
                onClick={doPush}
              >
                {busy === "push" ? "Pushing…" : "Commit & Push"}
              </button>
              <button className="btn btn-ghost py-1 shrink-0" disabled={busy !== ""} onClick={doDetach} data-testid="repo-detach">
                Detach
              </button>
            </div>
          )}
          <div className="flex gap-1.5 items-center">
            <input
              className="field text-[11.5px] flex-1 min-w-[180px]"
              type="password"
              placeholder="GitHub token (fine-grained PAT; needed for private repos & push)"
              value={token}
              data-testid="github-token"
              onChange={(e) => saveToken(e.target.value)}
            />
            <span className="text-[10px] text-[var(--color-ink-faint)]">stays in this browser</span>
          </div>
        </div>
      )}
    </div>
  );
}
