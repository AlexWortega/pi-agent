/**
 * Client-side GitHub repo integration — no server, BYO personal access token.
 *
 * Import: Git Trees API (one request) lists the branch tree; file contents come
 * from raw.githubusercontent.com (public repos, CORS *, no meaningful rate
 * limit) or the Git Blobs API (private repos, needs the token). Files land in
 * the OPFS workspace where the agent's tools operate.
 *
 * Push: the Git Data API — create blobs for changed files, a tree on top of
 * the imported base tree (with `sha: null` entries for deletions), a commit
 * with the imported head as parent, then a fast-forward ref update. No git
 * binary anywhere.
 */
import type { FsBackend } from "../pi/fs/backend";

export const GITHUB_TOKEN_STORAGE = "pi_github_token";
/** Import metadata lives OUTSIDE the workspace so agent tools never see it. */
export const REPO_META_PATH = "/.repo-meta.json";

const API = "https://api.github.com";
const MAX_FILE_BYTES = 512 * 1024;
const MAX_TOTAL_BYTES = 30 * 1024 * 1024;
const MAX_FILES = 2000;

/** Extensions we never import (binary / media — useless as text, breaks utf-8 round-trip). */
const BINARY_EXT =
  /\.(png|jpe?g|gif|webp|ico|icns|bmp|tiff?|woff2?|ttf|otf|eot|mp[34]|wav|ogg|webm|mov|avi|zip|gz|tgz|bz2|xz|zst|7z|rar|jar|war|pdf|exe|dll|so|dylib|bin|dat|pyc|class|wasm|gguf|onnx|pt|pth|safetensors|parquet|sqlite|db)$/i;

export interface RepoRef {
  owner: string;
  repo: string;
  branch?: string;
}

export interface RepoMeta {
  owner: string;
  repo: string;
  branch: string;
  /** Commit sha the import is based on (push parent). */
  headSha: string;
  /** Tree sha of that commit (push base_tree). */
  treeSha: string;
  /** Imported files: workspace-relative path → git blob sha. */
  files: Record<string, string>;
  /** Paths present in the repo but skipped at import (binary/too big) — never touched on push. */
  skipped: string[];
  importedAt: number;
}

export function getGithubToken(): string {
  try {
    return (localStorage.getItem(GITHUB_TOKEN_STORAGE) ?? "").trim();
  } catch {
    return "";
  }
}

/**
 * Accepts "owner/repo", "owner/repo@branch", and github.com URLs (incl.
 * /tree/<branch>[/subpath] — subpath is ignored).
 */
export function parseRepoRef(input: string): RepoRef | null {
  let s = input.trim();
  if (!s) return null;
  const url = s.match(/^(?:https?:\/\/)?(?:www\.)?github\.com\/([^/\s]+)\/([^/\s#?]+)(?:\/tree\/([^\s#?]+))?/i);
  if (url) {
    const [, owner, repoRaw, treePath] = url;
    const repo = repoRaw.replace(/\.git$/i, "");
    // /tree/<branch>/<sub/dir> — take just the first segment as the branch (best effort;
    // branch names with slashes must use owner/repo@branch/with/slash form).
    const branch = treePath?.split("/")[0];
    return { owner, repo, branch };
  }
  const at = s.match(/^([^/\s@]+)\/([^/\s@]+)(?:@(.+))?$/);
  if (at) {
    const [, owner, repoRaw, branch] = at;
    return { owner, repo: repoRaw.replace(/\.git$/i, ""), branch: branch?.trim() || undefined };
  }
  return null;
}

function ghHeaders(token: string): Record<string, string> {
  const h: Record<string, string> = { Accept: "application/vnd.github+json" };
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

async function gh(path: string, token: string, init?: RequestInit): Promise<any> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { ...ghHeaders(token), ...(init?.headers as Record<string, string>) },
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.message) msg += ` — ${body.message}`;
    } catch {
      /* keep status text */
    }
    throw new Error(`GitHub API ${path}: ${msg}`);
  }
  return res.json();
}

function b64ToUtf8(b64: string): string {
  const bin = atob(b64.replace(/\n/g, ""));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

function utf8ToB64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

/** Git blob sha1: sha1("blob <byteLen>\0" + content). */
export async function computeGitBlobSha(content: string): Promise<string> {
  const body = new TextEncoder().encode(content);
  const header = new TextEncoder().encode(`blob ${body.length}\0`);
  const all = new Uint8Array(header.length + body.length);
  all.set(header);
  all.set(body, header.length);
  const digest = await crypto.subtle.digest("SHA-1", all);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export interface ImportProgress {
  phase: "resolving" | "listing" | "downloading" | "writing";
  done: number;
  total: number;
}

export interface ImportResult {
  meta: RepoMeta;
  imported: number;
  skipped: string[];
}

export async function importRepo(
  ref: RepoRef,
  fs: FsBackend,
  workspaceRoot: string,
  token: string,
  onProgress?: (p: ImportProgress) => void,
): Promise<ImportResult> {
  onProgress?.({ phase: "resolving", done: 0, total: 0 });
  const { owner, repo } = ref;
  const repoInfo = await gh(`/repos/${owner}/${repo}`, token);
  const branch = ref.branch || repoInfo.default_branch;
  const isPrivate = !!repoInfo.private;

  const branchInfo = await gh(`/repos/${owner}/${repo}/branches/${encodeURIComponent(branch)}`, token);
  const headSha: string = branchInfo.commit.sha;
  const treeSha: string = branchInfo.commit.commit.tree.sha;

  onProgress?.({ phase: "listing", done: 0, total: 0 });
  const tree = await gh(`/repos/${owner}/${repo}/git/trees/${treeSha}?recursive=1`, token);
  if (tree.truncated) {
    throw new Error("Repo tree is too large for the GitHub trees API (truncated) — import a smaller repo/branch.");
  }

  const blobs = (tree.tree as Array<{ path: string; type: string; sha: string; size?: number }>).filter(
    (e) => e.type === "blob",
  );

  const skipped: string[] = [];
  const toFetch: typeof blobs = [];
  let total = 0;
  for (const e of blobs) {
    const size = e.size ?? 0;
    if (BINARY_EXT.test(e.path) || size > MAX_FILE_BYTES || e.path.startsWith(".git/")) {
      skipped.push(e.path);
      continue;
    }
    if (toFetch.length >= MAX_FILES || total + size > MAX_TOTAL_BYTES) {
      skipped.push(e.path);
      continue;
    }
    total += size;
    toFetch.push(e);
  }

  // Clear the workspace, then write files as they arrive.
  try {
    await fs.remove(workspaceRoot);
  } catch {
    /* fresh */
  }

  const files: Record<string, string> = {};
  let done = 0;
  onProgress?.({ phase: "downloading", done: 0, total: toFetch.length });

  const fetchOne = async (e: { path: string; sha: string }) => {
    let text: string;
    if (isPrivate || token) {
      const blob = await gh(`/repos/${owner}/${repo}/git/blobs/${e.sha}`, token);
      text = b64ToUtf8(blob.content);
    } else {
      const res = await fetch(
        `https://raw.githubusercontent.com/${owner}/${repo}/${headSha}/${e.path.split("/").map(encodeURIComponent).join("/")}`,
      );
      if (!res.ok) throw new Error(`raw fetch ${e.path}: ${res.status}`);
      text = await res.text();
    }
    await fs.writeText(`${workspaceRoot}/${e.path}`, text);
    files[e.path] = e.sha;
    done += 1;
    if (done % 10 === 0 || done === toFetch.length) {
      onProgress?.({ phase: "downloading", done, total: toFetch.length });
    }
  };

  // Modest concurrency; raw.githubusercontent handles it fine.
  const queue = [...toFetch];
  const workers = Array.from({ length: 8 }, async () => {
    while (queue.length) {
      const e = queue.shift()!;
      await fetchOne(e);
    }
  });
  await Promise.all(workers);

  const meta: RepoMeta = {
    owner,
    repo,
    branch,
    headSha,
    treeSha,
    files,
    skipped,
    importedAt: Date.now(),
  };
  await fs.writeText(REPO_META_PATH, JSON.stringify(meta));
  return { meta, imported: toFetch.length, skipped };
}

export async function loadRepoMeta(fs: FsBackend): Promise<RepoMeta | null> {
  try {
    return JSON.parse(await fs.readText(REPO_META_PATH)) as RepoMeta;
  } catch {
    return null;
  }
}

export async function clearRepoMeta(fs: FsBackend): Promise<void> {
  try {
    await fs.remove(REPO_META_PATH);
  } catch {
    /* absent */
  }
}

export interface WorkspaceDiff {
  changed: Array<{ path: string; content: string }>;
  added: Array<{ path: string; content: string }>;
  deleted: string[];
}

/** Compare current workspace text files against the imported snapshot. */
export async function diffWorkspace(meta: RepoMeta, fs: FsBackend, workspaceRoot: string): Promise<WorkspaceDiff> {
  const prefix = workspaceRoot + "/";
  const current = (await fs.walk(workspaceRoot)).filter((p) => p.startsWith(prefix)).map((p) => p.slice(prefix.length));

  const changed: WorkspaceDiff["changed"] = [];
  const added: WorkspaceDiff["added"] = [];
  const seen = new Set<string>();

  for (const rel of current) {
    if (BINARY_EXT.test(rel)) continue;
    seen.add(rel);
    let content: string;
    try {
      content = await fs.readText(prefix + rel);
    } catch {
      continue;
    }
    const baseSha = meta.files[rel];
    if (!baseSha) {
      added.push({ path: rel, content });
    } else if ((await computeGitBlobSha(content)) !== baseSha) {
      changed.push({ path: rel, content });
    }
  }

  const deleted = Object.keys(meta.files).filter((rel) => !seen.has(rel));
  return { changed, added, deleted };
}

export interface PushResult {
  commitSha: string;
  commitUrl: string;
  changed: number;
  added: number;
  deleted: number;
}

/**
 * Create a real commit from the workspace diff and fast-forward the branch.
 * Fails cleanly if the remote branch moved since import (no force push).
 */
export async function commitAndPush(
  meta: RepoMeta,
  fs: FsBackend,
  workspaceRoot: string,
  message: string,
  token: string,
  onProgress?: (msg: string) => void,
): Promise<PushResult> {
  if (!token) throw new Error("A GitHub token is required to push.");
  const { owner, repo, branch } = meta;

  // Refuse non-fast-forward: the remote must still be where we imported from.
  const remote = await gh(`/repos/${owner}/${repo}/git/ref/heads/${encodeURIComponent(branch)}`, token);
  if (remote.object.sha !== meta.headSha) {
    throw new Error(
      `Remote ${branch} moved (${meta.headSha.slice(0, 7)} → ${remote.object.sha.slice(0, 7)}) since import. Re-import and re-apply changes.`,
    );
  }

  onProgress?.("diffing workspace…");
  const diff = await diffWorkspace(meta, fs, workspaceRoot);
  const upserts = [...diff.changed, ...diff.added];
  if (upserts.length === 0 && diff.deleted.length === 0) {
    throw new Error("Nothing to push — workspace matches the imported snapshot.");
  }

  const treeEntries: Array<Record<string, unknown>> = [];
  let n = 0;
  for (const f of upserts) {
    n += 1;
    onProgress?.(`uploading blob ${n}/${upserts.length}: ${f.path}`);
    const blob = await gh(`/repos/${owner}/${repo}/git/blobs`, token, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: utf8ToB64(f.content), encoding: "base64" }),
    });
    treeEntries.push({ path: f.path, mode: "100644", type: "blob", sha: blob.sha });
  }
  for (const rel of diff.deleted) {
    treeEntries.push({ path: rel, mode: "100644", type: "blob", sha: null });
  }

  onProgress?.("creating tree…");
  const newTree = await gh(`/repos/${owner}/${repo}/git/trees`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_tree: meta.treeSha, tree: treeEntries }),
  });

  onProgress?.("creating commit…");
  const commit = await gh(`/repos/${owner}/${repo}/git/commits`, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, tree: newTree.sha, parents: [meta.headSha] }),
  });

  onProgress?.("updating ref…");
  await gh(`/repos/${owner}/${repo}/git/refs/heads/${encodeURIComponent(branch)}`, token, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sha: commit.sha, force: false }),
  });

  // Refresh the snapshot so subsequent pushes diff against the new base.
  const newMeta: RepoMeta = { ...meta, headSha: commit.sha, treeSha: newTree.sha, files: { ...meta.files } };
  for (const f of upserts) newMeta.files[f.path] = await computeGitBlobSha(f.content);
  for (const rel of diff.deleted) delete newMeta.files[rel];
  await fs.writeText(REPO_META_PATH, JSON.stringify(newMeta));

  return {
    commitSha: commit.sha,
    commitUrl: `https://github.com/${owner}/${repo}/commit/${commit.sha}`,
    changed: diff.changed.length,
    added: diff.added.length,
    deleted: diff.deleted.length,
  };
}
