/**
 * Browser POSIX path utilities — a minimal stand-in for the node:path subset
 * that pi's tools use (resolveToCwd / dirname / basename / relative …).
 *
 * The virtual filesystem is always POSIX-style rooted at "/". There is no
 * home directory and no file:// handling (both are node-only in pi); we keep
 * the `@`-prefix strip and unicode-space normalization that pi applies to
 * user-supplied paths so model output like "@src/app.ts" still resolves.
 */

const UNICODE_SPACES = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;

export interface PathInputOptions {
  trim?: boolean;
  stripAtPrefix?: boolean;
  normalizeUnicodeSpaces?: boolean;
}

export function isAbsolute(p: string): boolean {
  return p.startsWith("/");
}

/** Collapse ".", "..", and duplicate slashes. Mirrors node's posix normalize for absolute paths. */
export function normalize(p: string): string {
  const absolute = isAbsolute(p);
  const segments = p.split("/");
  const out: string[] = [];
  for (const seg of segments) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (out.length > 0 && out[out.length - 1] !== "..") {
        out.pop();
      } else if (!absolute) {
        out.push("..");
      }
      // for absolute paths, ".." at root is dropped
      continue;
    }
    out.push(seg);
  }
  const joined = out.join("/");
  if (absolute) return "/" + joined;
  return joined === "" ? "." : joined;
}

export function join(...parts: string[]): string {
  const joined = parts.filter((p) => p.length > 0).join("/");
  return normalize(joined);
}

/** Resolve `input` against `baseDir`, returning an absolute, normalized path. */
export function resolve(baseDir: string, input: string): string {
  if (isAbsolute(input)) return normalize(input);
  const base = isAbsolute(baseDir) ? baseDir : "/" + baseDir;
  return normalize(base + "/" + input);
}

export function dirname(p: string): string {
  const norm = normalize(p);
  const idx = norm.lastIndexOf("/");
  if (idx === -1) return ".";
  if (idx === 0) return "/";
  return norm.slice(0, idx);
}

export function basename(p: string): string {
  const norm = normalize(p);
  const idx = norm.lastIndexOf("/");
  return idx === -1 ? norm : norm.slice(idx + 1);
}

export function extname(p: string): string {
  const base = basename(p);
  const idx = base.lastIndexOf(".");
  return idx <= 0 ? "" : base.slice(idx);
}

/** Path of `target` relative to `from` (both treated as absolute dirs). */
export function relative(from: string, target: string): string {
  const fromParts = normalize(from).split("/").filter(Boolean);
  const toParts = normalize(target).split("/").filter(Boolean);
  let i = 0;
  while (i < fromParts.length && i < toParts.length && fromParts[i] === toParts[i]) i++;
  const up = fromParts.slice(i).map(() => "..");
  const down = toParts.slice(i);
  const rel = [...up, ...down].join("/");
  return rel;
}

export function normalizePath(input: string, options: PathInputOptions = {}): string {
  let normalized = options.trim ? input.trim() : input;
  if (options.normalizeUnicodeSpaces) {
    normalized = normalized.replace(UNICODE_SPACES, " ");
  }
  if (options.stripAtPrefix && normalized.startsWith("@")) {
    normalized = normalized.slice(1);
  }
  return normalized;
}

/**
 * Resolve a user/model-supplied path relative to cwd. Matches pi's
 * `resolveToCwd` option set (normalizeUnicodeSpaces + stripAtPrefix).
 */
export function resolveToCwd(filePath: string, cwd: string): string {
  const normalized = normalizePath(filePath, { normalizeUnicodeSpaces: true, stripAtPrefix: true });
  return resolve(cwd, normalized);
}

/** Relative path inside cwd, or undefined if the target escapes cwd. */
export function getCwdRelativePath(filePath: string, cwd: string): string | undefined {
  const resolvedCwd = normalize(cwd);
  const resolvedPath = resolve(resolvedCwd, filePath);
  const rel = relative(resolvedCwd, resolvedPath);
  const inside = rel === "" || (rel !== ".." && !rel.startsWith("../") && !isAbsolute(rel));
  return inside ? rel || "." : undefined;
}

/** Display path: relative to cwd when inside it, else absolute. */
export function formatPathRelativeToCwdOrAbsolute(filePath: string, cwd: string): string {
  const absolutePath = resolve(cwd, filePath);
  return getCwdRelativePath(absolutePath, cwd) ?? absolutePath;
}
