/**
 * Virtual filesystem backend for the in-browser pi agent.
 *
 * pi's read/write/edit tools talk to pluggable `*Operations` objects; this is
 * the storage those operations sit on. Primary impl is OPFS (origin-private
 * file system, persists across reloads). When OPFS is unavailable we fall back
 * to an in-memory store so the agent still runs (no persistence).
 *
 * All paths are absolute POSIX paths in the virtual tree; the tree root "/"
 * maps to the OPFS root directory.
 */

import { basename, dirname, join, normalize } from "./path";

export interface FsStat {
  kind: "file" | "directory";
  size: number;
}

export interface FsEntry {
  name: string;
  kind: "file" | "directory";
}

export interface FsBackend {
  readBytes(absPath: string): Promise<Uint8Array>;
  readText(absPath: string): Promise<string>;
  writeText(absPath: string, content: string): Promise<void>;
  mkdirp(absPath: string): Promise<void>;
  exists(absPath: string): Promise<boolean>;
  stat(absPath: string): Promise<FsStat | null>;
  list(absPath: string): Promise<FsEntry[]>;
  remove(absPath: string): Promise<void>;
  /** Absolute paths of every file under `absPath` (default root), sorted. */
  walk(absPath?: string): Promise<string[]>;
}

class FsError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = "FsError";
  }
}

function splitSegments(absPath: string): string[] {
  return normalize(absPath).split("/").filter(Boolean);
}

/* ------------------------------------------------------------------ OPFS --- */

class OpfsBackend implements FsBackend {
  private rootPromise: Promise<FileSystemDirectoryHandle>;

  constructor() {
    this.rootPromise = navigator.storage.getDirectory();
  }

  private async dirHandle(
    absPath: string,
    create: boolean,
  ): Promise<FileSystemDirectoryHandle> {
    let dir = await this.rootPromise;
    for (const seg of splitSegments(absPath)) {
      try {
        dir = await dir.getDirectoryHandle(seg, { create });
      } catch (e) {
        throw new FsError("ENOENT", `No such directory: ${absPath}`);
      }
    }
    return dir;
  }

  private async fileHandle(
    absPath: string,
    create: boolean,
  ): Promise<FileSystemFileHandle> {
    const dir = await this.dirHandle(dirname(absPath), create);
    try {
      return await dir.getFileHandle(basename(absPath), { create });
    } catch (e) {
      throw new FsError("ENOENT", `No such file: ${absPath}`);
    }
  }

  async readBytes(absPath: string): Promise<Uint8Array> {
    const handle = await this.fileHandle(absPath, false);
    const file = await handle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  }

  async readText(absPath: string): Promise<string> {
    const handle = await this.fileHandle(absPath, false);
    const file = await handle.getFile();
    return await file.text();
  }

  async writeText(absPath: string, content: string): Promise<void> {
    const handle = await this.fileHandle(absPath, true);
    const writable = await handle.createWritable();
    await writable.write(content);
    await writable.close();
  }

  async mkdirp(absPath: string): Promise<void> {
    await this.dirHandle(absPath, true);
  }

  async exists(absPath: string): Promise<boolean> {
    return (await this.stat(absPath)) !== null;
  }

  async stat(absPath: string): Promise<FsStat | null> {
    const segs = splitSegments(absPath);
    if (segs.length === 0) return { kind: "directory", size: 0 };
    const parent = await this.dirHandle(dirname(absPath), false).catch(() => null);
    if (!parent) return null;
    const name = basename(absPath);
    try {
      const fh = await parent.getFileHandle(name, { create: false });
      const file = await fh.getFile();
      return { kind: "file", size: file.size };
    } catch {
      /* not a file */
    }
    try {
      await parent.getDirectoryHandle(name, { create: false });
      return { kind: "directory", size: 0 };
    } catch {
      return null;
    }
  }

  async list(absPath: string): Promise<FsEntry[]> {
    const dir = await this.dirHandle(absPath, false);
    const entries: FsEntry[] = [];
    // @ts-expect-error - async iterator is standard on OPFS dir handles
    for await (const [name, handle] of dir.entries()) {
      entries.push({
        name,
        kind: handle.kind === "directory" ? "directory" : "file",
      });
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    return entries;
  }

  async remove(absPath: string): Promise<void> {
    const parent = await this.dirHandle(dirname(absPath), false);
    await parent.removeEntry(basename(absPath), { recursive: true });
  }

  async walk(absPath = "/"): Promise<string[]> {
    const out: string[] = [];
    const recurse = async (dirPath: string) => {
      let entries: FsEntry[];
      try {
        entries = await this.list(dirPath);
      } catch {
        return;
      }
      for (const entry of entries) {
        const childPath = join(dirPath, entry.name);
        if (entry.kind === "directory") {
          await recurse(childPath);
        } else {
          out.push(childPath);
        }
      }
    };
    await recurse(normalize(absPath));
    out.sort();
    return out;
  }
}

/* ---------------------------------------------------------------- memory --- */

class MemoryBackend implements FsBackend {
  private files = new Map<string, string>();
  private dirs = new Set<string>(["/"]);

  private ensureParentDirs(absPath: string): void {
    let dir = dirname(absPath);
    while (dir && dir !== "/" && !this.dirs.has(dir)) {
      this.dirs.add(dir);
      dir = dirname(dir);
    }
    this.dirs.add("/");
  }

  async readBytes(absPath: string): Promise<Uint8Array> {
    return new TextEncoder().encode(await this.readText(absPath));
  }

  async readText(absPath: string): Promise<string> {
    const key = normalize(absPath);
    if (!this.files.has(key)) throw new FsError("ENOENT", `No such file: ${absPath}`);
    return this.files.get(key)!;
  }

  async writeText(absPath: string, content: string): Promise<void> {
    const key = normalize(absPath);
    this.ensureParentDirs(key);
    this.files.set(key, content);
  }

  async mkdirp(absPath: string): Promise<void> {
    const key = normalize(absPath);
    this.dirs.add(key);
    this.ensureParentDirs(join(key, "_"));
  }

  async exists(absPath: string): Promise<boolean> {
    const key = normalize(absPath);
    return this.files.has(key) || this.dirs.has(key);
  }

  async stat(absPath: string): Promise<FsStat | null> {
    const key = normalize(absPath);
    if (this.files.has(key)) {
      return { kind: "file", size: new TextEncoder().encode(this.files.get(key)!).length };
    }
    if (this.dirs.has(key)) return { kind: "directory", size: 0 };
    return null;
  }

  async list(absPath: string): Promise<FsEntry[]> {
    const prefix = normalize(absPath) === "/" ? "/" : normalize(absPath) + "/";
    const seen = new Map<string, "file" | "directory">();
    for (const f of this.files.keys()) {
      if (!f.startsWith(prefix)) continue;
      const rest = f.slice(prefix.length);
      const slash = rest.indexOf("/");
      if (slash === -1) seen.set(rest, "file");
      else seen.set(rest.slice(0, slash), "directory");
    }
    for (const d of this.dirs) {
      if (d === normalize(absPath) || !d.startsWith(prefix)) continue;
      const rest = d.slice(prefix.length);
      const slash = rest.indexOf("/");
      seen.set(slash === -1 ? rest : rest.slice(0, slash), "directory");
    }
    return [...seen.entries()]
      .map(([name, kind]) => ({ name, kind }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  async remove(absPath: string): Promise<void> {
    const key = normalize(absPath);
    this.files.delete(key);
    this.dirs.delete(key);
    const prefix = key + "/";
    for (const f of [...this.files.keys()]) if (f.startsWith(prefix)) this.files.delete(f);
    for (const d of [...this.dirs]) if (d.startsWith(prefix)) this.dirs.delete(d);
  }

  async walk(absPath = "/"): Promise<string[]> {
    const prefix = normalize(absPath) === "/" ? "/" : normalize(absPath) + "/";
    return [...this.files.keys()].filter((f) => f.startsWith(prefix) || prefix === "/").sort();
  }
}

/* --------------------------------------------------------------- factory --- */

function opfsSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    !!navigator.storage &&
    typeof navigator.storage.getDirectory === "function"
  );
}

let backendSingleton: FsBackend | null = null;

export function getFsBackend(): FsBackend {
  if (backendSingleton) return backendSingleton;
  backendSingleton = opfsSupported() ? new OpfsBackend() : new MemoryBackend();
  return backendSingleton;
}

export const isPersistent = opfsSupported();
