/**
 * Browser Linux VM (v86) that backs the agent's real `bash` tool.
 *
 * Boots a small buildroot image (busybox userland: sh, grep, sed, awk, find,
 * diff, vi — no git/python/node) fully client-side. The image auto-mounts the
 * emulator's 9p filesystem at /mnt, which we use as the exchange point:
 * workspace files are pushed to /mnt/workspace before every command and pulled
 * back after, so `bash` and the read/write/edit tools always see the same tree.
 *
 * Commands run over the serial console. Arbitrary multi-line scripts are
 * written to a file via 9p and executed with `sh file` — no quoting/escaping
 * games on the serial line. A sentinel echo marks completion + exit code.
 */
import type { FsBackend } from "../pi/fs/backend";

export type VmPhase = "off" | "booting" | "ready" | "error";

export interface VmExecResult {
  output: string;
  exitCode: number;
}

const GUEST_WORKSPACE = "/mnt/workspace";
const MAX_OUTPUT = 50 * 1024;
/** Files we never sync into the VM (binary-ish — 9p+utf8 round-trip would corrupt). */
const SKIP_SYNC =
  /\.(png|jpe?g|gif|webp|ico|woff2?|ttf|otf|mp[34]|wav|ogg|zip|gz|pdf|wasm|gguf|bin|exe|so|dylib)$/i;

type Listener = () => void;

class BrowserVm {
  phase: VmPhase = "off";
  error = "";
  private emulator: any = null;
  private serialBuf = "";
  private bootPromise: Promise<void> | null = null;
  private execSeq = 0;
  private execChain: Promise<unknown> = Promise.resolve();
  private listeners = new Set<Listener>();

  onChange(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private setPhase(phase: VmPhase, error = "") {
    this.phase = phase;
    this.error = error;
    for (const fn of this.listeners) fn();
  }

  /** Boot the VM once; subsequent calls await the same boot. */
  boot(): Promise<void> {
    if (this.bootPromise) return this.bootPromise;
    this.bootPromise = this.doBoot().catch((e) => {
      this.bootPromise = null;
      this.setPhase("error", e instanceof Error ? e.message : String(e));
      throw e;
    });
    return this.bootPromise;
  }

  private async doBoot(): Promise<void> {
    this.setPhase("booting");
    // Lazy-load the emulator so the ~1.5MB libv86 chunk never blocks app start.
    const { V86 } = await import("v86");
    const base = import.meta.env.BASE_URL || "./";
    this.emulator = new V86({
      wasm_path: `${base}vm/v86.wasm`,
      bios: { url: `${base}vm/seabios.bin` },
      vga_bios: { url: `${base}vm/vgabios.bin` },
      cdrom: { url: `${base}vm/linux4.iso` },
      memory_size: 128 * 1024 * 1024,
      vga_memory_size: 2 * 1024 * 1024,
      filesystem: {},
      autostart: true,
      disable_keyboard: true,
      disable_mouse: true,
    });
    this.emulator.add_listener("serial0-output-byte", (byte: number) => {
      this.serialBuf += String.fromCharCode(byte);
      if (this.serialBuf.length > 400_000) this.serialBuf = this.serialBuf.slice(-200_000);
    });
    await this.waitSerial(/(~% |\/ # |# )/, 120_000, 0);
    // Quieten the shell and prepare the workspace mount point.
    await this.rawExec("stty -echo; mkdir -p /mnt/workspace", 10_000);
    this.setPhase("ready");
  }

  private waitSerial(re: RegExp, timeoutMs: number, from: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const t0 = Date.now();
      const iv = setInterval(() => {
        if (re.test(this.serialBuf.slice(from))) {
          clearInterval(iv);
          resolve();
        } else if (Date.now() - t0 > timeoutMs) {
          clearInterval(iv);
          reject(new Error(`VM timeout waiting for ${re}`));
        }
      }, 60);
    });
  }

  /** Run a short trusted command directly on the serial line. */
  private async rawExec(cmd: string, timeoutMs: number): Promise<VmExecResult> {
    this.execSeq += 1;
    const tag = `__DONE_${this.execSeq}__`;
    const mark = this.serialBuf.length;
    this.emulator.serial0_send(`${cmd}; echo; echo ${tag}$?\n`);
    await this.waitSerial(new RegExp(`${tag}\\d+`), timeoutMs, mark);
    const chunk = this.serialBuf.slice(mark);
    const m = chunk.match(new RegExp(`([\\s\\S]*?)\\n?${tag}(\\d+)`));
    let body = m ? m[1] : chunk;
    // Drop the echoed command line if the tty still echoes (pre-stty).
    const lines = body.replace(/\r/g, "").split("\n");
    if (lines[0]?.includes(cmd.slice(0, 30))) lines.shift();
    body = lines.join("\n").trim();
    return { output: body, exitCode: m ? parseInt(m[2], 10) : -1 };
  }

  /**
   * Execute an arbitrary (multi-line) script with the workspace synced in and
   * out. Serialized: one command at a time.
   */
  exec(fs: FsBackend, workspaceRoot: string, script: string, timeoutMs = 30_000): Promise<VmExecResult> {
    const run = async (): Promise<VmExecResult> => {
      await this.boot();
      await this.syncIn(fs, workspaceRoot);
      // Script goes through 9p — immune to quoting/newline issues on serial.
      await this.emulator.create_file(".cmd.sh", new TextEncoder().encode(script + "\n"));
      const result = await this.rawExec(`cd ${GUEST_WORKSPACE} && sh /mnt/.cmd.sh`, timeoutMs);
      await this.syncOut(fs, workspaceRoot);
      if (result.output.length > MAX_OUTPUT) {
        result.output = result.output.slice(0, MAX_OUTPUT) + `\n[output truncated at ${MAX_OUTPUT / 1024}KB]`;
      }
      return result;
    };
    const p = this.execChain.then(run, run);
    this.execChain = p.catch(() => {});
    return p;
  }

  /** Push current workspace text files into the guest (and drop stale ones). */
  private async syncIn(fs: FsBackend, workspaceRoot: string): Promise<void> {
    const prefix = workspaceRoot + "/";
    const files = (await fs.walk(workspaceRoot)).filter((p) => p.startsWith(prefix) && !SKIP_SYNC.test(p));
    const rels = files.map((p) => p.slice(prefix.length));

    // create_file() requires existing parent dirs — build them all in one guest
    // script (shipped over 9p, so no serial quoting limits).
    const dirs = new Set<string>();
    for (const rel of rels) {
      const parts = rel.split("/").slice(0, -1);
      for (let i = 1; i <= parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
    }
    const mkdirScript = [
      `rm -rf ${GUEST_WORKSPACE}`,
      `mkdir -p ${GUEST_WORKSPACE}`,
      ...[...dirs].sort().map((d) => `mkdir -p '${GUEST_WORKSPACE}/${d.replace(/'/g, "'\\''")}'`),
    ].join("\n");
    await this.emulator.create_file(".sync.sh", new TextEncoder().encode(mkdirScript + "\n"));
    await this.rawExec("sh /mnt/.sync.sh", 20_000);

    for (const rel of rels) {
      let text: string;
      try {
        text = await fs.readText(prefix + rel);
      } catch {
        continue;
      }
      await this.emulator.create_file(`workspace/${rel}`, new TextEncoder().encode(text));
    }
  }

  /** Pull the guest workspace back into OPFS (guest wins; deletions propagate). */
  private async syncOut(fs: FsBackend, workspaceRoot: string): Promise<void> {
    const listing = await this.rawExec(`find ${GUEST_WORKSPACE} -type f 2>/dev/null`, 15_000);
    const guestFiles = listing.output
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith(GUEST_WORKSPACE + "/"))
      .map((l) => l.slice(GUEST_WORKSPACE.length + 1));

    const prefix = workspaceRoot + "/";
    const hostFiles = new Set(
      (await fs.walk(workspaceRoot))
        .filter((p) => p.startsWith(prefix) && !SKIP_SYNC.test(p))
        .map((p) => p.slice(prefix.length)),
    );

    for (const rel of guestFiles) {
      let data: Uint8Array;
      try {
        data = await this.emulator.read_file(`workspace/${rel}`);
      } catch {
        continue;
      }
      const text = new TextDecoder().decode(data);
      let host: string | null = null;
      try {
        host = await fs.readText(prefix + rel);
      } catch {
        /* new file */
      }
      if (host !== text) await fs.writeText(prefix + rel, text);
      hostFiles.delete(rel);
    }
    // Files the guest deleted disappear from the workspace too.
    for (const rel of hostFiles) {
      try {
        await fs.remove(prefix + rel);
      } catch {
        /* already gone */
      }
    }
  }
}

export const browserVm = new BrowserVm();
