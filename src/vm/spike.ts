/**
 * v86 feasibility spike: boot the buildroot Linux image, expose a serial
 * exec(cmd) with a sentinel protocol, and probe 9p filesystem sharing.
 * Driven by Playwright via window.vmExec / window.vmStatus.
 */
import { V86 } from "v86";

const statusEl = document.getElementById("status")!;
const serialEl = document.getElementById("serial")!;

declare global {
  interface Window {
    vmStatus: string;
    vmExec: (cmd: string, timeoutMs?: number) => Promise<string>;
    vmCreateFile: (path: string, data: string) => Promise<void>;
    vmReadFile: (path: string) => Promise<string | null>;
  }
}

window.vmStatus = "booting";

let serialBuf = "";

const emulator = new V86({
  wasm_path: "/vm/v86.wasm",
  bios: { url: "/vm/seabios.bin" },
  vga_bios: { url: "/vm/vgabios.bin" },
  cdrom: { url: "/vm/linux4.iso" },
  memory_size: 128 * 1024 * 1024,
  vga_memory_size: 2 * 1024 * 1024,
  filesystem: {},
  autostart: true,
  disable_keyboard: true,
  disable_mouse: true,
});

emulator.add_listener("serial0-output-byte", (byte: number) => {
  const ch = String.fromCharCode(byte);
  serialBuf += ch;
  if (serialBuf.length > 200000) serialBuf = serialBuf.slice(-100000);
  serialEl.textContent = serialBuf.slice(-4000);
});

/** Wait until the serial buffer matches `re` (checked on a poll), or throw. */
function waitSerial(re: RegExp, timeoutMs: number): Promise<void> {
  const start = serialBuf.length;
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const iv = setInterval(() => {
      if (re.test(serialBuf.slice(start))) {
        clearInterval(iv);
        resolve();
      } else if (Date.now() - t0 > timeoutMs) {
        clearInterval(iv);
        reject(new Error(`timeout waiting for ${re}`));
      }
    }, 100);
  });
}

let execSeq = 0;
async function vmExec(cmd: string, timeoutMs = 30000): Promise<string> {
  execSeq += 1;
  const tag = `__DONE_${execSeq}__`;
  const mark = serialBuf.length;
  // Leading echo forces the sentinel onto its own line even when the command's
  // output doesn't end with a newline.
  emulator.serial0_send(`${cmd}; echo; echo ${tag}$?\n`);
  await waitSerial(new RegExp(`${tag}\\d+`), timeoutMs);
  const out = serialBuf.slice(mark);
  // Strip the echoed command line and the sentinel line.
  return out
    .split("\n")
    .filter((l) => !l.includes(tag) && !l.includes(cmd.slice(0, 40)))
    .join("\n")
    .trim();
}
window.vmExec = vmExec;

window.vmCreateFile = async (path: string, data: string) => {
  await emulator.create_file(path, new TextEncoder().encode(data));
};
window.vmReadFile = async (path: string) => {
  try {
    const data = await emulator.read_file(path);
    return new TextDecoder().decode(data);
  } catch {
    return null;
  }
};

(async () => {
  try {
    // Buildroot prints a login or shell prompt on ttyS0.
    await waitSerial(/(~% |\/ # |login: |# )/, 90000);
    window.vmStatus = "shell";
    statusEl.textContent = "shell ready";
  } catch (e) {
    window.vmStatus = "boot-timeout: " + String(e);
    statusEl.textContent = window.vmStatus;
  }
})();
