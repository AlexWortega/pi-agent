import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { createReadStream, existsSync, statSync } from "node:fs";
import { basename, join } from "node:path";

// Dev-only: serve local GGUF test models from ./models at /<name>.gguf for the
// headless test rig (scripts/harness-run.mjs). They live OUTSIDE public/ so
// `vite build` never copies the multi-GB files into dist/. Range requests are
// supported because wllama fetches model files in chunks.
function serveModels() {
  const handler = (req: any, res: any, next: any) => {
    const url: string = req.url?.split("?")[0] ?? "";
    if (!url.endsWith(".gguf")) return next();
    const file = join(process.cwd(), "models", basename(url));
    if (!existsSync(file)) return next();
    const size = statSync(file).size;
    const range = req.headers.range as string | undefined;
    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
    if (range) {
      const m = /bytes=(\d+)-(\d*)/.exec(range);
      const start = m ? parseInt(m[1], 10) : 0;
      const end = m && m[2] ? parseInt(m[2], 10) : size - 1;
      res.statusCode = 206;
      res.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
      res.setHeader("Content-Length", end - start + 1);
      createReadStream(file, { start, end }).pipe(res);
    } else {
      res.setHeader("Content-Length", size);
      createReadStream(file).pipe(res);
    }
  };
  return {
    name: "serve-models",
    configureServer(server: any) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server: any) {
      server.middlewares.use(handler);
    },
  };
}

// COOP/COEP are required for the multi-threaded WASM fallback path used by
// @reeselevine/wllama-webgpu. The WebGPU path works without them, but we set
// them so SharedArrayBuffer is available everywhere.
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
  configurePreviewServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

// Cloud-provider SDKs that pi-ai only loads via lazy dynamic import. The local
// WebGPU agent never touches them; keep them out of the bundle.
const CLOUD_SDKS = [
  "@anthropic-ai/sdk",
  "openai",
  "@aws-sdk/client-bedrock-runtime",
  "@smithy/node-http-handler",
  "@google/genai",
  "@mistralai/mistralai",
  "http-proxy-agent",
  "https-proxy-agent",
];

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss(), crossOriginIsolation, serveModels()],
  // wllama ships prebuilt .wasm; let Vite serve it untouched instead of
  // trying to pre-bundle the package.
  optimizeDeps: {
    exclude: ["@reeselevine/wllama-webgpu", ...CLOUD_SDKS],
  },
  build: {
    rollupOptions: {
      // pi-ai lazily imports cloud-provider SDKs (anthropic/openai/google/…)
      // via dynamic import. We always inject our own local WebGPU streamFn, so
      // those provider chunks are never loaded at runtime. Mark the heavy SDKs
      // external so the never-executed lazy chunks don't bundle ~1.5 MB of dead
      // code. Safe because nothing on the live code path imports them.
      external: CLOUD_SDKS,
    },
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5050,
  },
});
