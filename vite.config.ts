import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss(), crossOriginIsolation],
  // wllama ships prebuilt .wasm; let Vite serve it untouched instead of
  // trying to pre-bundle the package.
  optimizeDeps: {
    exclude: ["@reeselevine/wllama-webgpu"],
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5050,
  },
});
