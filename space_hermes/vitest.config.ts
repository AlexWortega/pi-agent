import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom", // gives us window + localStorage + fetch
    setupFiles: ["./test/setup.ts"],
    globals: false,
    include: ["test/**/*.test.ts"],
  },
});
