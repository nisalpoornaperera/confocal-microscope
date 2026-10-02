/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/** Backend started with `uv run confocal-server` (default host/port). */
const BACKEND = process.env.CONFOCAL_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs: the backend serves frontend/dist at "/" (StaticFiles).
  base: "./",
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: false },
      "/openapi.json": { target: BACKEND, changeOrigin: false },
      "/ws": { target: BACKEND.replace(/^http/, "ws"), ws: true, changeOrigin: false },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: false },
      "/ws": { target: BACKEND.replace(/^http/, "ws"), ws: true, changeOrigin: false },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
    // plotly.js-dist-min is one ~4.6 MB pre-minified file, loaded lazily in its own
    // chunk by the plot pages only; it cannot be split further.
    chunkSizeWarningLimit: 5000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("plotly.js-dist-min")) return "plotly";
          if (id.includes("node_modules/react") || id.includes("node_modules/scheduler")) {
            return "react";
          }
          return undefined;
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
