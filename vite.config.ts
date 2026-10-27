import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "./",
  worker: { format: "es" },
  server: { proxy: { '/api/native': { target: `http://127.0.0.1:${process.env.OTELLO_API_PORT ?? 8207}`, changeOrigin: true } } },
});
