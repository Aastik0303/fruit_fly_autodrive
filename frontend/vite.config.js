import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api/* is forwarded to the FastAPI backend, so the browser only talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
