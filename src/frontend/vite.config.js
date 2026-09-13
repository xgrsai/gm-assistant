import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies `/api` to the FastAPI backend, so the app uses relative
// URLs and needs no CORS. SSE responses are piped through http-proxy as they
// arrive (no compression middleware is added, so nothing buffers the stream).
export default defineConfig({
  plugins: [react()],
  server: {
    // Localhost only by default (the dev server is not hardened); Docker passes --host 0.0.0.0.
    host: process.env.VITE_HOST ?? '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        configure(proxy) {
          // If the backend drops a stream mid-response (crash, reload), http-proxy
          // leaves the browser connection hanging. Propagate the failure instead,
          // so the client sees a broken stream and can offer Retry.
          proxy.on('proxyRes', (proxyRes, _req, res) => {
            proxyRes.on('close', () => {
              if (!proxyRes.complete && !res.writableEnded) res.destroy()
            })
          })
        },
      },
    },
    // Bind-mounted sources on Docker Desktop for Windows do not emit fs events.
    watch: process.env.VITE_USE_POLLING ? { usePolling: true, interval: 300 } : undefined,
  },
})
