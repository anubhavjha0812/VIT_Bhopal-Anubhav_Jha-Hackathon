import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: the dashboard calls /api/*, which Vite forwards to FastAPI on :8000 without the prefix.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // echarts is the largest dependency: its own file, cached apart from our code.
  build: {
    // only the echarts chunk is above the default 500 kB, and it loads on chart pages alone
    chunkSizeWarningLimit: 650,
    rollupOptions: {
      output: {
        manualChunks: (id) => (/node_modules\/(echarts|zrender)/.test(id) ? 'echarts' : undefined),
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
