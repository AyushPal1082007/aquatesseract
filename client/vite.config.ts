import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// browser calls /api/* ; vite forwards to express (strips /api)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY ?? 'http://localhost:4000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
});
