import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    open: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/chat': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/upload': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      },
      '/suggestions': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
});

