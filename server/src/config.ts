import path from 'node:path';

export const config = {
  port: Number(process.env.PORT ?? 4000),
  resultsDir: path.resolve(process.env.RESULTS_DIR ?? path.join(__dirname, '../../results')),
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://aqua:aqua@localhost:5432/aquatesseract',
  corsOrigin: process.env.CORS_ORIGIN ?? 'http://localhost:5173',
};
