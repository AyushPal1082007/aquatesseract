import path from 'node:path';

const fromEnv = (name: string, fallback?: string): string | undefined => {
  const value = process.env[name];
  if (value) return path.resolve(value);
  return fallback ? path.resolve(fallback) : undefined;
};

export const config = {
  port: Number(process.env.PORT ?? 4000),
  resultsDir: path.resolve(process.env.RESULTS_DIR ?? path.join(__dirname, '../../results')),
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://aqua:aqua@localhost:5432/aquatesseract',
  corsOrigin: process.env.CORS_ORIGIN ?? 'http://localhost:5173',
  coordinateMetadataFile: fromEnv(
    'COORDINATE_METADATA_FILE',
    path.join(__dirname, '../verified_coordinate_metadata.json'),
  ),
  comparisonFile: fromEnv('COMPARISON_FILE'),
};
