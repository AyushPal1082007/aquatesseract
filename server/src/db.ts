import fs from 'node:fs';
import path from 'node:path';
import { Pool } from 'pg';
import { config } from './config';

export const pool = new Pool({ connectionString: config.databaseUrl, connectionTimeoutMillis: 2000 });
pool.on('error', (e) => console.warn('[db] pool error:', e.message));

export let dbReady = false;

export async function initDb(): Promise<void> {
  try {
    await pool.query(fs.readFileSync(path.resolve(__dirname, '../db/schema.sql'), 'utf8'));
    dbReady = true;
    console.log('[db] ready');
  } catch (e) {
    dbReady = false;
    console.warn('[db] not available, continuing file-only:', (e as Error).message);
  }
}
