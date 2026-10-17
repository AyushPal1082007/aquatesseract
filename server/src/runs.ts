import fs from 'node:fs';
import path from 'node:path';
import { config } from './config';
import { HttpError } from './errors';
import { openNpy, Npy } from './npy';

export const FILES = {
  truth: 'truth.npy',
  prediction: 'pred_mean.npy',
  uncertainty: 'pred_std.npy',
  observed: 'observed_mask.npy',
  ocean: 'ocean_mask.npy',
  density: 'observation_density.npy',
  physics: 'physics_audit.npy',
} as const;

export function listRuns(): string[] {
  if (!fs.existsSync(config.resultsDir)) return [];
  return fs
    .readdirSync(config.resultsDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && fs.existsSync(path.join(config.resultsDir, e.name, 'metadata.json')))
    .map((e) => e.name)
    .sort();
}

export function runDir(run: unknown): string {
  if (typeof run !== 'string' || !/^[\w.-]+$/.test(run)) throw new HttpError(400, 'bad_run', 'query param "run" missing or invalid');
  if (!listRuns().includes(run)) throw new HttpError(404, 'run_not_found', `no run folder "${run}"`);
  return path.join(config.resultsDir, run);
}

export const readJson = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));

export function arr(dir: string, key: keyof typeof FILES, required = true): Npy | null {
  const p = path.join(dir, FILES[key]);
  if (!fs.existsSync(p)) {
    if (required) throw new HttpError(404, 'array_missing', `${FILES[key]} not in this run`);
    return null;
  }
  return openNpy(p);
}
