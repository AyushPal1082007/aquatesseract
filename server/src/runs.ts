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

export type ArrayKey = keyof typeof FILES;

export interface VerifiedRunDefinition {
  model: string;
  uncertaintyAvailable: boolean;
  relativePath?: string;
}

/**
 * These IDs are the production artifacts expected by the read-only ML API.
 * The path is only a preferred location: the resolver also supports a direct
 * run folder and a bounded search, which keeps local result layouts flexible.
 */
export const VERIFIED_RUNS: Record<string, VerifiedRunDefinition> = {
  CESM2_omip1_spatial_transformer_full_seed1: {
    model: 'transformer',
    uncertaintyAvailable: false,
    relativePath: 'CESM2_omip1_spatial_transformer_full_seed1',
  },
  CESM2_omip1_spatial_hybrid_full_seed1: {
    model: 'hybrid',
    uncertaintyAvailable: true,
    relativePath: 'CESM2_omip1_spatial_hybrid_full_seed1',
  },
  CESM2_omip1_spatial_hybrid_xdrop_seed1: {
    model: 'hybrid_xdrop',
    uncertaintyAvailable: true,
    relativePath: 'CESM2_omip1_spatial_hybrid_xdrop_seed1',
  },
  CESM2_omip1_spatial_lstm_full_seed1: {
    model: 'lstm',
    uncertaintyAvailable: false,
    relativePath: 'CESM2_omip1_spatial_lstm_full_seed1',
  },
};

const RUN_ID = /^[\w.-]+$/;

function validRunId(run: unknown): run is string {
  return typeof run === 'string' && RUN_ID.test(run);
}

function hasMetadata(dir: string): boolean {
  return fs.existsSync(path.join(dir, 'metadata.json'));
}

function walkForRun(root: string, run: string, depth: number): string | null {
  if (depth > 4 || !fs.existsSync(root)) return null;
  const entries = fs.readdirSync(root, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const candidate = path.join(root, entry.name);
    if (entry.name === run && hasMetadata(candidate)) return candidate;
    const nested = walkForRun(candidate, run, depth + 1);
    if (nested) return nested;
  }
  return null;
}

function directRunDirectories(): string[] {
  if (!fs.existsSync(config.resultsDir)) return [];
  return fs.readdirSync(config.resultsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && validRunId(entry.name) && hasMetadata(path.join(config.resultsDir, entry.name)))
    .map((entry) => entry.name);
}

export function listRunIds(): string[] {
  if (!fs.existsSync(config.resultsDir)) return [];
  const ids = new Set<string>(directRunDirectories());
  for (const [run, definition] of Object.entries(VERIFIED_RUNS)) {
    if (definition.relativePath && hasMetadata(path.join(config.resultsDir, definition.relativePath))) ids.add(run);
    else if (walkForRun(config.resultsDir, run, 0)) ids.add(run);
  }
  return [...ids].sort();
}

export function resolveRunDirectory(run: unknown): string {
  if (!validRunId(run)) {
    throw new HttpError(400, 'bad_run', 'Run ID is missing or invalid.');
  }

  const definition = VERIFIED_RUNS[run];
  const candidates = [
    path.join(config.resultsDir, run),
    definition?.relativePath ? path.join(config.resultsDir, definition.relativePath) : null,
  ].filter((candidate): candidate is string => candidate !== null);

  for (const candidate of candidates) {
    if (hasMetadata(candidate)) return candidate;
  }

  const discovered = walkForRun(config.resultsDir, run, 0);
  if (discovered) return discovered;
  throw new HttpError(404, 'run_not_found', `No run folder "${run}" was found.`);
}

export const registeredRunDir = resolveRunDirectory;
export const runDir = resolveRunDirectory;
export const listRuns = listRunIds;

export function readJson(dir: string, file: string): any {
  const source = fs.readFileSync(path.join(dir, file), 'utf8');
  try {
    return JSON.parse(source);
  } catch (error) {
    // Some training configs were serialized by Python and contain bare
    // NaN/Infinity tokens, which are valid Python but invalid JSON. Only
    // normalize value positions; quoted strings remain untouched.
    const normalized = source.replace(
      /(:\s*|,\s*|\[\s*)(-?Infinity|NaN)(?=\s*[,}\]])/g,
      '$1null',
    );
    try {
      return JSON.parse(normalized);
    } catch {
      throw error;
    }
  }
}

export function readArray(dir: string, key: ArrayKey, required = true): Npy | null {
  const file = path.join(dir, FILES[key]);
  if (!fs.existsSync(file)) {
    if (required) throw new HttpError(404, 'array_missing', `${FILES[key]} is not available for this run.`);
    return null;
  }
  return openNpy(file);
}

export const arr = readArray;

export const fileRunArtifactRepository = {
  listRunIds,
  resolveRunDirectory,
  readJson,
  readArray,
};
