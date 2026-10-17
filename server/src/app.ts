import cors from 'cors';
import express, { NextFunction, Request, Response } from 'express';
import { config } from './config';
import { dbReady, pool } from './db';
import { HttpError } from './errors';
import { readAt, readBlock } from './npy';
import { arr, listRuns, readJson, runDir } from './runs';

const Z90 = 1.645;
const nn = (v: number) => (Number.isFinite(v) ? v : null);

function int(q: Request['query'], name: string, min: number, max: number): number {
  const v = Number(q[name]);
  if (q[name] === undefined || !Number.isInteger(v) || v < min || v > max)
    throw new HttpError(400, 'bad_param', `"${name}" must be an integer in [${min}, ${max}]`);
  return v;
}

const MAP_LAYERS = ['truth', 'prediction', 'uncertainty', 'error', 'observed', 'density'] as const;
type MapLayer = (typeof MAP_LAYERS)[number];

function plane(dir: string, layer: MapLayer, yi: number | null, di: number): Float64Array {
  const lead = (a: { shape: number[] }) => (a.shape.length === 4 ? [yi as number, di] : [di]);
  if (layer === 'error') {
    const p = arr(dir, 'prediction')!, t = arr(dir, 'truth')!;
    const a = readBlock(p, lead(p)), b = readBlock(t, lead(t));
    return a.map((v, i) => v - b[i]);
  }
  const a = arr(dir, layer as 'truth' | 'prediction' | 'uncertainty' | 'observed' | 'density')!;
  return readBlock(a, lead(a));
}

function oceanPlane(dir: string, di: number): Float64Array {
  return readBlock(arr(dir, 'ocean')!, [di]);
}

function mapResponse(req: Request, layer: MapLayer) {
  const dir = runDir(req.query.run);
  const meta = readJson(dir, 'metadata.json');
  const years: number[] = meta.years;
  const year = int(req.query, 'year', years[0], years[years.length - 1]);
  const yi = years.indexOf(year);
  const di = int(req.query, 'depth', 0, meta.depth_labels_m.length - 1);
  const stride = req.query.stride === undefined ? 1 : int(req.query, 'stride', 1, 16);
  const ref = arr(dir, 'truth')!;
  const [, , nlat, nlon] = ref.shape;
  const vals = plane(dir, layer, yi, di);
  const ocean = oceanPlane(dir, di);
  const out: (number | null)[] = [];
  let min = Infinity, max = -Infinity;
  const rows = Math.ceil(nlat / stride), cols = Math.ceil(nlon / stride);
  for (let r = 0; r < nlat; r += stride)
    for (let c = 0; c < nlon; c += stride) {
      const k = r * nlon + c;
      const v = ocean[k] === 0 ? NaN : vals[k];
      if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
      out.push(nn(v));
    }
  return { run: req.query.run, layer, year, depth_idx: di, depth_m: meta.depth_labels_m[di], shape: [rows, cols],
    stride, min: nn(min), max: nn(max), oxygen_units: meta.oxygen_units, simulation_caveat: true, values: out };
}

function cellCheck(req: Request, dir: string, nd: number, nlat: number, nlon: number) {
  const lat = int(req.query, 'lat', 0, nlat - 1), lon = int(req.query, 'lon', 0, nlon - 1);
  const ocean = arr(dir, 'ocean')!;
  const bad = (di: number) => readAt(ocean, [di, lat, lon]) === 0;
  return { lat, lon, bad, nd };
}

export function createApp() {
  const app = express();
  app.use(cors({ origin: config.corsOrigin }));

  app.get('/health', (_q, res) => res.json({ status: 'ok', db: dbReady, runs: listRuns().length, results_dir: config.resultsDir }));

  app.get('/runs', (_q, res) => res.json({ runs: listRuns() }));

  app.get('/metadata', (req, res) => {
    const dir = runDir(req.query.run);
    const grid = arr(dir, 'truth')!.shape;
    res.json({ ...readJson(dir, 'metadata.json'), config: readJson(dir, 'config.json'), array_order: ['year', 'depth', 'lat', 'lon'],
      shape: grid, layers: [...MAP_LAYERS], simulation_caveat: true,
      caveats: ['Simulation is not the real ocean.', 'Predictions are not new measurements.', 'Real data defines the sampling pattern.', 'Precomputed results, not live inference.'] });
  });

  app.get('/metrics', (req, res) => res.json(readJson(runDir(req.query.run), 'metrics.json')));

  app.get('/comparison', async (_q, res, next) => {
    try {
    const rows: Record<string, unknown>[] = [];
    if (dbReady) {
      const { rows: pub } = await pool.query('SELECT * FROM published_results ORDER BY dataset, rmse');
      for (const p of pub) rows.push({ source: 'published', dataset: p.dataset, split: p.split, model: p.model, feature_set: null, seed: null, rmse: p.rmse, r2: p.r2, reference: p.reference, scope: 'as reported in paper' });
    }
    for (const run of listRuns()) {
      const dir = runDir(run);
      const cfg = readJson(dir, 'config.json'), m = readJson(dir, 'metrics.json');
      rows.push({ source: cfg.source ?? 'hybrid', run, dataset: cfg.dataset, split: cfg.split, model: cfg.model, feature_set: cfg.feature_set, seed: cfg.seed,
        rmse: m.never_observed?.rmse ?? null, mae: m.never_observed?.mae ?? null, r2: m.never_observed?.r2 ?? null,
        rmse_all: m.all?.rmse ?? null, rmse_observed: m.observed?.rmse ?? null, coverage90: m.uncertainty?.coverage90 ?? null, scope: 'never-observed cells, shared evaluator' });
    }
    res.json({ note: 'published / reproduced / hybrid rows are separate sources. Do not mix.', rows });
    } catch (e) { next(e); }
  });

  app.get('/map', (req, res) => {
    const layer = req.query.layer as MapLayer;
    if (!MAP_LAYERS.includes(layer)) throw new HttpError(400, 'bad_param', `"layer" must be one of ${MAP_LAYERS.join(', ')}`);
    res.json(mapResponse(req, layer));
  });

  app.get('/density', (req, res) => {
    arr(runDir(req.query.run), 'density');
    res.json(mapResponse(req, 'density'));
  });

  app.get('/timeseries', (req, res) => {
    const dir = runDir(req.query.run);
    const meta = readJson(dir, 'metadata.json');
    const [ny, nd, nlat, nlon] = arr(dir, 'truth')!.shape;
    const di = int(req.query, 'depth', 0, nd - 1);
    const { lat, lon, bad } = cellCheck(req, dir, nd, nlat, nlon);
    if (bad(di)) throw new HttpError(422, 'land_cell', `cell (depth ${di}, lat ${lat}, lon ${lon}) is land or below seafloor`);
    const t = arr(dir, 'truth')!, p = arr(dir, 'prediction')!, s = arr(dir, 'uncertainty')!, o = arr(dir, 'observed')!;
    const rd = (a: typeof t) => Array.from({ length: ny }, (_, y) => readAt(a, [y, di, lat, lon]));
    const truth = rd(t), pred = rd(p), std = rd(s), obs = rd(o);
    res.json({ years: meta.years, depth_idx: di, depth_m: meta.depth_labels_m[di], lat, lon, truth: truth.map(nn), prediction: pred.map(nn),
      lower90: pred.map((v, i) => nn(v - Z90 * std[i])), upper90: pred.map((v, i) => nn(v + Z90 * std[i])), observed: obs.map((v) => v === 1), simulation_caveat: true });
  });

  app.get('/profile', (req, res) => {
    const dir = runDir(req.query.run);
    const meta = readJson(dir, 'metadata.json');
    const [, nd, nlat, nlon] = arr(dir, 'truth')!.shape;
    const year = int(req.query, 'year', meta.years[0], meta.years[meta.years.length - 1]);
    const yi = meta.years.indexOf(year);
    const { lat, lon, bad } = cellCheck(req, dir, nd, nlat, nlon);
    const t = arr(dir, 'truth')!, p = arr(dir, 'prediction')!, s = arr(dir, 'uncertainty')!, o = arr(dir, 'observed')!;
    const valid = Array.from({ length: nd }, (_, d) => !bad(d));
    if (!valid.some(Boolean)) throw new HttpError(422, 'land_cell', `column (lat ${lat}, lon ${lon}) is land`);
    const col = (a: typeof t) => Array.from({ length: nd }, (_, d) => (valid[d] ? readAt(a, [yi, d, lat, lon]) : NaN));
    res.json({ year, lat, lon, depth_m: meta.depth_labels_m, truth: col(t).map(nn), prediction: col(p).map(nn), std: col(s).map(nn), observed: col(o).map((v) => (Number.isNaN(v) ? null : v === 1)), simulation_caveat: true });
  });

  app.get('/physics-audit', (req, res) => {
    const dir = runDir(req.query.run);
    if (!arr(dir, 'physics', false)) throw new HttpError(501, 'not_available', 'physics_audit.npy not produced for this run yet');
    res.status(501).json({ error: { code: 'todo', message: 'wire once M3 defines physics_audit.npy layout' } });
  });

  app.use((_q, _r, next) => next(new HttpError(404, 'no_route', 'unknown endpoint')));
  app.use((e: Error, _q: Request, res: Response, _n: NextFunction) => {
    if (e instanceof HttpError) return res.status(e.status).json({ error: { code: e.code, message: e.message } });
    if (e instanceof RangeError) return res.status(400).json({ error: { code: 'bad_index', message: e.message } });
    console.error(e);
    res.status(500).json({ error: { code: 'internal', message: 'internal error' } });
  });
  return app;
}
