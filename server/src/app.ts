import cors from 'cors';
import express, { NextFunction, Request, Response } from 'express';
import { config } from './config';
import { dbReady, pool } from './db';
import { HttpError } from './errors';
import { createMlRouter } from './ml-routes';
import {
  dataSourceStatus,
  getCoordinates,
  mapPointToGrid,
  runInfo,
} from './ml-data';
import { Npy, readAt, readBlock } from './npy';
import { arr, listRuns, readJson, runDir } from './runs';

const Z90 = 1.645;
const MAP_LAYERS = ['truth', 'prediction', 'uncertainty', 'error', 'observed', 'density'] as const;
type MapLayer = (typeof MAP_LAYERS)[number];

const CAVEATS = [
  'Simulation is not the real ocean.',
  'Predictions are not new measurements.',
  'Real data defines the sampling pattern.',
  'Precomputed results, not live inference.',
];

const nullable = (value: number): number | null => Number.isFinite(value) ? value : null;

function queryText(req: Request, name: string): string | undefined {
  return typeof req.query[name] === 'string' ? req.query[name] : undefined;
}

function requiredRun(req: Request): string {
  const run = queryText(req, 'run') ?? queryText(req, 'runId');
  if (!run) throw new HttpError(400, 'bad_run', 'Query parameter "run" is missing or invalid.');
  return run;
}

function integerQuery(req: Request, name: string, min: number, max: number): number {
  const raw = queryText(req, name);
  const value = raw === undefined ? NaN : Number(raw);
  if (!Number.isInteger(value) || value < min || value > max)
    throw new HttpError(400, 'bad_param', `"${name}" must be an integer in [${min}, ${max}].`);
  return value;
}

function coordinateAxes(meta: Record<string, any>, nlat: number, nlon: number) {
  const axis = (values: unknown, length: number): number[] | null =>
    Array.isArray(values) && values.length === length &&
      values.every((value) => typeof value === 'number' && Number.isFinite(value))
      ? values
      : null;
  return { latitudes: axis(meta.lat, nlat), longitudes: axis(meta.lon, nlon) };
}

function yearAndDepth(req: Request, metadata: Record<string, any>) {
  const years = metadata.years as number[];
  const depths = metadata.depth_labels_m as number[];
  if (!Array.isArray(years) || !years.length || !Array.isArray(depths) || !depths.length)
    throw new HttpError(503, 'run_metadata_invalid', 'Run metadata is missing years or depth labels.');
  const year = integerQuery(req, 'year', years[0], years[years.length - 1]);
  const yearIndex = years.indexOf(year);
  if (yearIndex < 0) throw new HttpError(400, 'bad_param', `Year ${year} is not supported by this run.`);
  const depth = integerQuery(req, 'depth', 0, depths.length - 1);
  return { years, depths, year, yearIndex, depth };
}

function plane(dir: string, layer: MapLayer, yearIndex: number, depthIndex: number): Float64Array {
  if (layer === 'error') {
    const prediction = readBlock(arr(dir, 'prediction')!, [yearIndex, depthIndex]);
    const truth = readBlock(arr(dir, 'truth')!, [yearIndex, depthIndex]);
    return prediction.map((value, index) => Number.isFinite(value) && Number.isFinite(truth[index])
      ? value - truth[index]
      : NaN);
  }
  const key = layer === 'truth' || layer === 'prediction' || layer === 'uncertainty' ||
    layer === 'observed' || layer === 'density' ? layer : 'prediction';
  const source = arr(dir, key)!;
  return readBlock(source, source.shape.length === 4 ? [yearIndex, depthIndex] : [depthIndex]);
}

function mapResponse(req: Request, layer: MapLayer) {
  const run = requiredRun(req);
  const dir = runDir(run);
  const info = runInfo(run);
  if (layer === 'uncertainty' && !info.uncertaintyAvailable)
    throw new HttpError(422, 'uncertainty_unavailable', 'Uncertainty unavailable for this baseline.');

  const metadata = readJson(dir, 'metadata.json') as Record<string, any>;
  const { depths, year, yearIndex, depth } = yearAndDepth(req, metadata);
  const truth = arr(dir, 'truth')!;
  const [, , nlat, nlon] = truth.shape;
  const coordinates = getCoordinates(run);
  const values = plane(dir, layer, yearIndex, depth);
  const ocean = readBlock(arr(dir, 'ocean')!, [depth]);
  const output: (number | null)[] = [];
  let min = Infinity;
  let max = -Infinity;
  const stride = queryText(req, 'stride') === undefined ? 1 : integerQuery(req, 'stride', 1, 16);
  const rows = Math.ceil(nlat / stride);
  const cols = Math.ceil(nlon / stride);

  for (let lat = 0; lat < nlat; lat += stride) {
    for (let lon = 0; lon < nlon; lon += stride) {
      const index = lat * nlon + lon;
      const value = ocean[index] === 0 ? NaN : values[index];
      if (Number.isFinite(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
      output.push(nullable(value));
    }
  }

  const units = layer === 'observed'
    ? 'binary sampling mask'
    : layer === 'density'
      ? 'fraction of observed cells'
      : metadata.oxygen_units ?? 'unknown';

  return {
    run,
    layer,
    year,
    depth_idx: depth,
    depth_m: depths[depth],
    shape: [rows, cols],
    stride,
    min: nullable(min),
    max: nullable(max),
    oxygen_units: units,
    latitude_values: coordinates.latitudeValues.filter((_, index) => index % stride === 0),
    longitude_values: coordinates.longitudeValues.filter((_, index) => index % stride === 0),
    simulation_caveat: true,
    values: output,
  };
}

function metricResponse(run: string): Record<string, any> {
  const info = runInfo(run);
  const metrics = readJson(runDir(run), 'metrics.json') as Record<string, any>;
  return {
    ...metrics,
    uncertainty: info.uncertaintyAvailable ? metrics.uncertainty ?? null : null,
    uncertainty_available: info.uncertaintyAvailable,
    uncertainty_reason: info.uncertaintyReason,
  };
}

function legacyMetadata(run: string) {
  const dir = runDir(run);
  const info = runInfo(run);
  const metadata = readJson(dir, 'metadata.json') as Record<string, any>;
  const configFile = readJson(dir, 'config.json') as Record<string, any>;
  const shape = arr(dir, 'truth')!.shape;
  const coordinates = getCoordinates(run);
  const layers = MAP_LAYERS.filter((layer) =>
    (layer !== 'uncertainty' || info.uncertaintyAvailable) &&
    (layer !== 'density' || Boolean(arr(dir, 'density', false))),
  );
  return {
    ...metadata,
    config: configFile,
    array_order: ['year', 'depth', 'lat', 'lon'],
    shape,
    layers,
    coordinates_available: true,
    lat: coordinates.latitudeValues,
    lon: coordinates.longitudeValues,
    uncertainty_available: info.uncertaintyAvailable,
    uncertainty_reason: info.uncertaintyReason,
    simulation_caveat: true,
    caveats: CAVEATS,
  };
}

function legacyLocate(req: Request) {
  const run = requiredRun(req);
  const dir = runDir(run);
  const metadata = readJson(dir, 'metadata.json') as Record<string, any>;
  const shape = arr(dir, 'truth')!.shape;
  const [, depths, nlat, nlon] = shape;
  const latitude = Number(queryText(req, 'latitude'));
  const longitude = Number(queryText(req, 'longitude'));
  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude))
    throw new HttpError(400, 'bad_coordinate', 'latitude must be between -90 and 90 and longitude must be a valid number.');
  const depth = integerQuery(req, 'depth', 0, depths - 1);
  const coordinates = getCoordinates(run);
  const grid = mapPointToGrid(run, latitude, longitude, metadata.years[0], coordinates.depthValuesM[depth]);
  const ocean = readAt(arr(dir, 'ocean')!, [depth, grid.arrayIndices.latitudeIndex, grid.arrayIndices.longitudeIndex]) !== 0;
  if (!ocean) throw new HttpError(422, 'land_cell', 'No valid ocean value exists at this location and depth.');
  return {
    requested: { latitude, longitude },
    nearest: {
      latitude: grid.grid.latitude,
      longitude: grid.grid.longitude,
      lat_idx: grid.grid.latitudeIndex,
      lon_idx: grid.grid.longitudeIndex,
    },
    depth_idx: depth,
  };
}

function legacyTimeSeries(req: Request) {
  const run = requiredRun(req);
  const dir = runDir(run);
  const info = runInfo(run);
  const metadata = readJson(dir, 'metadata.json') as Record<string, any>;
  const truth = arr(dir, 'truth')!;
  const [ny, nd, nlat, nlon] = truth.shape;
  const depth = integerQuery(req, 'depth', 0, nd - 1);
  const lat = integerQuery(req, 'lat', 0, nlat - 1);
  const lon = integerQuery(req, 'lon', 0, nlon - 1);
  const ocean = arr(dir, 'ocean')!;
  if (readAt(ocean, [depth, lat, lon]) === 0)
    throw new HttpError(422, 'land_cell', `Cell (depth ${depth}, lat ${lat}, lon ${lon}) is land or below seafloor.`);
  const prediction = arr(dir, 'prediction')!;
  const std = info.uncertaintyAvailable ? arr(dir, 'uncertainty', false) : null;
  const observed = arr(dir, 'observed')!;
  const values = (source: Npy) => Array.from({ length: ny }, (_, yearIndex) =>
    readAt(source, [yearIndex, depth, lat, lon]),
  );
  const truthValues = values(truth);
  const predictionValues = values(prediction);
  const stdValues = std ? values(std) : [];
  return {
    years: metadata.years,
    depth_idx: depth,
    depth_m: metadata.depth_labels_m[depth],
    lat,
    lon,
    truth: truthValues.map(nullable),
    prediction: predictionValues.map(nullable),
    lower90: std ? predictionValues.map((value, index) => nullable(value - Z90 * stdValues[index])) : null,
    upper90: std ? predictionValues.map((value, index) => nullable(value + Z90 * stdValues[index])) : null,
    observed: values(observed).map((value) => value === 1),
    uncertainty_available: Boolean(std),
    simulation_caveat: true,
  };
}

function legacyProfile(req: Request) {
  const run = requiredRun(req);
  const dir = runDir(run);
  const info = runInfo(run);
  const metadata = readJson(dir, 'metadata.json') as Record<string, any>;
  const truth = arr(dir, 'truth')!;
  const [, nd, nlat, nlon] = truth.shape;
  const year = integerQuery(req, 'year', metadata.years[0], metadata.years[metadata.years.length - 1]);
  const yearIndex = metadata.years.indexOf(year);
  if (yearIndex < 0) throw new HttpError(400, 'bad_param', `Year ${year} is not supported by this run.`);
  const lat = integerQuery(req, 'lat', 0, nlat - 1);
  const lon = integerQuery(req, 'lon', 0, nlon - 1);
  const ocean = arr(dir, 'ocean')!;
  const valid = Array.from({ length: nd }, (_, depth) => readAt(ocean, [depth, lat, lon]) !== 0);
  if (!valid.some(Boolean)) throw new HttpError(422, 'land_cell', `Column (lat ${lat}, lon ${lon}) is land.`);
  const prediction = arr(dir, 'prediction')!;
  const std = info.uncertaintyAvailable ? arr(dir, 'uncertainty', false) : null;
  const observed = arr(dir, 'observed')!;
  const column = (source: Npy) => Array.from({ length: nd }, (_, depth) =>
    valid[depth] ? readAt(source, [yearIndex, depth, lat, lon]) : NaN,
  );
  const predictions = column(prediction);
  const truths = column(truth);
  return {
    year,
    lat,
    lon,
    depth_m: metadata.depth_labels_m,
    truth: truths.map(nullable),
    prediction: predictions.map(nullable),
    std: std ? column(std).map(nullable) : null,
    uncertainty_available: Boolean(std),
    observed: column(observed).map((value) => Number.isNaN(value) ? null : value === 1),
    simulation_caveat: true,
  };
}

async function legacyComparison(res: Response, next: NextFunction) {
  try {
    const rows: Record<string, unknown>[] = [];
    if (dbReady) {
      const result = await pool.query('SELECT * FROM published_results ORDER BY dataset, rmse');
      for (const row of result.rows) {
        rows.push({
          source: 'published', dataset: row.dataset, split: row.split, model: row.model,
          feature_set: null, seed: null, rmse: row.rmse, r2: row.r2,
          reference: row.reference, scope: 'as reported in paper',
        });
      }
    }
    for (const run of listRuns()) {
      const dir = runDir(run);
      const info = runInfo(run);
      const runConfig = readJson(dir, 'config.json') as Record<string, any>;
      const metrics = metricResponse(run);
      rows.push({
        source: runConfig.source ?? info.source,
        run,
        dataset: runConfig.dataset ?? info.dataset,
        split: runConfig.split ?? info.split,
        model: runConfig.model ?? info.model,
        feature_set: runConfig.feature_set ?? null,
        seed: runConfig.seed ?? info.seed,
        rmse: metrics.never_observed?.rmse ?? null,
        mae: metrics.never_observed?.mae ?? null,
        r2: metrics.never_observed?.r2 ?? null,
        valid_cells: metrics.never_observed?.n ?? null,
        rmse_all: metrics.all?.rmse ?? null,
        rmse_observed: metrics.observed?.rmse ?? null,
        coverage90: info.uncertaintyAvailable ? metrics.uncertainty?.coverage90 ?? null : null,
        uncertainty_available: info.uncertaintyAvailable,
        uncertainty_reason: info.uncertaintyReason,
        scope: 'never-observed cells, shared evaluator',
      });
    }
    res.json({ note: 'Published and local rows are separate sources. Do not mix.', rows });
  } catch (error) {
    next(error);
  }
}

export function createApp() {
  const app = express();
  app.use(cors({ origin: config.corsOrigin }));

  // Canonical ML API. All new frontend data access goes through this router.
  app.use('/ml', createMlRouter());

  // Legacy read-only API retained for the existing map explorer and contract docs.
  app.get('/health', (_req, res) => res.json({
    status: 'ok', db: dbReady, runs: listRuns().length, dataSourceStatus: dataSourceStatus(),
  }));
  app.get('/runs', (_req, res) => res.json({ runs: listRuns() }));
  app.get('/metadata', (req, res) => res.json(legacyMetadata(requiredRun(req))));
  app.get('/metrics', (req, res) => res.json(metricResponse(requiredRun(req))));
  app.get('/comparison', (req, res, next) => void legacyComparison(res, next));
  app.get('/locate', (req, res) => res.json(legacyLocate(req)));
  app.get('/map', (req, res) => {
    const layer = queryText(req, 'layer') as MapLayer | undefined;
    if (!layer || !MAP_LAYERS.includes(layer))
      throw new HttpError(400, 'bad_param', `"layer" must be one of ${MAP_LAYERS.join(', ')}.`);
    res.json(mapResponse(req, layer));
  });
  app.get('/density', (req, res) => res.json(mapResponse(req, 'density')));
  app.get('/timeseries', (req, res) => res.json(legacyTimeSeries(req)));
  app.get('/profile', (req, res) => res.json(legacyProfile(req)));
  app.get('/physics-audit', (req, res) => {
    const dir = runDir(requiredRun(req));
    if (!arr(dir, 'physics', false)) throw new HttpError(501, 'not_available', 'physics_audit.npy is not available for this run.');
    throw new HttpError(501, 'not_available', 'The physics audit layout is not yet defined.');
  });

  app.use((_req, _res, next) => next(new HttpError(404, 'no_route', 'Unknown endpoint.')));
  app.use((error: Error, _req: Request, res: Response, _next: NextFunction) => {
    if (error instanceof HttpError) return res.status(error.status).json({ error: { code: error.code, message: error.message } });
    if (error instanceof RangeError) return res.status(400).json({ error: { code: 'bad_index', message: error.message } });
    console.error(error);
    return res.status(500).json({ error: { code: 'internal', message: 'Internal error.' } });
  });

  return app;
}
