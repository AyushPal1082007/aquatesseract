import { Router, Request, Response, NextFunction } from 'express';
import { HttpError } from './errors';
import {
  availableRuns, comparisonData, configurationStatus, dataSourceStatus, getCoordinates, getGraphNodeData,
  getPoint, getSlice, pointProfile, pointTimeseries, provenance, recoverGraphCoordinates, runInfo, SLICE_LAYERS,
} from './ml-data';
import { getRegionData } from './region-service';
import { resolveRegion } from './regions';
import { fileRunArtifactRepository as repository, VERIFIED_RUNS } from './runs';

type Query = Request['query'];
const { readJson, resolveRunDirectory: runDir } = repository;

function text(q: Query, ...names: string[]): string | undefined {
  for (const name of names) {
    const value = q[name];
    if (typeof value === 'string' && value.length) return value;
  }
  return undefined;
}

function requiredText(q: Query, ...names: string[]): string {
  const value = text(q, ...names);
  if (value === undefined) throw new HttpError(400, 'invalid_request', `Missing query parameter "${names[0]}".`);
  return value;
}

function number(q: Query, name: string): number {
  const raw = text(q, name);
  const value = raw === undefined ? NaN : Number(raw);
  if (!Number.isFinite(value)) throw new HttpError(400, `invalid_${name}`, `"${name}" must be a finite number.`);
  return value;
}

function pointCoords(q: Query): { latitude: number; longitude: number } {
  if (text(q, 'x') !== undefined && text(q, 'y') !== undefined && text(q, 'z') !== undefined) {
    const recovered = recoverGraphCoordinates(number(q, 'x'), number(q, 'y'), number(q, 'z'));
    return { latitude: recovered.latitude, longitude: recovered.longitude0_to_360 };
  }
  return { latitude: number(q, 'latitude'), longitude: number(q, 'longitude') };
}

function boundedInteger(q: Query, name: string, defaultValue: number, min: number, max: number): number {
  const raw = text(q, name);
  const value = raw === undefined ? defaultValue : Number(raw);
  if (!Number.isInteger(value) || value < min || value > max)
    throw new HttpError(400, `invalid_${name}`, `"${name}" must be an integer in [${min}, ${max}].`);
  return value;
}

function runId(q: Query): string { return requiredText(q, 'runId', 'run'); }

function yearsForRun(run: string): number[] {
  const years = readJson(runDir(run), 'metadata.json').years;
  if (!Array.isArray(years) || years.length === 0 ||
      !years.every((value): value is number => typeof value === 'number' && Number.isInteger(value)))
    throw new HttpError(503, 'run_metadata_invalid', 'Run metadata must contain supported integer years.');
  return years;
}

function asyncRoute(fn: (req: Request, res: Response) => unknown) {
  return (req: Request, res: Response, next: NextFunction) => {
    try { Promise.resolve(fn(req, res)).catch(next); } catch (error) { next(error); }
  };
}

function yearValue(run: string, q: Query): number {
  const year = number(q, 'year');
  const years = yearsForRun(run);
  if (!years.includes(year)) throw new HttpError(400, 'invalid_year', 'Year is not supported by this run.');
  return year;
}

function layerValue(q: Query) {
  const layer = requiredText(q, 'layer');
  if (!SLICE_LAYERS.includes(layer as (typeof SLICE_LAYERS)[number]))
    throw new HttpError(400, 'unsupported_layer', `Layer must be one of: ${SLICE_LAYERS.join(', ')}.`);
  return layer as (typeof SLICE_LAYERS)[number];
}

function polygonValue(q: Query): unknown {
  const raw = text(q, 'polygon');
  if (!raw) return undefined;
  if (raw.length > 100_000) throw new HttpError(400, 'invalid_polygon', 'Polygon query is too large.');
  try { return JSON.parse(raw); } catch { throw new HttpError(400, 'invalid_polygon', 'Polygon must be JSON coordinate pairs.'); }
}

function regionSelection(q: Query) {
  const name = text(q, 'name', 'region');
  const rawBox = ['minLatitude', 'maxLatitude', 'minLongitude', 'maxLongitude'].map((name) => text(q, name));
  const hasBoxValue = rawBox.some((v) => v !== undefined);
  const box = hasBoxValue ? {
    minLatitude: number(q, 'minLatitude'),
    maxLatitude: number(q, 'maxLatitude'),
    minLongitude: number(q, 'minLongitude'),
    maxLongitude: number(q, 'maxLongitude'),
  } : undefined;
  return resolveRegion({ name, box, polygon: polygonValue(q) });
}

export function createMlRouter() {
  const router = Router();

  router.get('/runs', (_req, res) => {
    const available = availableRuns();
    const ids = new Set(available.map((run) => run.runId));
    const registered = Object.entries(VERIFIED_RUNS).map(([id, definition]) => ({
      runId: id,
      model: definition.model,
      uncertaintyAvailable: definition.uncertaintyAvailable,
      uncertaintyReason: definition.uncertaintyAvailable ? null : 'Uncertainty unavailable for this baseline',
      available: ids.has(id),
    }));
    res.json({ dataSourceStatus: dataSourceStatus(), configuration: configurationStatus(), available, registered });
  });

  router.get('/metadata', asyncRoute((req, res) => {
    const run = runId(req.query);
    const dir = runDir(run);
    const coordinates = getCoordinates(run);
    const shape = readJson(dir, 'metadata.json');
    const years = yearsForRun(run);
    const info = runInfo(run);
    res.json({
      ...info,
      dataSourceStatus: run.startsWith('MOCK_') ? 'mock' : dataSourceStatus(),
      dimensions: {
        year: years.length,
        depth: coordinates.depthValuesM.length,
        latitude: coordinates.latitudeValues.length,
        longitude: coordinates.longitudeValues.length,
      },
      years,
      coordinates,
      arrayOrder: ['year', 'depth', 'latitude', 'longitude'],
      oxygenUnits: shape.oxygen_units ?? null,
      availableLayers: SLICE_LAYERS.filter((layer) => layer !== 'uncertainty' || info.uncertaintyAvailable),
      contractValidationStatus: info.mock ? 'mock_fixture' : 'externally_contract_validated; runtime structure checked per request',
      provenance: provenance(run),
    });
  }));

  router.get('/slice', asyncRoute((req, res) => {
    const run = runId(req.query);
    const year = yearValue(run, req.query);
    const depthM = number(req.query, 'depth');
    const maxPoints = boundedInteger(req.query, 'maxPoints', 5000, 1, 10000);
    res.json(getSlice(run, year, depthM, layerValue(req.query), maxPoints));
  }));

  router.get('/point', asyncRoute((req, res) => {
    const coords = pointCoords(req.query);
    res.json(getPoint(
      runId(req.query), coords.latitude, coords.longitude,
      yearValue(runId(req.query), req.query), number(req.query, 'depth'),
    ));
  }));

  router.get('/point/profile', asyncRoute((req, res) => {
    const run = runId(req.query);
    const year = yearValue(run, req.query);
    const coords = pointCoords(req.query);
    res.json({
      run: runInfo(run), year,
      points: pointProfile(run, coords.latitude, coords.longitude, year),
      provenance: provenance(run, year),
    });
  }));

  router.get('/point/timeseries', asyncRoute((req, res) => {
    const run = runId(req.query);
    const requestedDepthM = number(req.query, 'depth');
    const coords = pointCoords(req.query);
    const points = pointTimeseries(run, coords.latitude, coords.longitude, requestedDepthM);
    const selectedDepthM = points[0]?.depthM ?? requestedDepthM;
    res.json({
      run: runInfo(run), requestedDepthM, depthM: selectedDepthM, points,
      provenance: provenance(run, undefined, selectedDepthM),
    });
  }));

  router.get('/graph/node', asyncRoute((req, res) => {
    const run = runId(req.query);
    const year = yearValue(run, req.query);
    const x = number(req.query, 'x');
    const y = number(req.query, 'y');
    const z = number(req.query, 'z');
    res.json(getGraphNodeData(run, year, x, y, z));
  }));

  router.get('/region', asyncRoute((req, res) => {
    const run = runId(req.query);
    const year = yearValue(run, req.query);
    const depthM = number(req.query, 'depth');
    const maxPoints = boundedInteger(req.query, 'maxPoints', 2000, 1, 10000);
    res.json(getRegionData(run, year, depthM, layerValue(req.query), maxPoints, regionSelection(req.query)));
  }));

  router.get('/comparison', (_req, res) => res.json(comparisonData()));
  router.get('/status', (_req, res) => res.json({
    dataSourceStatus: dataSourceStatus(), configuration: configurationStatus(),
  }));

  return router;
}
