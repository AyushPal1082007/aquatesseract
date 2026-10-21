import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, expect, test } from 'vitest';

const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aquatesseract-api-'));
const resultsRoot = path.join(temporaryRoot, 'runs');
const coordinateFile = path.join(temporaryRoot, 'coordinates.json');
const comparisonFile = path.join(temporaryRoot, 'comparison.csv');
const hybridId = 'CESM2_omip1_spatial_hybrid_full_seed1';
const baselineId = 'CESM2_omip1_spatial_lstm_full_seed1';
let server: ReturnType<import('node:http').Server['listen']>;
let baseUrl = '';

function writeNpy(file: string, shape: number[], values: number[], dtype: '<f4' | '|u1') {
  const descriptor = dtype === '<f4' ? 4 : 1;
  const tuple = `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  const prefix = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${tuple}, }`;
  const padding = 16 - ((10 + Buffer.byteLength(prefix) + 1) % 16);
  const header = Buffer.from(`${prefix}${' '.repeat(padding === 16 ? 0 : padding)}\n`, 'latin1');
  const output = Buffer.alloc(10 + header.length + values.length * descriptor);
  output.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  output.writeUInt16LE(header.length, 8);
  header.copy(output, 10);
  values.forEach((value, index) => {
    const offset = 10 + header.length + index * descriptor;
    if (dtype === '<f4') output.writeFloatLE(value, offset);
    else output.writeUInt8(value, offset);
  });
  fs.writeFileSync(file, output);
}

function product(values: number[]) { return values.reduce((a, b) => a * b, 1); }

function createRun(id: string, model: string) {
  const relative = id.includes('spatial_hybrid_') ? path.join('hybrid_fixed', id) : id;
  const directory = path.join(resultsRoot, relative);
  fs.mkdirSync(directory, { recursive: true });
  const shape = [2, 2, 2, 4];
  const size = product(shape);
  const truth = Array.from({ length: size }, (_, index) => index + 1);
  const prediction = truth.map((value) => value + 0.5);
  const observed = Array(size).fill(0);
  const ocean = Array(product(shape.slice(1))).fill(1);
  const std = Array(size).fill(0.25);
  // Coordinate longitude 270 maps to artifact longitude 0 after the configured single-cell shift.
  ocean[0] = 0;
  truth[0] = NaN;
  prediction[0] = NaN;
  // One ocean cell contains invalid values and must not contribute to summaries.
  const invalidIndex = ((0 * 2 + 0) * 2 + 1) * 4 + 2;
  truth[invalidIndex] = NaN;
  prediction[invalidIndex] = NaN;
  observed[0] = 1;
  observed[8] = 1;
  observed[12] = 1;
  writeNpy(path.join(directory, 'truth.npy'), shape, truth, '<f4');
  writeNpy(path.join(directory, 'pred_mean.npy'), shape, prediction, '<f4');
  writeNpy(path.join(directory, 'pred_std.npy'), shape, std, '<f4');
  writeNpy(path.join(directory, 'observed_mask.npy'), shape, observed, '|u1');
  writeNpy(path.join(directory, 'ocean_mask.npy'), shape.slice(1), ocean, '|u1');
  fs.writeFileSync(path.join(directory, 'metadata.json'), JSON.stringify({
    dataset: 'CESM2_omip1', split: 'spatial', model, seed: 1, years: [2000, 2001],
    depth_labels_m: [0, 100], oxygen_units: 'synthetic fixture', simulation_caveat: true,
  }));
  fs.writeFileSync(path.join(directory, 'config.json'), JSON.stringify({
    dataset: 'CESM2_omip1', split: 'spatial', model, feature_set: 'full', seed: 1, source: 'hybrid',
  }));
  fs.writeFileSync(path.join(directory, 'metrics.json'), JSON.stringify({
    all: { rmse: 0.1, mae: 0.05, r2: 0.5, n: 10 },
    observed: { rmse: 0.1, mae: 0.05, r2: 0.5, n: 2 },
    never_observed: { rmse: 0.1, mae: 0.05, r2: 0.5, n: 8 },
    uncertainty: { coverage90: 0.99 },
  }));
}

beforeAll(async () => {
  fs.mkdirSync(resultsRoot, { recursive: true });
  createRun(hybridId, 'hybrid');
  createRun(baselineId, 'lstm');
  fs.writeFileSync(coordinateFile, JSON.stringify({
    latitudeValues: [-10, 10],
    latitudeOrder: 'south_to_north',
    longitudeValues: [0, 90, 180, 270],
    longitudeOrder: 'west_to_east',
    longitudeConvention: '0-360',
    depthValuesM: [0, 100],
    depthUnits: 'm',
    longitudeShiftCells: 1,
    oceanMaskAligned: true,
  }));
  fs.writeFileSync(comparisonFile, 'model,rmse\nhybrid_spatial_full,0.1\n');
  process.env.RESULTS_DIR = resultsRoot;
  process.env.COORDINATE_METADATA_FILE = coordinateFile;
  process.env.COMPARISON_FILE = comparisonFile;
  process.env.AQUATESSERACT_ALLOW_SYNTHETIC_FIXTURES = '1';
  const { createApp } = await import('./app.js');
  server = createApp().listen(0);
  await new Promise<void>((resolve) => server.once('listening', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('test HTTP server did not bind a port');
  baseUrl = `http://127.0.0.1:${address.port}/ml`;
});

afterAll(async () => {
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  fs.rmSync(temporaryRoot, { recursive: true, force: true });
});

async function get(pathname: string) {
  const response = await fetch(`${baseUrl}${pathname}`);
  return { response, body: await response.json() as Record<string, any> };
}

test('validates coordinate metadata, longitude conventions, and explicit shift', async () => {
  const { normalizeLongitude, parseCoordinateMetadata } = await import('./ml-data.js');
  expect(normalizeLongitude(370, '0-360')).toBe(10);
  expect(normalizeLongitude(-190, '-180-180')).toBe(170);
  expect(() => normalizeLongitude(Number.NaN, '0-360')).toThrow();
  expect(() => parseCoordinateMetadata({ latitudeValues: [0], longitudeValues: [0], depthValuesM: [0] })).toThrow();
  const { response, body } = await get(`/point?runId=${hybridId}&latitude=9&longitude=-91&year=2000&depth=95`);
  expect(response.status).toBe(200);
  expect(body.normalized.longitude).toBe(269);
  expect(body.grid.longitudeIndex).toBe(3);
  expect(body.arrayIndices.longitudeIndex).toBe(0);
  expect(body.grid.latitude).toBe(10);
  expect(body.grid.latitudeIndex).toBe(1);
  expect(body.arrayIndices.depthIndex).toBe(1);
  expect(body.grid.depthM).toBe(100);
  expect(body.values.prediction).toBe(13.5);
  expect(body.values.observedStatus).toBe('observed');
  expect(body.provenance.datasetType).toBe('simulation');
});

test('returns point status and uncertainty only for registry-approved hybrid runs', async () => {
  const { response, body } = await get(`/point?runId=${hybridId}&latitude=-10&longitude=270&year=2000&depth=0`);
  expect(response.status).toBe(200);
  expect(body.status).toBe('land');
  expect(body.values.oceanStatus).toBe('land');
  expect(body.values.uncertaintyAvailable).toBe(true);
  expect(body.values.uncertainty).toBeNull();

  const baseline = await get(`/point?runId=${baselineId}&latitude=10&longitude=270&year=2000&depth=0`);
  expect(baseline.response.status).toBe(200);
  expect(baseline.body.values.uncertainty).toBeNull();
  expect(baseline.body.values.uncertaintyAvailable).toBe(false);
  expect(baseline.body.values.uncertaintyReason).toBe('Uncertainty unavailable for this baseline');

  const noBaselineUncertainty = await get(`/slice?runId=${baselineId}&year=2000&depth=0&layer=uncertainty`);
  expect(noBaselineUncertainty.response.status).toBe(422);
  expect(noBaselineUncertainty.body.error.code).toBe('uncertainty_unavailable');
});

test('serves profiles, time series, named regions, bounding boxes, and polygons', async () => {
  const profile = await get(`/point/profile?runId=${hybridId}&latitude=10&longitude=270&year=2000`);
  expect(profile.response.status).toBe(200);
  expect(profile.body.points).toHaveLength(2);

  const series = await get(`/point/timeseries?runId=${hybridId}&latitude=10&longitude=270&depth=0`);
  expect(series.response.status).toBe(200);
  expect(series.body.points.map((point: { year: number }) => point.year)).toEqual([2000, 2001]);

  for (const selection of [
    'region=Atlantic%20Ocean',
    'minLatitude=-20&maxLatitude=20&minLongitude=-10&maxLongitude=20',
    `polygon=${encodeURIComponent(JSON.stringify([[-10, -20], [20, -20], [20, 20], [-10, 20]]))}`,
  ]) {
    const region = await get(`/region?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&maxPoints=2&${selection}`);
    expect(region.response.status).toBe(200);
    expect(region.body.summary.validOceanCellCount).toBeGreaterThan(0);
    expect(region.body.visualization.points.length).toBeLessThanOrEqual(2);
  }
});

test('excludes invalid values from region metrics and reports empty or malformed regions', async () => {
  const empty = await get(`/region?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&minLatitude=70&maxLatitude=80&minLongitude=0&maxLongitude=10`);
  expect(empty.response.status).toBe(422);
  expect(empty.body.error.code).toBe('empty_region');

  const malformed = await get(`/region?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&polygon=not-json`);
  expect(malformed.response.status).toBe(400);
  expect(malformed.body.error.code).toBe('invalid_polygon');

  const result = await get(`/region?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&region=Pacific%20Ocean`);
  expect(result.body.summary.validOceanCellCount).toBeLessThan(result.body.summary.regionCellCount);
  expect(result.body.summary.rmse).toBeTypeOf('number');

  const finiteOnly = await get(`/region?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&minLatitude=0&maxLatitude=20&minLongitude=0&maxLongitude=100`);
  expect(finiteOnly.response.status).toBe(200);
  expect(finiteOnly.body.summary.regionCellCount).toBe(2);
  expect(finiteOnly.body.summary.validOceanCellCount).toBe(1);
});

test('rejects invalid coordinates, run IDs, years, depths, layers, and oversized requests', async () => {
  const checks = [
    [`/point?runId=${hybridId}&latitude=91&longitude=0&year=2000&depth=0`, 'invalid_latitude'],
    [`/point?runId=${hybridId}&latitude=0&longitude=Infinity&year=2000&depth=0`, 'invalid_longitude'],
    ['/point?runId=../../etc/passwd&latitude=0&longitude=0&year=2000&depth=0', 'bad_run'],
    [`/point?runId=${hybridId}&latitude=0&longitude=0&year=1900&depth=0`, 'invalid_year'],
    [`/point?runId=${hybridId}&latitude=0&longitude=0&year=2000&depth=1000`, 'invalid_depth'],
    [`/slice?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&maxPoints=10001`, 'invalid_maxPoints'],
    [`/slice?runId=${hybridId}&year=2000&depth=0&layer=unknown`, 'unsupported_layer'],
  ];
  for (const [url, code] of checks) {
    const { response, body } = await get(url);
    expect(response.status).toBeGreaterThanOrEqual(400);
    expect(body.error.code).toBe(code);
  }
});

test('limits slice point counts and reads configured comparison CSV without synthesizing metrics', async () => {
  const slice = await get(`/slice?runId=${hybridId}&year=2000&depth=0&layer=reconstruction&maxPoints=2`);
  expect(slice.response.status).toBe(200);
  expect(slice.body.points.length).toBeLessThanOrEqual(2);
  expect(slice.body.exact).toBe(false);

  const comparison = await get('/comparison');
  expect(comparison.body.dataSourceStatus).toBe('verified');
  expect(comparison.body.rows).toEqual([{ model: 'hybrid_spatial_full', rmse: '0.1' }]);
});

test('keeps baseline placeholder uncertainty out of legacy API surfaces', async () => {
  const metadata = await get(`/../metadata?run=${baselineId}`);
  expect(metadata.response.status).toBe(200);
  expect(metadata.body.layers).not.toContain('uncertainty');
  expect(metadata.body.uncertainty_available).toBe(false);

  const map = await get(`/../map?run=${baselineId}&layer=uncertainty&year=2000&depth=0`);
  expect(map.response.status).toBe(422);
  expect(map.body.error.code).toBe('uncertainty_unavailable');

  const metrics = await get(`/../metrics?run=${baselineId}`);
  expect(metrics.response.status).toBe(200);
  expect(metrics.body.uncertainty).toBeNull();
  expect(metrics.body.uncertainty_available).toBe(false);

  const profile = await get(`/../profile?run=${baselineId}&year=2000&lat=1&lon=3`);
  expect(profile.response.status).toBe(200);
  expect(profile.body.std).toBeNull();
  expect(profile.body.uncertainty_available).toBe(false);

  const series = await get(`/../timeseries?run=${baselineId}&depth=0&lat=1&lon=3`);
  expect(series.response.status).toBe(200);
  expect(series.body.lower90).toBeNull();
  expect(series.body.uncertainty_available).toBe(false);

  const comparison = await get('/../comparison');
  const baselineComparison = (comparison.body.rows as Array<Record<string, unknown>>)
    .find((row) => row.run === baselineId);
  expect(baselineComparison?.coverage90).toBeNull();
  expect(baselineComparison?.uncertainty_available).toBe(false);
});

test('handles graph unit-sphere coordinates on /point and /graph/node', async () => {
  // Query /point with x, y, z instead of latitude, longitude
  // z = sin(10 deg), x = cos(10 deg) * cos(0 deg), y = cos(10 deg) * sin(0 deg)
  const latRad = (10 * Math.PI) / 180;
  const z = Math.sin(latRad);
  const x = Math.cos(latRad);
  const y = 0;

  const pointRes = await get(`/point?runId=${hybridId}&x=${x}&y=${y}&z=${z}&year=2000&depth=0`);
  expect(pointRes.response.status).toBe(200);
  expect(pointRes.body.grid.latitude).toBe(10);

  const nodeRes = await get(`/graph/node?runId=${hybridId}&x=${x}&y=${y}&z=${z}&year=2000`);
  expect(nodeRes.response.status).toBe(200);
  expect(nodeRes.body.recovered.latitude).toBeCloseTo(10, 4);
  expect(nodeRes.body.recovered.longitude0_to_360).toBeCloseTo(0, 4);
  expect(nodeRes.body.profile.length).toBe(2); // In test fixture, depth levels length is 2
});

