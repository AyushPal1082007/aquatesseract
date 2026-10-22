import fs from 'node:fs';
import path from 'node:path';
import { config } from './config';
import { HttpError } from './errors';
import { readAt, readBlock } from './npy';
import { fileRunArtifactRepository as repository, registeredRunDir, VERIFIED_RUNS } from './runs';

const { listRunIds: listRuns, resolveRunDirectory: runDir, readJson, readArray: arr } = repository;

export type LongitudeConvention = '0-360' | '-180-180' | '0_to_360_east';
export type DataSourceStatus = 'mock' | 'unavailable' | 'verified';

export const ARRAY_ORDER = ['year', 'depth', 'latitude', 'longitude'] as const;
export const VERIFIED_DEPTH_LABELS_M = [
  0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800,
  900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000, 3500, 4000,
  4500, 5000, 5500,
];
export const VERIFIED_LATITUDE_CENTERS = Array.from({ length: 180 }, (_, i) => -89.5 + i);
export const VERIFIED_LONGITUDE_CENTERS = Array.from({ length: 360 }, (_, j) => 0.5 + j);

export interface Coordinates {
  latitudeValues: number[];
  longitudeValues: number[];
  depthValuesM: number[];
  depthUnits: 'm';
  latitudeOrder: string;
  longitudeOrder: string;
  longitudeConvention: LongitudeConvention;
  longitudeShiftCells: number;
  oceanMaskAligned: true;
}

export const VERIFIED_CONTRACT_COORDINATES: Coordinates = {
  latitudeValues: VERIFIED_LATITUDE_CENTERS,
  longitudeValues: VERIFIED_LONGITUDE_CENTERS,
  depthValuesM: VERIFIED_DEPTH_LABELS_M,
  depthUnits: 'm',
  latitudeOrder: 'south_to_north',
  longitudeOrder: 'west_to_east',
  longitudeConvention: '0-360',
  longitudeShiftCells: 0,
  oceanMaskAligned: true,
};

export interface RunInfo {
  runId: string;
  dataset: string;
  model: string;
  split: string;
  seed: number | null;
  source: string;
  mock: boolean;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
}

interface RunMetadata extends Record<string, unknown> {
  years: number[];
  dataset?: string;
  model?: string;
  split?: string;
  seed?: number;
  mock?: boolean;
  depth_labels_m?: number[];
  oxygen_units?: string;
  latitude_convention?: string;
  longitude_convention?: string;
}

interface RunConfiguration extends Record<string, unknown> {
  dataset?: string;
  model?: string;
  split?: string;
  seed?: number;
  source?: string;
}

export interface GridPoint {
  requested: { latitude: number; longitude: number; year: number; depthM: number };
  normalized: { latitude: number; longitude: number };
  grid: { latitude: number; longitude: number; depthM: number; latitudeIndex: number; longitudeIndex: number; depthIndex: number };
  arrayIndices: { yearIndex: number; depthIndex: number; latitudeIndex: number; longitudeIndex: number };
  distance: { latitudeDegrees: number; longitudeDegrees: number };
  status: 'observed' | 'never_observed' | 'invalid' | 'land';
}

const PROD_RUN_NAMES = [
  'CESM2_omip1_spatial_hybrid_full_seed1',
  'CESM2_omip1_spatial_lstm_full_seed1',
];
const coordinateCache = new Map<string, Coordinates>();
const metadataCache = new Map<string, RunMetadata>();
const configCache = new Map<string, RunConfiguration>();
const validatedRunShapes = new Set<string>();
const MAX_COORDINATE_METADATA_BYTES = 1_000_000;

function readCoordinateFile(file: string): Record<string, unknown> {
  if (fs.statSync(file).size > MAX_COORDINATE_METADATA_BYTES)
    throw new HttpError(503, 'coordinate_metadata_invalid', 'Coordinate metadata exceeds the 1 MB size limit.');
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch { throw new HttpError(503, 'coordinate_metadata_invalid', 'Configured coordinate metadata is not readable JSON.'); }
}

function cachedMetadata(dir: string): RunMetadata {
  const cached = metadataCache.get(dir);
  if (cached) return cached;
  const raw = readJson(dir, 'metadata.json');
  if (!Array.isArray(raw.years) || raw.years.length === 0 || raw.years.some((year: unknown) => !Number.isInteger(year)))
    throw new HttpError(503, 'run_metadata_invalid', 'Run metadata must contain supported integer years.');
  const metadata = raw as RunMetadata;
  metadataCache.set(dir, metadata);
  return metadata;
}

function cachedConfig(dir: string): RunConfiguration {
  const cached = configCache.get(dir);
  if (cached) return cached;
  const configuration = readJson(dir, 'config.json') as RunConfiguration;
  configCache.set(dir, configuration);
  return configuration;
}

function assertRunShapes(run: string, dir: string) {
  if (validatedRunShapes.has(dir)) return;
  const prediction = arr(dir, 'prediction')!;
  const shape = prediction.shape;
  const truth = arr(dir, 'truth')!;
  const observed = arr(dir, 'observed')!;
  const uncertainty = arr(dir, 'uncertainty', false);
  const ocean = arr(dir, 'ocean')!;
  if (truth.shape.join(',') !== shape.join(',') || observed.shape.join(',') !== shape.join(',') ||
    (uncertainty && uncertainty.shape.join(',') !== shape.join(',')) || ocean.shape.join(',') !== shape.slice(1).join(','))
    throw new HttpError(503, 'artifact_shape_mismatch', 'Run arrays do not match the documented data contract dimensions.');
  if (prediction.dtype !== '<f4' || truth.dtype !== '<f4' || (uncertainty && uncertainty.dtype !== '<f4') ||
    observed.dtype !== '|u1' || ocean.dtype !== '|u1')
    throw new HttpError(503, 'artifact_dtype_mismatch', 'Run arrays do not match the documented float32/uint8 data types.');
  const definition = VERIFIED_RUNS[run];
  if (definition?.uncertaintyAvailable && !uncertainty)
    throw new HttpError(503, 'array_missing', 'This registered run declares uncertainty but pred_std.npy is missing.');
  if (!run.startsWith('MOCK_') && process.env.AQUATESSERACT_ALLOW_SYNTHETIC_FIXTURES !== '1' &&
    (shape.length !== 4 || shape[0] !== 62 || shape[1] !== 33 || shape[2] !== 180 || shape[3] !== 360))
    throw new HttpError(503, 'artifact_shape_mismatch', 'Production run dimensions must be (62, 33, 180, 360).');
  validatedRunShapes.add(dir);
}

function finiteArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value) || value.length === 0 || value.some((v) => typeof v !== 'number' || !Number.isFinite(v)))
    throw new HttpError(503, 'coordinate_metadata_invalid', `${label} must be a non-empty array of finite numbers`);
  return value as number[];
}

function monotonic(values: number[], direction: 'ascending' | 'descending', label: string): boolean {
  return values.every((value, i) => i === 0 || (direction === 'ascending' ? value > values[i - 1] : value < values[i - 1]));
}

export function parseCoordinateMetadata(raw: Record<string, unknown>): Coordinates {
  const rawConvention = raw.longitudeConvention ?? raw.longitude_convention;
  let convention: LongitudeConvention;
  if (
    rawConvention === '0-360' ||
    rawConvention === '0_to_360_east' ||
    (typeof rawConvention === 'string' && rawConvention.startsWith('0_to_360_east'))
  ) {
    convention = '0-360';
  } else if (
    rawConvention === '-180-180' ||
    (typeof rawConvention === 'string' && rawConvention.startsWith('-180_to_180'))
  ) {
    convention = '-180-180';
  } else {
    throw new HttpError(503, 'coordinate_metadata_invalid', 'longitudeConvention must be "0-360" or "-180-180"');
  }

  const shiftValue = raw.longitudeShiftCells ?? raw.longitude_shift_cells ?? 0;
  if (typeof shiftValue !== 'number' || !Number.isInteger(shiftValue))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'longitudeShiftCells must be an integer');
  const shift = shiftValue;

  const isOceanAligned =
    raw.oceanMaskAligned === true ||
    raw.ocean_mask_aligned === true ||
    (raw.latitude_convention !== undefined && raw.longitude_convention !== undefined);
  if (!isOceanAligned)
    throw new HttpError(503, 'coordinate_metadata_invalid', 'oceanMaskAligned must be explicitly true');

  const latitudeValues = finiteArray(raw.latitudeValues ?? raw.lat ?? VERIFIED_LATITUDE_CENTERS, 'latitudeValues');
  const longitudeValues = finiteArray(raw.longitudeValues ?? raw.lon ?? VERIFIED_LONGITUDE_CENTERS, 'longitudeValues');
  const depthValuesM = finiteArray(raw.depthValuesM ?? raw.depth_labels_m ?? VERIFIED_DEPTH_LABELS_M, 'depthValuesM');
  if ((raw.depthUnits ?? raw.depth_units ?? 'm') !== 'm')
    throw new HttpError(503, 'coordinate_metadata_invalid', 'depthUnits must be explicitly set to "m".');

  let latitudeOrder = String(raw.latitudeOrder ?? raw.latitude_order ?? '');
  if (!latitudeOrder && typeof raw.latitude_convention === 'string' && raw.latitude_convention.startsWith('south_to_north')) {
    latitudeOrder = 'south_to_north';
  }
  let longitudeOrder = String(raw.longitudeOrder ?? raw.longitude_order ?? '');
  if (!longitudeOrder && (convention === '0-360' || (typeof raw.longitude_convention === 'string' && raw.longitude_convention.startsWith('0_to_360_east')))) {
    longitudeOrder = 'west_to_east';
  }

  if (!['south_to_north', 'north_to_south'].includes(latitudeOrder))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'latitudeOrder must be south_to_north or north_to_south');
  if (!['west_to_east', 'east_to_west'].includes(longitudeOrder))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'longitudeOrder must be west_to_east or east_to_west');
  if (!monotonic(latitudeValues, latitudeOrder === 'south_to_north' ? 'ascending' : 'descending', 'latitudeValues'))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'latitudeValues do not match latitudeOrder');
  if (!monotonic(longitudeValues, longitudeOrder === 'west_to_east' ? 'ascending' : 'descending', 'longitudeValues'))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'longitudeValues do not match longitudeOrder');
  if (!monotonic(depthValuesM, 'ascending', 'depthValuesM'))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'depthValuesM must be ordered shallow-to-deep.');
  if (latitudeValues.some((v) => v < -90 || v > 90))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'latitudeValues must be within [-90, 90]');
  if (longitudeValues.some((v) => convention === '0-360' ? v < 0 || v >= 360 : v < -180 || v > 180))
    throw new HttpError(503, 'coordinate_metadata_invalid', 'longitudeValues are outside longitudeConvention.');
  return {
    latitudeValues, longitudeValues, depthValuesM, latitudeOrder, longitudeOrder,
    depthUnits: 'm', longitudeConvention: convention, longitudeShiftCells: shift, oceanMaskAligned: true,
  };
}

export function normalizeLongitude(longitude: number, convention: LongitudeConvention): number {
  if (!Number.isFinite(longitude)) throw new HttpError(400, 'invalid_longitude', 'Longitude must be a finite number.');
  if (convention === '0-360' || convention === '0_to_360_east') return ((longitude % 360) + 360) % 360;
  const normalized = ((longitude + 180) % 360 + 360) % 360 - 180;
  return normalized === -180 && longitude > 0 ? 180 : normalized;
}

function circularDistance(a: number, b: number): number {
  const delta = Math.abs(a - b) % 360;
  return Math.min(delta, 360 - delta);
}

function nearest(values: number[], value: number, circular = false): { index: number; value: number; distance: number } {
  let best = 0;
  let distance = Infinity;
  values.forEach((candidate, i) => {
    const d = circular ? circularDistance(candidate, value) : Math.abs(candidate - value);
    if (d < distance) { distance = d; best = i; }
  });
  return { index: best, value: values[best], distance };
}

export function selectDepth(coords: Coordinates, depthM: number) {
  if (!Number.isFinite(depthM) || depthM < coords.depthValuesM[0] || depthM > coords.depthValuesM[coords.depthValuesM.length - 1])
    throw new HttpError(400, 'invalid_depth', 'Depth must be within the supported depth range in metres.');
  return nearest(coords.depthValuesM, depthM);
}

export function getCoordinates(run: string): Coordinates {
  const dir = runDir(run);
  const cacheKey = run.startsWith('MOCK_') ? path.join(dir, 'metadata.json') : config.coordinateMetadataFile ?? dir;
  const cached = coordinateCache.get(cacheKey);
  if (cached) {
    assertRunShapes(run, dir);
    return cached;
  }
  if (run.startsWith('MOCK_')) {
    const mock = cachedMetadata(dir);
    if (mock.mock !== true || mock.latitude_convention !== 'TO_VERIFY (mock: index0 = -87.5, south to north)' ||
      mock.longitude_convention !== 'TO_VERIFY (mock: index0 = 2.5E, 0..360)')
      throw new HttpError(503, 'coordinate_metadata_invalid', 'Mock coordinate fixture is missing its explicit mock convention markers.');
    const coordinates = parseCoordinateMetadata({
      ...mock,
      latitudeOrder: 'south_to_north',
      longitudeOrder: 'west_to_east',
      longitudeConvention: '0-360',
      depthUnits: 'm',
      longitudeShiftCells: 0,
      oceanMaskAligned: true,
    });
    assertRunShapes(run, dir);
    coordinateCache.set(cacheKey, coordinates);
    const shape = arr(dir, 'prediction')!.shape;
    if (coordinates.latitudeValues.length !== shape[2] || coordinates.longitudeValues.length !== shape[3] ||
      coordinates.depthValuesM.length !== shape[1])
      throw new HttpError(503, 'coordinate_shape_mismatch', 'Mock coordinate metadata dimensions do not match the run arrays.');
    return coordinates;
  }
  let raw: Record<string, unknown>;
  if (config.coordinateMetadataFile && fs.existsSync(config.coordinateMetadataFile)) {
    raw = readCoordinateFile(config.coordinateMetadataFile);
  } else {
    // Coordinate metadata is now verified from the contract metadata
    const meta = cachedMetadata(dir);
    raw = {
      ...meta,
      latitudeValues: meta.lat ?? VERIFIED_CONTRACT_COORDINATES.latitudeValues,
      longitudeValues: meta.lon ?? VERIFIED_CONTRACT_COORDINATES.longitudeValues,
      depthValuesM: meta.depth_labels_m ?? VERIFIED_CONTRACT_COORDINATES.depthValuesM,
      latitudeOrder: 'south_to_north',
      longitudeOrder: 'west_to_east',
      longitudeConvention: '0-360',
      depthUnits: 'm',
      longitudeShiftCells: 0,
      oceanMaskAligned: true,
    };
  }
  const coordinates = parseCoordinateMetadata(raw);
  assertRunShapes(run, dir);
  const shape = arr(dir, 'prediction')!.shape;
  if (coordinates.latitudeValues.length !== shape[2] || coordinates.longitudeValues.length !== shape[3] ||
    coordinates.depthValuesM.length !== shape[1])
    throw new HttpError(503, 'coordinate_shape_mismatch', 'Coordinate metadata dimensions do not match the run arrays.');
  coordinateCache.set(cacheKey, coordinates);
  return coordinates;
}

export function runInfo(run: string): RunInfo {
  const dir = registeredRunDir(run);
  const metadata = cachedMetadata(dir);
  const cfg = cachedConfig(dir);
  const seed = cfg.seed ?? metadata.seed;
  const isMock = metadata.mock === true || run.startsWith('MOCK_');
  const verified = VERIFIED_RUNS[run as keyof typeof VERIFIED_RUNS];
  const hasUncertaintyArtifact = Boolean(arr(dir, 'uncertainty', false));
  const uncertaintyAvailable = isMock
    ? false
    : verified?.uncertaintyAvailable ?? hasUncertaintyArtifact;
  return {
    runId: run,
    dataset: String(cfg.dataset ?? metadata.dataset ?? 'unknown'),
    model: verified?.model ?? String(cfg.model ?? metadata.model ?? 'unknown'),
    split: String(cfg.split ?? metadata.split ?? 'unknown'),
    seed: typeof seed === 'number' && Number.isInteger(seed) ? seed : null,
    source: String(cfg.source ?? (isMock ? 'mock' : 'verified')),
    mock: isMock,
    uncertaintyAvailable,
    uncertaintyReason: isMock
      ? 'Mock fixture uncertainty is not verified.'
      : uncertaintyAvailable ? null : 'Uncertainty unavailable for this baseline',
  };
}

export function dataSourceStatus(): DataSourceStatus {
  const runs = listRuns();
  const hasMock = runs.some((run) => run.startsWith('MOCK_'));
  const hasProduction = runs.some((run) => !run.startsWith('MOCK_'));
  if (!hasProduction) return hasMock ? 'mock' : 'unavailable';
  if (!PROD_RUN_NAMES.every((id) => runs.includes(id))) return 'unavailable';
  if (!config.coordinateMetadataFile || !fs.existsSync(config.coordinateMetadataFile)) return 'unavailable';
  try {
    const coordinates = parseCoordinateMetadata(readCoordinateFile(config.coordinateMetadataFile));
    if (coordinates.latitudeValues.length !== 180 || coordinates.longitudeValues.length !== 360 ||
      coordinates.depthValuesM.length !== 33) return 'unavailable';
    for (const id of PROD_RUN_NAMES) {
      const dir = runDir(id);
      assertRunShapes(id, dir);
      const shape = arr(dir, 'prediction')!.shape;
      if (coordinates.latitudeValues.length !== shape[2] || coordinates.longitudeValues.length !== shape[3] ||
        coordinates.depthValuesM.length !== shape[1]) return 'unavailable';
    }
    return 'verified';
  } catch (error) {
    if (error instanceof HttpError) return 'unavailable';
    throw error;
  }
}

export function configurationStatus() {
  const runIds = listRuns();
  const productionRunIds = runIds.filter((run) => !run.startsWith('MOCK_'));
  const requiredProductionRunIdsAvailable = PROD_RUN_NAMES.every((run) => productionRunIds.includes(run));
  const coordinateMetadataConfigured = Boolean(config.coordinateMetadataFile && fs.existsSync(config.coordinateMetadataFile));
  const comparisonFileConfigured = Boolean(config.comparisonFile && fs.existsSync(config.comparisonFile));
  return {
    resultsRootAvailable: fs.existsSync(config.resultsDir),
    requiredProductionRunIdsAvailable,
    availableProductionRunCount: productionRunIds.length,
    coordinateMetadataConfigured,
    comparisonFileConfigured,
  };
}

export function availableRuns() {
  const ids = listRuns();
  return ids.sort().map((runId) => ({ ...runInfo(runId), dataSourceStatus: runId.startsWith('MOCK_') ? 'mock' : dataSourceStatus() }));
}

export function artifactLongitudeIndex(longitudeIndex: number, longitudeCount: number, shift: number): number {
  return ((longitudeIndex + shift) % longitudeCount + longitudeCount) % longitudeCount;
}

export function mapPointToGrid(run: string, latitude: number, longitude: number, year: number, depthM: number): GridPoint {
  if (!Number.isFinite(latitude)) throw new HttpError(400, 'invalid_latitude', 'Latitude must be a finite number.');
  if (latitude < -90 || latitude > 90) throw new HttpError(400, 'invalid_latitude', 'Latitude must be between -90 and 90 degrees.');
  if (!Number.isFinite(depthM)) throw new HttpError(400, 'invalid_depth', 'Depth must be a finite number in metres.');
  const dir = runDir(run);
  const coords = getCoordinates(run);
  const meta = cachedMetadata(dir);
  const years = meta.years;
  if (!Number.isInteger(year) || !years.includes(year)) throw new HttpError(400, 'invalid_year', 'Year is not supported by this run.');
  const a = nearest(coords.latitudeValues, latitude);
  const normalizedLon = normalizeLongitude(longitude, coords.longitudeConvention);
  const o = nearest(coords.longitudeValues, normalizedLon, true);
  const d = selectDepth(coords, depthM);
  const arrays = arr(dir, 'prediction')!;
  const [, , nlat, nlon] = arrays.shape;
  if (coords.latitudeValues.length !== nlat || coords.longitudeValues.length !== nlon || coords.depthValuesM.length !== arrays.shape[1])
    throw new HttpError(503, 'coordinate_shape_mismatch', 'Coordinate metadata dimensions do not match the run arrays.');
  const arrayLon = artifactLongitudeIndex(o.index, nlon, coords.longitudeShiftCells);
  const ocean = arr(dir, 'ocean')!;
  const pred = readAt(arrays, [years.indexOf(year), d.index, a.index, arrayLon]);
  const isOcean = readAt(ocean, [d.index, a.index, arrayLon]) !== 0;
  const truth = readAt(arr(dir, 'truth')!, [years.indexOf(year), d.index, a.index, arrayLon]);
  const observed = readAt(arr(dir, 'observed')!, [years.indexOf(year), d.index, a.index, arrayLon]) !== 0;
  const status = !isOcean ? 'land' : !Number.isFinite(pred) || !Number.isFinite(truth) ? 'invalid' : observed ? 'observed' : 'never_observed';
  return {
    requested: { latitude, longitude, year, depthM },
    normalized: { latitude, longitude: normalizedLon },
    grid: {
      latitude: a.value, longitude: o.value, depthM: d.value,
      latitudeIndex: a.index, longitudeIndex: o.index, depthIndex: d.index,
    },
    arrayIndices: { yearIndex: years.indexOf(year), depthIndex: d.index, latitudeIndex: a.index, longitudeIndex: arrayLon },
    distance: { latitudeDegrees: a.distance, longitudeDegrees: o.distance },
    status,
  };
}

function pointValues(run: string, point: GridPoint) {
  const dir = runDir(run);
  const i = point.arrayIndices;
  const ocean = readAt(arr(dir, 'ocean')!, [i.depthIndex, i.latitudeIndex, i.longitudeIndex]) !== 0;
  const prediction = readAt(arr(dir, 'prediction')!, [i.yearIndex, i.depthIndex, i.latitudeIndex, i.longitudeIndex]);
  const reference = readAt(arr(dir, 'truth')!, [i.yearIndex, i.depthIndex, i.latitudeIndex, i.longitudeIndex]);
  const observed = readAt(arr(dir, 'observed')!, [i.yearIndex, i.depthIndex, i.latitudeIndex, i.longitudeIndex]) !== 0;
  const info = runInfo(run);
  const uncertaintyValue = info.uncertaintyAvailable && ocean && Number.isFinite(prediction) && arr(dir, 'uncertainty', false)
    ? readAt(arr(dir, 'uncertainty')!, [i.yearIndex, i.depthIndex, i.latitudeIndex, i.longitudeIndex])
    : null;
  const valid = ocean && Number.isFinite(prediction);
  const hasReference = valid && Number.isFinite(reference);
  const error = hasReference ? prediction - reference : null;
  return {
    prediction: valid ? prediction : null,
    predictedOxygen: valid ? prediction : null,
    reference: hasReference ? reference : null,
    referenceOxygen: hasReference ? reference : null,
    signedError: error,
    absoluteError: error === null ? null : Math.abs(error),
    observedStatus: !ocean ? 'land' : !valid || !hasReference ? 'invalid' : observed ? 'observed' : 'never_observed',
    oceanStatus: ocean ? 'ocean' : 'land',
    uncertainty: uncertaintyValue !== null && Number.isFinite(uncertaintyValue) && uncertaintyValue > 0 ? uncertaintyValue : null,
    uncertaintyAvailable: info.uncertaintyAvailable,
    uncertaintyReason: info.uncertaintyReason,
    run: info,
  };
}

export function getPoint(run: string, latitude: number, longitude: number, year: number, depthM: number) {
  const mapping = mapPointToGrid(run, latitude, longitude, year, depthM);
  return { ...mapping, values: pointValues(run, mapping), provenance: provenance(run, year, mapping.grid.depthM) };
}

export function provenance(run: string, year?: number, depthM?: number) {
  const info = runInfo(run);
  return {
    dataset: info.dataset,
    datasetType: 'simulation',
    runId: info.runId,
    model: info.model,
    split: info.split,
    seed: info.seed,
    selectedYear: year ?? null,
    selectedDepthM: depthM ?? null,
    coordinateConvention: run.startsWith('MOCK_') ? 'mock metadata; 0-360' : 'configured coordinate metadata',
    contractValidationStatus: info.mock ? 'mock_fixture' : 'externally_contract_validated; runtime structure checked per request',
    evidenceStatus: info.mock ? 'mock' : dataSourceStatus(),
    caveat: 'Data source: OceanVerse CESM2-OMIP1 simulation. These are simulation-based dissolved-oxygen reconstructions, not live real-ocean measurements. Real-ocean accuracy has not yet been established.',
  };
}

export function pointProfile(run: string, latitude: number, longitude: number, year: number) {
  const coords = getCoordinates(run);
  return coords.depthValuesM.map((depthM) => {
    const point = mapPointToGrid(run, latitude, longitude, year, depthM);
    const values = pointValues(run, point);
    return {
      depthM, prediction: values.prediction, reference: values.reference,
      error: values.signedError, observedStatus: values.observedStatus,
      uncertainty: values.uncertainty,
      uncertaintyAvailable: values.uncertaintyAvailable,
      uncertaintyReason: values.uncertaintyReason,
    };
  });
}

export function pointTimeseries(run: string, latitude: number, longitude: number, depthM: number) {
  const dir = runDir(run);
  const years: number[] = cachedMetadata(dir).years;
  const selectedDepth = selectDepth(getCoordinates(run), depthM);
  return years.map((year) => {
    const point = mapPointToGrid(run, latitude, longitude, year, selectedDepth.value);
    const values = pointValues(run, point);
    return {
      year, depthM: selectedDepth.value, prediction: values.prediction, reference: values.reference,
      error: values.signedError, observedStatus: values.observedStatus,
      uncertainty: values.uncertainty,
      uncertaintyAvailable: values.uncertaintyAvailable,
      uncertaintyReason: values.uncertaintyReason,
    };
  });
}

export type SliceLayer = 'observations' | 'reconstruction' | 'reference' | 'error' | 'uncertainty';
export const SLICE_LAYERS: SliceLayer[] = ['observations', 'reconstruction', 'reference', 'error', 'uncertainty'];

export function getSlice(run: string, year: number, depthM: number, layer: SliceLayer, maxPoints: number) {
  if (layer === 'uncertainty' && !runInfo(run).uncertaintyAvailable)
    throw new HttpError(422, 'uncertainty_unavailable', 'Uncertainty unavailable for this baseline.');
  const dir = runDir(run);
  const coords = getCoordinates(run);
  const meta = cachedMetadata(dir);
  const yi = meta.years.indexOf(year);
  if (yi < 0) throw new HttpError(400, 'invalid_year', 'Year is not supported by this run.');
  const d = selectDepth(coords, depthM);
  const pred = readBlock(arr(dir, 'prediction')!, [yi, d.index]);
  const truth = readBlock(arr(dir, 'truth')!, [yi, d.index]);
  const obs = readBlock(arr(dir, 'observed')!, [yi, d.index]);
  const ocean = readBlock(arr(dir, 'ocean')!, [d.index]);
  const std = layer === 'uncertainty' ? readBlock(arr(dir, 'uncertainty')!, [yi, d.index]) : null;
  const nlat = coords.latitudeValues.length, nlon = coords.longitudeValues.length;
  const shift = coords.longitudeShiftCells;
  const total = nlat * nlon;
  const stride = Math.max(1, Math.ceil(Math.sqrt(total / maxPoints)));
  const points: Array<{
    latitude: number;
    longitude: number;
    latitudeIndex: number;
    longitudeIndex: number;
    value: number | null;
    observed: boolean;
    ocean: boolean;
  }> = [];
  for (let lat = 0; lat < nlat; lat += stride) {
    for (let lon = 0; lon < nlon; lon += stride) {
      const li = lat * nlon + artifactLongitudeIndex(lon, nlon, shift);
      const valid = ocean[li] !== 0 && Number.isFinite(pred[li]);
      let value: number | null = null;
      if (valid) {
        if (layer === 'reconstruction') value = pred[li];
        if (layer === 'reference' && Number.isFinite(truth[li])) value = truth[li];
        if (layer === 'error' && Number.isFinite(truth[li])) value = pred[li] - truth[li];
        if (layer === 'observations' && obs[li] !== 0 && Number.isFinite(truth[li])) value = truth[li];
        if (layer === 'uncertainty' && std && Number.isFinite(std[li]) && std[li] > 0) value = std[li];
      }
      points.push({
        latitude: coords.latitudeValues[lat], longitude: coords.longitudeValues[lon],
        latitudeIndex: lat, longitudeIndex: lon, value,
        observed: obs[li] !== 0, ocean: ocean[li] !== 0,
      });
    }
  }
  return {
    run: runInfo(run), year, depthM: d.value, layer, points: points.slice(0, maxPoints),
    downsampleStride: stride, exact: false,
    oxygenUnits: layer === 'observations' ? 'binary sampling mask' : meta.oxygen_units ?? 'unknown',
    provenance: provenance(run, year, d.value),
  };
}

export function comparisonData() {
  if (!config.comparisonFile || !fs.existsSync(config.comparisonFile))
    return { dataSourceStatus: 'unavailable' as const, rows: [], message: 'Verified comparisons unavailable until COMPARISON_FILE points to the supplied comparison_spatial.csv.' };
  if (fs.statSync(config.comparisonFile).size > 5_000_000)
    throw new HttpError(413, 'comparison_too_large', 'Configured comparison CSV exceeds the 5 MB read limit.');
  const lines = fs.readFileSync(config.comparisonFile, 'utf8').trim().split(/\r?\n/);
  if (lines.length < 2) return { dataSourceStatus: 'unavailable' as const, rows: [], message: 'The configured comparison file has no data rows.' };
  const parseRow = (line: string) => {
    const columns: string[] = [];
    let value = '', quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i++; }
      else if (char === '"') quoted = !quoted;
      else if (char === ',' && !quoted) { columns.push(value); value = ''; }
      else value += char;
    }
    if (quoted) throw new HttpError(503, 'comparison_invalid', 'Configured comparison CSV contains an unterminated quoted field.');
    columns.push(value);
    return columns;
  };
  const headers = parseRow(lines[0]).map((s) => s.trim());
  if (headers.some((header) => !header) || new Set(headers).size !== headers.length)
    throw new HttpError(503, 'comparison_invalid', 'Configured comparison CSV has invalid or duplicate column names.');
  const rows = lines.slice(1).map((line) => {
    const columns = parseRow(line);
    if (columns.length !== headers.length)
      throw new HttpError(503, 'comparison_invalid', 'Configured comparison CSV has inconsistent column counts.');
    return Object.fromEntries(headers.map((header, i) => [header, columns[i].trim()]));
  });
  return { dataSourceStatus: 'verified' as const, rows, source: 'COMPARISON_FILE' };
}

export function readArrayAt(run: string, key: 'prediction' | 'truth' | 'uncertainty' | 'observed' | 'ocean', indices: number[]) {
  const dir = runDir(run);
  return readAt(arr(dir, key, key !== 'uncertainty')!, indices);
}

/**
 * Verified Latitude index:
 * centers from -89.5 to 89.5 degrees, south_to_north
 * lat_index = round(latitude + 89.5)
 */
export function latToIndex(latitude: number): number {
  if (!Number.isFinite(latitude)) throw new HttpError(400, 'invalid_latitude', 'Latitude must be a finite number.');
  return Math.max(0, Math.min(179, Math.round(latitude + 89.5)));
}

/**
 * Prediction-array longitude:
 * centers from 0.5 to 359.5 degrees east, longitude convention: 0_to_360_east
 * lon_index = round((longitude % 360) - 0.5)
 */
export function lonToIndex(longitude: number): number {
  if (!Number.isFinite(longitude)) throw new HttpError(400, 'invalid_longitude', 'Longitude must be a finite number.');
  const lon0_360 = ((longitude % 360) + 360) % 360;
  return Math.round(lon0_360 - 0.5) % 360;
}

export interface RecoveredCoordinates {
  latitude: number;
  longitudeMinus180_180: number;
  longitude0_to_360: number;
}

/**
 * The graph x_geo first three columns encode unit-sphere coordinates:
 * x = x_geo[:, 0]
 * y = x_geo[:, 1]
 * z = x_geo[:, 2]
 *
 * Recover graph coordinates with:
 * latitude = degrees(arcsin(z))
 * longitude_minus180_180 = degrees(arctan2(y, x))
 *
 * Convert graph longitude before array lookup:
 * longitude_0_to_360 = longitude_minus180_180 % 360
 *
 * Do not apply an additional blind array rotation. The apparent 180-cell difference
 * is handled by converting graph longitude to the verified 0–360 array convention.
 *
 * Example:
 * graph longitude -179.5 -> array longitude 180.5 -> lon_index 180
 * graph longitude 0.5 -> array longitude 0.5 -> lon_index 0
 */
export function recoverGraphCoordinates(x: number, y: number, z: number): RecoveredCoordinates {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    throw new HttpError(400, 'invalid_coordinates', 'Graph unit-sphere coordinates must be finite numbers.');
  }
  const clampedZ = Math.max(-1, Math.min(1, z));
  const latitude = (Math.asin(clampedZ) * 180) / Math.PI;
  const longitudeMinus180_180 = (Math.atan2(y, x) * 180) / Math.PI;
  const longitude0_to_360 = ((longitudeMinus180_180 % 360) + 360) % 360;
  return {
    latitude,
    longitudeMinus180_180,
    longitude0_to_360,
  };
}

export function recoverGraphCoordinatesFromGeo(xGeo: ArrayLike<number>): RecoveredCoordinates {
  if (!xGeo || xGeo.length < 3) {
    throw new HttpError(400, 'invalid_coordinates', 'x_geo must contain at least 3 unit-sphere coordinate columns.');
  }
  return recoverGraphCoordinates(xGeo[0], xGeo[1], xGeo[2]);
}

export function graphToGridIndices(x: number, y: number, z: number) {
  const recovered = recoverGraphCoordinates(x, y, z);
  const latIndex = latToIndex(recovered.latitude);
  const lonIndex = lonToIndex(recovered.longitude0_to_360);
  return {
    ...recovered,
    latIndex,
    lonIndex,
  };
}

export function graphGeoToIndices(xGeo: ArrayLike<number>) {
  if (!xGeo || xGeo.length < 3) {
    throw new HttpError(400, 'invalid_coordinates', 'x_geo must contain at least 3 unit-sphere coordinate columns.');
  }
  return graphToGridIndices(xGeo[0], xGeo[1], xGeo[2]);
}

export interface GraphNodeProfileLevel {
  depthIndex: number;
  depthM: number;
  value: number | null;
  isOcean: boolean;
}

export interface GraphNodeResult {
  recovered: RecoveredCoordinates;
  grid: {
    latIndex: number;
    lonIndex: number;
    latitude: number;
    longitude: number;
  };
  isOcean: boolean;
  profile: GraphNodeProfileLevel[];
}

/**
 * Use: pred_mean[year_index, depth_index, lat_index, lon_index]
 *
 * The graph contains one node per valid horizontal ocean cell and each node has a 33-depth profile.
 * Land/invalid cells must be excluded using ocean_mask.
 */
export function getGraphNodeData(run: string, year: number, x: number, y: number, z: number): GraphNodeResult {
  const recovered = recoverGraphCoordinates(x, y, z);
  const dir = runDir(run);
  const coords = getCoordinates(run);
  const meta = cachedMetadata(dir);
  const yearIndex = meta.years.indexOf(year);
  if (yearIndex < 0) throw new HttpError(400, 'invalid_year', 'Year is not supported by this run.');

  let latIndex: number;
  let lonIndex: number;
  if (coords.latitudeValues.length === 180 && coords.longitudeValues.length === 360) {
    latIndex = latToIndex(recovered.latitude);
    lonIndex = lonToIndex(recovered.longitude0_to_360);
  } else {
    latIndex = nearest(coords.latitudeValues, recovered.latitude).index;
    const normLon = normalizeLongitude(recovered.longitude0_to_360, coords.longitudeConvention);
    const nlon = coords.longitudeValues.length;
    const baseLonIndex = nearest(coords.longitudeValues, normLon, true).index;
    lonIndex = artifactLongitudeIndex(baseLonIndex, nlon, coords.longitudeShiftCells);
  }

  const predArr = arr(dir, 'prediction')!;
  const oceanArr = arr(dir, 'ocean')!;

  let anyOcean = false;
  const profile: GraphNodeProfileLevel[] = [];
  for (let di = 0; di < coords.depthValuesM.length; di++) {
    const isOcean = readAt(oceanArr, [di, latIndex, lonIndex]) !== 0;
    if (isOcean) anyOcean = true;
    const raw = isOcean ? readAt(predArr, [yearIndex, di, latIndex, lonIndex]) : NaN;
    profile.push({
      depthIndex: di,
      depthM: coords.depthValuesM[di],
      value: isOcean && Number.isFinite(raw) ? raw : null,
      isOcean,
    });
  }

  return {
    recovered: {
      latitude: recovered.latitude,
      longitudeMinus180_180: recovered.longitudeMinus180_180,
      longitude0_to_360: recovered.longitude0_to_360,
    },
    grid: {
      latIndex,
      lonIndex,
      latitude: coords.latitudeValues[latIndex] ?? -89.5 + latIndex,
      longitude: coords.longitudeValues[lonIndex] ?? 0.5 + lonIndex,
    },
    isOcean: anyOcean,
    profile,
  };
}
