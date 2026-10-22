const BASE = '/api';

export interface MapData {
  layer: string;
  year: number;
  depth_m: number;
  shape: [number, number];
  stride: number;
  min: number | null;
  max: number | null;
  values: (number | null)[];
  oxygen_units: string;
  latitude_values: number[] | null;
  longitude_values: number[] | null;
}

export interface Metadata {
  years: number[];
  depth_labels_m: number[];
  oxygen_units: string;
  caveats: string[];
  dataset: string;
  split: string;
  model: string;
  layers: string[];
  shape: number[];
  uncertainty_available: boolean;
  latitude_convention?: string;
  longitude_convention?: string;
  lat?: number[];
  lon?: number[];
  mock?: boolean;
  contract_validated?: boolean;
  coordinates_available: boolean;
  config?: { model?: string; dataset?: string; split?: string; feature_set?: string; seed?: number; source?: string; contract_validated?: boolean };
}

export interface CompRow {
  source: 'published' | 'reproduced' | 'hybrid' | 'mock';
  dataset: string;
  split: string;
  model: string;
  feature_set: string | null;
  rmse: number | null;
  r2: number | null;
  scope: string;
  valid_cells?: number | null;
}

export interface ProfileData {
  year: number;
  lat: number;
  lon: number;
  depth_m: number[];
  truth: (number | null)[];
  prediction: (number | null)[];
  std: (number | null)[];
  observed: (boolean | null)[];
  simulation_caveat: boolean;
}

export interface TimeSeriesData {
  years: number[];
  depth_idx: number;
  depth_m: number;
  lat: number;
  lon: number;
  truth: (number | null)[];
  prediction: (number | null)[];
  lower90: (number | null)[];
  upper90: (number | null)[];
  observed: boolean[];
  simulation_caveat: boolean;
}

export interface LocateData {
  requested: { latitude: number; longitude: number };
  nearest: { latitude: number; longitude: number; lat_idx: number; lon_idx: number };
  depth_idx: number;
}

export interface MLRun {
  runId: string;
  dataset: string;
  model: string;
  split: string;
  seed: number | null;
  source: string;
  mock: boolean;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
  dataSourceStatus: 'mock' | 'unavailable' | 'verified';
}

export interface MLCoordinates {
  latitudeValues: number[];
  longitudeValues: number[];
  depthValuesM: number[];
  depthUnits: 'm';
  latitudeOrder: string;
  longitudeOrder: string;
  longitudeConvention: string;
  longitudeShiftCells: number;
  oceanMaskAligned: boolean;
}

export interface MLMetadata extends MLRun {
  dataSourceStatus: 'mock' | 'unavailable' | 'verified';
  dimensions: { year: number; depth: number; latitude: number; longitude: number };
  years: number[];
  coordinates: MLCoordinates;
  arrayOrder: string[];
  oxygenUnits: string | null;
  availableLayers: ('observations' | 'reconstruction' | 'reference' | 'error' | 'uncertainty')[];
  contractValidationStatus: string;
  provenance: Record<string, unknown>;
}

export interface MLSlicePoint {
  latitude: number;
  longitude: number;
  latitudeIndex: number;
  longitudeIndex: number;
  value: number | null;
  observed: boolean;
  ocean: boolean;
}

export interface MLSlice {
  run: MLRun;
  year: number;
  depthM: number;
  layer: 'observations' | 'reconstruction' | 'reference' | 'error' | 'uncertainty';
  points: MLSlicePoint[];
  downsampleStride: number;
  exact: boolean;
  oxygenUnits: string;
  provenance: Record<string, unknown>;
}

export interface MLPointValues {
  prediction: number | null;
  predictedOxygen: number | null;
  reference: number | null;
  referenceOxygen: number | null;
  signedError: number | null;
  absoluteError: number | null;
  observedStatus: 'observed' | 'never_observed' | 'invalid' | 'land';
  oceanStatus: 'ocean' | 'land';
  uncertainty: number | null;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
  run: MLRun;
}

export interface MLPoint {
  requested: { latitude: number; longitude: number; year: number; depthM: number };
  normalized: { latitude: number; longitude: number };
  grid: { latitude: number; longitude: number; depthM: number; latitudeIndex: number; longitudeIndex: number; depthIndex: number };
  arrayIndices: { yearIndex: number; depthIndex: number; latitudeIndex: number; longitudeIndex: number };
  distance: { latitudeDegrees: number; longitudeDegrees: number };
  status: 'observed' | 'never_observed' | 'invalid' | 'land';
  values: MLPointValues;
  provenance: Record<string, unknown>;
}

export interface MLProfilePoint {
  depthM: number;
  prediction: number | null;
  reference: number | null;
  error: number | null;
  observedStatus: 'observed' | 'never_observed' | 'invalid' | 'land';
  uncertainty: number | null;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
}

export interface MLProfile {
  run: MLRun;
  year: number;
  points: MLProfilePoint[];
  provenance: Record<string, unknown>;
}

export interface MLTimeseriesPoint {
  year: number;
  depthM: number;
  prediction: number | null;
  reference: number | null;
  error: number | null;
  observedStatus: 'observed' | 'never_observed' | 'invalid' | 'land';
  uncertainty: number | null;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
}

export interface MLTimeseries {
  run: MLRun;
  requestedDepthM: number;
  depthM: number;
  points: MLTimeseriesPoint[];
  provenance: Record<string, unknown>;
}

export interface MLMetrics {
  all?: { rmse?: number; mae?: number; r2?: number; n?: number };
  observed?: { rmse?: number; mae?: number; r2?: number; n?: number };
  never_observed?: { rmse?: number; mae?: number; r2?: number; n?: number };
  uncertainty?: { coverage90?: number; mean_width90?: number; coverage90_observed?: number; coverage90_never_observed?: number } | null;
  uncertaintyAvailable: boolean;
  uncertaintyReason: string | null;
  provenance: Record<string, unknown>;
}

async function get<T>(path: string, params: Record<string, string | number> = {}, signal?: AbortSignal): Promise<T> {
  const query = new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)])).toString();
  const response = await fetch(`${BASE}/${path}${query ? `?${query}` : ''}`, { signal });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
  return body as T;
}

export const mlApi = {
  runs: (signal?: AbortSignal) => get<{ dataSourceStatus: string; configuration: Record<string, unknown>; available: MLRun[]; registered: Array<Record<string, unknown>> }>('ml/runs', {}, signal),
  metadata: (runId: string, signal?: AbortSignal) => get<MLMetadata>('ml/metadata', { runId }, signal),
  slice: (runId: string, year: number, depth: number, layer: MLSlice['layer'], maxPoints = 10000, signal?: AbortSignal) =>
    get<MLSlice>('ml/slice', { runId, year, depth, layer, maxPoints }, signal),
  point: (runId: string, latitude: number, longitude: number, year: number, depth: number, signal?: AbortSignal) =>
    get<MLPoint>('ml/point', { runId, latitude, longitude, year, depth }, signal),
  profile: (runId: string, latitude: number, longitude: number, year: number, signal?: AbortSignal) =>
    get<MLProfile>('ml/point/profile', { runId, latitude, longitude, year }, signal),
  timeseries: (runId: string, latitude: number, longitude: number, depth: number, signal?: AbortSignal) =>
    get<MLTimeseries>('ml/point/timeseries', { runId, latitude, longitude, depth }, signal),
  metrics: async (runId: string, signal?: AbortSignal): Promise<MLMetrics> => {
    try {
      return await get<MLMetrics>('ml/metrics', { runId }, signal);
    } catch (error) {
      // Keep the dashboard usable while an older server container is being
      // replaced: the legacy read-only endpoint exposes the same artifact
      // metrics under `run` with snake_case uncertainty fields.
      const legacy = await get<Record<string, any>>('metrics', { run: runId }, signal);
      return {
        ...legacy,
        uncertaintyAvailable: Boolean(legacy.uncertainty_available),
        uncertaintyReason: legacy.uncertainty_reason ?? null,
        provenance: legacy.provenance ?? {},
      } as MLMetrics;
    }
  },
};

export const api = {
  runs: () => get<{ runs: string[] }>('runs'),
  metadata: (run: string) => get<Metadata>('metadata', { run }),
  map: (run: string, layer: string, year: number, depth: number, stride = 1, signal?: AbortSignal) =>
    get<MapData>('map', { run, layer, year, depth, stride }, signal),
  comparison: () => get<{ rows: CompRow[] }>('comparison'),
  profile: (run: string, year: number, lat: number, lon: number, signal?: AbortSignal) =>
    get<ProfileData>('profile', { run, year, lat, lon }, signal),
  timeseries: (run: string, depth: number, lat: number, lon: number, signal?: AbortSignal) =>
    get<TimeSeriesData>('timeseries', { run, depth, lat, lon }, signal),
  locate: (run: string, latitude: number, longitude: number, depth: number, nearest = true) =>
    get<LocateData>('locate', { run, latitude, longitude, depth, nearest: nearest ? 1 : 0 }),
  ml: mlApi,
};
