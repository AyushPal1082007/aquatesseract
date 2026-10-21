const BASE = '/api';

export interface MapData {
  layer: string; year: number; depth_m: number; shape: [number, number];
  stride: number;
  min: number | null; max: number | null; values: (number | null)[]; oxygen_units: string;
  latitude_values: number[] | null; longitude_values: number[] | null;
}
export interface Metadata {
  years: number[]; depth_labels_m: number[]; oxygen_units: string; caveats: string[];
  dataset: string; split: string; model: string; layers: string[]; shape: number[];
  uncertainty_available: boolean; latitude_convention?: string; longitude_convention?: string;
  lat?: number[]; lon?: number[]; mock?: boolean; contract_validated?: boolean; coordinates_available: boolean;
  config?: { model?: string; dataset?: string; split?: string; feature_set?: string; seed?: number; source?: string; contract_validated?: boolean };
}
export interface CompRow {
  source: 'published' | 'reproduced' | 'hybrid' | 'mock'; dataset: string; split: string; model: string;
  feature_set: string | null; rmse: number | null; r2: number | null; scope: string;
  valid_cells?: number | null;
}

export interface ProfileData {
  year: number; lat: number; lon: number; depth_m: number[];
  truth: (number | null)[]; prediction: (number | null)[]; std: (number | null)[];
  observed: (boolean | null)[]; simulation_caveat: boolean;
}
export interface TimeSeriesData {
  years: number[]; depth_idx: number; depth_m: number; lat: number; lon: number;
  truth: (number | null)[]; prediction: (number | null)[]; lower90: (number | null)[];
  upper90: (number | null)[]; observed: boolean[]; simulation_caveat: boolean;
}
export interface LocateData {
  requested: { latitude: number; longitude: number };
  nearest: { latitude: number; longitude: number; lat_idx: number; lon_idx: number };
  depth_idx: number;
}

async function get<T>(path: string, params: Record<string, string | number> = {}, signal?: AbortSignal): Promise<T> {
  const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  const res = await fetch(`${BASE}/${path}${qs ? `?${qs}` : ''}`, { signal });
  const body = await res.json();
  if (!res.ok) throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
  return body as T;
}

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
};
