const BASE = '/api';

export interface MapData {
  layer: string; year: number; depth_m: number; shape: [number, number];
  min: number | null; max: number | null; values: (number | null)[]; oxygen_units: string;
}
export interface Metadata {
  years: number[]; depth_labels_m: number[]; oxygen_units: string; caveats: string[];
  dataset: string; split: string; model: string; layers: string[]; shape: number[];
}
export interface CompRow {
  source: 'published' | 'reproduced' | 'hybrid' | 'mock'; dataset: string; split: string; model: string;
  feature_set: string | null; rmse: number | null; r2: number | null; scope: string;
}

async function get<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  const res = await fetch(`${BASE}/${path}${qs ? `?${qs}` : ''}`);
  const body = await res.json();
  if (!res.ok) throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
  return body as T;
}

export const api = {
  runs: () => get<{ runs: string[] }>('runs'),
  metadata: (run: string) => get<Metadata>('metadata', { run }),
  map: (run: string, layer: string, year: number, depth: number, stride = 1) =>
    get<MapData>('map', { run, layer, year, depth, stride }),
  comparison: () => get<{ rows: CompRow[] }>('comparison'),
};
