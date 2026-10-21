/* ================================================================
   GLOBE DASHBOARD — MOCK DATA ADAPTER (demo mode)
   ----------------------------------------------------------------
   Small, deterministic, clearly-labeled fixtures. NO real
   OceanVerse CESM2-OMIP1 artifacts are bundled here. When the
   backend contract is connected, replace the functions below with
   real `api` calls — the shapes in types.ts already match.
   ================================================================ */

import {
  CellValue, DEPTHS_M, GlobeFrame, LayerId, ModelKind, PointProfile,
  RunSpec, YEARS,
} from './types.js';

/* ---------- deterministic PRNG (mulberry32) ---------- */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- runs (the four project model types) ---------- */
export const RUNS: RunSpec[] = [
  { id: 'ov_hybrid_s42',    label: 'Hybrid LSTM+Transformer',                 model: 'hybrid',          seed: 42,  hasUncertainty: true },
  { id: 'ov_hybriddo_s7',   label: 'Hybrid LSTM+Transformer · dropout',       model: 'hybrid_dropout',  seed: 7,   hasUncertainty: true },
  { id: 'ov_lstm_s42',      label: 'LSTM baseline',                           model: 'lstm',            seed: 42,  hasUncertainty: false },
  { id: 'ov_transform_s42', label: 'Transformer baseline',                    model: 'transformer',     seed: 42,  hasUncertainty: false },
];

export const MODEL_LABEL: Record<ModelKind, string> = {
  hybrid: 'Hybrid LSTM + Transformer',
  hybrid_dropout: 'Hybrid LSTM + Transformer (dropout)',
  lstm: 'LSTM baseline',
  transformer: 'Transformer baseline',
};

/* ---------- coarse grid geometry ---------- */
const LAT_STEP = 10, LON_STEP = 10;
const LATS: number[] = [];
for (let lat = -80; lat <= 80; lat += LAT_STEP) LATS.push(lat);
const LONS: number[] = [];
for (let lon = -180; lon <= 170; lon += LON_STEP) LONS.push(lon);

/** Very rough ocean/land mask (lat,lon) — demo only, not geography-accurate. */
function isOcean(lat: number, lon: number): boolean {
  const a = Math.abs(lat);
  // crude continent boxes (lonMin, lonMax, latMin, latMax)
  const continents: [number, number, number, number][] = [
    [-170, -60, 10, 75],   // N. America
    [-85, -30, -58, 14],   // S. America
    [-15, 50, 34, 72],     // Europe
    [-20, 55, -37, 34],    // Africa
    [55, 145, 5, 78],      // Asia
    [110, 155, -45, -8],   // Australia
    [-75, 20, -90, -60],   // Antarctica band
  ];
  return !continents.some(([lo, hi, la, ha]) =>
    lon >= lo && lon <= hi && lat >= la && lat <= ha && a > 0);
}

/* ---------- smooth pseudo-oxygen field ---------- */
function baseOxygen(lat: number, lon: number, depthM: number): number {
  // low-lat warm surface waters are oxygen-poor; high-lat + deep are richer
  const latF = Math.sin((lat * Math.PI) / 180);
  const depthF = Math.min(depthM / 2000, 1);
  const v =
    3.4 + 1.9 * (0.5 + 0.5 * latF) * (1 - 0.25 * depthF) +
    1.2 * Math.sin(lon * 0.04 + lat * 0.02) * (1 - depthF);
  return Math.max(0.2, v);
}

/** Sparse observation network: ~7% of ocean cells, biased to shipping lanes. */
function observationMask(seed: number, year: number): Set<string> {
  const r = rng(seed + year * 17);
  const set = new Set<string>();
  for (const lat of LATS) {
    for (const lon of LONS) {
      if (!isOcean(lat, lon)) continue;
      const laneBias =
        (Math.abs(lat) < 45 && Math.abs(Math.sin(lon * 0.017)) < 0.35) ? 0.16 : 0.035;
      if (r() < laneBias) set.add(`${lat},${lon}`);
    }
  }
  return set;
}

/* ---------- frames ---------- */
export function getFrame(run: RunSpec, layer: LayerId, year: number, depthM: number): GlobeFrame {
  const r = rng(run.seed * 1000 + year + depthM);
  const noiseScale = run.model === 'hybrid' ? 0.12 : run.model === 'hybrid_dropout' ? 0.16 : 0.24;
  const cells: CellValue[] = [];
  let min = Infinity, max = -Infinity;
  const obs = observationMask(run.seed, year);

  for (const lat of LATS) {
    for (const lon of LONS) {
      if (!isOcean(lat, lon)) continue;
      const truth = baseOxygen(lat, lon, depthM) + 0.3 * (r() - 0.5);
      const pred = truth + noiseScale * (r() - 0.5) * 3.2;
      let value = Number.NaN;
      switch (layer) {
        case 'prediction':  value = pred; break;
        case 'truth':       value = truth; break;
        case 'error':       value = Math.abs(pred - truth); break;
        case 'uncertainty':
          value = run.hasUncertainty ? 0.05 + 0.28 * Math.abs(pred - truth) : NaN;
          break;
        case 'observations': value = NaN; break; // handled via point cloud
      }
      if (Number.isFinite(value)) {
        cells.push({ lat, lon, value, observed: obs.has(`${lat},${lon}`) });
        min = Math.min(min, value); max = Math.max(max, value);
      }
    }
  }
  if (layer === 'observations') {
    // layer shows only observed cells as glowing points; value unused
    for (const key of obs) {
      const [lat, lon] = key.split(',').map(Number);
      cells.push({ lat, lon, value: NaN, observed: true });
    }
    min = 0; max = 1;
  }
  return { runId: run.id, layer, year, depthM, cells, min: Number.isFinite(min) ? min : 0, max: Number.isFinite(max) ? max : 1 };
}

/* ---------- per-point profiles for the analysis drawer ---------- */
export function getPointProfile(run: RunSpec, lat: number, lon: number, depthM: number): PointProfile {
  const depthIdx = DEPTHS_M.indexOf(depthM as (typeof DEPTHS_M)[number]);
  const di = depthIdx < 0 ? 3 : depthIdx;
  const truth: number[] = [], pred: number[] = [], std: number[] = [];
  const observedAtDepth: boolean[] = [];
  const r = rng(run.seed + Math.round((lat + 90) * 977 + (lon + 180) * 131));

  for (const d of DEPTHS_M) {
    const t = baseOxygen(lat, lon, d) + 0.2 * (r() - 0.5);
    truth.push(t);
    pred.push(t + (run.model.startsWith('hybrid') ? 0.18 : 0.34) * (r() - 0.5) * 2.6);
    std.push(run.hasUncertainty ? 0.06 + 0.22 * Math.abs(r() - 0.5) : 0);
    observedAtDepth.push(d === 0 || (d === 100 && r() < 0.4));
  }

  const timeYears: number[] = [], timeSeries: number[] = [], errorSeries: number[] = [];
  for (const y of YEARS) {
    const yr = rng(run.seed + y * 3 + Math.round(lat * 7.3 + lon * 2.1));
    const t = baseOxygen(lat, lon, DEPTHS_M[di]) + 0.25 * (yr() - 0.5);
    const p = t + (run.model.startsWith('hybrid') ? 0.2 : 0.38) * (yr() - 0.5) * 2.4;
    timeYears.push(y);
    timeSeries.push(p);
    errorSeries.push(Math.abs(p - t));
  }

  const rmse = Math.sqrt(
    errorSeries.reduce((s, e) => s + e * e, 0) / Math.max(errorSeries.length, 1),
  );

  return {
    lat, lon,
    depthLabelsM: [...DEPTHS_M],
    prediction: pred,
    truth,
    uncertainty: run.hasUncertainty ? std : null,
    observedAtDepth,
    timeYears,
    timeSeries,
    errorSeries,
    rmse,
  };
}

/* ---------- search helper: nearest ocean cell ---------- */
export function findNearestOceanCell(lat: number, lon: number): { lat: number; lon: number } {
  let best = { lat: 0, lon: 0 }, bestD = Infinity;
  for (const la of LATS) {
    for (const lo of LONS) {
      if (!isOcean(la, lo)) continue;
      const d = (la - lat) ** 2 + (lo - lon) ** 2;
      if (d < bestD) { bestD = d; best = { lat: la, lon: lo }; }
    }
  }
  return best;
}

export function cellValueAt(frame: GlobeFrame, lat: number, lon: number): CellValue | null {
  return frame.cells.find((c) => c.lat === lat && c.lon === lon) ?? null;
}
