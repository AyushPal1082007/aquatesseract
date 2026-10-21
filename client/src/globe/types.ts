/* ================================================================
   GLOBE DASHBOARD — shared types
   Mirrors the conventions of src/api.ts so the mock adapter can be
   swapped for the real /api backend without touching the UI.
   ================================================================ */

export type ModelKind =
  | 'hybrid'          // Hybrid LSTM + Transformer (has model-estimated uncertainty)
  | 'hybrid_dropout'  // Hybrid LSTM + Transformer with dropout
  | 'lstm'            // LSTM baseline (no uncertainty provided)
  | 'transformer';    // Transformer baseline (no uncertainty provided)

export interface RunSpec {
  id: string;            // e.g. 'ov_hybrid_s42'
  label: string;         // human-friendly
  model: ModelKind;
  seed: number;
  /** Model-estimated uncertainty is scientifically valid only when true. */
  hasUncertainty: boolean;
}

export type LayerId = 'prediction' | 'truth' | 'error' | 'uncertainty' | 'observations';

export interface LayerInfo {
  id: LayerId;
  label: string;        // exact transparency wording shown in UI
  units: string | null; // null => dimensionless / not applicable
}

export const LAYERS: LayerInfo[] = [
  { id: 'prediction',  label: 'Hybrid reconstruction',            units: 'mL/L' },
  { id: 'truth',       label: 'Simulation reference field',       units: 'mL/L' },
  { id: 'error',       label: 'Reconstruction error',             units: 'mL/L' },
  { id: 'uncertainty', label: 'Model-estimated uncertainty',      units: 'mL/L' },
  { id: 'observations',label: 'Sparse simulated observations',    units: null },
];

export const YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020] as const;
export const DEPTHS_M = [0, 25, 50, 100, 200, 500, 1000, 2000] as const;

export interface CellValue {
  lat: number;          // degrees, -90..90
  lon: number;          // degrees, -180..180
  value: number;        // layer value at this cell (NaN = no data / land)
  observed: boolean;    // observed cell vs never-observed simulated cell
}

/** One grid frame for a (run, layer, year, depth). */
export interface GlobeFrame {
  runId: string;
  layer: LayerId;
  year: number;
  depthM: number;
  cells: CellValue[];       // sparse list of ocean cells (land omitted)
  min: number;
  max: number;
}

export interface PointProfile {
  lat: number;
  lon: number;
  depthLabelsM: number[];
  /** per depth: prediction, truth, ±uncertainty (null when not provided) */
  prediction: number[];
  truth: number[];
  uncertainty: number[] | null;
  observedAtDepth: boolean[];
  timeYears: number[];
  timeSeries: number[];     // prediction at selected depth across years
  errorSeries: number[];    // |pred - truth| across years
  rmse: number;
}

export const SIMULATION_NOTICE =
  'Data source: OceanVerse CESM2-OMIP1 simulation. These are simulation-based dissolved-oxygen ' +
  'reconstructions, not live real-ocean measurements. Real-ocean accuracy has not yet been established.';

export const DEMO_NOTICE =
  'Demo mode — mock data. Verified CESM2-OMIP1 artifacts are not connected.';
