import { mlApi, MLMetadata, MLRun } from '../api';
import { CellValue, GlobeFrame, LayerId, PointProfile, RunSpec } from './types';

export function toRunSpec(run: MLRun): RunSpec {
  return {
    id: run.runId,
    label: `${run.model} · ${run.dataset} · ${run.split}`,
    model: run.model,
    seed: run.seed,
    hasUncertainty: run.uncertaintyAvailable,
    uncertaintyReason: run.uncertaintyReason,
    source: run.source,
    dataset: run.dataset,
    split: run.split,
    mock: run.mock,
  };
}

export async function loadRuns(signal?: AbortSignal): Promise<RunSpec[]> {
  const response = await mlApi.runs(signal);
  if (!response || !Array.isArray(response.available)) {
    throw new Error('The ML runs endpoint returned no run list. Check the server logs and RESULTS_DIR.');
  }
  return response.available.map(toRunSpec);
}

export async function loadMetadata(runId: string, signal?: AbortSignal): Promise<MLMetadata> {
  return mlApi.metadata(runId, signal);
}

export async function getFrame(
  run: RunSpec,
  layer: LayerId,
  year: number,
  depthM: number,
  signal?: AbortSignal,
): Promise<GlobeFrame> {
  const response = await mlApi.slice(run.id, year, depthM, layer, 10000, signal);
  const cells: CellValue[] = response.points.map((point) => ({
    lat: point.latitude,
    lon: point.longitude > 180 ? point.longitude - 360 : point.longitude,
    value: point.value ?? Number.NaN,
    observed: point.observed,
    ocean: point.ocean,
  }));
  const finiteValues = cells.map((cell) => cell.value).filter(Number.isFinite);
  return {
    runId: run.id,
    layer,
    year: response.year,
    depthM: response.depthM,
    cells,
    min: finiteValues.length ? Math.min(...finiteValues) : 0,
    max: finiteValues.length ? Math.max(...finiteValues) : 1,
    units: response.oxygenUnits,
    downsampleStride: response.downsampleStride,
  };
}

export async function resolvePoint(
  run: RunSpec,
  latitude: number,
  longitude: number,
  year: number,
  depthM: number,
  signal?: AbortSignal,
) {
  return mlApi.point(run.id, latitude, longitude, year, depthM, signal);
}

export async function getPointProfile(
  run: RunSpec,
  latitude: number,
  longitude: number,
  year: number,
  depthM: number,
  signal?: AbortSignal,
): Promise<PointProfile> {
  const [profile, series] = await Promise.all([
    mlApi.profile(run.id, latitude, longitude, year, signal),
    mlApi.timeseries(run.id, latitude, longitude, depthM, signal),
  ]);
  const profilePoints = profile.points;
  const depthLabelsM = profilePoints.map((point) => point.depthM);
  const prediction = profilePoints.map((point) => point.prediction ?? Number.NaN);
  const truth = profilePoints.map((point) => point.reference ?? Number.NaN);
  const hasUncertainty = profilePoints.some((point) => point.uncertaintyAvailable);
  const uncertainty = hasUncertainty
    ? profilePoints.map((point) => point.uncertainty ?? Number.NaN)
    : null;
  const errors = series.points.map((point) => Math.abs(point.error ?? Number.NaN));
  const finiteErrors = errors.filter(Number.isFinite);
  return {
    lat: Number(profile.provenance.selectedLatitude ?? latitude),
    lon: Number(profile.provenance.selectedLongitude ?? longitude),
    depthLabelsM,
    prediction,
    truth,
    uncertainty,
    observedAtDepth: profilePoints.map((point) => point.observedStatus === 'observed'),
    timeYears: series.points.map((point) => point.year),
    timeSeries: series.points.map((point) => point.prediction ?? Number.NaN),
    errorSeries: errors,
    rmse: finiteErrors.length
      ? Math.sqrt(finiteErrors.reduce((sum, error) => sum + error * error, 0) / finiteErrors.length)
      : 0,
    provenance: profile.provenance,
  };
}

export function cellValueAt(frame: GlobeFrame, lat: number, lon: number): CellValue | null {
  return frame.cells.find((cell) => cell.lat === lat && cell.lon === lon) ?? null;
}
