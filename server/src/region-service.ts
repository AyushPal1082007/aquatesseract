import { HttpError } from './errors';
import { getCoordinates, provenance, runInfo, selectDepth, type SliceLayer } from './ml-data';
import { readBlock } from './npy';
import { fileRunArtifactRepository as repository } from './runs';
import { containsCellCenter, type RegionSelection } from './regions';

const { resolveRunDirectory: runDir, readJson, readArray: arr } = repository;

function shiftedIndex(index: number, width: number, shift: number): number {
  return ((index + shift) % width + width) % width;
}

export function getRegionData(
  run: string,
  year: number,
  depthM: number,
  layer: SliceLayer,
  maxPoints: number,
  region: RegionSelection,
) {
  const info = runInfo(run);
  if (layer === 'uncertainty' && !info.uncertaintyAvailable)
    throw new HttpError(422, 'uncertainty_unavailable', 'Uncertainty unavailable for this baseline.');
  const dir = runDir(run);
  const meta = readJson(dir, 'metadata.json');
  const coords = getCoordinates(run);
  const yearIndex = (meta.years as number[]).indexOf(year);
  if (yearIndex < 0) throw new HttpError(400, 'invalid_year', 'Year is not supported by this run.');
  const depth = selectDepth(coords, depthM);
  const depthIndex = depth.index;
  const selectedDepthM = depth.value;
  const prediction = readBlock(arr(dir, 'prediction')!, [yearIndex, depthIndex]);
  const reference = readBlock(arr(dir, 'truth')!, [yearIndex, depthIndex]);
  const observed = readBlock(arr(dir, 'observed')!, [yearIndex, depthIndex]);
  const ocean = readBlock(arr(dir, 'ocean')!, [depthIndex]);
  const uncertainty = info.uncertaintyAvailable ? readBlock(arr(dir, 'uncertainty')!, [yearIndex, depthIndex]) : null;
  const nlat = coords.latitudeValues.length, nlon = coords.longitudeValues.length;
  const selectedCells: number[] = [];
  for (let lat = 0; lat < nlat; lat++) {
    for (let lon = 0; lon < nlon; lon++) {
      if (containsCellCenter(region, coords.latitudeValues[lat], coords.longitudeValues[lon], coords.longitudeConvention))
        selectedCells.push(lat * nlon + lon);
    }
  }

  const regionCellCount = selectedCells.length;
  let validOceanCellCount = 0;
  let observedCellCount = 0;
  let neverObservedCellCount = 0;
  let sumPrediction = 0, minPrediction = Infinity, maxPrediction = -Infinity;
  let sumUncertainty = 0, uncertaintyCount = 0;
  let sumSquaredError = 0, sumAbsoluteError = 0, sumReference = 0, sumReferenceSquared = 0, errorCount = 0;

  for (const cell of selectedCells) {
    const lat = Math.floor(cell / nlon), lon = cell % nlon;
    const index = lat * nlon + shiftedIndex(lon, nlon, coords.longitudeShiftCells);
    const pred = prediction[index], truth = reference[index];
    if (ocean[index] === 0 || !Number.isFinite(pred) || !Number.isFinite(truth)) continue;
    validOceanCellCount++;
    sumPrediction += pred;
    minPrediction = Math.min(minPrediction, pred);
    maxPrediction = Math.max(maxPrediction, pred);
    if (observed[index] !== 0) observedCellCount++;
    else neverObservedCellCount++;
    if (uncertainty && Number.isFinite(uncertainty[index]) && uncertainty[index] > 0) {
      sumUncertainty += uncertainty[index];
      uncertaintyCount++;
    }
    const error = pred - truth;
    sumSquaredError += error * error;
    sumAbsoluteError += Math.abs(error);
    sumReference += truth;
    sumReferenceSquared += truth * truth;
    errorCount++;
  }
  if (validOceanCellCount === 0)
    throw new HttpError(422, 'empty_region', 'No valid ocean cells were found in this region.');

  const rmse = errorCount ? Math.sqrt(sumSquaredError / errorCount) : null;
  const mae = errorCount ? sumAbsoluteError / errorCount : null;
  const variance = errorCount ? sumReferenceSquared - (sumReference * sumReference) / errorCount : 0;
  const r2 = errorCount && variance > 0 ? 1 - sumSquaredError / variance : null;

  const pointCount = Math.min(maxPoints, regionCellCount);
  const points: Array<{ latitude: number; longitude: number; value: number | null }> = [];
  for (let i = 0; i < pointCount; i++) {
    const cell = selectedCells[Math.floor(i * regionCellCount / pointCount)];
    const lat = Math.floor(cell / nlon), lon = cell % nlon;
    const index = lat * nlon + shiftedIndex(lon, nlon, coords.longitudeShiftCells);
    const pred = prediction[index], truth = reference[index];
    const valid = ocean[index] !== 0 && Number.isFinite(pred) && Number.isFinite(truth);
    let value: number | null = null;
    if (valid) {
      if (layer === 'reconstruction') value = pred;
      if (layer === 'reference') value = truth;
      if (layer === 'error') value = pred - truth;
      if (layer === 'observations' && observed[index] !== 0) value = truth;
      if (layer === 'uncertainty' && uncertainty && Number.isFinite(uncertainty[index]) && uncertainty[index] > 0) value = uncertainty[index];
    }
    points.push({ latitude: coords.latitudeValues[lat], longitude: coords.longitudeValues[lon], value });
  }

  return {
    region: { name: region.name, approximate: region.approximate, boundary: region.boundary },
    run: info,
    year,
    depthM: selectedDepthM,
    layer,
    summary: {
      regionCellCount,
      validOceanCellCount,
      meanPrediction: sumPrediction / validOceanCellCount,
      minimumPrediction: minPrediction,
      maximumPrediction: maxPrediction,
      meanUncertainty: uncertaintyCount ? sumUncertainty / uncertaintyCount : null,
      uncertaintyAvailable: info.uncertaintyAvailable,
      rmse,
      mae,
      r2,
      observedCellCount,
      neverObservedCellCount,
    },
    visualization: { points, downsampleMethod: 'deterministic_even_sample', exact: false, maxPoints },
    provenance: provenance(run, year, selectedDepthM),
  };
}
