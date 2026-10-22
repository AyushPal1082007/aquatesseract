import { HttpError } from './errors';
import type { LongitudeConvention } from './ml-data';

export interface BoundingBox {
  minLatitude: number;
  maxLatitude: number;
  minLongitude: number;
  maxLongitude: number;
}

export type RegionBoundary =
  | { kind: 'boxes'; boxes: BoundingBox[] }
  | { kind: 'polygon'; coordinates: Array<[number, number]> };

export interface RegionSelection {
  name: string;
  approximate: boolean;
  boundary: RegionBoundary;
}

export const NAMED_REGIONS: Record<string, RegionSelection> = {
  'atlantic ocean': {
    name: 'Atlantic Ocean', approximate: true,
    boundary: { kind: 'boxes', boxes: [
      { minLatitude: -60, maxLatitude: 70, minLongitude: -70, maxLongitude: 20 },
    ] },
  },
  'pacific ocean': {
    name: 'Pacific Ocean', approximate: true,
    boundary: { kind: 'boxes', boxes: [
      { minLatitude: -60, maxLatitude: 65, minLongitude: 120, maxLongitude: 180 },
      { minLatitude: -60, maxLatitude: 65, minLongitude: -180, maxLongitude: -70 },
    ] },
  },
  'indian ocean': {
    name: 'Indian Ocean', approximate: true,
    boundary: { kind: 'boxes', boxes: [
      { minLatitude: -60, maxLatitude: 30, minLongitude: 20, maxLongitude: 120 },
    ] },
  },
  'southern ocean': {
    name: 'Southern Ocean', approximate: true,
    boundary: { kind: 'boxes', boxes: [
      { minLatitude: -90, maxLatitude: -60, minLongitude: -180, maxLongitude: 180 },
    ] },
  },
  'arctic ocean': {
    name: 'Arctic Ocean', approximate: true,
    boundary: { kind: 'boxes', boxes: [
      { minLatitude: 70, maxLatitude: 90, minLongitude: -180, maxLongitude: 180 },
    ] },
  },
};

export function validateBoundingBox(box: BoundingBox): BoundingBox {
  const values = [box.minLatitude, box.maxLatitude, box.minLongitude, box.maxLongitude];
  if (values.some((v) => !Number.isFinite(v)) ||
      box.minLatitude < -90 || box.maxLatitude > 90 || box.minLatitude >= box.maxLatitude ||
      box.minLongitude < -180 || box.maxLongitude > 360 || box.minLongitude >= box.maxLongitude)
    throw new HttpError(400, 'invalid_region', 'Bounding box coordinates are invalid or out of range.');
  return box;
}

export function validatePolygon(value: unknown): Array<[number, number]> {
  if (!Array.isArray(value) || value.length < 3 || value.length > 1000)
    throw new HttpError(400, 'invalid_polygon', 'Polygon must contain between 3 and 1000 coordinate pairs.');
  const points: Array<[number, number]> = value.map((point) => {
    if (!Array.isArray(point) || point.length !== 2 || point.some((n) => typeof n !== 'number' || !Number.isFinite(n)))
      throw new HttpError(400, 'invalid_polygon', 'Polygon coordinates must be finite [longitude, latitude] pairs.');
    const [longitude, latitude] = point;
    if (longitude < -180 || longitude > 180 || latitude < -90 || latitude > 90)
      throw new HttpError(400, 'invalid_polygon', 'Polygon coordinates must be within longitude [-180, 180] and latitude [-90, 90].');
    return [longitude, latitude];
  });
  return points;
}

export function resolveRegion(input: {
  name?: string;
  box?: BoundingBox;
  polygon?: unknown;
}): RegionSelection {
  const count = Number(Boolean(input.name)) + Number(Boolean(input.box)) + Number(Boolean(input.polygon));
  if (count !== 1) throw new HttpError(400, 'invalid_region', 'Provide exactly one named region, bounding box, or polygon.');
  if (input.name) {
    const region = NAMED_REGIONS[input.name.trim().toLowerCase()];
    if (!region) throw new HttpError(400, 'invalid_region', 'Unknown named region.');
    return region;
  }
  if (input.box) {
    const box = validateBoundingBox(input.box);
    return { name: 'Custom bounding box', approximate: false, boundary: { kind: 'boxes', boxes: [box] } };
  }
  const coordinates = validatePolygon(input.polygon);
  return { name: 'Custom polygon', approximate: false, boundary: { kind: 'polygon', coordinates } };
}

function normalizedLongitudeDelta(longitude: number, center: number): number {
  return ((longitude - center + 540) % 360) - 180;
}

function inPolygon(latitude: number, longitude: number, coordinates: Array<[number, number]>): boolean {
  const points = coordinates.map(([lon, lat]) => [normalizedLongitudeDelta(lon, longitude), lat] as [number, number]);
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    const intersects = (yi > latitude) !== (yj > latitude) &&
      0 < ((xj - xi) * (latitude - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

export function containsCellCenter(region: RegionSelection, latitude: number, longitude: number, _convention: LongitudeConvention): boolean {
  if (region.boundary.kind === 'polygon') return inPolygon(latitude, longitude, region.boundary.coordinates);
  const lon = normalizedLongitudeDelta(longitude, 0);
  return region.boundary.boxes.some((box) => {
    const width = box.maxLongitude - box.minLongitude;
    const west = normalizedLongitudeDelta(box.minLongitude, 0);
    const east = normalizedLongitudeDelta(box.maxLongitude, 0);
    const longitudeInBox = width >= 360 || (west <= east
      ? lon >= west && lon <= east
      : lon >= west || lon <= east);
    return latitude >= box.minLatitude && latitude <= box.maxLatitude && longitudeInBox;
  });
}
