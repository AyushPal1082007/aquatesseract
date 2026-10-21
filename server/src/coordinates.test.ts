import { describe, expect, test } from 'vitest';
import {
  latToIndex,
  lonToIndex,
  recoverGraphCoordinates,
  recoverGraphCoordinatesFromGeo,
  graphToGridIndices,
  parseCoordinateMetadata,
  ARRAY_ORDER,
  VERIFIED_LATITUDE_CENTERS,
  VERIFIED_LONGITUDE_CENTERS,
  VERIFIED_DEPTH_LABELS_M,
  VERIFIED_CONTRACT_COORDINATES,
} from './ml-data';

describe('Verified Contract Coordinates and Metadata', () => {
  test('verified array order is (year, depth, latitude, longitude)', () => {
    expect(ARRAY_ORDER).toEqual(['year', 'depth', 'latitude', 'longitude']);
  });

  test('latitude centers from -89.5 to 89.5 degrees (180 cells, south_to_north)', () => {
    expect(VERIFIED_LATITUDE_CENTERS.length).toBe(180);
    expect(VERIFIED_LATITUDE_CENTERS[0]).toBe(-89.5);
    expect(VERIFIED_LATITUDE_CENTERS[179]).toBe(89.5);

    // lat_index = round(latitude + 89.5)
    expect(latToIndex(-89.5)).toBe(0);
    expect(latToIndex(-89.0)).toBe(1);
    expect(latToIndex(0.0)).toBe(90);
    expect(latToIndex(89.5)).toBe(179);
  });

  test('prediction-array longitude centers from 0.5 to 359.5 degrees east (360 cells)', () => {
    expect(VERIFIED_LONGITUDE_CENTERS.length).toBe(360);
    expect(VERIFIED_LONGITUDE_CENTERS[0]).toBe(0.5);
    expect(VERIFIED_LONGITUDE_CENTERS[359]).toBe(359.5);

    // lon_index = round((longitude % 360) - 0.5)
    expect(lonToIndex(0.5)).toBe(0);
    expect(lonToIndex(1.5)).toBe(1);
    expect(lonToIndex(180.5)).toBe(180);
    expect(lonToIndex(359.5)).toBe(359);
    expect(lonToIndex(360.5)).toBe(0);
  });

  test('depth labels match contract 33 levels (0 to 5500 m)', () => {
    expect(VERIFIED_DEPTH_LABELS_M.length).toBe(33);
    expect(VERIFIED_DEPTH_LABELS_M[0]).toBe(0);
    expect(VERIFIED_DEPTH_LABELS_M[32]).toBe(5500);
  });

  test('parses contract metadata directly without requiring manual configuration', () => {
    const contractMeta = {
      dataset: 'CESM2_omip1',
      split: 'spatial',
      model: 'hybrid',
      seed: 1,
      start_year: 1948,
      years: [1948, 1949],
      depth_labels_m: VERIFIED_DEPTH_LABELS_M,
      latitude_convention:
        'south_to_north: row index i = latitude -89.5 + i degrees north (1-degree cell centres, 180 rows). Verified.',
      longitude_convention:
        '0_to_360_east: column index j = longitude 0.5 + j degrees east (1-degree cell centres, 360 columns), NOT -180..180. Verified.',
      lat: VERIFIED_LATITUDE_CENTERS,
      lon: VERIFIED_LONGITUDE_CENTERS,
      simulation_caveat: true,
    };

    const parsed = parseCoordinateMetadata(contractMeta);
    expect(parsed.latitudeOrder).toBe('south_to_north');
    expect(parsed.longitudeOrder).toBe('west_to_east');
    expect(parsed.longitudeConvention).toBe('0-360');
    expect(parsed.longitudeShiftCells).toBe(0);
    expect(parsed.oceanMaskAligned).toBe(true);
    expect(parsed.depthUnits).toBe('m');
    expect(parsed.latitudeValues.length).toBe(180);
    expect(parsed.longitudeValues.length).toBe(360);
    expect(parsed.depthValuesM.length).toBe(33);
  });

  test('recovers graph coordinates from unit-sphere x_geo', () => {
    // x = cos(lat) * cos(lon), y = cos(lat) * sin(lon), z = sin(lat)
    // Test case 1: Equator (lat=0, z=0), graph longitude -179.5
    // longitude_minus180_180 = degrees(arctan2(y, x)) = -179.5
    // longitude_0_to_360 = -179.5 % 360 = 180.5
    // lon_index = round(180.5 - 0.5) = 180
    const lon1 = -179.5;
    const rad1 = (lon1 * Math.PI) / 180;
    const x1 = Math.cos(rad1);
    const y1 = Math.sin(rad1);
    const z1 = 0;

    const recovered1 = recoverGraphCoordinates(x1, y1, z1);
    expect(recovered1.latitude).toBeCloseTo(0, 5);
    expect(recovered1.longitudeMinus180_180).toBeCloseTo(-179.5, 5);
    expect(recovered1.longitude0_to_360).toBeCloseTo(180.5, 5);

    const grid1 = graphToGridIndices(x1, y1, z1);
    expect(grid1.latIndex).toBe(90); // round(0 + 89.5) = 90
    expect(grid1.lonIndex).toBe(180); // prompt example: graph lon -179.5 -> array lon 180.5 -> lon_index 180

    // Test case 2: Equator (lat=0, z=0), graph longitude 0.5
    // longitude_minus180_180 = 0.5
    // longitude_0_to_360 = 0.5
    // lon_index = round(0.5 - 0.5) = 0
    const lon2 = 0.5;
    const rad2 = (lon2 * Math.PI) / 180;
    const x2 = Math.cos(rad2);
    const y2 = Math.sin(rad2);
    const z2 = 0;

    const recovered2 = recoverGraphCoordinates(x2, y2, z2);
    expect(recovered2.latitude).toBeCloseTo(0, 5);
    expect(recovered2.longitudeMinus180_180).toBeCloseTo(0.5, 5);
    expect(recovered2.longitude0_to_360).toBeCloseTo(0.5, 5);

    const grid2 = graphToGridIndices(x2, y2, z2);
    expect(grid2.latIndex).toBe(90);
    expect(grid2.lonIndex).toBe(0); // prompt example: graph lon 0.5 -> array lon 0.5 -> lon_index 0
  });

  test('recovers coordinates from x_geo vector', () => {
    const xGeo = [0, 1, 0]; // (lat=0, lon=90)
    const recovered = recoverGraphCoordinatesFromGeo(xGeo);
    expect(recovered.latitude).toBeCloseTo(0, 5);
    expect(recovered.longitudeMinus180_180).toBeCloseTo(90, 5);
    expect(recovered.longitude0_to_360).toBeCloseTo(90, 5);
  });
});
