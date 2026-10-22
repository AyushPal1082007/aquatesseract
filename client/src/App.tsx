import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { api, CompRow, MapData, Metadata, ProfileData, TimeSeriesData } from './api';

type PickedCell = { latIdx: number; lonIdx: number; latitude: number; longitude: number };
type OceanRegion = { id: string; name: string; south: number; north: number; west: number; east: number };

function layerLabel(layer: string) {
  const labels: Record<string, string> = {
    truth: 'Reference oxygen', prediction: 'Reconstructed oxygen', error: 'Reconstruction error',
    uncertainty: 'Model uncertainty', observed: 'Observation coverage', density: 'Sampling density',
  };
  return labels[layer] ?? layer[0]?.toUpperCase() + layer.slice(1);
}

function blend(a: number[], b: number[], t: number) {
  const f = Math.max(0, Math.min(1, t));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(',')})`;
}

function color(value: number, min: number, max: number, layer: string) {
  if (layer === 'error') {
    const limit = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const t = (value / limit + 1) / 2;
    return t < .5 ? blend([66, 125, 182], [232, 242, 239], t * 2) : blend([232, 242, 239], [224, 112, 100], (t - .5) * 2);
  }
  if (layer === 'uncertainty') return blend([211, 224, 244], [112, 93, 174], (value - min) / (max - min || 1));
  if (layer === 'observed') return value >= .5 ? '#16857e' : '#b8c9c8';
  if (layer === 'density') return blend([231, 241, 239], [43, 128, 151], (value - min) / (max - min || 1));
  const stops = [[125, 164, 161], [157, 192, 174], [192, 207, 174], [226, 216, 183], [248, 241, 218]];
  const t = Math.max(0, Math.min(.999, (value - min) / (max - min || 1))) * (stops.length - 1);
  const i = Math.floor(t);
  return blend(stops[i], stops[i + 1], t - i);
}

function MapCanvas({ data, selected, onSelect, compact = false }: {
  data: MapData; selected?: PickedCell | null; onSelect?: (cell: PickedCell) => void; compact?: boolean;
}) {
  const [rows, cols] = data.shape;
  const px = compact ? 1 : 2;
  const click = (event: MouseEvent<HTMLCanvasElement>) => {
    if (!onSelect) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.max(0, Math.min(cols - 1, Math.floor((event.clientX - rect.left) / rect.width * cols)));
    const row = rows - 1 - Math.max(0, Math.min(rows - 1, Math.floor((event.clientY - rect.top) / rect.height * rows)));
    if (data.values[row * cols + col] == null) return;
    onSelect({ latIdx: row, lonIdx: col, latitude: data.latitude_values?.[row] ?? row, longitude: data.longitude_values?.[col] ?? col });
  };
  return <canvas className={`field-canvas${compact ? ' field-canvas--compact' : ''}`} width={cols * px} height={rows * px}
    onClick={click} role={onSelect ? 'button' : undefined}
    aria-label={onSelect ? 'Click an ocean cell to inspect it' : `${data.layer} map`}
    ref={(canvas) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.fillStyle = '#dce9ec'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const min = data.min ?? 0, max = data.max ?? 1;
      data.values.forEach((value, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        ctx.fillStyle = value == null ? '#dce9ec' : color(value, min, max, data.layer);
        ctx.fillRect(c * px, (rows - 1 - r) * px, px, px);
      });
      if (selected && !compact) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.strokeRect(selected.lonIdx * px, (rows - 1 - selected.latIdx) * px, px, px);
      }
    }} />;
}

function Legend({ data }: { data: MapData }) {
  const diverging = data.layer === 'error';
  const gradient = diverging ? 'linear-gradient(90deg,#427db6,#e8f2ef 50%,#e07064)' :
    data.layer === 'uncertainty' ? 'linear-gradient(90deg,#d3e0f4,#705dae)' :
      'linear-gradient(90deg,#7da4a1,#9dc0ae,#c0cfae,#e2d8b7,#f8f1da)';
  const extent = Math.max(Math.abs(data.min ?? 0), Math.abs(data.max ?? 0));
  return <div className="field-legend"><div className="field-legend__bar" style={{ background: gradient }} />
    <div className="field-legend__labels"><span>{diverging ? `-${extent.toFixed(3)}` : data.min?.toFixed(3)}</span><span>{diverging ? 'zero error' : data.oxygen_units}</span><span>{diverging ? `+${extent.toFixed(3)}` : data.max?.toFixed(3)}</span></div>
  </div>;
}

function LayerDeepDive({ data, year, depthM, region, onExpand, onClose, expanded = false }: { data: MapData; year: number; depthM: number; region: OceanRegion; onExpand?: () => void; onClose?: () => void; expanded?: boolean }) {
  const { rowIds, colIds } = regionGrid(data, region);
  const values = rowIds.flatMap((row) => colIds.map((col) => data.values[row * data.shape[1] + col]))
    .filter((value): value is number => value != null && Number.isFinite(value));
  if (!values.length) return <article className="glass-card layer-insight-panel"><div className="layer-insight-head"><div><p className="section-kicker">SELECTED LAYER ANALYSIS</p><h2>{data.layer} · {region.name}</h2><p>{year} · {depthM} m · {data.oxygen_units}</p></div></div><div className="chart-empty">No valid values are available for this region in the selected layer and slice.</div></article>;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const std = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  const isError = data.layer === 'error', isObserved = data.layer === 'observed', isUncertainty = data.layer === 'uncertainty', isDensity = data.layer === 'density';
  const regionalMin = Math.min(...values), regionalMax = Math.max(...values);
  const extent = isError ? Math.max(Math.abs(regionalMin), Math.abs(regionalMax)) || 1 : Math.max(regionalMax - regionalMin, 1e-9);
  const low = isError ? -extent : regionalMin, high = isError ? extent : regionalMax;
  const bins = Array.from({ length: 14 }, (_, index) => ({ start: low + (high - low) * index / 14, end: low + (high - low) * (index + 1) / 14, count: 0 }));
  values.forEach((value) => { const index = Math.min(13, Math.max(0, Math.floor((value - low) / (high - low || 1) * 14))); bins[index].count++; });
  const maxBin = Math.max(...bins.map((bin) => bin.count), 1);
  const bands = [{ label: 'South', min: -90, max: -45 }, { label: 'Tropics S', min: -45, max: 0 }, { label: 'Tropics N', min: 0, max: 45 }, { label: 'North', min: 45, max: 90 }];
  const latStats = bands.map((band) => {
    let total = 0, count = 0;
    rowIds.forEach((row) => {
      const latitude = data.latitude_values?.[row];
      if (latitude == null || latitude < band.min || latitude >= band.max) return;
      for (const col of colIds) {
        const value = data.values[row * data.shape[1] + col];
        if (value != null && Number.isFinite(value)) { total += isError ? Math.abs(value) : value; count++; }
      }
    });
    return { ...band, mean: count ? total / count : null, count };
  });
  const bandMax = Math.max(...latStats.map((band) => band.mean ?? 0), 1e-9);
  const lonBands = Array.from({ length: 4 }, (_, bandIndex) => {
    const start = Math.floor(bandIndex * colIds.length / 4), end = Math.floor((bandIndex + 1) * colIds.length / 4);
    let total = 0, count = 0;
    for (const row of rowIds) for (const col of colIds.slice(start, Math.max(start + 1, end))) {
      const value = data.values[row * data.shape[1] + col];
      if (value != null && Number.isFinite(value)) { total += isError ? Math.abs(value) : value; count++; }
    }
    const first = data.longitude_values?.[colIds[start]];
    const last = data.longitude_values?.[colIds[Math.max(start, end - 1)]];
    return { label: first == null || last == null ? `Sector ${bandIndex + 1}` : `${Math.round(first)}°–${Math.round(last)}°`, mean: count ? total / count : null, count };
  });
  const sortedValues = [...values].sort((a, b) => a - b);
  const quantile = (q: number) => sortedValues[Math.round((sortedValues.length - 1) * q)];
  const quantiles = [{ label: 'Min', value: sortedValues[0] }, { label: '25th', value: quantile(.25) }, { label: 'Median', value: quantile(.5) }, { label: '75th', value: quantile(.75) }, { label: 'Max', value: sortedValues[sortedValues.length - 1] }];
  const quantileSpan = quantiles[4].value - quantiles[0].value || 1;
  const layerName = layerLabel(data.layer);
  const headlineMetric = isError ? { label: 'Mean absolute error', value: (values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length).toFixed(3) }
    : isObserved ? { label: 'Observed coverage', value: `${(mean * 100).toFixed(1)}%` }
      : isUncertainty ? { label: 'Mean uncertainty', value: mean.toFixed(3) }
        : isDensity ? { label: 'Mean observed fraction', value: mean.toFixed(3) }
          : { label: 'Mean oxygen', value: mean.toFixed(3) };
  return <article className={`glass-card layer-insight-panel layer-insight-panel--${data.layer}${expanded ? ' layer-insight-panel--expanded' : ''}`}>
    <div className="layer-insight-head"><div><p className="section-kicker">SELECTED LAYER ANALYSIS</p><h2>{layerName} distribution</h2><p>{region.name} · {year} · {depthM} m · {data.oxygen_units}</p></div><div className="layer-insight-actions"><span className="layer-insight-badge">{values.length.toLocaleString()} valid regional cells</span>{onExpand && <button className="analysis-open-button" onClick={onExpand}>Expand analysis ⤢</button>}{onClose && <button className="analysis-close-button" onClick={onClose} aria-label="Close analysis">×</button>}</div></div>
    <div className="layer-stat-row"><div><span>{headlineMetric.label}</span><b>{headlineMetric.value} {isObserved ? '' : data.oxygen_units}</b></div><div><span>{isError ? 'Mean signed error' : isObserved ? 'Observed cells' : 'Spread (standard deviation)'}</span><b>{isObserved ? `${Math.round(mean * values.length).toLocaleString()} / ${values.length.toLocaleString()}` : `${(isError ? mean : std).toFixed(3)} ${data.oxygen_units}`}</b></div><div><span>Valid ocean cells</span><b>{values.length.toLocaleString()}</b></div></div>
    <div className="layer-insight-grid"><section className="layer-insight-chart"><h3>{isError ? 'Signed error distribution' : isObserved ? 'Observation coverage distribution' : isUncertainty ? 'Uncertainty distribution' : isDensity ? 'Sampling density distribution' : 'Oxygen value distribution'}</h3><p>Cell counts for the selected region and map slice</p>
      <div className="layer-histogram" role="img" aria-label={`${layerName} value distribution histogram`}>{bins.map((bin, index) => <div className="layer-histogram__column" key={index} title={`${bin.count.toLocaleString()} cells`}><i className={isError && bin.start < 0 ? 'is-negative' : ''} style={{ height: `${Math.max(2, bin.count / maxBin * 100)}%` }}/></div>)}</div>
      <div className="layer-histogram__axis"><span>{low.toFixed(2)}</span><span>{isError ? '0 error' : data.oxygen_units}</span><span>{high.toFixed(2)}</span></div>
    </section><section className="layer-insight-chart"><h3>{isError ? 'Mean absolute error by latitude band' : isObserved ? 'Observed coverage by latitude band' : isUncertainty ? 'Mean uncertainty by latitude band' : isDensity ? 'Mean sampling density by latitude band' : 'Mean oxygen by latitude band'}</h3><p>Average across valid cells at each latitude</p>
      <div className="latitude-bars">{latStats.map((band) => <div className="latitude-bars__item" key={band.label}><span>{band.label}</span><div><i style={{ width: `${Math.max(band.mean == null ? 0 : 3, (band.mean ?? 0) / bandMax * 100)}%` }}/></div><b>{band.mean == null ? '—' : isObserved ? `${(band.mean * 100).toFixed(1)}%` : band.mean.toFixed(2)}</b></div>)}</div>
      <small>{latStats.reduce((sum, band) => sum + band.count, 0).toLocaleString()} valid cells grouped by latitude</small>
    </section><section className="layer-insight-chart"><h3>{isError ? 'Mean absolute error by longitude sector' : 'Mean value by longitude sector'}</h3><p>Regional east–west variation</p><div className="latitude-bars">{lonBands.map((band) => <div className="latitude-bars__item" key={band.label}><span>{band.label}</span><div><i style={{ width: `${Math.max(band.mean == null ? 0 : 3, (band.mean ?? 0) / Math.max(...lonBands.map((item) => item.mean ?? 0), 1e-9) * 100)}%` }}/></div><b>{band.mean == null ? '—' : isObserved ? `${(band.mean * 100).toFixed(1)}%` : band.mean.toFixed(2)}</b></div>)}</div><small>{lonBands.reduce((sum, band) => sum + band.count, 0).toLocaleString()} valid cells grouped by longitude</small></section>
    <section className="layer-insight-chart"><h3>Value spread</h3><p>Minimum, quartiles, median, and maximum</p><div className="quantile-plot">{quantiles.map((item) => <div className="quantile-plot__item" key={item.label}><span>{item.label}</span><div><i style={{ left: `${(item.value - quantiles[0].value) / quantileSpan * 100}%` }}/></div><b>{item.value.toFixed(3)}</b></div>)}</div><small>{data.oxygen_units} · {isError ? 'signed values' : 'selected layer values'}</small></section></div>
  </article>;
}

function seriesPath(values: (number | null)[], x: (i: number, v: number) => number, y: (i: number, v: number) => number) {
  let active = false;
  return values.map((value, i) => {
    if (value == null || !Number.isFinite(value)) { active = false; return ''; }
    const command = active ? 'L' : 'M'; active = true;
    return `${command}${x(i, value).toFixed(1)},${y(i, value).toFixed(1)}`;
  }).filter(Boolean).join(' ');
}

function DepthProfile({ data }: { data: ProfileData | null }) {
  if (!data) return <div className="chart-empty">Click an ocean cell to see its oxygen profile by depth.</div>;
  const values = [...data.truth, ...data.prediction].filter((v): v is number => v != null && Number.isFinite(v));
  if (!values.length) return <div className="chart-empty">No profile values are available for this cell.</div>;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const maxDepth = Math.max(...data.depth_m, 1), w = 620, h = 220, l = 44, r = 12, t = 12, b = 28;
  const x = (_: number, v: number) => l + (v - min) / span * (w - l - r);
  const y = (i: number) => t + data.depth_m[i] / maxDepth * (h - t - b);
  return <div className="chart-wrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Depth profile: reconstruction and reference">
    <line x1={l} y1={t} x2={l} y2={h-b} className="chart-axis"/><line x1={l} y1={h-b} x2={w-r} y2={h-b} className="chart-axis"/>
    <path d={seriesPath(data.truth, x, y)} className="chart-line chart-line--reference"/><path d={seriesPath(data.prediction, x, y)} className="chart-line chart-line--prediction"/>
    <text x={l} y={h-6} className="chart-tick">{min.toFixed(2)}</text><text x={w-r} y={h-6} textAnchor="end" className="chart-tick">{max.toFixed(2)} {data.oxygen_units}</text>
    <text x={l-7} y={t+5} textAnchor="end" className="chart-tick">0 m</text><text x={l-7} y={h-b} textAnchor="end" className="chart-tick">{maxDepth} m</text>
  </svg><div className="chart-legend"><span><i className="legend-swatch legend-swatch--prediction"/>Reconstruction</span><span><i className="legend-swatch legend-swatch--reference"/>Reference</span></div></div>;
}

function TimeTrend({ data }: { data: TimeSeriesData | null }) {
  if (!data) return <div className="chart-empty">Select a cell to see its oxygen trend through time.</div>;
  const values = [...data.truth, ...data.prediction].filter((v): v is number => v != null && Number.isFinite(v));
  if (!values.length) return <div className="chart-empty">No time-series values are available.</div>;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const w = 620, h = 220, l = 38, r = 12, t = 12, b = 28;
  const x = (i: number) => l + i / Math.max(data.years.length - 1, 1) * (w-l-r);
  const y = (_: number, v: number) => t + (max-v)/span * (h-t-b);
  return <div className="chart-wrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Time trend: reconstruction and reference">
    <line x1={l} y1={t} x2={l} y2={h-b} className="chart-axis"/><line x1={l} y1={h-b} x2={w-r} y2={h-b} className="chart-axis"/>
    <path d={seriesPath(data.truth, (i) => x(i), y)} className="chart-line chart-line--reference"/><path d={seriesPath(data.prediction, (i) => x(i), y)} className="chart-line chart-line--prediction"/>
    <text x={l} y={h-6} className="chart-tick">{data.years[0]}</text><text x={w-r} y={h-6} textAnchor="end" className="chart-tick">{data.years[data.years.length-1]}</text>
    <text x={l-6} y={t+5} textAnchor="end" className="chart-tick">{max.toFixed(2)}</text><text x={l-6} y={h-b} textAnchor="end" className="chart-tick">{min.toFixed(2)}</text>
  </svg><div className="chart-legend"><span><i className="legend-swatch legend-swatch--prediction"/>Reconstruction</span><span><i className="legend-swatch legend-swatch--reference"/>Reference</span></div></div>;
}

const OCEAN_REGIONS: OceanRegion[] = [
  { id: 'north-atlantic', name: 'North Atlantic', south: 0, north: 70, west: -80, east: 20 },
  { id: 'south-atlantic', name: 'South Atlantic', south: -60, north: 0, west: -70, east: 20 },
  { id: 'north-pacific', name: 'North Pacific', south: 0, north: 65, west: 120, east: -100 },
  { id: 'south-pacific', name: 'South Pacific', south: -60, north: 0, west: 120, east: -70 },
  { id: 'indian', name: 'Indian Ocean', south: -60, north: 25, west: 20, east: 120 },
  { id: 'southern', name: 'Southern Ocean', south: -80, north: -45, west: -180, east: 180 },
];

function regionGrid(data: MapData, region: OceanRegion) {
  const rowIds = (data.latitude_values ?? []).map((lat, i) => ({ lat, i }))
    .filter(({ lat }) => lat >= region.south && lat <= region.north)
    .sort((a, b) => a.lat - b.lat).map(({ i }) => i);
  const colIds = (data.longitude_values ?? []).map((lon, i) => {
    const normalized = ((lon + 180) % 360 + 360) % 360 - 180;
    const inside = region.west <= region.east ? normalized >= region.west && normalized <= region.east : normalized >= region.west || normalized <= region.east;
    const distance = normalized >= region.west ? normalized - region.west : normalized + 360 - region.west;
    return { i, inside, distance };
  }).filter(({ inside }) => inside).sort((a, b) => a.distance - b.distance).map(({ i }) => i);
  return { rowIds, colIds };
}

function RegionalLayerGraph({ run, meta, year, onYearChange, layer, onLayerChange, regionId, onRegionChange }: {
  run: string; meta: Metadata; year: number; onYearChange: (value: number) => void;
  layer: string; onLayerChange: (value: string) => void; regionId: string; onRegionChange: (value: string) => void;
}) {
  const [layers, setLayers] = useState<MapData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const region = OCEAN_REGIONS.find((item) => item.id === regionId) ?? OCEAN_REGIONS[0];
  const regionValues = layers.flatMap((data) => {
    const [, cols] = data.shape, { rowIds, colIds } = regionGrid(data, region);
    return rowIds.flatMap((row) => colIds.map((col) => data.values[row * cols + col]))
      .filter((value): value is number => value != null && Number.isFinite(value));
  });
  const regionalMin = regionValues.length ? Math.min(...regionValues) : null;
  const regionalMax = regionValues.length ? Math.max(...regionValues) : null;
  const errorExtent = layer === 'error' ? Math.max(Math.abs(regionalMin ?? 0), Math.abs(regionalMax ?? 0)) : 0;
  const oxygenMin = layer === 'error' ? -errorExtent : regionalMin;
  const oxygenMax = layer === 'error' ? errorExtent : regionalMax;

  useEffect(() => {
    if (!run || !meta.depth_labels_m.length) return;
    const count = meta.depth_labels_m.length;
    const indices = [...new Set(Array.from({ length: Math.min(6, count) }, (_, i) => Math.round(i * (count - 1) / (Math.min(6, count) - 1 || 1))))];
    const controller = new AbortController();
    setLoading(true); setError('');
    const requestTimer = window.setTimeout(() => {
      Promise.all(indices.map((depthIndex) => api.map(run, layer, year, depthIndex, 4, controller.signal)))
        .then(setLayers)
        .catch((reason: Error) => { if (reason.name !== 'AbortError') setError(reason.message); })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    }, 100);
    return () => { window.clearTimeout(requestTimer); controller.abort(); };
  }, [run, meta, year, layer]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layers.length) return;
    const width = 800, height = 560, left = 104, top = 22, planeWidth = 610, planeDepth = 36, layerGap = 88, skew = 60, wave = 18;
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    const min = oxygenMin ?? 0, max = oxygenMax ?? 1, span = max - min || 1;
    for (let layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
      const data = layers[layerIndex], [rows, cols] = data.shape;
      const { rowIds, colIds } = regionGrid(data, region);
      if (!rowIds.length || !colIds.length) continue;
      const point = (ri: number, ci: number) => {
        const row = rowIds[ri], col = colIds[ci], value = data.values[row * cols + col];
        const norm = value == null ? 0.5 : (value - min) / span;
        const u = ci / Math.max(colIds.length - 1, 1), v = ri / Math.max(rowIds.length - 1, 1);
        return { x: left + u * planeWidth + (v - 0.5) * skew, y: top + layerIndex * layerGap + v * planeDepth + (norm - 0.5) * wave, value };
      };
      ctx.lineWidth = 0.5;
      for (let ri = 0; ri < rowIds.length - 1; ri++) for (let ci = 0; ci < colIds.length - 1; ci++) {
        const corners = [point(ri, ci), point(ri, ci + 1), point(ri + 1, ci + 1), point(ri + 1, ci)];
        const present = corners.map(({ value }) => value).filter((value): value is number => value != null);
        if (!present.length) continue;
        const mean = present.reduce((sum, value) => sum + value, 0) / present.length;
        ctx.beginPath(); ctx.moveTo(corners[0].x, corners[0].y); corners.slice(1).forEach(({ x, y }) => ctx.lineTo(x, y)); ctx.closePath();
        ctx.fillStyle = color(mean, min, max, layer); ctx.globalAlpha = 0.68; ctx.fill();
        ctx.globalAlpha = 0.34; ctx.strokeStyle = 'rgba(38,87,105,.58)'; ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.fillStyle = '#557b8a'; ctx.font = '600 12px DM Sans, sans-serif';
      ctx.fillText(`${data.depth_m.toLocaleString()} m`, 18, top + layerIndex * layerGap + planeDepth / 2 + 4);
    }
    ctx.globalAlpha = 1;
  }, [layers, region, oxygenMin, oxygenMax, layer]);

  const valid = regionValues.length > 0;

  return <article className="glass-card regional-panel">
    <div className="regional-head"><div><p className="section-kicker">REGIONAL DEPTH STRUCTURE</p><h2>Layered {layerLabel(layer).toLowerCase()} field</h2>
      <p>Six depth surfaces across the selected ocean region. Choose a layer and year to explore its structure.</p></div>
      <div className="regional-selects"><label className="regional-select"><span>Layer to visualize</span><select value={layer} onChange={(event) => onLayerChange(event.target.value)}>
        {meta.layers.map((item) => <option key={item} value={item}>{layerLabel(item)}</option>)}
      </select></label><label className="regional-select"><span>Ocean region</span><select value={regionId} onChange={(event) => onRegionChange(event.target.value)}>
        {OCEAN_REGIONS.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label></div>
    </div>
    {error ? <div className="chart-empty">{error}</div> : loading && !layers.length ? <div className="chart-empty">Loading {region.name} {layer} layers for {year}…</div> : !valid ? <div className="chart-empty">No valid {layer} cells are available in this region.</div> : <div className="regional-visual-layout"><div className="regional-layer-canvas"><canvas ref={canvasRef} aria-label={`${region.name} ${layer} layers at ${year}`}/></div><aside className="oxygen-scale" aria-label={`${layer} intensity, ${meta.oxygen_units}`}><span className="oxygen-scale__title">{layer === 'error' ? 'ERROR' : layer === 'uncertainty' ? 'UNCERTAINTY' : layer === 'observed' ? 'OBSERVED' : 'OXYGEN'}</span><b>{oxygenMax?.toFixed(3) ?? '—'}</b><i className="oxygen-scale__bar" style={{ background: layer === 'error' ? 'linear-gradient(to top,#427db6,#e8f2ef,#e07064)' : layer === 'uncertainty' ? 'linear-gradient(to top,#d3e0f4,#705dae)' : layer === 'observed' ? 'linear-gradient(to top,#b8c9c8,#16857e)' : layer === 'density' ? 'linear-gradient(to top,#b8c9c8,#2b8097)' : undefined }}/><b>{oxygenMin?.toFixed(3) ?? '—'}</b><small>{meta.oxygen_units}</small></aside></div>}
    <div className="regional-footer"><label className="regional-year-control"><span className="regional-year-readout"><span>YEAR</span><b>{year}</b></span><input className="glass-year-slider" type="range" min={0} max={Math.max(meta.years.length - 1, 0)} step={1} value={Math.max(meta.years.indexOf(year), 0)} onChange={(event) => onYearChange(meta.years[Number(event.target.value)])}/><span className="regional-year-endpoints"><i>{meta.years[0]}</i><i>{meta.years[meta.years.length - 1]}</i></span></label></div>
  </article>;
}

function ModelBars({ rows, dataset }: { rows: CompRow[]; dataset: string }) {
  const items = rows.filter((row) => row.dataset === dataset && row.rmse != null && Number.isFinite(row.rmse));
  if (!items.length) return <div className="chart-empty">No model comparison metrics are available.</div>;
  const max = Math.max(...items.map((row) => row.rmse ?? 0)) || 1;
  return <div className="model-bars">{items.map((row, i) => <div className="model-bar" key={`${row.model}-${row.feature_set}-${i}`}>
    <div className="model-bar__label"><span>{row.model}{row.feature_set ? ` · ${row.feature_set}` : ''}{row.scope ? ` · ${row.scope}` : ''}</span><b>{row.rmse?.toFixed(3)}</b></div>
    <div className="model-bar__track"><i style={{ width: `${Math.max(4, (row.rmse ?? 0) / max * 100)}%` }}/></div>
  </div>)}<p className="chart-note">Lower RMSE means a closer match for the reported evaluation scope.</p></div>;
}

export default function App() {
  const [runs, setRuns] = useState<string[]>([]);
  const [run, setRun] = useState('');
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [layer, setLayer] = useState('prediction');
  const [regionId, setRegionId] = useState(OCEAN_REGIONS[0].id);
  const [year, setYear] = useState(0);
  const [depth, setDepth] = useState(0);
  const [maps, setMaps] = useState<Record<string, MapData>>({});
  const [rows, setRows] = useState<CompRow[]>([]);
  const [selected, setSelected] = useState<PickedCell | null>(null);
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [trend, setTrend] = useState<TimeSeriesData | null>(null);
  const [err, setErr] = useState('');
  const [loadingMaps, setLoadingMaps] = useState(false);
  const [analysisOpen, setAnalysisOpen] = useState(false);

  useEffect(() => {
    if (!analysisOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setAnalysisOpen(false); };
    document.addEventListener('keydown', closeOnEscape);
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', closeOnEscape); document.body.style.overflow = ''; };
  }, [analysisOpen]);

  useEffect(() => {
    api.runs().then((result) => { setRuns(result.runs); setRun(result.runs[0] ?? ''); }).catch((error: Error) => setErr(error.message));
    api.comparison().then((result) => setRows(result.rows)).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!run) return;
    setMeta(null); setMaps({});
    api.metadata(run).then((next) => {
      setMeta(next); setYear(next.years[0]); setLayer(next.layers.includes('prediction') ? 'prediction' : next.layers[0]);
      setSelected(null); setProfile(null); setTrend(null);
    }).catch((error: Error) => setErr(error.message));
  }, [run]);
  useEffect(() => {
    if (!run || !meta) return;
    let active = true; setLoadingMaps(true);
    const requested = [...new Set(['truth', 'prediction', 'error', layer])];
    Promise.all(requested.map((name) => api.map(run, name, year, depth, 1))).then((results) => {
      if (!active) return;
      const next: Record<string, MapData> = Object.fromEntries(results.map((result) => [result.layer, result]));
      const reference = next.truth, reconstruction = next.prediction;
      if (reference && reconstruction) {
        const sharedMin = Math.min(reference.min ?? 0, reconstruction.min ?? 0);
        const sharedMax = Math.max(reference.max ?? 1, reconstruction.max ?? 1);
        next.truth = { ...reference, min: sharedMin, max: sharedMax };
        next.prediction = { ...reconstruction, min: sharedMin, max: sharedMax };
      }
      setMaps(next); setErr('');
    }).catch((error: Error) => { if (active) setErr(error.message); }).finally(() => { if (active) setLoadingMaps(false); });
    return () => { active = false; };
  }, [run, meta, layer, year, depth]);
  useEffect(() => {
    if (!run || !meta || !selected) return;
    api.profile(run, year, selected.latIdx, selected.lonIdx).then(setProfile).catch((error: Error) => setErr(error.message));
    api.timeseries(run, depth, selected.latIdx, selected.lonIdx).then(setTrend).catch((error: Error) => setErr(error.message));
  }, [run, year, depth, selected, meta]);

  const depthM = meta?.depth_labels_m[depth] ?? 0;
  const rawSelectedLayerMap = maps[layer];
  const selectedLayerMap = rawSelectedLayerMap?.year === year && rawSelectedLayerMap.depth_m === depthM ? rawSelectedLayerMap : undefined;
  const map = selectedLayerMap;
  const datasetRows = rows.filter((row) => row.dataset === meta?.dataset);
  const selectCell = (cell: PickedCell) => { setSelected(cell); setProfile(null); setTrend(null); setErr(''); };
  const runLabel = run.replaceAll('_', ' ');

  return <div className="explorer-page">
    <header className="explorer-topbar"><a className="explorer-brand" href="#/">AQUA<span>TESSERACT</span></a>
      <div className="explorer-nav"><a href="#/">Landing</a><a className="explorer-nav__globe" href="#/globe">Open 3D Globe <span>↗</span></a></div>
    </header>
    <main>
      <section className="explorer-hero"><div><p className="explorer-kicker">OCEANVERSE / FIELD ANALYSIS</p><h1>Reconstruction Explorer</h1>
        <p>Compare the reference, reconstruction, and error. Select a cell to inspect its depth profile and time trend.</p></div>
        <div className="hero-status"><span className="status-dot"/>Simulation-backed results</div>
      </section>
      <div className="science-note"><b>Simulation data</b><span>Precomputed model results, not new ocean measurements.</span></div>
      {err && <p className="err">{err}</p>}
      {!runs.length && !err && <p className="empty-state">No reconstruction runs are available yet. Add a validated run folder to results.</p>}
      {meta && <>
        <section className="controls glass-card" aria-label="Map controls">
          <label className="control-field control-field--run"><span>Model run</span><select value={run} onChange={(event) => setRun(event.target.value)}>{runs.map((item) => <option key={item} value={item}>{item.replaceAll('_',' ')}</option>)}</select></label>
          <label className="control-field"><span>Layer</span><select value={layer} onChange={(event) => setLayer(event.target.value)}>{meta.layers.map((item) => <option key={item} value={item}>{layerLabel(item)}</option>)}</select></label>
          <label className="control-field control-field--slider"><span>Year <b>{year}</b></span><input type="range" min={meta.years[0]} max={meta.years[meta.years.length-1]} value={year} onChange={(event) => setYear(Number(event.target.value))}/></label>
          <label className="control-field control-field--slider"><span>Depth <b>{depthM} m</b></span><input type="range" min={0} max={meta.depth_labels_m.length-1} value={depth} onChange={(event) => setDepth(Number(event.target.value))}/></label>
        </section>
        <section className="map-layout">
          <article className="glass-card map-panel"><div className="section-title"><div><p className="section-kicker">SELECTED FIELD</p><h2>{layerLabel(layer)}</h2></div><span className="map-caption">{meta.dataset} / {year} / {depthM} m</span></div>
            {map ? <MapCanvas data={map} selected={selected} onSelect={selectCell}/> : <div className="map-placeholder">{loadingMaps ? 'Loading ocean field...' : 'No map available for this run.'}</div>}
            {map && <Legend data={map}/>}<div className="map-footer"><span>Click an ocean cell to inspect its values.</span><span>{map?.shape[0]} × {map?.shape[1]} grid</span></div>
            {selected && <div className="selected-cell"><span className="status-dot"/>Selected: {selected.latitude.toFixed(1)}°, {selected.longitude.toFixed(1)}°</div>}
          </article>
          <aside className="glass-card context-panel"><p className="section-kicker">READING THE MAP</p><h2>One slice at a time</h2>
            <p>Each cell is one ocean location for the selected year and depth. Reference and reconstruction use the same color scale.</p>
            <div className="metric-pill"><span>RUN</span><b>{runLabel}</b></div><div className="metric-pill"><span>GRID</span><b>{meta.shape?.[2]} × {meta.shape?.[3]}</b></div><div className="metric-pill"><span>UNITS</span><b>{meta.oxygen_units}</b></div>
            <div className="no-data-key"><i/>Land or no valid ocean value</div>
          </aside>
        </section>
        <section className="analysis-section"><div className="section-heading"><div><p className="section-kicker">LAYER COMPARISON</p><h2>Same slice, three views</h2></div><p>Linked to the selected run, year, and depth.</p></div>
          <div className="comparison-grid">{[
            { title: 'Reference', subtitle: 'Simulation truth field', data: maps.truth },
            { title: 'Reconstruction', subtitle: 'Model-predicted oxygen', data: maps.prediction },
            { title: 'Error', subtitle: 'Prediction minus reference', data: maps.error },
          ].map((item) => <article className="glass-card comparison-card" key={item.title}><div className="comparison-card__head"><div><h3>{item.title}</h3><p>{item.subtitle}</p></div></div>
            {item.data ? <MapCanvas data={item.data} compact onSelect={selectCell}/> : <div className="comparison-placeholder">Loading field...</div>}{item.data && <Legend data={item.data}/>}
          </article>)}</div>
        </section>
        <section className="analysis-section">
          <RegionalLayerGraph run={run} meta={meta} year={year} onYearChange={setYear} layer={layer} onLayerChange={setLayer} regionId={regionId} onRegionChange={setRegionId} />
        </section>
        <section className="analysis-section">{selectedLayerMap
          ? <LayerDeepDive data={selectedLayerMap} year={year} depthM={depthM} region={OCEAN_REGIONS.find((item) => item.id === regionId) ?? OCEAN_REGIONS[0]} onExpand={() => setAnalysisOpen(true)}/>
          : <article className="glass-card layer-insight-panel"><div className="layer-insight-head"><div><p className="section-kicker">SELECTED LAYER ANALYSIS</p><h2>Loading {layerLabel(layer)} details</h2><p>{OCEAN_REGIONS.find((item) => item.id === regionId)?.name} · {year} · {depthM} m · {meta.oxygen_units}</p></div></div><div className="chart-empty">The selected layer slice is loading. Its regional metrics and charts will appear here.</div></article>}
        </section>
        <section className="analysis-section"><div className="section-heading"><div><p className="section-kicker">CELL ANALYSIS</p><h2>{selected ? `${selected.latitude.toFixed(1)}°, ${selected.longitude.toFixed(1)}°` : 'Explore a location'}</h2></div><p>Choose a cell to reveal its profile.</p></div>
          <div className="analysis-grid"><article className="glass-card chart-card"><div className="chart-card__head"><div><h3>Oxygen by depth</h3><p>Reference and reconstruction at the selected cell</p></div></div><DepthProfile data={profile}/></article>
            <article className="glass-card chart-card"><div className="chart-card__head"><div><h3>Change through time</h3><p>{depthM} m at the selected cell</p></div></div><TimeTrend data={trend}/></article>
            <article className="glass-card chart-card"><div className="chart-card__head"><div><h3>Model RMSE</h3><p>Lower values indicate a closer reconstruction</p></div></div><ModelBars rows={datasetRows} dataset={meta.dataset}/></article>
          </div>
        </section>
        <section className="analysis-section"><div className="section-heading"><div><p className="section-kicker">RUN DETAILS</p><h2>Model comparison</h2></div><p>Metrics reported by the shared evaluator.</p></div>
          <div className="glass-card table-wrap"><table><thead><tr><th>Source</th><th>Dataset</th><th>Split</th><th>Model</th><th>Features</th><th>RMSE</th><th>R²</th><th>Scope</th></tr></thead>
            <tbody>{datasetRows.map((row, i) => <tr key={`${row.model}-${row.feature_set}-${i}`}><td>{row.source}</td><td>{row.dataset}</td><td>{row.split}</td><td>{row.model}</td><td>{row.feature_set ?? '—'}</td><td>{row.rmse?.toFixed(3) ?? '—'}</td><td>{row.r2?.toFixed(3) ?? '—'}</td><td>{row.scope}</td></tr>)}</tbody>
          </table></div>
        </section>
      </>}
    </main>
    {analysisOpen && selectedLayerMap && <div className="analysis-overlay" role="dialog" aria-modal="true" aria-label="Selected layer analysis" onMouseDown={(event) => { if (event.target === event.currentTarget) setAnalysisOpen(false); }}>
      <LayerDeepDive data={selectedLayerMap} year={year} depthM={depthM} region={OCEAN_REGIONS.find((item) => item.id === regionId) ?? OCEAN_REGIONS[0]} expanded onClose={() => setAnalysisOpen(false)}/>
    </div>}
  </div>;
}
