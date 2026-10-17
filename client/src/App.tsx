import { useEffect, useRef, useState } from 'react';
import { api, CompRow, MapData, Metadata } from './api';

function color(t: number): string {
  // simple blue -> yellow ramp
  const r = Math.round(40 + 215 * t), g = Math.round(70 + 170 * t), b = Math.round(160 - 120 * t);
  return `rgb(${r},${g},${b})`;
}

function MapCanvas({ data }: { data: MapData }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const [rows, cols] = data.shape, px = 6;
    cv.width = cols * px; cv.height = rows * px;
    const ctx = cv.getContext('2d')!;
    const lo = data.min ?? 0, span = (data.max ?? 1) - lo || 1;
    data.values.forEach((v, i) => {
      const r = Math.floor(i / cols), c = i % cols;
      ctx.fillStyle = v === null ? '#2b2b2b' : color((v - lo) / span); // land = dark grey
      ctx.fillRect(c * px, (rows - 1 - r) * px, px, px); // north up (lat index 0 = south, TO VERIFY with M3)
    });
  }, [data]);
  return <canvas ref={ref} className="map" />;
}

export default function App() {
  const [runs, setRuns] = useState<string[]>([]);
  const [run, setRun] = useState('');
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [layer, setLayer] = useState('prediction');
  const [year, setYear] = useState(0);
  const [depth, setDepth] = useState(0);
  const [map, setMap] = useState<MapData | null>(null);
  const [rows, setRows] = useState<CompRow[]>([]);
  const [err, setErr] = useState('');

  useEffect(() => {
    api.runs().then((r) => { setRuns(r.runs); setRun(r.runs[0] ?? ''); }).catch((e) => setErr(e.message));
    api.comparison().then((r) => setRows(r.rows)).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!run) return;
    api.metadata(run).then((m) => { setMeta(m); setYear(m.years[0]); }).catch((e) => setErr(e.message));
  }, [run]);
  useEffect(() => {
    if (!run || !meta) return;
    api.map(run, layer, year, depth, 1).then((m) => { setMap(m); setErr(''); }).catch((e) => setErr(e.message));
  }, [run, meta, layer, year, depth]);

  return (
    <>
      <div className="banner">
        ⚠ Evaluated on a <b>simulated</b> ocean (OceanVerse digital twin). Predictions are not new measurements.
        Precomputed results, not live inference.
      </div>
      <main>
        <h1>AquaTesseract · Reconstruction Explorer</h1>
        {err && <p className="err">{err}</p>}
        {!runs.length && !err && <p>No runs found. Run <code>make mock</code>.</p>}
        {meta && (
          <section className="controls">
            <label>Run
              <select value={run} onChange={(e) => setRun(e.target.value)}>{runs.map((r) => <option key={r}>{r}</option>)}</select>
            </label>
            <label>Layer
              <select value={layer} onChange={(e) => setLayer(e.target.value)}>{meta.layers.map((l) => <option key={l}>{l}</option>)}</select>
            </label>
            <label>Year {year}
              <input type="range" min={meta.years[0]} max={meta.years[meta.years.length - 1]} value={year} onChange={(e) => setYear(+e.target.value)} />
            </label>
            <label>Depth {meta.depth_labels_m[depth]} m
              <input type="range" min={0} max={meta.depth_labels_m.length - 1} value={depth} onChange={(e) => setDepth(+e.target.value)} />
            </label>
          </section>
        )}
        {map && <MapCanvas data={map} />}
        {map && <p className="legend">{map.layer}: {map.min?.toFixed(3)} → {map.max?.toFixed(3)} · units: {map.oxygen_units}</p>}
        <h2>Model comparison</h2>
        <table>
          <thead><tr><th>Source</th><th>Dataset</th><th>Split</th><th>Model</th><th>Features</th><th>RMSE</th><th>R²</th><th>Scope</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td><span className={`tag ${r.source}`}>{r.source}</span></td><td>{r.dataset}</td><td>{r.split}</td><td>{r.model}</td>
                <td>{r.feature_set ?? '—'}</td><td>{r.rmse?.toFixed(3) ?? '—'}</td><td>{r.r2?.toFixed(3) ?? '—'}</td><td>{r.scope}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </main>
    </>
  );
}
