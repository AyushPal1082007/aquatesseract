/* ================================================================
   GLOBE DASHBOARD — /#/globe route
   Layout per reference images: dominant 3D globe center, floating
   glass control panel right, big numeric stats, bottom summary
   strip, expandable analysis drawer. Dark navy / cyan scientific.
   ================================================================ */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createGlobeScene, GlobeSceneHandle } from './GlobeScene.js';
import {
  cellValueAt, findNearestOceanCell, getFrame, getPointProfile, RUNS,
} from './mockAdapter.js';
import {
  DEPTHS_M, DEMO_NOTICE, LayerId, LAYERS, PointProfile, RunSpec, SIMULATION_NOTICE, YEARS,
} from './types.js';
import './globe.css';

type Sel = { lat: number; lon: number } | null;

/* ─────────── tiny SVG chart components (no chart lib) ─────────── */

function Chart({ title, unit, series, height = 110, band }: {
  title: string; unit: string; height?: number;
  series: { label: string; color: string; values: number[]; dashed?: boolean }[];
  band?: { upper: number[]; lower: number[]; label: string };
}) {
  const xs = series[0]?.values.length ?? 0;
  if (!xs) return null;
  const all = [...series.flatMap((s) => s.values), ...(band ? [...band.upper, ...band.lower] : [])];
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.12 || 0.1;
  const y0 = lo - pad, y1 = hi + pad;
  const W = 320, H = height;
  const px = (i: number) => 34 + (i / (xs - 1)) * (W - 44);
  const py = (v: number) => H - 22 - ((v - y0) / (y1 - y0)) * (H - 38);
  const path = (vs: number[]) => vs.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
  return (
    <div className="gd-chart">
      <div className="gd-chart__title">{title}<span className="gd-chart__unit">{unit}</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="gd-chart__svg">
        <line x1={34} y1={H - 22} x2={W - 10} y2={H - 22} className="gd-chart__axis" />
        <line x1={34} y1={12} x2={34} y2={H - 22} className="gd-chart__axis" />
        {band && (
          <path
            d={`${band.upper.map((v, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')} ${band.lower.slice().reverse().map((v, i) => `L${px(xs - 1 - i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')} Z`}
            className="gd-chart__band"
          />
        )}
        {series.map((s) => (
          <path key={s.label} d={path(s.values)} fill="none"
            className={`gd-chart__line${s.dashed ? ' gd-chart__line--dashed' : ''}`}
            style={{ stroke: s.color }} />
        ))}
        <text x={36} y={20} className="gd-chart__tlabel">{y1.toFixed(2)}</text>
        <text x={36} y={H - 24} className="gd-chart__tlabel">{y0.toFixed(2)}</text>
      </svg>
      <div className="gd-chart__legend">
        {series.map((s) => (
          <span key={s.label} className="gd-chart__lg">
            <i style={{ background: s.color, opacity: s.dashed ? 0.55 : 1 }} />{s.label}
          </span>
        ))}
        {band && <span className="gd-chart__lg"><i className="gd-chart__lg-band" />{band.label}</span>}
      </div>
    </div>
  );
}

/* ─────────── main component ─────────── */

export default function GlobeDashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<GlobeSceneHandle | null>(null);

  const [runIdx, setRunIdx] = useState(0);
  const [layer, setLayer] = useState<LayerId>('prediction');
  const [year, setYear] = useState<number>(YEARS[0]);
  const [depthIdx, setDepthIdx] = useState(2);
  const [opacity, setOpacity] = useState(0.4);
  const [speed, setSpeed] = useState(1);
  const [autoRotate, setAutoRotate] = useState(true);
  const [sel, setSel] = useState<Sel>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [searchNote, setSearchNote] = useState('');

  const run: RunSpec = RUNS[runIdx];
  const depthM = DEPTHS_M[depthIdx];
  const layerInfo = LAYERS.find((l) => l.id === layer)!;

  const frame = useMemo(
    () => getFrame(run, layer, year, depthM),
    [run, layer, year, depthIdx, depthM],
  );

  const profile: PointProfile | null = useMemo(
    () => (sel ? getPointProfile(run, sel.lat, sel.lon, depthM) : null),
    [run, sel, depthM],
  );

  const selectedCell = sel ? cellValueAt(frame, sel.lat, sel.lon) : null;

  /* ---------- init three.js scene ---------- */
  useEffect(() => {
    if (!canvasRef.current) return;
    const handle = createGlobeScene({
      canvas: canvasRef.current,
      onPick: (lat, lon) => {
        const near = findNearestOceanCell(lat, lon);
        setSel(near);
        setDrawerOpen(true);
      },
    });
    sceneRef.current = handle;
    return () => { handle.dispose(); sceneRef.current = null; };
  }, []);

  useEffect(() => { sceneRef.current?.setFrame(frame, opacity); }, [frame, opacity]);
  useEffect(() => { sceneRef.current?.setAutoRotate(autoRotate); }, [autoRotate]);
  useEffect(() => { sceneRef.current?.setSpeed(speed); }, [speed]);
  useEffect(() => { sceneRef.current?.setSelected(sel?.lat ?? null, sel?.lon ?? null); }, [sel]);

  /* ---------- search ---------- */
  const doSearch = useCallback(() => {
    const m = query.trim().match(/^(-?\d+(?:\.\d+)?)[, ]\s*(-?\d+(?:\.\d+)?)$/);
    if (!m) { setSearchNote('Use "lat, lon" e.g. "22, -140"'); return; }
    const near = findNearestOceanCell(+m[1], +m[2]);
    setSel(near);
    setDrawerOpen(true);
    sceneRef.current?.focusPoint(near.lat, near.lon);
    setSearchNote(`→ cell ${near.lat.toFixed(0)}°, ${near.lon.toFixed(0)}°`);
  }, [query]);

  const resetView = () => {
    sceneRef.current?.resetView();
    setSel(null); setDrawerOpen(false); setSearchNote('');
  };

  const observedCount = frame.cells.filter((c) => c.observed).length;
  const rmse = profile?.rmse ?? 0.18;

  return (
    <div className="gd-root">
      {/* ---------- persistent transparency notices ---------- */}
      <div className="gd-notices">
        <p className="gd-notice gd-notice--sim">⚠ {SIMULATION_NOTICE}</p>
        <p className="gd-notice gd-notice--demo">● {DEMO_NOTICE}</p>
      </div>

      {/* ---------- top header ---------- */}
      <header className="gd-header">
        <div className="gd-header__brand">
          <span className="gd-logo">◈</span>
          <div>
            <h1 className="gd-title">AQUA<span>TESSERACT</span></h1>
            <p className="gd-subtitle">Simulation-Based Oxygen Reconstruction</p>
          </div>
        </div>
        <div className="gd-search">
          <input
            value={query}
            placeholder="Search coordinate or region — e.g. 22, -140"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && doSearch()}
          />
          <button onClick={doSearch}>LOCATE</button>
          {searchNote && <span className="gd-search__note">{searchNote}</span>}
        </div>
        <div className="gd-status">
          <a className="gd-nav-link" href="#/explorer">Reconstruction Explorer</a>
          <span className="gd-badge gd-badge--demo">DEMO MODE</span>
          <span className="gd-badge gd-badge--src">CESM2-OMIP1 · SIM</span>
        </div>
      </header>

      {/* ---------- main stage ---------- */}
      <div className="gd-stage">
        {/* globe */}
        <div className="gd-globewrap">
          <canvas ref={canvasRef} className="gd-globe" />
          {/* active layer label + legend */}
          <div className="gd-layerlabel">
            <span className="gd-layerlabel__name">{layerInfo.label}</span>
            <span className="gd-layerlabel__meta">
              {layerInfo.units ?? 'categorical'} · {layer === 'observations' ? 'obs markers' : `${frame.min.toFixed(2)} → ${frame.max.toFixed(2)}`}
            </span>
          </div>
          <div className="gd-legend">
            <div className="gd-legend__bar" />
            <div className="gd-legend__ends">
              <span>{layer === 'observations' ? 'observed' : frame.min.toFixed(2)}</span>
              <span>{layer === 'observations' ? '·' : frame.max.toFixed(2)}</span>
            </div>
          </div>
          {selectedCell && (
            <div className="gd-selchip">
              <i /> {sel!.lat.toFixed(0)}°, {sel!.lon.toFixed(0)}° · {selectedCell.observed ? 'observed cell' : 'never-observed simulated cell'}
            </div>
          )}
        </div>

        {/* stat counters (reference-2 style) */}
        <div className="gd-stats">
          <div className="gd-stat">
            <span className="gd-stat__num">{observedCount}</span>
            <span className="gd-stat__lbl">Observed cells<br />(sparse, simulated)</span>
          </div>
          <div className="gd-stat">
            <span className="gd-stat__num">{frame.cells.length}</span>
            <span className="gd-stat__lbl">Simulated grid<br />cells on globe</span>
          </div>
          <div className="gd-stat">
            <span className="gd-stat__num">{rmse.toFixed(2)}</span>
            <span className="gd-stat__lbl">RMSE · mock<br /><em>not verified</em></span>
          </div>
        </div>

        {/* ---------- right control panel ---------- */}
        <aside className="gd-panel">
          <div className="gd-panel__head">
            <span>INSTRUMENT CONTROL</span>
            <i className="gd-beacon" />
          </div>

          <label className="gd-field">
            <span>Model / Run</span>
            <select value={runIdx} onChange={(e) => setRunIdx(+e.target.value)}>
              {RUNS.map((r, i) => <option key={r.id} value={i}>{r.label} · s{r.seed}</option>)}
            </select>
          </label>

          <label className="gd-field">
            <span>Data layer</span>
            <select value={layer} onChange={(e) => setLayer(e.target.value as LayerId)}>
              {LAYERS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
            </select>
            {layer === 'uncertainty' && !run.hasUncertainty && (
              <em className="gd-warn">Uncertainty unavailable — not model-provided for baseline runs.</em>
            )}
          </label>

          <label className="gd-field">
            <span>Year <b>{year}</b></span>
            <input type="range" min={YEARS[0]} max={YEARS[YEARS.length - 1]} step={5}
              value={year} onChange={(e) => setYear(+e.target.value)} />
          </label>

          <label className="gd-field">
            <span>Depth <b>{depthM} m</b></span>
            <input type="range" min={0} max={DEPTHS_M.length - 1} step={1}
              value={depthIdx} onChange={(e) => setDepthIdx(+e.target.value)} />
          </label>

          <label className="gd-field">
            <span>Layer opacity <b>{Math.round(opacity * 100)}%</b></span>
            <input type="range" min={0.05} max={1} step={0.01}
              value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
          </label>

          <label className="gd-field">
            <span>Animation speed <b>{speed.toFixed(1)}×</b></span>
            <input type="range" min={0.2} max={3} step={0.1}
              value={speed} onChange={(e) => setSpeed(+e.target.value)} />
          </label>

          <div className="gd-toggles">
            <button
              className={`gd-toggle ${autoRotate ? 'gd-toggle--on' : ''}`}
              onClick={() => setAutoRotate((v) => !v)}>
              AUTO-ROTATE {autoRotate ? 'ON' : 'OFF'}
            </button>
            <button className="gd-toggle" onClick={resetView}>RESET VIEW</button>
          </div>

          <div className="gd-panel__foot">
            <p>Orbit: drag · Zoom: scroll</p>
            <p>Click a glowing point to inspect</p>
          </div>
        </aside>

        {/* ---------- bottom summary strip ---------- */}
        <div className="gd-strip">
          <div className="gd-strip__item">
            <span>SELECTED O₂</span>
            <b>{selectedCell && Number.isFinite(selectedCell.value) && layer !== 'observations'
              ? `${selectedCell.value.toFixed(2)} mL/L` : '—'}</b>
          </div>
          <div className="gd-strip__item">
            <span>UNCERTAINTY</span>
            <b>{run.hasUncertainty && layer === 'uncertainty'
              ? (selectedCell && Number.isFinite(selectedCell.value) ? `±${selectedCell.value.toFixed(2)}` : '—')
              : run.hasUncertainty ? 'model-estimated' : 'unavailable'}</b>
          </div>
          <div className="gd-strip__item">
            <span>STATUS</span>
            <b>{selectedCell
              ? selectedCell.observed ? 'observed' : 'reconstructed'
              : 'no selection'}</b>
          </div>
          <div className="gd-strip__item">
            <span>RMSE (run)</span>
            <b className="gd-dim">{rmse.toFixed(3)} <em>mock</em></b>
          </div>
          <div className="gd-strip__item">
            <span>YEAR</span><b>{year}</b>
          </div>
          <div className="gd-strip__item">
            <span>DEPTH</span><b>{depthM} m</b>
          </div>
        </div>

        {/* ---------- expandable analysis drawer ---------- */}
        <section className={`gd-drawer ${drawerOpen ? 'gd-drawer--open' : ''}`}>
          <button className="gd-drawer__tab" onClick={() => setDrawerOpen((v) => !v)}>
            <span>ANALYSIS</span>
            <i className={`gd-chevron ${drawerOpen ? 'gd-chevron--up' : ''}`} />
          </button>
          <div className="gd-drawer__body">
            {!sel || !profile ? (
              <p className="gd-drawer__empty">
                Select a point on the globe (or search a coordinate) to load its analysis.
              </p>
            ) : (
              <>
                <div className="gd-drawer__head">
                  <h3>Point {profile.lat.toFixed(0)}°, {profile.lon.toFixed(0)}°</h3>
                  <span className="gd-tag">
                    {selectedCell?.observed ? 'observed cell' : 'never-observed simulated cell'}
                  </span>
                  <span className="gd-tag gd-tag--model">{run.label}</span>
                  <span className="gd-tag gd-tag--mock">mock data</span>
                </div>
                <div className="gd-drawer__grid">
                  <Chart title="Depth profile" unit="mL/L"
                    series={[
                      { label: 'hybrid reconstruction', color: '#00e5ff', values: profile.prediction },
                      { label: 'simulation reference', color: '#9ca3af', values: profile.truth, dashed: true },
                    ]}
                    band={profile.uncertainty ? {
                      label: 'model-estimated ±90%',
                      upper: profile.prediction.map((p, i) => p + 1.645 * profile.uncertainty![i]),
                      lower: profile.prediction.map((p, i) => p - 1.645 * profile.uncertainty![i]),
                    } : undefined} />
                  <Chart title={`Time series @ ${depthM} m`} unit="mL/L"
                    series={[{ label: 'reconstruction', color: '#38bdf8', values: profile.timeSeries }]} />
                  <Chart title="Reconstruction error" unit="mL/L"
                    series={[{ label: '|pred − reference|', color: '#f59e0b', values: profile.errorSeries }]} />
                  <Chart title="Model-estimated uncertainty" unit="mL/L"
                    series={profile.uncertainty
                      ? [{ label: '±σ', color: '#c084fc', values: profile.uncertainty }]
                      : [{ label: 'unavailable — baseline run', color: '#475569', values: profile.prediction.map(() => 0), dashed: true }]} />
                </div>
                <div className="gd-provenance">
                  <h4>Provenance</h4>
                  <ul>
                    <li><b>Dataset:</b> OceanVerse CESM2-OMIP1 (simulation) — demo fixture values, not verified artifacts.</li>
                    <li><b>Model:</b> {run.label} · seed {run.seed}</li>
                    <li><b>Uncertainty:</b> {run.hasUncertainty ? 'model-estimated (hybrid run)' : 'unavailable — baseline runs do not provide scientifically validated uncertainty.'}</li>
                    <li><b>Units:</b> dissolved oxygen, mL/L. NaN = land / below seafloor.</li>
                    <li><b>Caveat:</b> not live real-ocean measurements; real-ocean accuracy not established.</li>
                  </ul>
                </div>
              </>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
