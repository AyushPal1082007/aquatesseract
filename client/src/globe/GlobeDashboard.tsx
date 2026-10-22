import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { mlApi } from '../api';
import earthTextureUrl from '../data/earth.png';
import { createGlobeScene, GlobeSceneHandle } from './GlobeScene.js';
import {
  cellValueAt,
  getFrame,
  getPointProfile,
  loadMetadata,
  loadRuns,
  resolvePoint,
} from './mockAdapter.js';
import {
  DEMO_NOTICE,
  GlobeFrame,
  LAYERS,
  LayerId,
  PointProfile,
  RunSpec,
  SIMULATION_NOTICE,
} from './types.js';
import './globe.css';

type Sel = { lat: number; lon: number } | null;

function displayValues(values: number[]): number[] {
  const finite = values.filter(Number.isFinite);
  const fallback = finite[0] ?? 0;
  return values.map((value) => Number.isFinite(value) ? value : fallback);
}

function normalizeLongitude(value: number): number {
  return value > 180 ? value - 360 : value;
}

function Chart({ title, unit, series, height = 110, band }: {
  title: string;
  unit: string;
  height?: number;
  series: { label: string; color: string; values: number[]; dashed?: boolean }[];
  band?: { upper: number[]; lower: number[]; label: string };
}) {
  const xs = series[0]?.values.length ?? 0;
  if (!xs) return null;
  const all = [...series.flatMap((item) => item.values), ...(band ? [...band.upper, ...band.lower] : [])].filter(Number.isFinite);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.12 || 0.1;
  const y0 = lo - pad;
  const y1 = hi + pad;
  const W = 320;
  const H = height;
  const px = (index: number) => 34 + (index / Math.max(xs - 1, 1)) * (W - 44);
  const py = (value: number) => H - 22 - ((value - y0) / (y1 - y0)) * (H - 38);
  const path = (values: number[]) => values.map((value, index) => `${index ? 'L' : 'M'}${px(index).toFixed(1)},${py(value).toFixed(1)}`).join(' ');
  return (
    <div className="gd-chart">
      <div className="gd-chart__title">{title}<span className="gd-chart__unit">{unit}</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="gd-chart__svg">
        <line x1={34} y1={H - 22} x2={W - 10} y2={H - 22} className="gd-chart__axis" />
        <line x1={34} y1={12} x2={34} y2={H - 22} className="gd-chart__axis" />
        {band && (
          <path
            d={`${band.upper.map((value, index) => `${index ? 'L' : 'M'}${px(index).toFixed(1)},${py(value).toFixed(1)}`).join(' ')} ${band.lower.slice().reverse().map((value, index) => `L${px(xs - 1 - index).toFixed(1)},${py(value).toFixed(1)}`).join(' ')} Z`}
            className="gd-chart__band"
          />
        )}
        {series.map((item) => (
          <path key={item.label} d={path(item.values)} fill="none"
            className={`gd-chart__line${item.dashed ? ' gd-chart__line--dashed' : ''}`}
            style={{ stroke: item.color }} />
        ))}
        <text x={36} y={20} className="gd-chart__tlabel">{y1.toFixed(2)}</text>
        <text x={36} y={H - 24} className="gd-chart__tlabel">{y0.toFixed(2)}</text>
      </svg>
      <div className="gd-chart__legend">
        {series.map((item) => (
          <span key={item.label} className="gd-chart__lg">
            <i style={{ background: item.color, opacity: item.dashed ? 0.55 : 1 }} />{item.label}
          </span>
        ))}
        {band && <span className="gd-chart__lg"><i className="gd-chart__lg-band" />{band.label}</span>}
      </div>
    </div>
  );
}

export default function GlobeDashboard() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<GlobeSceneHandle | null>(null);
  const fallbackRef = useRef<HTMLDivElement>(null);
  const fallbackViewRef = useRef({ pitch: 0, yaw: 0, scale: 1, dragging: false, moved: false, x: 0, y: 0 });
  const selectPointRef = useRef<(latitude: number, longitude: number) => void>(() => undefined);
  const [runs, setRuns] = useState<RunSpec[]>([]);
  const [runIdx, setRunIdx] = useState(0);
  const [metadata, setMetadata] = useState<Awaited<ReturnType<typeof loadMetadata>> | null>(null);
  const [metrics, setMetrics] = useState<Awaited<ReturnType<typeof mlApi.metrics>> | null>(null);
  const [layer, setLayer] = useState<LayerId>('reconstruction');
  const [year, setYear] = useState(0);
  const [depthIdx, setDepthIdx] = useState(0);
  const [opacity, setOpacity] = useState(0.4);
  const [speed, setSpeed] = useState(1);
  const [autoRotate, setAutoRotate] = useState(true);
  const [sel, setSel] = useState<Sel>(null);
  const [profile, setProfile] = useState<PointProfile | null>(null);
  const [frame, setFrame] = useState<GlobeFrame | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [sceneError, setSceneError] = useState('');

  const run = runs[runIdx];
  const years = metadata?.years ?? [];
  const depths = metadata?.coordinates.depthValuesM ?? [];
  const depthM = depths[depthIdx] ?? 0;
  const availableLayers = (metadata?.availableLayers ?? ['reconstruction', 'reference', 'error', 'observations']) as LayerId[];
  const availableLayersKey = availableLayers.join(',');
  const layerInfo = LAYERS.find((item) => item.id === layer) ?? LAYERS[0];
  const selectedCell = frame && sel ? cellValueAt(frame, sel.lat, sel.lon) : null;
  const rmse = metrics?.never_observed?.rmse ?? metrics?.all?.rmse ?? profile?.rmse ?? null;

  const paintFallback = useCallback(() => {
    const element = fallbackRef.current;
    if (!element) return;
    const view = fallbackViewRef.current;
    element.style.setProperty('--fallback-pitch', `${view.pitch}deg`);
    element.style.setProperty('--fallback-yaw', `${view.yaw}deg`);
    element.style.setProperty('--fallback-scale', String(view.scale));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    loadRuns(controller.signal)
      .then((items) => {
        setRuns(items);
        setRunIdx(0);
        if (!items.length) setError('No ML runs are available. Generate or mount a results run first.');
      })
      .catch((reason: Error) => {
        if (reason.name !== 'AbortError') setError(reason.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    setMetadata(null);
    setMetrics(null);
    setFrame(null);
    setProfile(null);
    setSel(null);
    loadMetadata(run.id, controller.signal)
      .then((nextMetadata) => {
        setMetadata(nextMetadata);
        setYear(nextMetadata.years[0]);
        setDepthIdx(0);
        const preferred = nextMetadata.availableLayers.includes('reconstruction')
          ? 'reconstruction'
          : nextMetadata.availableLayers[0];
        if (preferred) setLayer(preferred as LayerId);
        setError('');
        // Metrics are informative; a missing metrics route must not prevent
        // the actual ML field from loading and rendering on the globe.
        mlApi.metrics(run.id, controller.signal)
          .then(setMetrics)
          .catch(() => setMetrics(null));
      })
      .catch((reason: Error) => {
        if (reason.name !== 'AbortError') setError(reason.message);
      });
    return () => controller.abort();
  }, [run]);

  useEffect(() => {
    if (!run || !metadata || !availableLayers.includes(layer) || !years.length) return;
    const controller = new AbortController();
    setFrame(null);
    getFrame(run, layer, year, depthM, controller.signal)
      .then(setFrame)
      .catch((reason: Error) => {
        if (reason.name !== 'AbortError') setError(reason.message);
      });
    return () => controller.abort();
  }, [run, metadata, layer, year, depthM, availableLayersKey, years.length]);

  useEffect(() => {
    if (!run || !metadata || !sel) {
      setProfile(null);
      return;
    }
    const controller = new AbortController();
    getPointProfile(run, sel.lat, sel.lon, year, depthM, controller.signal)
      .then(setProfile)
      .catch((reason: Error) => {
        if (reason.name !== 'AbortError') setNote(reason.message);
      });
    return () => controller.abort();
  }, [run, metadata, sel, year, depthM]);

  useEffect(() => {
    if (!canvasRef.current) return;
    try {
      const handle = createGlobeScene({
        canvas: canvasRef.current,
        onPick: (lat, lon) => { void selectPointRef.current(lat, lon); },
        onError: (message) => setSceneError(message),
      });
      sceneRef.current = handle;
      setSceneError('');
      return () => { handle.dispose(); sceneRef.current = null; };
    } catch (reason) {
      setSceneError(reason instanceof Error ? reason.message : 'WebGL could not initialize.');
      return undefined;
    }
  }, []);

  useEffect(() => {
    if (!sceneError) return;
    const element = fallbackRef.current;
    if (!element) return;
    const view = fallbackViewRef.current;
    const down = (event: PointerEvent) => {
      if (event.button !== 0) return;
      view.dragging = true;
      view.moved = false;
      view.x = event.clientX;
      view.y = event.clientY;
      element.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      if (!view.dragging) return;
      const dx = event.clientX - view.x;
      const dy = event.clientY - view.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) view.moved = true;
      view.yaw += dx * 0.45;
      view.pitch = Math.max(-55, Math.min(55, view.pitch + dy * 0.28));
      view.x = event.clientX;
      view.y = event.clientY;
      paintFallback();
    };
    const up = (event: PointerEvent) => {
      if (!view.dragging) return;
      view.dragging = false;
      if (!view.moved) {
        const rect = element.getBoundingClientRect();
        const radius = Math.min(rect.width, rect.height) / 2;
        const nx = (event.clientX - (rect.left + rect.width / 2)) / radius;
        const ny = (event.clientY - (rect.top + rect.height / 2)) / radius;
        if (nx * nx + ny * ny <= 1.05) {
          const latitude = Math.max(-89, Math.min(89, -ny * 90 - view.pitch * 0.35));
          const longitude = ((nx * 180 - view.yaw + 540) % 360) - 180;
          void selectPointRef.current(latitude, longitude);
        }
      }
    };
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      view.scale = Math.max(0.72, Math.min(1.85, view.scale - event.deltaY * 0.001));
      paintFallback();
    };
    element.addEventListener('pointerdown', down);
    element.addEventListener('pointermove', move);
    element.addEventListener('pointerup', up);
    element.addEventListener('pointercancel', up);
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('pointerdown', down);
      element.removeEventListener('pointermove', move);
      element.removeEventListener('pointerup', up);
      element.removeEventListener('pointercancel', up);
      element.removeEventListener('wheel', wheel);
    };
  }, [sceneError, paintFallback]);

  useEffect(() => {
    if (!sceneError || !autoRotate) return;
    const timer = window.setInterval(() => {
      fallbackViewRef.current.yaw += 0.7 * speed;
      paintFallback();
    }, 16);
    return () => window.clearInterval(timer);
  }, [sceneError, autoRotate, speed, paintFallback]);

  useEffect(() => {
    if (!frame || !sceneRef.current) return;
    try {
      sceneRef.current.setFrame(frame, opacity);
    } catch (reason) {
      setSceneError(reason instanceof Error ? reason.message : 'ML texture rendering failed.');
    }
  }, [frame, opacity]);
  useEffect(() => { sceneRef.current?.setAutoRotate(autoRotate); }, [autoRotate]);
  useEffect(() => { sceneRef.current?.setSpeed(speed); }, [speed]);
  useEffect(() => { sceneRef.current?.setSelected(sel?.lat ?? null, sel?.lon ?? null); }, [sel]);

  const selectPoint = useCallback(async (latitude: number, longitude: number) => {
    if (!run || !metadata || !years.length) return;
    try {
      const point = await resolvePoint(run, latitude, longitude, year, depthM);
      if (point.status === 'land' || point.status === 'invalid') {
        setNote('No valid ocean reconstruction exists at the nearest cell for this depth.');
        return;
      }
      const next = { lat: point.grid.latitude, lon: normalizeLongitude(point.grid.longitude) };
      setSel(next);
      setDrawerOpen(true);
      setNote(`Cell ${next.lat.toFixed(1)}°, ${next.lon.toFixed(1)}° · ${point.status.replace('_', '-')}`);
      sceneRef.current?.focusPoint(next.lat, next.lon);
      const fallback = fallbackViewRef.current;
      fallback.yaw = -next.lon;
      fallback.pitch = Math.max(-35, Math.min(35, next.lat * 0.35));
      fallback.scale = 1.35;
      paintFallback();
    } catch (reason) {
      setNote(reason instanceof Error ? reason.message : 'Unable to locate that coordinate.');
    }
  }, [run, metadata, years.length, year, depthM, paintFallback]);

  useEffect(() => { selectPointRef.current = selectPoint; }, [selectPoint]);

  const doSearch = useCallback(() => {
    const match = query.trim().match(/^(-?\d+(?:\.\d+)?)[, ]\s*(-?\d+(?:\.\d+)?)$/);
    if (!match) { setNote('Use "lat, lon" e.g. "22, -140"'); return; }
    void selectPoint(Number(match[1]), Number(match[2]));
  }, [query, selectPoint]);

  const resetView = () => {
    sceneRef.current?.resetView();
    fallbackViewRef.current.pitch = 0;
    fallbackViewRef.current.yaw = 0;
    fallbackViewRef.current.scale = 1;
    paintFallback();
    setSel(null);
    setProfile(null);
    setDrawerOpen(false);
    setNote('');
  };

  if (loading && !runs.length) return <div className="gd-root gd-root--loading">Loading ML runs…</div>;

  const observedCount = frame?.cells.filter((cell) => cell.ocean && cell.observed).length ?? 0;
  const oceanCellCount = frame?.cells.filter((cell) => cell.ocean).length ?? 0;
  const plotUnit = frame?.units ?? 'model units';
  const profilePrediction = profile ? displayValues(profile.prediction) : [];
  const profileTruth = profile ? displayValues(profile.truth) : [];
  const profileUncertainty = profile?.uncertainty ? displayValues(profile.uncertainty) : null;
  const timeSeries = profile ? displayValues(profile.timeSeries) : [];
  const errorSeries = profile ? displayValues(profile.errorSeries) : [];

  return (
    <div className="gd-root">
      <div className="gd-notices">
        <p className="gd-notice gd-notice--sim">⚠ {SIMULATION_NOTICE}</p>
        {run?.mock && <p className="gd-notice gd-notice--demo">● {DEMO_NOTICE}</p>}
      </div>

      <header className="gd-header">
        <div className="gd-header__brand">
          <span className="gd-logo">◈</span>
          <div><h1 className="gd-title">AQUA<span>TESSERACT</span></h1><p className="gd-subtitle">ML-backed simulation reconstruction</p></div>
        </div>
        <div className="gd-search">
          <input value={query} placeholder="Search coordinate — e.g. 22, -140" onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && doSearch()} />
          <button onClick={doSearch}>LOCATE</button>
          {note && <span className="gd-search__note">{note}</span>}
        </div>
        <div className="gd-status">
          <a className="gd-nav-link" href="#/explorer">← Back to Explorer</a>
          <span className="gd-badge gd-badge--src">{run ? `${run.dataset} · ${run.split}` : 'ML API'}</span>
          <span className={`gd-badge ${run?.mock ? 'gd-badge--demo' : ''}`}>{run?.mock ? 'MOCK FIXTURE' : 'READ-ONLY ML'}</span>
        </div>
      </header>

      {error && <div className="gd-error">{error}</div>}

      <div className="gd-stage">
        <div className={`gd-globewrap ${sceneError ? 'gd-globewrap--fallback' : ''}`}>
          {sceneError && <div ref={fallbackRef} className="gd-globe-fallback-visual" aria-label="Interactive Earth fallback">
            <img src={earthTextureUrl} alt="" draggable="false" />
          </div>}
          <canvas ref={canvasRef} className="gd-globe" />
          {sceneError && <div className="gd-globe-fallback"><strong>3D globe unavailable</strong><span>{sceneError}</span><small>Enable WebGL / hardware acceleration, then reload.</small></div>}
          {frame && <>
            <div className="gd-layerlabel"><span className="gd-layerlabel__name">{layerInfo.label}</span><span className="gd-layerlabel__meta">{plotUnit} · stride {frame.downsampleStride}</span></div>
            <div className="gd-legend"><div className="gd-legend__bar" /><div className="gd-legend__ends"><span>{frame.min.toFixed(2)}</span><span>{frame.max.toFixed(2)}</span></div></div>
          </>}
          {selectedCell && <div className="gd-selchip"><i /> {sel!.lat.toFixed(1)}°, {sel!.lon.toFixed(1)}° · {selectedCell.observed ? 'observed cell' : 'never-observed simulated cell'}</div>}
        </div>

        <div className="gd-stats">
          <div className="gd-stat"><span className="gd-stat__num">{observedCount}</span><span className="gd-stat__lbl">Observed cells<br />(sparse, simulated)</span></div>
          <div className="gd-stat"><span className="gd-stat__num">{oceanCellCount}</span><span className="gd-stat__lbl">Ocean cells<br />in ML slice</span></div>
          <div className="gd-stat"><span className="gd-stat__num">{rmse == null ? '—' : rmse.toFixed(3)}</span><span className="gd-stat__lbl">RMSE · {rmse == null ? 'unavailable' : 'shared evaluator'}</span></div>
        </div>

        <aside className="gd-panel">
          <div className="gd-panel__head"><span>INSTRUMENT CONTROL</span><i className="gd-beacon" /></div>
          <label className="gd-field"><span>Model / Run</span><select value={runIdx} onChange={(event) => setRunIdx(Number(event.target.value))}>{runs.map((item, index) => <option key={item.id} value={index}>{item.label}{item.seed == null ? '' : ` · s${item.seed}`}</option>)}</select></label>
          <label className="gd-field"><span>Data layer</span><select value={layer} onChange={(event) => setLayer(event.target.value as LayerId)}>{LAYERS.filter((item) => availableLayers.includes(item.id)).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select>{layer === 'uncertainty' && !run?.hasUncertainty && <em className="gd-warn">Uncertainty is unavailable for this baseline.</em>}</label>
          <label className="gd-field"><span>Year <b>{year || '—'}</b></span><select value={year} onChange={(event) => setYear(Number(event.target.value))}>{years.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label className="gd-field"><span>Depth <b>{depthM} m</b></span><input type="range" min={0} max={Math.max(depths.length - 1, 0)} step={1} value={depthIdx} onChange={(event) => setDepthIdx(Number(event.target.value))} /></label>
          <label className="gd-field"><span>Layer opacity <b>{Math.round(opacity * 100)}%</b></span><input type="range" min={0.05} max={1} step={0.01} value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /></label>
          <label className="gd-field"><span>Animation speed <b>{speed.toFixed(1)}×</b></span><input type="range" min={0.2} max={3} step={0.1} value={speed} onChange={(event) => setSpeed(Number(event.target.value))} /></label>
          <div className="gd-toggles"><button className={`gd-toggle ${autoRotate ? 'gd-toggle--on' : ''}`} onClick={() => setAutoRotate((value) => !value)}>AUTO-ROTATE {autoRotate ? 'ON' : 'OFF'}</button><button className="gd-toggle" onClick={resetView}>RESET VIEW</button></div>
          <div className="gd-panel__foot"><p>Orbit: drag · Zoom: scroll</p><p>Click a glowing point to inspect ML values</p></div>
        </aside>

        <div className="gd-strip">
          <div className="gd-strip__item"><span>SELECTED VALUE</span><b>{selectedCell && Number.isFinite(selectedCell.value) ? `${selectedCell.value.toFixed(3)} ${plotUnit}` : '—'}</b></div>
          <div className="gd-strip__item"><span>UNCERTAINTY</span><b>{run?.hasUncertainty ? (layer === 'uncertainty' ? 'model-estimated' : 'available') : 'unavailable'}</b></div>
          <div className="gd-strip__item"><span>STATUS</span><b>{selectedCell ? selectedCell.observed ? 'observed' : 'reconstructed' : 'no selection'}</b></div>
          <div className="gd-strip__item"><span>RMSE</span><b className="gd-dim">{rmse == null ? '—' : rmse.toFixed(3)}</b></div>
          <div className="gd-strip__item"><span>YEAR</span><b>{year || '—'}</b></div>
          <div className="gd-strip__item"><span>DEPTH</span><b>{depthM} m</b></div>
        </div>

        <section className={`gd-drawer ${drawerOpen ? 'gd-drawer--open' : ''}`}>
          <button className="gd-drawer__tab" onClick={() => setDrawerOpen((value) => !value)}><span>ANALYSIS</span><i className={`gd-chevron ${drawerOpen ? 'gd-chevron--up' : ''}`} /></button>
          <div className="gd-drawer__body">
            {!sel || !profile ? <p className="gd-drawer__empty">Select a point on the globe or search a coordinate to load its ML analysis.</p> : <>
              <div className="gd-drawer__head"><h3>Point {profile.lat.toFixed(1)}°, {profile.lon.toFixed(1)}°</h3><span className="gd-tag">{selectedCell?.observed ? 'observed cell' : 'never-observed simulated cell'}</span><span className="gd-tag gd-tag--model">{run?.label}</span>{run?.mock && <span className="gd-tag gd-tag--mock">mock fixture</span>}</div>
              <div className="gd-drawer__grid">
                <Chart title="Depth profile" unit={plotUnit} series={[{ label: 'ML reconstruction', color: '#00e5ff', values: profilePrediction }, { label: 'simulation reference', color: '#9ca3af', values: profileTruth, dashed: true }]} band={profileUncertainty ? { label: 'model-estimated ±90%', upper: profilePrediction.map((value, index) => value + 1.645 * profileUncertainty[index]), lower: profilePrediction.map((value, index) => value - 1.645 * profileUncertainty[index]) } : undefined} />
                <Chart title={`Time series @ ${depthM} m`} unit={plotUnit} series={[{ label: 'ML reconstruction', color: '#38bdf8', values: timeSeries }]} />
                <Chart title="Reconstruction error" unit={plotUnit} series={[{ label: '|pred − reference|', color: '#f59e0b', values: errorSeries }]} />
                <Chart title="Model-estimated uncertainty" unit={plotUnit} series={profileUncertainty ? [{ label: '±σ', color: '#c084fc', values: profileUncertainty }] : [{ label: 'unavailable — baseline run', color: '#475569', values: profilePrediction.map(() => 0), dashed: true }]} />
              </div>
              <div className="gd-provenance"><h4>Provenance</h4><ul><li><b>Dataset:</b> {run?.dataset} · simulation-backed</li><li><b>Model:</b> {run?.model} · seed {run?.seed ?? 'not recorded'}</li><li><b>Source:</b> {run?.source} · {run?.mock ? 'mock fixture' : 'read-only artifact'}</li><li><b>Uncertainty:</b> {run?.hasUncertainty ? 'model-provided' : 'unavailable for this baseline'}</li><li><b>Caveat:</b> precomputed simulation output, not live real-ocean measurements.</li></ul></div>
            </>}
          </div>
        </section>
      </div>
    </div>
  );
}
