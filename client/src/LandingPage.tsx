import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import './landing.css';

/* ── Asset imports (Vite resolves these to hashed URLs) ── */
import aboveOceanSrc from './data/aboveOcean.mp4';
import belowOceanSrc from './data/belowOcen.mp4';
import deepOceanSrc from './data/aquatesseract_deep_ocean_2000m.png';
import abyssSrc from './data/aquatesseract_abyss_5500m.png';

/* ─────────── Constants & Types ─────────── */

interface ScrollState {
  section: number;
  sectionProgress: number;
  sectionFloat: number;
}

interface TelemetryData {
  depth: number;
  needleDeg: number;
  temp: string;
  o2: string;
  pressure: number;
  salinity: string;
  obs: number;
  status: string;
  zoneName: string;
  zoneColor: string;
  zoneDesc: string;
}

const SECTIONS = [
  { id: 0, depth: 0, label: 'SURFACE', depthStr: '0 m' },
  { id: 1, depth: 200, label: 'UNDERWATER', depthStr: '200 m' },
  { id: 2, depth: 2000, label: 'DEEP OCEAN', depthStr: '2,000 m' },
  { id: 3, depth: 5500, label: 'THE ABYSS', depthStr: '5,500 m' },
  { id: 4, depth: 6000, label: 'EXPLORER', depthStr: '6,000 m' },
];

/* ─────────── Component ─────────── */

interface Props {
  onEnterExplorer: () => void;
}

export default function LandingPage({ onEnterExplorer }: Props) {
  /* ── scroll tracking ── */
  const [scroll, setScroll] = useState<ScrollState>({
    section: 0,
    sectionProgress: 0,
    sectionFloat: 0,
  });
  const sectionRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        const vh = window.innerHeight;
        const viewportCenter = y + vh / 2;
        const sectionCenters = sectionRefs.current.map((section) =>
          section
            ? section.getBoundingClientRect().top + y + section.offsetHeight / 2
            : null,
        );

        let sf = 0;
        for (let i = 0; i < sectionCenters.length - 1; i += 1) {
          const currentCenter = sectionCenters[i];
          const nextCenter = sectionCenters[i + 1];
          if (currentCenter === null || nextCenter === null) continue;
          if (viewportCenter >= currentCenter && viewportCenter < nextCenter) {
            sf = i + (viewportCenter - currentCenter) / (nextCenter - currentCenter);
            break;
          }
          if (viewportCenter >= nextCenter) sf = i + 1;
        }

        sf = Math.min(Math.max(sf, 0), SECTIONS.length - 1);
        const s = Math.min(Math.floor(sf), SECTIONS.length - 1);
        const sp = Math.min(sf - s, 1);
        setScroll({ section: s, sectionProgress: sp, sectionFloat: sf });
        ticking = false;
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  /* ── continuous telemetry data calculation ── */
  const telemetry: TelemetryData = useMemo(() => {
    const { sectionFloat: sf } = scroll;

    // Calculate continuous depth in meters
    let depth = 0;
    if (sf <= 0) depth = 0;
    else if (sf <= 1) depth = sf * 200;
    else if (sf <= 2) depth = 200 + (sf - 1) * 1800;
    else if (sf <= 3) depth = 2000 + (sf - 2) * 3500;
    else depth = Math.min(6000, 5500 + (sf - 3) * 500);

    // Speedometer needle angle: sweeps from -125deg (0m) to +125deg (5500m+)
    const norm = Math.min(Math.max(depth / 5500, 0), 1);
    const needleDeg = -125 + norm * 250;

    // Temperature (18.3°C down to 1.2°C)
    let temp = 18.3;
    if (sf <= 1) temp = 18.3 - sf * (18.3 - 14.1);
    else if (sf <= 2) temp = 14.1 - (sf - 1) * (14.1 - 2.8);
    else temp = Math.max(1.1, 2.8 - (sf - 2) * (2.8 - 1.2));

    // Dissolved O2 Saturation %
    let o2 = 97.2;
    if (sf <= 1) o2 = 97.2 - sf * (97.2 - 82.5);
    else if (sf <= 2) o2 = 82.5 - (sf - 1) * (82.5 - 44.1);
    else o2 = 44.1 + (sf - 2) * (58.2 - 44.1);

    // Hydrostatic Pressure (1 atm at surface, +1 atm per 10m)
    const pressure = Math.round(1 + depth / 10);

    // Salinity (PSU)
    let salinity = 34.7;
    if (sf <= 1) salinity = 34.7 + sf * 0.4;
    else if (sf <= 2) salinity = 35.1 - (sf - 1) * 0.2;
    else salinity = 34.9 - (sf - 2) * 0.1;

    // Zone metadata
    let obs = 62;
    let status = 'Direct Buoy Sensors';
    let zoneName = 'SURFACE LAYER';
    let zoneColor = '#7dd3fc';
    let zoneDesc = 'High-frequency satellite & surface buoy validation';

    if (sf < 0.6) {
      obs = Math.round(62 - sf * 10);
      status = 'Direct Buoy Sensors';
      zoneName = 'SURFACE / SUNLIT';
      zoneColor = '#93c5fd';
      zoneDesc = 'Full atmospheric interaction & primary production';
    } else if (sf < 1.6) {
      obs = Math.round(52 - (sf - 0.6) * 14);
      status = 'Argo Float Profiles';
      zoneName = 'EPIPELAGIC (200 m)';
      zoneColor = '#00e5ff';
      zoneDesc = 'Photochemical gradient & mixed layer boundary';
    } else if (sf < 2.6) {
      obs = Math.round(38 - (sf - 1.6) * 24);
      status = 'Sparse Transect Data';
      zoneName = 'MESOPELAGIC (2,000 m)';
      zoneColor = '#818cf8';
      zoneDesc = 'Oxygen minimum zone; sparse shipboard transects';
    } else {
      obs = Math.max(2, Math.round(14 - (sf - 2.6) * 12));
      status = 'Physics-Constrained ML';
      zoneName = 'ABYSSOPELAGIC (5,500 m)';
      zoneColor = '#c084fc';
      zoneDesc = 'Extreme depth; reconstruction governed by physics constraints';
    }

    return {
      depth: Math.round(depth),
      needleDeg,
      temp: temp.toFixed(1),
      o2: o2.toFixed(1),
      pressure,
      salinity: salinity.toFixed(1),
      obs,
      status,
      zoneName,
      zoneColor,
      zoneDesc,
    };
  }, [scroll]);

  /* ── background layer opacity (crossfade as user dives deeper) ── */
  const layerOpacity = useCallback(
    (idx: number): number => {
      if (idx === 0) return 1; // surface is always base
      const { sectionFloat: sf } = scroll;
      const start = idx - 0.45;
      const end = idx + 0.15;
      if (sf <= start) return 0;
      if (sf >= end) return 1;
      return (sf - start) / (end - start);
    },
    [scroll],
  );

  /* ── zoom-in / zoom-out dynamic styling for each section text ── */
  const getZoomStyle = useCallback(
    (idx: number): React.CSSProperties => {
      const { sectionFloat: sf } = scroll;
      const diff = sf - idx;
      const smoothstep = (start: number, end: number, value: number) => {
        const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
        return t * t * (3 - 2 * t);
      };

      // Keep each section readable as it enters and leaves the viewport. The
      // fade follows scroll directly, with no CSS transition lag between frames.
      const fadeIn = smoothstep(-0.9, -0.1, diff);
      const fadeOut = 1 - smoothstep(0.1, 0.9, diff);
      const opacity = Math.max(0.72, fadeIn * fadeOut);
      const scale = diff < 0
        ? 0.96 + smoothstep(-1, 0, diff) * 0.04
        : 1 + Math.min(diff, 1) * 0.04;

      return {
        opacity,
        transform: `scale(${scale.toFixed(3)})`,
        pointerEvents: 'auto',
        willChange: 'transform, opacity',
      };
    },
    [scroll],
  );

  /* ── helpers ── */
  const scrollToSection = (i: number) => {
    sectionRefs.current[i]?.scrollIntoView({ behavior: 'smooth' });
  };

  const navBg = `rgba(5, 12, 24, ${Math.min(0.25 + scroll.sectionFloat * 0.15, 0.92)})`;
  const overlayOpacity = 0.2 + Math.min(scroll.sectionFloat * 0.14, 0.6);

  /* ── particles ── */
  const particles = useMemo(
    () =>
      Array.from({ length: 32 }, (_, i) => ({
        id: i,
        left: Math.random() * 100,
        delay: Math.random() * 20,
        duration: 11 + Math.random() * 18,
        size: 1.5 + Math.random() * 3.5,
        opacity: 0.1 + Math.random() * 0.35,
        drift: (Math.random() - 0.5) * 60,
      })),
    [],
  );

  return (
    <div className="landing-root">
      {/* ════ FIXED BACKGROUND STACK ════ */}
      <div className="landing-bg">
        <div className="landing-bg__layer" style={{ opacity: layerOpacity(0), zIndex: 0 }}>
          <video src={aboveOceanSrc} autoPlay muted loop playsInline />
        </div>
        <div className="landing-bg__layer" style={{ opacity: layerOpacity(1), zIndex: 1 }}>
          <video src={belowOceanSrc} autoPlay muted loop playsInline />
        </div>
        <div className="landing-bg__layer" style={{ opacity: layerOpacity(2), zIndex: 2 }}>
          <img src={deepOceanSrc} alt="Deep ocean 2000m" />
        </div>
        <div className="landing-bg__layer" style={{ opacity: layerOpacity(3), zIndex: 3 }}>
          <img src={abyssSrc} alt="Abyss 5500m" />
        </div>
        <div className="landing-bg__overlay" style={{ opacity: overlayOpacity, zIndex: 4 }} />
      </div>

      {/* ════ PARTICLES ════ */}
      <div className="particles" aria-hidden="true">
        {particles.map((p) => (
          <div
            key={p.id}
            className="particle"
            style={
              {
                left: `${p.left}%`,
                '--delay': `${p.delay}s`,
                '--duration': `${p.duration}s`,
                '--size': `${p.size}px`,
                '--p-opacity': p.opacity,
                '--drift': p.drift,
              } as React.CSSProperties
            }
          />
        ))}
      </div>

      {/* ════ TOP NAVBAR ════ */}
      <nav className="landing-nav" style={{ background: navBg }}>
        <div className="landing-nav__brand" onClick={() => scrollToSection(0)}>
          <div className="landing-nav__logo-icon">◈</div>
          <span>AQUA</span>TESSERACT
        </div>
        <div className="landing-nav__right">
          <ul className="landing-nav__links">
            {SECTIONS.slice(0, 4).map((s) => (
              <li key={s.id}>
                <a
                  href={`#section-${s.id}`}
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection(s.id);
                  }}
                  className={scroll.section === s.id ? 'active' : ''}
                >
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
          <button className="nav-enter-btn" onClick={onEnterExplorer} id="nav-enter-explorer-btn">
            Enter Explorer
          </button>
        </div>
      </nav>

      {/* ════ FIXED OBSERVATION WINDOW (LEFT SIDE) ════ */}
      <aside className="fixed-obs-window" aria-label="Real-time Ocean Observation HUD">
        <div className="obs-header">
          <div className="obs-live-tag">
            <span className="obs-beacon" />
            <span className="obs-live-text">TELEMETRY // LIVE HUD</span>
          </div>
          <span className="obs-zone-badge" style={{ color: telemetry.zoneColor, borderColor: `${telemetry.zoneColor}44` }}>
            {telemetry.zoneName}
          </span>
        </div>

        <div className="obs-depth-hero">
          <div className="obs-depth-val">
            <span className="obs-depth-number">{telemetry.depth.toLocaleString()}</span>
            <span className="obs-depth-unit">m</span>
          </div>
          <div className="obs-depth-desc">{telemetry.zoneDesc}</div>
        </div>

        <div className="obs-metrics-grid">
          <div className="obs-metric">
            <span className="obs-metric__label">Dissolved O₂</span>
            <span className="obs-metric__val obs-val--aqua">{telemetry.o2} %</span>
            <div className="obs-bar-track">
              <div
                className="obs-bar-fill"
                style={{
                  width: `${Math.min(100, Math.max(10, parseFloat(telemetry.o2)))}%`,
                  background: telemetry.zoneColor,
                }}
              />
            </div>
          </div>

          <div className="obs-metric">
            <span className="obs-metric__label">Temperature</span>
            <span className="obs-metric__val">{telemetry.temp} °C</span>
          </div>

          <div className="obs-metric">
            <span className="obs-metric__label">Pressure</span>
            <span className="obs-metric__val">{telemetry.pressure} atm</span>
          </div>

          <div className="obs-metric">
            <span className="obs-metric__label">Salinity</span>
            <span className="obs-metric__val">{telemetry.salinity} PSU</span>
          </div>
        </div>

        <div className="obs-footer">
          <div className="obs-trust-row">
            <span className="obs-trust-label">Observation Coverage</span>
            <span className="obs-trust-val">{telemetry.obs}%</span>
          </div>
          <div className="obs-source-tag">
            <span className="obs-source-dot" style={{ background: telemetry.zoneColor }} />
            {telemetry.status}
          </div>
        </div>
      </aside>

      {/* ════ FIXED SPEEDOMETER DEPTH GAUGE (RIGHT SIDE) ════ */}
      <aside className="fixed-speedometer" aria-label="Bathy-Speedometer Depth Gauge">
        <div className="speedometer-box">
          <svg className="speedometer-svg" viewBox="0 0 200 200">
            <defs>
              <linearGradient id="gaugeArcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#38bdf8" />
                <stop offset="35%" stopColor="#00e5ff" />
                <stop offset="70%" stopColor="#818cf8" />
                <stop offset="100%" stopColor="#c084fc" />
              </linearGradient>
              <linearGradient id="needleGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#ff3366" />
                <stop offset="40%" stopColor="#00e5ff" />
                <stop offset="100%" stopColor="#ffffff" />
              </linearGradient>
              <filter id="gaugeGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Gauge outer rim */}
            <circle cx="100" cy="100" r="88" className="gauge-rim" />

            {/* Inactive track arc (-125deg to +125deg) */}
            <path
              d="M 37.8 157.3 A 80 80 0 1 1 162.2 157.3"
              fill="none"
              stroke="rgba(255, 255, 255, 0.08)"
              strokeWidth="5"
              strokeLinecap="round"
            />

            {/* Active illuminated depth arc */}
            <path
              d="M 37.8 157.3 A 80 80 0 1 1 162.2 157.3"
              fill="none"
              stroke="url(#gaugeArcGrad)"
              strokeWidth="5"
              strokeDasharray="349"
              strokeDashoffset={349 - (349 * Math.min(telemetry.depth / 5500, 1))}
              strokeLinecap="round"
              filter="url(#gaugeGlow)"
              className="gauge-active-arc"
            />

            {/* Tick marks around speedometer */}
            {Array.from({ length: 11 }).map((_, i) => {
              const tickAngle = -125 + i * 25; // -125 to +125 deg
              const rad = ((tickAngle - 90) * Math.PI) / 180;
              const isMajor = i % 2 === 0;
              const r1 = 80;
              const r2 = isMajor ? 69 : 74;
              const x1 = 100 + r1 * Math.cos(rad);
              const y1 = 100 + r1 * Math.sin(rad);
              const x2 = 100 + r2 * Math.cos(rad);
              const y2 = 100 + r2 * Math.sin(rad);
              const isPassed = tickAngle <= telemetry.needleDeg;
              return (
                <line
                  key={i}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={isPassed ? '#00e5ff' : 'rgba(255,255,255,0.22)'}
                  strokeWidth={isMajor ? 2.2 : 1.2}
                />
              );
            })}

            {/* Depth label numbers on dial */}
            <text x="36" y="174" className="dial-num">0</text>
            <text x="26" y="92" className="dial-num">1k</text>
            <text x="56" y="44" className="dial-num">2k</text>
            <text x="144" y="44" className="dial-num">3k</text>
            <text x="174" y="92" className="dial-num">4k</text>
            <text x="156" y="174" className="dial-num">5.5k</text>

            {/* ROTATING PIN / SPEEDOMETER NEEDLE */}
            <g
              className="speedometer-needle-group"
              style={{
                transform: `rotate(${telemetry.needleDeg}deg)`,
                transformOrigin: '100px 100px',
              }}
            >
              {/* Tapered glowing needle pointer */}
              <polygon
                points="97.5,108 102.5,108 101,30 99,30"
                fill="url(#needleGrad)"
                filter="url(#gaugeGlow)"
              />
              {/* Illuminated sharp needle pin tip */}
              <circle cx="100" cy="28" r="2.5" fill="#00e5ff" filter="url(#gaugeGlow)" />
              {/* Counterbalance tail */}
              <polygon points="98,108 102,108 100.8,124 99.2,124" fill="#475569" />
              {/* Center Hub */}
              <circle cx="100" cy="100" r="10" fill="#0b172a" stroke="#00e5ff" strokeWidth="2" />
              <circle cx="100" cy="100" r="4" fill="#ffffff" />
            </g>
          </svg>

          {/* Speedometer Center Readout */}
          <div className="speedometer-readout">
            <div className="speedo-title">BATHYMETRY</div>
            <div className="speedo-main-val">
              <span>{telemetry.depth.toLocaleString()}</span>
              <span className="speedo-unit">M</span>
            </div>
            <div className="speedo-sub-label">OCEAN DEPTH</div>
          </div>

          {/* Quick Jump Buttons */}
          <div className="speedo-quick-jumps">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                className={`speedo-jump-btn ${scroll.section === s.id ? 'active' : ''}`}
                onClick={() => scrollToSection(s.id)}
                title={`Jump to ${s.label} (${s.depthStr})`}
              >
                {s.label[0]}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* ══════════════════════════════════════════════════════════
          SCROLL SECTIONS WITH SMOOTH ZOOM-IN / ZOOM-OUT TYPOGRAPHY
          ══════════════════════════════════════════════════════════ */}

      {/* ── SECTION 0: SURFACE (Reference Image Style) ── */}
      <section
        id="section-0"
        className="depth-section depth-section--hero"
        ref={(el) => { sectionRefs.current[0] = el; }}
      >
        <div className="depth-section__inner" style={getZoomStyle(0)}>
          <div className="hero-content">
            <p className="hero-kicker">
              JOIN US AND DISCOVER A WORLD BENEATH THE WAVES
            </p>

            <h1 className="hero-title hero-title--ref">
              <span className="hero-title__line1">EXPLORE THE DEPTHS</span>
              <span className="hero-title__line2">OF THE OCEAN</span>
            </h1>

            <div className="hero-actions">
              <button
                className="hero-pill-btn"
                onClick={() => scrollToSection(1)}
                id="start-adventure-btn"
              >
                START YOUR ADVENTURE
              </button>
            </div>

            <div className="scroll-dive-cue" onClick={() => scrollToSection(1)}>
              <span className="scroll-dive-cue__text">SCROLL TO DIVE</span>
              <div className="scroll-dive-cue__arrow" />
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 1: UNDERWATER (200 M) ── */}
      <section
        id="section-1"
        className="depth-section"
        ref={(el) => { sectionRefs.current[1] = el; }}
      >
        <div className="depth-section__inner" style={getZoomStyle(1)}>
          <div className="zone-narrative zone-narrative--underwater">
            <span className="zone-kicker kicker-aqua">200 METERS · SUNLIT GRADIENT</span>
            <h2 className="zone-title zone-title--aqua">
              BENEATH THE SURFACE
            </h2>
            <p className="zone-lead">
              Where solar rays fragment into crystalline Tyndall rays. Phytoplankton blooms
              saturate the upper photic layer with oxygen.
            </p>
            <div className="zone-highlight-card">
              <div className="zone-card-stat">
                <span className="stat-big">97.2%</span>
                <span className="stat-label">O₂ Saturation Equilibrium</span>
              </div>
              <p className="stat-note">
                Direct float profiles provide dense coverage, anchoring model boundary conditions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 2: DEEP OCEAN (2,000 M) ── */}
      <section
        id="section-2"
        className="depth-section"
        ref={(el) => { sectionRefs.current[2] = el; }}
      >
        <div className="depth-section__inner" style={getZoomStyle(2)}>
          <div className="zone-narrative zone-narrative--deep">
            <span className="zone-kicker kicker-blue">2,000 METERS · MESOPELAGIC TWILIGHT</span>
            <h2 className="zone-title zone-title--blue">
              THE DEEP OCEAN
            </h2>
            <p className="zone-lead">
              Sunlight ceases to penetrate. Microbial respiration consumes dissolved oxygen,
              creating vast Oxygen Minimum Zones that traverse entire ocean basins.
            </p>
            <div className="zone-highlight-card">
              <div className="zone-card-stat">
                <span className="stat-big stat-blue">±12.8 µmol</span>
                <span className="stat-label">Model Uncertainty Window</span>
              </div>
              <p className="stat-note">
                Only 12% observation density. Sparse matrix completion reconstructs the unseen.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 3: THE ABYSS (5,500 M) ── */}
      <section
        id="section-3"
        className="depth-section"
        ref={(el) => { sectionRefs.current[3] = el; }}
      >
        <div className="depth-section__inner" style={getZoomStyle(3)}>
          <div className="zone-narrative zone-narrative--abyss">
            <span className="zone-kicker kicker-purple">5,500 METERS · ABYSSOPELAGIC REALM</span>
            <h2 className="zone-title zone-title--purple">
              THE ABYSS
            </h2>
            <p className="zone-lead">
              Over 550 atmospheres of crushing hydrostatic pressure. Cold Antarctic Bottom Water
              flows silently along trench floors in near-total darkness.
            </p>
            <div className="zone-highlight-card">
              <div className="zone-card-stat">
                <span className="stat-big stat-purple">73%</span>
                <span className="stat-label">Never Directly Sampled</span>
              </div>
              <p className="stat-note">
                Trust matters most where data vanishes. Physics-informed constraints guarantee mass conservation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 4: CTA EXPLORER LAUNCH ── */}
      <section
        id="section-4"
        className="depth-section depth-section--cta"
        ref={(el) => { sectionRefs.current[4] = el; }}
      >
        <div className="depth-section__inner" style={getZoomStyle(4)}>
          <div className="cta-portal">
            <span className="zone-kicker kicker-gold">READY TO INVESTIGATE</span>
            <h2 className="cta-headline">
              Enter AquaTesseract
            </h2>
            <p className="cta-subtext">
              Reconstructing ocean dissolved oxygen across 4D space-time with quantified trust,
              uncertainty envelopes, and spatial validation.
            </p>
            <button className="cta-launch-btn" onClick={onEnterExplorer} id="enter-explorer-main-btn">
              <span>EXPLORE THE RECONSTRUCTION</span>
              <span className="cta-arrow-icon">→</span>
            </button>
            <div className="cta-specs-row">
              <div className="cta-spec-item">
                <span className="spec-val">0 – 6,000 m</span>
                <span className="spec-lbl">Continuous Depth Column</span>
              </div>
              <div className="cta-spec-item">
                <span className="spec-val">OceanVerse</span>
                <span className="spec-lbl">Validation Benchmark</span>
              </div>
              <div className="cta-spec-item">
                <span className="spec-val">100% Physics Checked</span>
                <span className="spec-lbl">Conserved Oxygen Budget</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
