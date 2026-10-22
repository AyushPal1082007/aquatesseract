/* ================================================================
   GLOBE DASHBOARD — three.js scene
   Dark scientific instrument: textured sphere + oxygen-field
   equirectangular data overlay + clean orbital arcs
   (visual only, NOT real ocean currents) + raycast picking.
   ================================================================ */

import * as THREE from 'three';
import earthTextureUrl from '../data/earth.png';
import { LAYERS, LayerId } from './types.js';
import { GlobeFrame } from './types.js';

export interface GlobeSceneOptions {
  canvas: HTMLCanvasElement;
  onPick: (lat: number, lon: number) => void;
  onFrame?: () => void;
}

export interface GlobeSceneHandle {
  setFrame(frame: GlobeFrame, opacity: number): void;
  setAutoRotate(on: boolean): void;
  setSpeed(mult: number): void;
  setSelected(lat: number | null, lon: number | null): void;
  resetView(): void;
  focusPoint(lat: number, lon: number): void;
  dispose(): void;
}

/* Soft marine palette keeps the modeled oxygen layer legible over the pale earth. */
function oxyColor(t: number, out: THREE.Color): THREE.Color {
  t = Math.min(1, Math.max(0, t));
  const stops: [number, number[]][] = [
    [0.00, [125, 164, 161]],
    [0.35, [157, 192, 174]],
    [0.60, [192, 207, 174]],
    [0.85, [226, 216, 183]],
    [1.00, [248, 241, 218]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0] || i === stops.length - 1) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1)));
      out.setRGB(
        (c0[0] + (c1[0] - c0[0]) * f) / 255,
        (c0[1] + (c1[1] - c0[1]) * f) / 255,
        (c0[2] + (c1[2] - c0[2]) * f) / 255,
      );
      return out;
    }
  }
  return out;
}

const ATLAS_W = 1024, ATLAS_H = 512;
const GRID_ROWS = 17, GRID_COLS = 36;

function buildFieldTexture(frame: GlobeFrame): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = ATLAS_W; cv.height = ATLAS_H;
  const ctx = cv.getContext('2d')!;
  const values = new Float32Array(GRID_ROWS * GRID_COLS);
  const valid = new Uint8Array(GRID_ROWS * GRID_COLS);
  for (const cell of frame.cells) {
    if (!Number.isFinite(cell.value)) continue;
    const row = Math.round((cell.lat + 80) / 10);
    const col = ((Math.round((cell.lon + 180) / 10) % GRID_COLS) + GRID_COLS) % GRID_COLS;
    if (row < 0 || row >= GRID_ROWS) continue;
    const index = row * GRID_COLS + col;
    values[index] = cell.value;
    valid[index] = 1;
  }

  // The demo samples are spaced 10° apart. Interpolate them into a smooth
  // texture and fade coverage beside missing (land) cells instead of drawing
  // opaque square tiles.
  const pixels = ctx.createImageData(ATLAS_W, ATLAS_H);
  const span = frame.max - frame.min || 1;
  const color = new THREE.Color();
  for (let y = 0; y < ATLAS_H; y++) {
    const lat = 90 - ((y + 0.5) / ATLAS_H) * 180;
    const gridY = (lat + 80) / 10;
    if (gridY < 0 || gridY > GRID_ROWS - 1) continue;
    const row0 = Math.floor(gridY), row1 = Math.min(GRID_ROWS - 1, row0 + 1);
    const fy = gridY - row0;
    for (let x = 0; x < ATLAS_W; x++) {
      const lon = ((x + 0.5) / ATLAS_W) * 360 - 180;
      const gridX = (lon + 180) / 10;
      const col0 = Math.floor(gridX) % GRID_COLS;
      const col1 = (col0 + 1) % GRID_COLS;
      const fx = gridX - Math.floor(gridX);
      let weightedValue = 0, coverage = 0;
      let index = row0 * GRID_COLS + col0, weight = (1 - fx) * (1 - fy);
      if (valid[index]) { weightedValue += values[index] * weight; coverage += weight; }
      index = row0 * GRID_COLS + col1; weight = fx * (1 - fy);
      if (valid[index]) { weightedValue += values[index] * weight; coverage += weight; }
      index = row1 * GRID_COLS + col0; weight = (1 - fx) * fy;
      if (valid[index]) { weightedValue += values[index] * weight; coverage += weight; }
      index = row1 * GRID_COLS + col1; weight = fx * fy;
      if (valid[index]) { weightedValue += values[index] * weight; coverage += weight; }
      if (!coverage) continue;
      oxyColor((weightedValue / coverage - frame.min) / span, color);
      const offset = (y * ATLAS_W + x) * 4;
      pixels.data[offset] = Math.round(color.r * 255);
      pixels.data[offset + 1] = Math.round(color.g * 255);
      pixels.data[offset + 2] = Math.round(color.b * 255);
      pixels.data[offset + 3] = Math.round(coverage * 255);
    }
  }
  ctx.putImageData(pixels, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function buildObservationTexture(frame: GlobeFrame): THREE.CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = ATLAS_W;
  cv.height = ATLAS_H;
  const ctx = cv.getContext('2d')!;
  ctx.fillStyle = 'rgba(232, 95, 61, 0.92)';
  for (const cell of frame.cells) {
    if (!cell.observed) continue;
    const x = ((cell.lon + 180) / 360) * ATLAS_W;
    const y = ((90 - cell.lat) / 180) * ATLAS_H;
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/* Lat/lon → 3D position on sphere (three.js equirect mapping). */
export function latLonToVec3(lat: number, lon: number, r: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta),
  );
}

export function createGlobeScene(opts: GlobeSceneOptions): GlobeSceneHandle {
  const { canvas, onPick } = opts;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0.62, 3.45);

  /* ---------- lights ---------- */
  scene.add(new THREE.AmbientLight(0xe8f1ed, 1.05));
  const key = new THREE.DirectionalLight(0xfff4dc, 1.65);
  key.position.set(3, 2, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xa3d5d8, 0.8);
  rim.position.set(-4, -1, -3);
  scene.add(rim);

  /* ---------- globe group (rotates) ---------- */
  const globe = new THREE.Group();
  scene.add(globe);

  const baseTex = new THREE.TextureLoader().load(earthTextureUrl);
  baseTex.colorSpace = THREE.SRGBColorSpace;
  const fieldTexHolder: { tex: THREE.CanvasTexture | null } = { tex: null };

  const sphereMat = new THREE.MeshPhongMaterial({
    map: baseTex,
    color: 0xffffff,
    emissive: 0x344c48,
    emissiveIntensity: 0.12,
    shininess: 24,
    specular: 0x9cb8b4,
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 96), sphereMat);
  sphere.renderOrder = 0;
  globe.add(sphere);

  /* field overlay: second sphere slightly larger, transparent */
  const fieldMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.55, depthWrite: false,
  });
  const fieldSphere = new THREE.Mesh(new THREE.SphereGeometry(1.002, 96, 96), fieldMat);
  fieldSphere.renderOrder = 1;
  globe.add(fieldSphere);

  /* Tiny moving crystal flecks sit just above the surface, like fine ice dust. */
  const crystalCount = 220;
  const crystalSeeds = Array.from({ length: crystalCount }, () => ({
    theta: Math.random() * Math.PI * 2,
    phi: Math.acos(2 * Math.random() - 1),
    drift: (Math.random() - 0.5) * 0.025,
    phase: Math.random() * Math.PI * 2,
    size: 0.72 + Math.random() * 0.56,
  }));
  const crystalGeometry = new THREE.OctahedronGeometry(0.0042, 0);
  const crystalMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.58, depthWrite: false,
  });
  const crystals = new THREE.InstancedMesh(crystalGeometry, crystalMaterial, crystalCount);
  crystals.frustumCulled = false;
  crystals.renderOrder = 2;
  globe.add(crystals);
  const crystalTransform = new THREE.Object3D();
  function moveCrystals(time: number) {
    for (let i = 0; i < crystalCount; i++) {
      const seed = crystalSeeds[i];
      const theta = seed.theta + seed.drift * time;
      const phi = Math.max(0.04, Math.min(Math.PI - 0.04,
        seed.phi + Math.sin(time * 0.32 + seed.phase) * 0.035));
      const radius = 1.015 + Math.sin(time * 0.7 + seed.phase) * 0.0012;
      crystalTransform.position.set(
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.cos(theta),
      );
      const twinkle = seed.size * (0.82 + 0.18 * Math.sin(time * 1.4 + seed.phase));
      crystalTransform.scale.setScalar(twinkle);
      crystalTransform.rotation.set(seed.phase + time * 0.15, theta, phi);
      crystalTransform.updateMatrix();
      crystals.setMatrixAt(i, crystalTransform.matrix);
    }
    crystals.instanceMatrix.needsUpdate = true;
  }
  moveCrystals(0);

  /* ---------- selected point marker ---------- */
  const selGroup = new THREE.Group();
  globe.add(selGroup);
  const selRing = new THREE.Mesh(
    new THREE.RingGeometry(0.028, 0.036, 40),
    new THREE.MeshBasicMaterial({ color: 0xf5ffff, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false }),
  );
  const selDot = new THREE.Mesh(
    new THREE.SphereGeometry(0.014, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
  const selBeam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.002, 0.002, 0.55, 8),
    new THREE.MeshBasicMaterial({ color: 0xb9dfdc, transparent: true, opacity: 0.5, depthWrite: false }),
  );
  selBeam.position.y = 0.275;
  selBeam.rotation.z = 0;
  selGroup.add(selRing, selDot, selBeam);
  selGroup.visible = false;

  /* ---------- orbital ring arcs (reference-2 style) ---------- */
  const arcs = new THREE.Group();
  for (let i = 0; i < 2; i++) {
    const curve = new THREE.EllipseCurve(0, 0, 1.55 + i * 0.35, 1.55 + i * 0.35, 0.3, Math.PI * 2 - 0.6, false);
    const pts = curve.getPoints(128).map((p) => new THREE.Vector3(p.x, 0.18 * (i === 0 ? 1 : -1), p.y));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    arcs.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: 0xf8ffff, transparent: true, opacity: 0.36 + i * 0.08,
    })));
  }
  arcs.rotation.x = Math.PI / 2.4;
  scene.add(arcs);

  /* ---------- controls: manual orbit (no OrbitControls dep) ---------- */
  let drag = false, px = 0, py = 0;
  let rotY = 0.6, rotX = 0.25; // globe tilt/yaw
  let camDist = 3.45, targetDist = 3.45;
  let autoRotate = true, speedMult = 1;
  const down = (e: PointerEvent) => {
    if (e.button !== 0) return;
    drag = true; px = e.clientX; py = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const move = (e: PointerEvent) => {
    if (!drag) return;
    rotY += (e.clientX - px) * 0.005;
    rotX = Math.max(-1.1, Math.min(1.1, rotX + (e.clientY - py) * 0.004));
    px = e.clientX; py = e.clientY;
  };
  const up = () => { drag = false; };
  const wheel = (e: WheelEvent) => {
    e.preventDefault();
    targetDist = Math.max(1.8, Math.min(6, targetDist + e.deltaY * 0.0016));
  };
  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', wheel as EventListener, { passive: false });

  /* ---------- picking ---------- */
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const click = (e: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObject(sphere, false)[0];
    if (!hit) return;
    // local point → lat/lon (sphere is inside rotated globe group)
    const p = sphere.worldToLocal(hit.point.clone()).normalize();
    const lat = 90 - (Math.acos(p.y) * 180) / Math.PI;
    const lon = ((Math.atan2(p.z, -p.x) * 180) / Math.PI) - 180;
    onPick(lat, Math.round((lon + 540) % 360 - 180));
  };
  canvas.addEventListener('click', click);

  /* ---------- resize ---------- */
  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  /* ---------- render loop ---------- */
  let raf = 0, t0 = performance.now();
  let disposed = false;
  const tick = (now: number) => {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    const t = (now - t0) / 1000;
    if (autoRotate && !drag) rotY += 0.0016 * speedMult * 60 / 60; // steady spin
    globe.rotation.y = rotY;
    globe.rotation.x = rotX;
    moveCrystals(t);
    arcs.rotation.z = t * 0.02;
    // idle marker pulse
    const pulse = 1 + 0.14 * Math.sin(t * 3.2);
    selRing.scale.setScalar(pulse);
    // camera dolly
    camDist += (targetDist - camDist) * 0.08;
    camera.position.set(0, camDist * 0.18, camDist);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    opts.onFrame?.();
  };
  raf = requestAnimationFrame(tick);

  /* ---------- API ---------- */
  const handle: GlobeSceneHandle = {
    setFrame(frame, opacity) {
      // swap field texture
      fieldTexHolder.tex?.dispose();
      const info = LAYERS.find((l) => l.id === (frame.layer as LayerId));
      if (info) {
        fieldTexHolder.tex = frame.layer === 'observations'
          ? buildObservationTexture(frame)
          : buildFieldTexture(frame);
        (fieldMat as THREE.MeshBasicMaterial & { map: THREE.Texture | null }).map = fieldTexHolder.tex;
        fieldMat.opacity = opacity;
        fieldMat.visible = true;
        fieldMat.needsUpdate = true;
      } else {
        fieldMat.visible = false;
      }
      // The same slider cross-fades the satellite map out as the data layer
      // becomes stronger. At 100%, only modeled data (and the tiny flecks) remain.
      sphereMat.transparent = opacity > 0;
      sphereMat.opacity = 1 - opacity;
      sphereMat.depthWrite = false;
      sphereMat.needsUpdate = true;
    },
    setAutoRotate(on) { autoRotate = on; },
    setSpeed(m) { speedMult = m; },
    setSelected(lat, lon) {
      if (lat == null || lon == null) { selGroup.visible = false; return; }
      const v = latLonToVec3(lat, lon, 1.02);
      selGroup.position.copy(v);
      selGroup.lookAt(0, 0, 0);
      selGroup.visible = true;
    },
    resetView() { rotX = 0.25; rotY = 0.6; targetDist = 3.45; },
    focusPoint(lat, lon) {
      // rotate globe so the point faces the camera
      rotY = ((-(lon + 180)) * Math.PI) / 180 + Math.PI / 2;
      rotX = Math.max(-1.1, Math.min(1.1, (lat * Math.PI) / 180 * 0.9));
      targetDist = Math.min(targetDist, 3.1);
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointercancel', up);
      canvas.removeEventListener('wheel', wheel);
      canvas.removeEventListener('click', click);
      fieldTexHolder.tex?.dispose();
      baseTex.dispose();
      sphereMat.dispose();
      fieldMat.dispose();
      crystalGeometry.dispose();
      crystalMaterial.dispose();
      arcs.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
    },
  };
  return handle;
}
