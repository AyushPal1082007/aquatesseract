"""Make a small FAKE run folder in the exact real format, so server/client work from day 1.
Fake numbers. Folder name starts with MOCK_ and metadata.mock=true."""
import argparse
import json
from pathlib import Path

import numpy as np

from aquatesseract.contract import DEPTH_LEVELS_M, N_DEPTH, metrics_for

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--years", type=int, default=10)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
Y, D, LAT, LON = a.years, N_DEPTH, 36, 72
lat = np.linspace(-87.5, 87.5, LAT)
lon = np.linspace(2.5, 357.5, LON)
la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
land = (np.sin(2 * lo) + np.cos(3 * la)) > 1.1
bottom = (14 + 18 * np.abs(np.sin(lo + la))).astype(int)
ocean = (~land)[None] & (np.arange(D)[:, None, None] <= bottom[None])  # (D,LAT,LON)

depth = np.array(DEPTH_LEVELS_M, dtype=float)[None, :, None, None]
yr = np.arange(Y, dtype=float)[:, None, None, None]
truth = (0.9 * np.exp(-depth / 1500) + 0.1 * np.cos(la)[None, None] - 0.004 * yr
         + 0.03 * np.sin(3 * lo)[None, None]).astype(np.float32)
truth = np.broadcast_to(truth, (Y, D, LAT, LON)).copy()
truth[:, ~ocean] = np.nan

p = (0.01 + 0.15 * np.exp(-(((lat - 40) / 25) ** 2)))[None, None, :, None]
p = p * (0.3 + yr / Y) * np.exp(-depth / 800)
obs = ((rng.random((Y, D, LAT, LON)) < p) & ocean[None]).astype(np.uint8)

dens = np.zeros(obs.shape, dtype=np.float32)
n = 0
for dy in (-1, 0, 1):
    for dl in (-1, 0, 1):
        for dn in (-1, 0, 1):
            dens += np.roll(obs, (dy, dl, dn), axis=(0, 2, 3))
            n += 1
dens /= n
sigma = (0.01 + 0.06 * (1 - np.clip(dens / 0.05, 0, 1))).astype(np.float32)
pred = (truth + rng.normal(size=truth.shape) * sigma).astype(np.float32)
std = (sigma * 0.9).astype(np.float32)  # deliberately a bit overconfident
for arr in (pred, std, dens):
    arr[:, ~ocean] = np.nan if arr is not dens else 0

name = "MOCK_CESM2_omip1_temporal_hybrid_base_seed0"
d = Path(a.out) / name
d.mkdir(parents=True, exist_ok=True)
for f, arr in [("pred_mean", pred), ("pred_std", std), ("truth", truth), ("observed_mask", obs),
               ("ocean_mask", ocean.astype(np.uint8)), ("observation_density", dens)]:
    np.save(d / f"{f}.npy", arr)

(d / "metrics.json").write_text(json.dumps(metrics_for(pred, truth, std, obs), indent=2))
years = list(range(1948, 1948 + Y))
(d / "metadata.json").write_text(json.dumps({
    "dataset": "CESM2_omip1", "split": "temporal", "model": "hybrid", "seed": 0,
    "start_year": 1948, "years": years, "depth_labels_m": DEPTH_LEVELS_M,
    "latitude_convention": "TO_VERIFY (mock: index0 = -87.5, south to north)",
    "longitude_convention": "TO_VERIFY (mock: index0 = 2.5E, 0..360)",
    "oxygen_units": "TO_VERIFY (mock: normalised 0..1)", "normalization": "TO_VERIFY",
    "lat": [float(x) for x in lat], "lon": [float(x) for x in lon],
    "simulation_caveat": True, "mock": True}, indent=2))
(d / "config.json").write_text(json.dumps({
    "dataset": "CESM2_omip1", "split": "temporal", "model": "hybrid", "feature_set": "base",
    "seed": 0, "source": "mock", "note": "fake numbers for plumbing only"}, indent=2))
rows = ["year,rmse,r2"]
for i, y in enumerate(years):
    m = metrics_for(pred[i], truth[i])["all"]
    rows.append(f"{y},{m['rmse']:.5f},{m['r2']:.5f}")
(d / "summary_by_year.csv").write_text("\n".join(rows) + "\n")
print("wrote", d)
