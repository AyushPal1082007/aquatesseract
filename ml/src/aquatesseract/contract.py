"""Single source of truth for the shared data contract. See docs/DATA_CONTRACT.md.

Nobody changes array order, units, masking or coordinates silently.
Change here + docs + server together, in one PR, reviewed by all ML members.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAY_ORDER = ("year", "depth", "lat", "lon")
N_DEPTH = 33
# Table 5 of the OceanVerse paper: depth level centres (m)
DEPTH_LEVELS_M = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800,
                  900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000, 3500, 4000,
                  4500, 5000, 5500]
assert len(DEPTH_LEVELS_M) == N_DEPTH

REQUIRED_FILES = ["pred_mean.npy", "pred_std.npy", "truth.npy", "observed_mask.npy",
                  "ocean_mask.npy", "metrics.json", "config.json", "metadata.json"]
OPTIONAL_FILES = ["observation_density.npy", "physics_audit.npy", "summary_by_year.csv"]
METADATA_KEYS = ["dataset", "split", "model", "seed", "start_year", "years", "depth_labels_m",
                 "latitude_convention", "longitude_convention", "oxygen_units", "normalization",
                 "simulation_caveat"]
METRIC_GROUPS = ["all", "observed", "never_observed"]
METRIC_KEYS = ["rmse", "mae", "r2", "n"]
UNC_KEYS = ["coverage90", "mean_width90", "coverage90_observed", "coverage90_never_observed"]
Z90 = 1.645
RUN_NAME_FMT = "{dataset}_{split}_{model}_{feature_set}_seed{seed}"


def metrics_for(pred: np.ndarray, truth: np.ndarray, std: np.ndarray | None = None,
                observed: np.ndarray | None = None) -> dict:
    """THE evaluator core. Land = NaN in truth and is excluded. error = prediction - truth."""
    ocean = np.isfinite(truth)
    obs = (observed.astype(bool) & ocean) if observed is not None else np.zeros_like(ocean)
    groups = {"all": ocean, "observed": obs, "never_observed": ocean & ~obs}
    out: dict = {}
    for name, m in groups.items():
        n = int(m.sum())
        if n == 0:
            out[name] = {"rmse": None, "mae": None, "r2": None, "n": 0}
            continue
        e = pred[m].astype(np.float64) - truth[m]
        ss_tot = float(((truth[m] - truth[m].mean()) ** 2).sum())
        out[name] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                     "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else None, "n": n}
    if std is not None:
        inside = np.abs(pred - truth) <= Z90 * std

        def cov(m):
            return float(inside[m].mean()) if m.any() else None

        out["uncertainty"] = {"coverage90": cov(groups["all"]),
                              "mean_width90": float((2 * Z90 * std[groups["all"]]).mean()),
                              "coverage90_observed": cov(groups["observed"]),
                              "coverage90_never_observed": cov(groups["never_observed"])}
    return out


def validate_run(run_dir: str | Path) -> list[str]:
    """Return list of problems. Empty list = run folder obeys contract."""
    d = Path(run_dir)
    errs: list[str] = []
    for f in REQUIRED_FILES:
        if not (d / f).exists():
            errs.append(f"missing file: {f}")
    if errs:
        return errs
    meta = json.loads((d / "metadata.json").read_text())
    errs += [f"metadata missing key: {k}" for k in METADATA_KEYS if k not in meta]
    if meta.get("simulation_caveat") is not True:
        errs.append("metadata.simulation_caveat must be true")
    L = {k: np.load(d / f, mmap_mode="r") for k, f in
         [("mean", "pred_mean.npy"), ("std", "pred_std.npy"), ("truth", "truth.npy"),
          ("obs", "observed_mask.npy"), ("ocean", "ocean_mask.npy")]}
    shape = L["truth"].shape
    if len(shape) != 4:
        return errs + [f"truth must be 4D {ARRAY_ORDER}, got {shape}"]
    for k in ("mean", "std", "obs"):
        if L[k].shape != shape:
            errs.append(f"{k} shape {L[k].shape} != truth shape {shape}")
    if L["ocean"].shape != shape[1:]:
        errs.append(f"ocean_mask shape {L['ocean'].shape} != {shape[1:]} (depth,lat,lon)")
    if shape[1] != N_DEPTH or len(meta.get("depth_labels_m", [])) != shape[1]:
        errs.append(f"depth axis must be {N_DEPTH} and match depth_labels_m")
    if len(meta.get("years", [])) != shape[0]:
        errs.append("len(metadata.years) != shape[0]")
    for k in ("mean", "std", "truth"):
        if L[k].dtype != np.float32:
            errs.append(f"{k} dtype must be float32, got {L[k].dtype}")
    for k in ("obs", "ocean"):
        if L[k].dtype != np.uint8:
            errs.append(f"{k} dtype must be uint8 (0/1), got {L[k].dtype}")
    if errs:
        return errs
    ocean = np.broadcast_to(np.asarray(L["ocean"]).astype(bool), shape)
    mean, std, truth = (np.asarray(L[k]) for k in ("mean", "std", "truth"))
    if not np.isfinite(mean[ocean]).all():
        errs.append("pred_mean has NaN/Inf on ocean cells")
    if not (np.isfinite(std[ocean]).all() and (std[ocean] > 0).all()):
        errs.append("pred_std must be finite and > 0 on ocean cells")
    if not np.isfinite(truth[ocean]).all():
        errs.append("truth has NaN on ocean cells")
    if np.isfinite(truth[~ocean]).any():
        errs.append("land/invalid cells must be NaN in truth (never 0)")
    if np.asarray(L["obs"])[~ocean].any():
        errs.append("observed_mask set on land cells")
    m = json.loads((d / "metrics.json").read_text())
    for g in METRIC_GROUPS:
        if g not in m or any(k not in m[g] for k in METRIC_KEYS):
            errs.append(f"metrics.json group '{g}' incomplete")
    return errs
