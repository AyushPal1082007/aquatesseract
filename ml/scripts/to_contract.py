#!/usr/bin/env python
"""
Convert a run folder made by hybrid_train.py into the AquaTesseract run-folder contract
(docs/DATA_CONTRACT.md), then run the shared evaluator (contract.metrics_for) and the validator
(contract.validate_run) on it.  Only needs numpy + contract.py.

  python to_contract.py --run_dir /content/out/spatial_drop \
         --gt data/CESM2_omip1/omip1_ground_truth/OMIP1_do.npy --results_dir results

Output folder: results/{dataset}_{split}_{model}_{feature_set}_seed{seed}/
Needs a lot of RAM at the metrics step (about 5 GB); use a High-RAM Colab runtime if it crashes.
"""
import argparse
import json
import os
import sys

import numpy as np
from numpy.lib.format import open_memmap

try:
    from aquatesseract.contract import DEPTH_LEVELS_M, RUN_NAME_FMT, metrics_for, validate_run
except ImportError:  # contract.py next to this script (e.g. in Colab)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from contract import DEPTH_LEVELS_M, RUN_NAME_FMT, metrics_for, validate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="folder written by hybrid_train.py")
    ap.add_argument("--gt", required=True, help="OMIP1_do.npy (ground truth)")
    ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--feature_set", default=None, help="default: 'xdrop' if input dropout was used, else 'full'")
    ap.add_argument("--placeholder_std", type=float, default=1.0,
                    help="models without a std head (lstm/transformer/mlp) still need a pred_std file")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.run_dir, "run_info.json")))
    dataset, split, seed = info["dataset"], info["split"], int(info["seed"])
    model = info.get("model", "hybrid")
    Y, start = int(info["num_years"]), int(info["start_year"])
    if sorted(info["inferred_year_idx"]) != list(range(Y)):
        sys.exit("ERROR: this run was not inferred for every year (quick run / year_stride > 1). The contract needs all years.")
    has_unc = model == "hybrid"
    feature_set = a.feature_set or ("xdrop" if (info.get("x_drop") or info.get("do_drop")) else "full")
    run_name = RUN_NAME_FMT.format(dataset=dataset, split=split, model=model, feature_set=feature_set, seed=seed)
    out = os.path.join(a.results_dir, run_name)
    if os.path.exists(out) and not a.overwrite:
        sys.exit(f"ERROR: {out} exists (use --overwrite)")
    os.makedirs(out, exist_ok=True)
    print("run name:", run_name)

    gt = np.load(a.gt, mmap_mode="r")
    pm = np.load(os.path.join(a.run_dir, "pred_mean.npy"), mmap_mode="r")
    ps = np.load(os.path.join(a.run_dir, "pred_std.npy"), mmap_mode="r")
    ob = np.load(os.path.join(a.run_dir, "obs_mask.npy"), mmap_mode="r")
    shape = gt.shape
    if len(shape) != 4 or shape[0] != Y or shape[1] != len(DEPTH_LEVELS_M):
        sys.exit(f"ERROR: ground truth shape {shape} does not match (years={Y}, depth={len(DEPTH_LEVELS_M)}, lat, lon)")
    for name, arr in (("pred_mean", pm), ("pred_std", ps), ("obs_mask", ob)):
        if arr.shape != shape:
            sys.exit(f"ERROR: {name} shape {arr.shape} != ground truth shape {shape}")

    fin = np.isfinite(gt)
    ocean = fin[0].copy()
    if not ((fin.all(axis=0) == ocean).all() and (fin.any(axis=0) == ocean).all()):
        sys.exit("ERROR: the NaN pattern of the ground truth changes between years; the contract needs a static ocean_mask")
    del fin
    print(f"ocean cells per year: {int(ocean.sum())} of {ocean.size}")

    t = open_memmap(os.path.join(out, "truth.npy"), mode="w+", dtype=np.float32, shape=shape)
    m_ = open_memmap(os.path.join(out, "pred_mean.npy"), mode="w+", dtype=np.float32, shape=shape)
    s_ = open_memmap(os.path.join(out, "pred_std.npy"), mode="w+", dtype=np.float32, shape=shape)
    o_ = open_memmap(os.path.join(out, "observed_mask.npy"), mode="w+", dtype=np.uint8, shape=shape)
    stat = {"ocean_cells_without_prediction": 0, "bad_std_on_ocean": 0, "prediction_on_land_removed": 0, "obs_on_land_removed": 0}
    for yi in range(Y):
        g = np.asarray(gt[yi], dtype=np.float32)
        p = np.asarray(pm[yi], dtype=np.float32)
        s = np.asarray(ps[yi], dtype=np.float32)
        o = np.asarray(ob[yi]).astype(bool)
        stat["ocean_cells_without_prediction"] += int((ocean & ~np.isfinite(p)).sum())
        stat["prediction_on_land_removed"] += int((~ocean & np.isfinite(p)).sum())
        stat["obs_on_land_removed"] += int((~ocean & o).sum())
        if has_unc:
            stat["bad_std_on_ocean"] += int((ocean & ~(np.isfinite(s) & (s > 0))).sum())
            s = np.where(ocean, s, np.nan)
        else:
            s = np.where(ocean, a.placeholder_std, np.nan)
        t[yi] = g
        m_[yi] = np.where(ocean, p, np.nan)
        s_[yi] = s.astype(np.float32)
        o_[yi] = (o & ocean).astype(np.uint8)
    for mm in (t, m_, s_, o_):
        mm.flush()
    np.save(os.path.join(out, "ocean_mask.npy"), ocean.astype(np.uint8))
    print("conversion checks:", json.dumps(stat))
    if stat["ocean_cells_without_prediction"] or stat["bad_std_on_ocean"]:
        sys.exit("ERROR: some ocean cells have no valid prediction/std (see counts above). "
                 "Not writing metrics. Send these counts to the team; the grid mapping or node coverage needs a look.")

    n_lat, n_lon = shape[2], shape[3]
    meta = {"dataset": dataset, "split": split, "model": model, "seed": seed, "start_year": start,
            "years": list(range(start, start + Y)), "depth_labels_m": DEPTH_LEVELS_M,
            "latitude_convention": "south_to_north: row index i = latitude -89.5 + i degrees north (1-degree cell centres, 180 rows). Verified.",
            "longitude_convention": "0_to_360_east: column index j = longitude 0.5 + j degrees east (1-degree cell centres, 360 columns), NOT -180..180. Verified.",
            "oxygen_units": "NOT confirmed: raw values of the OceanVerse ground-truth file (range about 0 to 0.41)",
            "normalization": "NOT confirmed: values are used exactly as stored in the ground-truth file; no extra scaling applied",
            "simulation_caveat": True,
            "lat": [float(-89.5 + i) for i in range(n_lat)], "lon": [float(0.5 + j) for j in range(n_lon)]}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)

    cfg = {"dataset": dataset, "split": split, "model": model, "feature_set": feature_set, "seed": seed,
           "source": "hybrid" if model == "hybrid" else "reproduced", "has_uncertainty": has_unc,
           "pred_std_is_placeholder": (not has_unc), "placeholder_std": (None if has_unc else a.placeholder_std),
           "training": {"x_drop": info.get("x_drop", 0.0), "do_drop": info.get("do_drop", 0.0),
                        "epochs_run": len(info.get("history", [])), "mapping_exact_pct": info.get("mapping_exact_pct")},
           "pipeline_metrics_UNOFFICIAL_not_for_slides": info.get("metrics"),
           "notes": "ground truth is a CESM2 simulation. Model inputs include the 11-year DO window and the x features of the "
                    "same cell; x is zero where no measurement exists (see input dropout in training.x_drop)."}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    print("computing shared-evaluator metrics (needs RAM) ...")
    truth = np.load(os.path.join(out, "truth.npy"))
    pred = np.load(os.path.join(out, "pred_mean.npy"))
    obs = np.load(os.path.join(out, "observed_mask.npy"))
    std = np.load(os.path.join(out, "pred_std.npy")) if has_unc else None
    metrics = metrics_for(pred, truth, std, obs)
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    del truth, pred, obs, std
    print(json.dumps(metrics, indent=2))

    nm = info.get("metrics", {})
    if "full_field_rmse" in nm:
        print(f"cross-check vs our own numbers: all.rmse {metrics['all']['rmse']:.5f} vs full_field_rmse {nm['full_field_rmse']:.5f} | "
              f"never_observed.rmse {metrics['never_observed']['rmse']:.5f} vs unobserved_cells_rmse {nm['unobserved_cells_rmse']:.5f}")

    errs = validate_run(out)
    print("\nvalidate_run:", "OK, obeys the contract" if not errs else "PROBLEMS:")
    for e in errs:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
