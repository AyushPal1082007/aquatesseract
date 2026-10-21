#!/usr/bin/env python
"""
Make the visualisation graphs from the output of hybrid_train.py (only needs numpy + matplotlib).

    python plot_results.py --out_dir hybrid_out/CESM2_omip1_spatial_seed1 ^
        --gt data/CESM2_omip1/omip1_ground_truth/OMIP1_do.npy

Optional: --lat 10.5 --lon -140.5 --depth_idx 5 --year 1990
(if lat/lon are not given, the most-observed ocean cell at that depth is used)
Plots are saved to <out_dir>/plots/
"""
import argparse
import csv
import json
import os
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LATS = np.arange(-89.5, 90, 1.0)


def lon_to_idx(lon):
    """-180..180 degrees -> index on the ground-truth axis (0..360, index j = j+0.5 deg E)."""
    return (int(round(lon + 179.5)) + 180) % 360


def idx_to_lon(i):
    """index on the ground-truth axis -> longitude in -180..180."""
    return ((i + 0.5 + 180) % 360) - 180


def nanf(v):
    try:
        return float(v)
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gt", required=True, help="path to the ground-truth .npy (e.g. OMIP1_do.npy)")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--depth_idx", type=int, default=5, help="0 = surface, 32 = deepest level")
    ap.add_argument("--year", type=int, default=None, help="calendar year for map / depth profile")
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.out_dir, "run_info.json")))
    start, Y = info["start_year"], info["num_years"]
    years = np.arange(start, start + Y)
    done = list(info["inferred_year_idx"])
    pm = np.load(os.path.join(a.out_dir, "pred_mean.npy"), mmap_mode="r")
    ps = np.load(os.path.join(a.out_dir, "pred_std.npy"), mmap_mode="r")
    ob = np.load(os.path.join(a.out_dir, "obs_mask.npy"), mmap_mode="r")
    gt = np.load(a.gt, mmap_mode="r")
    plots = os.path.join(a.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    d = a.depth_idx
    temporal = info["split"] == "temporal"
    val_start_year = start + info["n_train_years"]

    # choose a location: given, or the most-observed ocean cell at this depth
    if a.lat is None or a.lon is None:
        cnt = np.zeros((180, 360), dtype=np.int32)
        for yi in done:
            cnt += np.asarray(ob[yi, d]).astype(np.int32)
        li, oi = np.unravel_index(np.argmax(cnt), cnt.shape)
        print(f"auto-picked location: lat={LATS[li]:.1f} lon={idx_to_lon(oi):.1f} ({cnt[li, oi]} observed years)")
    else:
        li, oi = int(round(a.lat + 89.5)), lon_to_idx(a.lon)
    lat_c, lon_c = LATS[li], idx_to_lon(oi)
    yi_map = done[len(done) // 2] if a.year is None else int(a.year - start)
    if yi_map not in done:
        yi_map = min(done, key=lambda v: abs(v - yi_map))
    print(f"map / profile year: {start + yi_map}")

    # ---- 1. time series at one location and depth ---------------------------------------------
    g = np.asarray(gt[:, d, li, oi], dtype=float)
    p = np.asarray(pm[:, d, li, oi], dtype=float)
    s = np.asarray(ps[:, d, li, oi], dtype=float)
    o = np.asarray(ob[:, d, li, oi], dtype=bool)
    if np.isnan(g).all():
        raise SystemExit("that cell is land in the ground truth - pick another --lat/--lon")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(years, g, "k-", lw=1.5, label="ground truth (simulation)")
    ax.plot(years, p, "-o", color="tab:blue", ms=3, lw=1.5, label="hybrid prediction")
    ax.fill_between(years, p - 1.645 * s, p + 1.645 * s, color="tab:blue", alpha=0.2, label="90% interval")
    if o.any():
        ax.scatter(years[o], g[o], s=28, color="tab:orange", zorder=5, label="year had an observation")
    if temporal:
        ax.axvspan(val_start_year, years[-1] + 0.5, color="grey", alpha=0.12, label="validation years")
    ax.set_xlabel("year")
    ax.set_ylabel("dissolved oxygen (dataset units)")
    ax.set_title(f"DO time series  lat {lat_c:.1f}, lon {lon_c:.1f}, depth level {d}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "1_timeseries_location.png"), dpi=150)
    plt.close(fig)

    # ---- 2. global (area-weighted) mean time series --------------------------------------------
    w2 = np.cos(np.deg2rad(LATS))[:, None] * np.ones((1, 360))
    def wmean(field2d_or_3d, other):
        ok = ~np.isnan(field2d_or_3d) & ~np.isnan(other)
        ww = np.broadcast_to(w2, field2d_or_3d.shape)
        return np.sum(ww[ok] * field2d_or_3d[ok]) / max(np.sum(ww[ok]), 1e-12) if ok.any() else np.nan
    gl, pl, ga, pa = [], [], [], []
    for yi in done:
        gg, pp = np.asarray(gt[yi], dtype=np.float32), np.asarray(pm[yi], dtype=np.float32)
        gl.append(wmean(gg[d], pp[d]))
        pl.append(wmean(pp[d], gg[d]))
        ga.append(wmean(gg, pp))
        pa.append(wmean(pp, gg))
    yrs = years[done]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (gv, pv, ttl) in zip(axs, [(gl, pl, f"depth level {d}"), (ga, pa, "all depth levels")]):
        ax.plot(yrs, gv, "k-", label="ground truth")
        ax.plot(yrs, pv, "-o", ms=3, color="tab:blue", label="hybrid prediction")
        if temporal:
            ax.axvspan(val_start_year, yrs[-1] + 0.5, color="grey", alpha=0.12)
        ax.set_title(f"Area-weighted global mean DO, {ttl}")
        ax.set_xlabel("year")
        ax.legend()
    axs[0].set_ylabel("DO (dataset units)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "2_global_mean_timeseries.png"), dpi=150)
    plt.close(fig)

    # ---- 3. maps: truth / prediction / error / uncertainty -------------------------------------
    def roll(x):
        return np.roll(np.asarray(x, dtype=float), 180, axis=-1)   # 0..360 axis -> -180..180 for display
    gm, pmap, smap = roll(gt[yi_map, d]), roll(pm[yi_map, d]), roll(ps[yi_map, d])
    vmin, vmax = np.nanmin(gm), np.nanmax(gm)
    ext = [-180, 180, -90, 90]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    panels = [(gm, "ground truth", "viridis", vmin, vmax), (pmap, "prediction", "viridis", vmin, vmax),
              (pmap - gm, "error (pred - truth)", "RdBu_r", -np.nanmax(np.abs(pmap - gm)), np.nanmax(np.abs(pmap - gm))),
              (smap, "predicted uncertainty (1 sigma)", "magma", 0, np.nanmax(smap))]
    for ax, (arr, ttl, cm, lo, hi) in zip(axs.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(f"{ttl}  (year {start + yi_map}, depth level {d})")
        fig.colorbar(im, ax=ax, fraction=0.025)
    oy, ox = np.nonzero(np.roll(np.asarray(ob[yi_map, d]), 180, axis=-1))
    axs[0, 1].scatter(ox - 179.5, LATS[oy], s=3, c="red", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "3_maps.png"), dpi=150)
    plt.close(fig)

    # ---- 4. depth profile at the location ------------------------------------------------------
    gp, pp_, sp = (np.asarray(x[yi_map, :, li, oi], dtype=float) for x in (gt, pm, ps))
    op = np.asarray(ob[yi_map, :, li, oi], dtype=bool)
    lv = np.arange(33)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(gp, lv, "k-", label="ground truth")
    ax.plot(pp_, lv, "-o", ms=3, color="tab:blue", label="prediction")
    ax.fill_betweenx(lv, pp_ - 1.645 * sp, pp_ + 1.645 * sp, color="tab:blue", alpha=0.2, label="90% interval")
    if op.any():
        ax.scatter(gp[op], lv[op], s=25, color="tab:orange", zorder=5, label="observed")
    ax.invert_yaxis()
    ax.set_xlabel("DO (dataset units)")
    ax.set_ylabel("depth level (0 = surface)")
    ax.set_title(f"Depth profile, lat {lat_c:.1f} lon {lon_c:.1f}, {start + yi_map}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "4_depth_profile.png"), dpi=150)
    plt.close(fig)

    # ---- sample cells for plots 5 and 7 --------------------------------------------------------
    rng = np.random.default_rng(0)
    sg, sp_, ss, so = [], [], [], []
    for yi in done[::max(1, len(done) // 6)][:6]:
        G, P, S = (np.asarray(x[yi], dtype=np.float32).ravel() for x in (gt, pm, ps))
        obs_flat = np.asarray(ob[yi]).ravel()
        okc = np.flatnonzero(~np.isnan(G) & ~np.isnan(P) & ~np.isnan(S))
        if okc.size:
            pick = rng.choice(okc, size=min(40000, okc.size), replace=False)
            sg.append(G[pick])
            sp_.append(P[pick])
            ss.append(S[pick])
            so.append(obs_flat[pick])
    sg, sp_, ss, so = np.concatenate(sg), np.concatenate(sp_), np.concatenate(ss), np.concatenate(so)

    # ---- 5. predicted vs true scatter ----------------------------------------------------------
    r2 = 1 - np.sum((sp_ - sg) ** 2) / np.sum((sg - sg.mean()) ** 2)
    rmse = np.sqrt(np.mean((sp_ - sg) ** 2))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(sg, sp_, gridsize=60, bins="log", cmap="viridis")
    lim = [min(sg.min(), sp_.min()), max(sg.max(), sp_.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("prediction")
    ax.set_title(f"All ocean cells (sampled): R2={r2:.3f}, RMSE={rmse:.4f}")
    fig.colorbar(hb, ax=ax, label="count (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "5_scatter_pred_vs_true.png"), dpi=150)
    plt.close(fig)

    # ---- 6. error vs sparsity (the MNAR story) -------------------------------------------------
    rows = list(csv.DictReader(open(os.path.join(a.out_dir, "per_year.csv"))))
    yr = np.array([nanf(r["year"]) for r in rows])
    frac = np.array([nanf(r["obs_frac"]) for r in rows]) * 100
    rf = np.array([nanf(r["rmse_full"]) for r in rows])
    ru = np.array([nanf(r["rmse_unobserved"]) for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(yr, rf, "-o", ms=3, color="tab:blue", label="RMSE, all ocean cells")
    axs[0].plot(yr, ru, "-o", ms=3, color="tab:green", label="RMSE, never-observed cells")
    axs[0].set_xlabel("year")
    axs[0].set_ylabel("RMSE (dataset units)")
    tw = axs[0].twinx()
    tw.bar(yr, frac, color="tab:orange", alpha=0.25, width=0.8)
    tw.set_ylabel("observed cells (% of ocean cells)", color="tab:orange")
    axs[0].legend(loc="upper left", fontsize=8)
    axs[0].set_title("Error and data availability per year")
    sc = axs[1].scatter(frac, ru, c=yr, cmap="plasma")
    axs[1].set_xlabel("observed cells (% of ocean cells)")
    axs[1].set_ylabel("RMSE, never-observed cells")
    axs[1].set_title("Does more data mean less error?")
    fig.colorbar(sc, ax=axs[1], label="year")
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "6_error_vs_sparsity.png"), dpi=150)
    plt.close(fig)

    # ---- 7. uncertainty calibration (observed vs never-observed cells) -------------------------
    z = np.abs(sg - sp_) / np.maximum(ss, 1e-8)
    nominal = np.linspace(0.05, 0.99, 30)
    zc = [NormalDist().inv_cdf(0.5 + q / 2) for q in nominal]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for mask, lab, col in [(~so, "never-observed cells", "tab:red"), (so, "observed cells (train+val)", "tab:blue")]:
        if mask.sum() > 100:
            emp = [np.mean(z[mask] <= c) for c in zc]
            ax.plot(nominal, emp, "-o", ms=3, color=col, label=f"{lab} (n={int(mask.sum())})")
            i90 = int(np.argmin(np.abs(nominal - 0.90)))
            print(f"calibration, {lab}: nominal {nominal[i90]:.2f} -> actual {emp[i90]:.2f}")
    ax.set_xlabel("nominal coverage of the interval")
    ax.set_ylabel("actual coverage")
    ax.set_title("Uncertainty calibration\n(below the line = overconfident)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots, "7_calibration.png"), dpi=150)
    plt.close(fig)

    print("saved plots to", plots)
    for f in sorted(os.listdir(plots)):
        print("  ", f)


if __name__ == "__main__":
    main()
