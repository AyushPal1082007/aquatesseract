#!/usr/bin/env python
"""
Train the LSTM+Transformer hybrid on OceanVerse and write a full 4D reconstruction.

Example (from the folder that contains `data/CESM2_omip1/...`):
    python hybrid_train.py --dataset CESM2_omip1 --split spatial --quick        # smoke test
    python hybrid_train.py --dataset CESM2_omip1 --split spatial --epochs 30    # real run

Outputs in <out_dir>:
    pred_mean.npy   [years, 33, 180, 360]  predicted DO (NaN = land / year skipped)
    pred_std.npy    same shape             predicted uncertainty (1 sigma)
    obs_mask.npy    same shape, bool       True where the graph actually had an observation
    per_year.csv    per-year error and observation fraction
    run_info.json   settings + metrics
    best_model.pt   weights
Does not need torch-sparse / pyg-lib (it does its own batching), only torch + torch_geometric
(to unpickle the .pt graph files) + numpy.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap

from hybrid_model import LSTMTransformerHybrid

DATASETS = {
    "CESM2_omip1": dict(num_years=62, start_year=1948, tag="omip1", gt="OMIP1_do.npy"),
    "CESM2_omip2": dict(num_years=61, start_year=1958, tag="omip2", gt="OMIP2_do.npy"),
    "GFDL_ESM4": dict(num_years=95, start_year=1920, tag="GFDL", gt="GFDL_do.npy"),
}
KEYS = ("xf", "do", "geo", "m", "y")
NAN = float("nan")


def build_model(args, factor_dim=8, geo_dim=5, time_len=11, profile_len=33):
    if args.model == "hybrid":
        return LSTMTransformerHybrid(time_len=time_len, profile_len=profile_len,
                                     factor_dim=factor_dim, geo_dim=geo_dim)
    from baselines import BaselineAdapter   # authors' baselines, no torch-sparse needed
    return BaselineAdapter(args.model, args.hidden_dim, args.num_layers, factor_dim, geo_dim, time_len)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_year(root, tag, year):
    p = os.path.join(root, f"{tag}_graph", f"{year}.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def split_nodes(split, yi, n_nodes, n_train_years, tr_idx, va_idx):
    """Boolean masks [n_nodes] of which nodes are train / val in year index yi."""
    if split == "temporal":
        is_train_year = yi < n_train_years
        return (torch.full((n_nodes,), is_train_year, dtype=torch.bool),
                torch.full((n_nodes,), not is_train_year, dtype=torch.bool))
    tr = torch.zeros(n_nodes, dtype=torch.bool)
    va = torch.zeros(n_nodes, dtype=torch.bool)
    # random-split index files are numpy arrays, spatial ones are torch tensors -> accept both
    tr[torch.as_tensor(np.asarray(tr_idx[yi]), dtype=torch.long)] = True
    va[torch.as_tensor(np.asarray(va_idx[yi]), dtype=torch.long)] = True
    return tr, va


def pack(d, sel):
    return {"xf": d.x[sel].float(), "do": d.time_series_profile[sel].float(),
            "geo": d.x_geo[sel].float(), "m": d.mask[sel].float(), "y": d.y[sel].float()}


def build_sets(root, tag, start, years_used, split, n_train_years, tr_idx, va_idx):
    tr_parts, va_parts = [], []
    for k, yi in enumerate(years_used):
        d = load_year(root, tag, start + yi)
        n = d.y.shape[0]
        has_obs = (~torch.isnan(d.y)).any(dim=1)
        tr, va = split_nodes(split, yi, n, n_train_years, tr_idx, va_idx)
        if (tr & has_obs).any():
            tr_parts.append(pack(d, tr & has_obs))
        if (va & has_obs).any():
            va_parts.append(pack(d, va & has_obs))
        if (k + 1) % 10 == 0 or k + 1 == len(years_used):
            print(f"  loaded {k + 1}/{len(years_used)} years", flush=True)

    def cat(parts):
        if not parts:
            raise RuntimeError("empty train/val set - check split / years")
        return {key: torch.cat([p[key] for p in parts]) for key in KEYS}

    return cat(tr_parts), cat(va_parts)


def epoch_pass(model, data, device, bs, opt=None, mse_only=False, x_drop=0.0, do_drop=0.0):
    train = opt is not None
    model.train(train)
    n = data["y"].shape[0]
    order = torch.randperm(n) if train else torch.arange(n)
    se = ae = loss_sum = 0.0
    cnt = nb = 0
    with torch.enable_grad() if train else torch.no_grad():
        for i in range(0, n, bs):
            b = order[i:i + bs]
            xb = {k: data[k][b].to(device) for k in KEYS}
            if train and (x_drop > 0 or do_drop > 0):   # input dropout: teach the model what "no information" looks like
                nb_ = xb["xf"].shape[0]
                if x_drop > 0:
                    xb["xf"] = xb["xf"] * (torch.rand(nb_, 1, 1, device=device) >= x_drop).float()
                if do_drop > 0:
                    xb["do"] = xb["do"] * (torch.rand(nb_, 1, 1, device=device) >= do_drop).float()
            mu, lv = model(xb["xf"], xb["do"], xb["geo"], xb["m"])
            ok = ~torch.isnan(xb["y"])
            ys = xb["y"][ok] * model.do_scale
            err = mu[ok] - ys
            if mse_only:
                loss = (err ** 2).mean()
            else:  # Gaussian negative log-likelihood (observed cells only)
                loss = 0.5 * (lv[ok] + err ** 2 * torch.exp(-lv[ok])).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            e = err.detach() / model.do_scale  # back to original units
            se += (e ** 2).sum().item()
            ae += e.abs().sum().item()
            cnt += e.numel()
            loss_sum += loss.item()
            nb += 1
    return {"loss": loss_sum / max(nb, 1), "rmse": math.sqrt(se / max(cnt, 1)), "mae": ae / max(cnt, 1)}


def gaussian_crps(y, mu, sd):
    sd = sd.clamp_min(1e-8)
    z = (y - mu) / sd
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = torch.special.ndtr(z)
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def val_metrics(ys, mus, sds):
    if len(ys) == 0:
        return {}
    y, mu, sd = (torch.from_numpy(np.concatenate(a)).double() for a in (ys, mus, sds))
    err = mu - y
    sse = (err ** 2).sum().item()
    sst = ((y - y.mean()) ** 2).sum().item()
    z = err.abs() / sd.clamp_min(1e-8)
    return {"n_cells": int(y.numel()),
            "rmse": math.sqrt(sse / y.numel()),
            "mae": err.abs().mean().item(),
            "r2": 1 - sse / sst if sst > 0 else NAN,
            "crps": gaussian_crps(y, mu, sd).mean().item(),
            "coverage_90": (z <= 1.6449).double().mean().item(),
            "coverage_50": (z <= 0.6745).double().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--dataset", default="CESM2_omip1", choices=list(DATASETS))
    ap.add_argument("--split", default="spatial", choices=["random", "temporal", "spatial"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=2, help="MSE-only epochs before switching to NLL")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--model", default="hybrid", choices=["hybrid", "lstm", "transformer", "mlp"],
                    help="hybrid = ours; lstm/transformer/mlp = the authors' baselines run in OUR pipeline")
    ap.add_argument("--hidden_dim", type=int, default=None, help="baselines only (authors: lstm 256, transformer 32)")
    ap.add_argument("--num_layers", type=int, default=None, help="baselines only (authors: lstm 2, transformer 1)")
    ap.add_argument("--x_drop", type=float, default=0.0,
                    help="training only: probability of zeroing the whole x-feature block of a profile (mimics unobserved cells)")
    ap.add_argument("--do_drop", type=float, default=0.0,
                    help="training only: probability of zeroing the whole DO window of a profile")
    ap.add_argument("--select_by", default="rmse", choices=["rmse", "mae", "nll"],
                    help="which val number picks the best epoch (nll = val Gaussian NLL, ignores the MSE warm-up epochs)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--year_stride", type=int, default=1, help="use every k-th year (train AND infer)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, every 6th year")
    ap.add_argument("--infer_only", action="store_true",
                    help="skip training, load best_model.pt from --out_dir and only redo inference/metrics")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_years", type=int, default=None, help="override (testing only)")
    ap.add_argument("--start_year", type=int, default=None, help="override (testing only)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    if args.model != "hybrid":
        args.hidden_dim = args.hidden_dim or {"lstm": 256, "transformer": 32, "mlp": 256}[args.model]
        args.num_layers = args.num_layers or {"lstm": 2, "transformer": 1, "mlp": 2}[args.model]
    Y = args.num_years or cfg["num_years"]
    start = args.start_year or cfg["start_year"]
    tag = cfg["tag"]
    root = os.path.join(args.data_root, args.dataset)
    if args.quick:
        args.epochs, args.warmup_epochs, args.year_stride = 2, 1, max(args.year_stride, 6)
    out = args.out_dir or os.path.join("hybrid_out", f"{args.dataset}_{args.split}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  dataset={args.dataset}  split={args.split}  years={Y}  out={out}")
    set_seed(args.seed)

    n_train_years = int(Y * 0.7) + 1  # same rule as the authors' temporal split (44 of 62)
    tr_idx = va_idx = None
    if args.split != "temporal":
        fn = "spatial_split.pt" if args.split == "spatial" else f"random_split_seed{args.seed}.pt"
        s = torch.load(os.path.join(root, f"{tag}_indices", fn), map_location="cpu", weights_only=False)
        tr_idx, va_idx = s["train_indices"], s["val_indices"]

    years_used = list(range(0, Y, args.year_stride))
    if args.infer_only:
        model = build_model(args).to(device)
        model.load_state_dict(torch.load(os.path.join(out, "best_model.pt"), map_location=device))
        history, best_rmse = [], float("nan")
    else:
        print("loading data ...")
        train, val = build_sets(root, tag, start, years_used, args.split, n_train_years, tr_idx, va_idx)
        print(f"train profiles={train['y'].shape[0]}  val profiles={val['y'].shape[0]}")

        model = build_model(args, factor_dim=train["xf"].shape[-1], geo_dim=train["geo"].shape[-1],
                            time_len=train["do"].shape[-1], profile_len=train["y"].shape[-1]).to(device)
        print("parameters:", sum(p.numel() for p in model.parameters()))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        history, best_rmse, best_state, bad = [], 1e9, None, 0
        best_crit = 1e9
        for ep in range(args.epochs):
            t0 = time.time()
            mse_only = ep < args.warmup_epochs or not getattr(model, "has_uncertainty", True)
            tr = epoch_pass(model, train, device, args.batch_size, opt, mse_only, args.x_drop, args.do_drop)
            va = epoch_pass(model, val, device, args.batch_size * 4, None, mse_only)
            history.append({"epoch": ep, "train": tr, "val": va})
            print(f"epoch {ep:02d} {'MSE' if mse_only else 'NLL'}  train_loss={tr['loss']:.4f}  "
                  f"val_loss={va['loss']:.4f}  val_rmse={va['rmse']:.5f}  val_mae={va['mae']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
            crit = {"rmse": va["rmse"], "mae": va["mae"],
                    "nll": float("inf") if mse_only else va["loss"]}[args.select_by]
            if crit < best_crit:
                best_crit, best_rmse, bad = crit, va["rmse"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(out, "best_model.pt"))
            elif ep >= args.warmup_epochs:
                bad += 1
                if bad >= args.patience:
                    print("early stopping")
                    break
        model.load_state_dict(best_state)
    model.eval()

    # ------------------------------------------------------------------ inference on ALL nodes
    gt_path = os.path.join(root, f"{tag}_ground_truth", cfg["gt"])
    gt = np.load(gt_path, mmap_mode="r")
    assert gt.shape[0] == Y, f"ground truth has {gt.shape[0]} years, expected {Y}"
    shape = (Y, 33, 180, 360)
    mean_mm = open_memmap(os.path.join(out, "pred_mean.npy"), mode="w+", dtype=np.float32, shape=shape)
    std_mm = open_memmap(os.path.join(out, "pred_std.npy"), mode="w+", dtype=np.float32, shape=shape)
    obs_mm = open_memmap(os.path.join(out, "obs_mask.npy"), mode="w+", dtype=np.bool_, shape=shape)
    mean_mm[:] = np.nan
    std_mm[:] = np.nan
    obs_mm[:] = False

    rows, v_y, v_mu, v_sd = [], [], [], []
    map_pct = None
    fse = fn = use = un = 0.0
    hit90 = hit50 = 0
    print("inference on all nodes ...")
    for k, yi in enumerate(years_used):
        d = load_year(root, tag, start + yi)
        n = d.y.shape[0]
        xf, dos, geo = d.x.float(), d.time_series_profile.float(), d.x_geo.float()
        m, y = d.mask.float(), d.y.float()
        means, stds = [], []
        for i in range(0, n, 2048):
            sl = slice(i, i + 2048)
            mu, sd = model.predict(xf[sl].to(device), dos[sl].to(device), geo[sl].to(device), m[sl].to(device))
            means.append(mu.cpu())
            stds.append(sd.cpu())
        mean = torch.cat(means).clamp(-0.1, 1.0)
        std = torch.cat(stds)
        valid = m > 0.5
        nan_t = torch.full_like(mean, NAN)
        mean_v = torch.where(valid, mean, nan_t)
        std_v = torch.where(valid, std, nan_t)

        # node -> (lat, lon) grid cell, same convention as the authors' inference code
        lat = torch.rad2deg(torch.asin(geo[:, 2].clamp(-1, 1)))
        lon = torch.rad2deg(torch.atan2(geo[:, 1], geo[:, 0]))
        li = torch.round(lat + 89.5).long().clamp(0, 179).numpy()
        # ground-truth longitude axis is 0..360 (index j = j+0.5 deg E), node lon is -180..180,
        # so shift by 180 cells. Verified 100% exact match on the 1948 graph.
        oi = ((torch.round(lon + 179.5).long().clamp(0, 359) + 180) % 360).numpy()

        obs = ~torch.isnan(y)
        mg = np.full((33, 180, 360), np.nan, np.float32)
        sg = np.full((33, 180, 360), np.nan, np.float32)
        og = np.zeros((33, 180, 360), dtype=bool)
        mg[:, li, oi] = mean_v.numpy().T
        sg[:, li, oi] = std_v.numpy().T
        og[:, li, oi] = obs.numpy().T
        mean_mm[yi], std_mm[yi], obs_mm[yi] = mg, sg, og

        g = np.asarray(gt[yi], dtype=np.float32)
        if k == 0:  # sanity check: observed y must equal the ground truth at the mapped grid cell
            o = obs.numpy()
            if o.any():
                diff = np.abs(g[:, li, oi].T[o] - y.numpy()[o])
                diff = diff[~np.isnan(diff)]
                if diff.size:
                    map_pct = 100.0 * float(np.mean(diff < 1e-4))
                    print(f"  mapping check: {map_pct:.2f}% of observed cells match ground truth exactly "
                          f"(median diff {np.median(diff):.4f}, max {diff.max():.2e}, n={diff.size})")
                    print("    -> " + ("OK" if map_pct > 99 else
                                       "WARNING: mapping or y-vs-ground-truth mismatch (full-field metrics/plots not reliable)"))

        ok_full = ~np.isnan(g) & ~np.isnan(mg)
        err = mg - g
        se_full = float(np.sum(err[ok_full].astype(np.float64) ** 2))
        n_full = int(ok_full.sum())
        ok_un = ok_full & ~og
        se_un = float(np.sum(err[ok_un].astype(np.float64) ** 2))
        n_un = int(ok_un.sum())
        fse, fn, use, un = fse + se_full, fn + n_full, use + se_un, un + n_un
        okc = ok_un & ~np.isnan(sg)
        zc = np.abs(err[okc]) / np.maximum(sg[okc], 1e-8)
        hit90 += int(np.sum(zc <= 1.6449))
        hit50 += int(np.sum(zc <= 0.6745))

        tr_m, va_m = split_nodes(args.split, yi, n, n_train_years, tr_idx, va_idx)
        vm = (va_m.unsqueeze(1) & obs).numpy()
        val_rmse = NAN
        if vm.any():
            yy, mm_, ss = y.numpy()[vm], mean_v.numpy()[vm], std_v.numpy()[vm]
            v_y.append(yy), v_mu.append(mm_), v_sd.append(ss)
            val_rmse = float(np.sqrt(np.mean((mm_ - yy) ** 2)))
        role = ("train" if yi < n_train_years else "val") if args.split == "temporal" else "mixed"
        rows.append({"year": start + yi, "role": role, "obs_cells": int(obs.sum().item()),
                     "obs_frac": float(obs.sum().item() / max(int(valid.sum().item()), 1)),
                     "rmse_full": math.sqrt(se_full / max(n_full, 1)),
                     "rmse_unobserved": math.sqrt(se_un / max(n_un, 1)),
                     "rmse_val_cells": val_rmse})
        if (k + 1) % 5 == 0 or k + 1 == len(years_used):
            print(f"  inferred {k + 1}/{len(years_used)} years", flush=True)
    for mm in (mean_mm, std_mm, obs_mm):
        mm.flush()

    with open(os.path.join(out, "per_year.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = {"val_cells": val_metrics(v_y, v_mu, v_sd),
               "full_field_rmse": math.sqrt(fse / max(fn, 1)),
               "unobserved_cells_rmse": math.sqrt(use / max(un, 1)),
               "unobserved_coverage_90": hit90 / max(un, 1),
               "unobserved_coverage_50": hit50 / max(un, 1),
               "best_val_rmse_during_training": best_rmse}
    if not getattr(model, "has_uncertainty", True):   # baselines give no uncertainty
        for k in ("crps", "coverage_90", "coverage_50"):
            if metrics["val_cells"]:
                metrics["val_cells"][k] = None
        metrics["unobserved_coverage_90"] = metrics["unobserved_coverage_50"] = None
    info = {"dataset": args.dataset, "model": args.model, "split": args.split, "seed": args.seed, "num_years": Y,
            "start_year": start, "n_train_years": n_train_years, "year_stride": args.year_stride, "x_drop": args.x_drop, "do_drop": args.do_drop,
            "inferred_year_idx": years_used, "quick": args.quick, "units": "raw dataset units (same as y)", "lon_axis": "ground-truth convention: index j = (j+0.5) deg E (0..360)",
            "mapping_exact_pct": map_pct, "metrics": metrics, "history": history}
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("\n=== RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
