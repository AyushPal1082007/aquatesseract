# Shared data contract (v0 DRAFT until M1/M2/M3/M4/M5 sign off)

Items marked **TO VERIFY** need M3 to confirm against real OceanVerse before freeze.

## Run folder
`results/{dataset}_{split}_{model}_{feature_set}_seed{n}/`

| File | Shape / dtype | Notes |
|---|---|---|
| `truth.npy` | `(Y, 33, LAT, LON)` float32 | NaN on land / below seafloor. Never 0 |
| `pred_mean.npy` | same | NaN on land |
| `pred_std.npy` | same | finite and > 0 on ocean cells |
| `observed_mask.npy` | same, uint8 | 1 = real-sampling-pattern observation exists |
| `ocean_mask.npy` | `(33, LAT, LON)` uint8 | 1 = valid ocean cell (static over years) |
| `observation_density.npy` | `(Y, 33, LAT, LON)` float32 | optional. fraction of observed cells in neighbourhood |
| `physics_audit.npy` | TBD by M3 | optional. SAS layer |
| `metrics.json` | see below | written by shared evaluator only |
| `config.json` | | `dataset, split, model, feature_set, seed, source` (`reproduced` or `hybrid`) |
| `metadata.json` | | see keys below |
| `summary_by_year.csv` | | optional |

- Array order: `(year, depth, latitude, longitude)`. Depth = 33 levels from paper Table 5 (0..5500 m).
- Missing: NaN in NumPy, `null` in JSON.
- `error = prediction - truth`.
- 90% interval: `mean ± 1.645 * std` (Gaussian head).

## metadata.json keys
`dataset, split, model, seed, start_year, years[], depth_labels_m[], latitude_convention, longitude_convention, oxygen_units, normalization, simulation_caveat (must be true)`; optional `lat[]`, `lon[]`, `mock`.

TO VERIFY: latitude order (south->north?), longitude range (0..360 or -180..180), oxygen units + normalisation (paper RMSE ~0.05 implies normalised), whether the grid is 180x360.

## metrics.json
```json
{
  "all":            {"rmse":0,"mae":0,"r2":0,"n":0},
  "observed":       {"rmse":0,"mae":0,"r2":0,"n":0},
  "never_observed": {"rmse":0,"mae":0,"r2":0,"n":0},
  "uncertainty": {"coverage90":0,"mean_width90":0,"coverage90_observed":0,"coverage90_never_observed":0}
}
```
`uncertainty` only for models with a std head.

## API (read-only, precomputed)
`GET /health /runs /metadata /metrics /comparison /map /timeseries /profile /density /physics-audit`

Params: `run` (folder name), `year` (calendar), `depth` (level index 0..32), `lat`/`lon` (array indices), `layer` = `truth|prediction|uncertainty|error|observed|density`, optional `stride` (downsample).
Errors: `{ "error": { "code", "message" } }`. Land cell -> 422 `land_cell`.

## Validate
`make check` (runs `validate_run()` on every folder).
