# AquaTesseract

Sparse ocean dissolved-oxygen reconstruction on OceanVerse + trust analysis (uncertainty, observation density, physics audits) + Reconstruction Explorer.

> Results are evaluated on a **simulated** ocean. Predictions are not new measurements.

## Repo map

| Path | Owner | What |
|---|---|---|
| `ml/src/aquatesseract/contract.py` | M1+M2+M3 (shared) | data contract constants, `metrics_for()` shared evaluator core, `validate_run()` |
| `ml/scripts/` | ML team | `make_mock_run.py`, `validate_run.py`; put `train_hybrid.py` etc. here |
| `server/` | M4 | Express API, Postgres (run registry, published numbers). Reads run folders from `results/` |
| `client/` | M5 | Vite + React Reconstruction Explorer |
| `results/` | all | run folders (git-ignored). `{dataset}_{split}_{model}_{feature_set}_seed{n}/` |
| `data/` | all | raw OceanVerse (git-ignored) + `CHECKSUMS.txt` |
| `docs/DATA_CONTRACT.md` | all | the frozen contract. Read first |

Flow: `ML -> results/<run>/ -> read-only API -> frontend`. No live inference.

## Pinned toolchain (same for everyone)

| Tool | Version | Where pinned |
|---|---|---|
| Node | 20.15.1 | `.nvmrc`, `engines`, Dockerfiles |
| npm deps | exact versions + `package-lock.json` | always `npm ci`, never `npm install` |
| Python | 3.11 | `.python-version`, CI |
| Python deps | exact | `ml/requirements.txt` |
| torch 2.3.1 | per hardware | `ml/requirements-torch.txt` |
| Postgres | 16.3 | `docker-compose.yml` |

## Quick start

### A. Docker (M4/M5, anyone who only needs the app)
```bash
cp .env.example .env
python3 -m venv ml/.venv && ml/.venv/bin/pip install -r ml/requirements.txt && ml/.venv/bin/pip install -e ml
make mock        # fake run in real format
make up          # db :5432, api :4000, web :5173
```
Open http://localhost:5173 . API check: http://localhost:4000/health

### B. Local (ML members, or no Docker)
```bash
nvm use                 # node 20.15.1
make setup              # npm ci x2, python venv, deps
# then install torch for your hardware: ml/requirements-torch.txt
make mock
docker compose up db    # only postgres in docker
make dev-server         # terminal 1  (works without db too, file-only)
make dev-client         # terminal 2
```

## Rules that kill "works on my machine"

1. **Install only from lockfiles.** `npm ci`, `pip install -r ml/requirements.txt`. Need new dep? Add exact version, commit lockfile, say so in chat.
2. **No absolute paths.** Use env vars (`RESULTS_DIR`, `DATABASE_URL`) or repo-relative paths. Config from `.env`, never committed.
3. **No big files in git.** `.npy/.nc/.pt` ignored; pre-commit blocks >2 MB. Share data by checksum (`data/CHECKSUMS.txt`).
4. **One contract.** Array order `(year, depth, lat, lon)`, NaN on land, `error = prediction - truth`. Change only via PR touching `contract.py` + `docs/DATA_CONTRACT.md` + server together.
5. **Every run folder must pass `make check`** before M4/M5 touch it.
6. **One evaluator.** Metrics come from `contract.metrics_for()` only. No number in slides unless M2's evaluator made it.
7. **Label sources.** `published` (paper) vs `reproduced` (ours, baselines) vs `hybrid` (ours). Never mixed.
8. **Seeds fixed and recorded** in `config.json`. Run naming: `{dataset}_{split}_{model}_{feature_set}_seed{n}`.
9. **CI must be green** to merge. `pre-commit install` once.
10. **PRs, small, 1 review.** Branches `m1/...`, `m4/...`. Never push to `main`.

## Daily status post (twice a day)
```
Done: / Blocked: / Next: / Files changed: / Result or log location:
```

## Troubleshooting
- `/health` says `db:false` -> postgres not up. API still serves files; `/comparison` loses published rows.
- `make check` FAIL "NaN" -> land filled with 0. Must be NaN in `truth`/`pred_*`.
- Map upside-down / shifted 180° -> lat/lon convention unverified. M3 verifies, fix `metadata.json`, not the client.
- Port in use -> change `SERVER_PORT` / `CLIENT_PORT` in `.env`.
