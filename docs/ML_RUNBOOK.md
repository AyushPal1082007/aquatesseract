# ML Runbook

## Scope

AquaTesseract reconstructs dissolved oxygen from sparse observations using
OceanVerse CESM2-OMIP1 simulated data. Results are simulation-based and are
not real-ocean validation.

Primary array order:

(year, depth, lat, lon) = (62, 33, 180, 360)

## Repository locations

- `ml/scripts/hybrid_model.py`
- `ml/scripts/hybrid_train.py`
- `ml/scripts/baselines.py`
- `ml/scripts/plot_results.py`
- `ml/scripts/to_contract.py`
- `ml/src/aquatesseract/contract.py`
- `ml/scripts/validate_run.py`
- `ml/tests/test_contract.py`

Raw data belongs in `data/oceanverse/CESM2_omip1/`.

Run artifacts belong in `results/<run_name>/`.

Large datasets, checkpoints, and NumPy arrays must not be committed to Git.

## Environment

The pinned environment is Python 3.11 with Torch 2.3.1.

## Validation and tests

Validate result folders with:

```bash
make check
