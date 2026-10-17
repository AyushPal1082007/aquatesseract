import subprocess
import sys
from pathlib import Path

import numpy as np

from aquatesseract.contract import metrics_for, validate_run

ROOT = Path(__file__).resolve().parents[2]


def test_metrics_known_values():
    t = np.array([1, 2, 3, np.nan], dtype=np.float32)
    p = np.array([1, 2, 4, 9], dtype=np.float32)
    m = metrics_for(p, t, np.full(4, 1.0, dtype=np.float32), np.array([1, 0, 0, 0]))
    assert m["all"]["n"] == 3 and m["observed"]["n"] == 1
    assert abs(m["all"]["rmse"] - (1 / 3) ** 0.5) < 1e-9


def test_mock_run_obeys_contract(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "ml/scripts/make_mock_run.py"), "--out",
                    str(tmp_path), "--years", "4"], check=True)
    runs = list(tmp_path.glob("MOCK_*"))
    assert runs and validate_run(runs[0]) == []


def test_zero_on_land_is_rejected(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "ml/scripts/make_mock_run.py"), "--out",
                    str(tmp_path), "--years", "3"], check=True)
    run = next(tmp_path.glob("MOCK_*"))
    t = np.load(run / "truth.npy")
    t[np.isnan(t)] = 0.0  # the classic NaN-vs-zero bug
    np.save(run / "truth.npy", t)
    assert any("NaN" in e for e in validate_run(run))
