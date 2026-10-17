"""Usage: python ml/scripts/validate_run.py results/<run>/ [...]   Exit 1 if any run breaks contract."""
import sys

from aquatesseract.contract import validate_run

bad = 0
for p in sys.argv[1:]:
    errs = validate_run(p)
    print(("OK   " if not errs else "FAIL ") + p)
    for e in errs:
        print("   -", e)
    bad += bool(errs)
sys.exit(1 if bad else 0)
