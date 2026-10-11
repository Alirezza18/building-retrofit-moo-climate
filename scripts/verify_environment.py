#!/usr/bin/env python
"""Check that every required dependency imports and that the data files exist."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
            "catboost", "xgboost", "shap", "pymoo", "bayes_opt", "openpyxl"]


def main() -> int:
    ok = True
    print("Python:", sys.version.split()[0])
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  [OK]   {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:  # pragma: no cover
            ok = False
            print(f"  [FAIL] {mod}: {exc}")

    from building_retrofit_moo import config
    for path in (config.BASE_CSV, config.FUTURE_CSV):
        exists = path.exists()
        ok &= exists
        print(f"  [{'OK' if exists else 'MISSING'}]   {path}")

    print("\nEnvironment READY" if ok else "\nEnvironment INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
