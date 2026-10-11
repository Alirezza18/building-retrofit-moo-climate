#!/usr/bin/env python
"""Convenience wrapper so the pipeline runs without installing the package.

    python scripts/run_pipeline.py --quick
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when running straight from a clone.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from building_retrofit_moo.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
