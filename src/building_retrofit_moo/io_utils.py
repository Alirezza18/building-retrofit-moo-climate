"""Filesystem helpers that survive Windows' 260-character MAX_PATH limit.

The study folder name can be long enough that ``results/...`` paths exceed
MAX_PATH. On Windows we transparently switch to the extended-length ``\\\\?\\``
prefix so pandas/matplotlib ``open`` calls still succeed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def fs_path(path: Path | str) -> str:
    """Return a path string usable by ``open`` even when it exceeds MAX_PATH."""
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\") and len(s) >= 240:
        return "\\\\?\\" + s
    return s


def ensure_dir(path: Path | str) -> None:
    os.makedirs(fs_path(path), exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> None:
    """``DataFrame.to_csv`` with long-path support."""
    ensure_dir(Path(path).parent)
    df.to_csv(fs_path(path), index=index, **kwargs)
