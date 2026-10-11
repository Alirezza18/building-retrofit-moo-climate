"""Loading and cleaning of the parametric-simulation dataset.

Each CSV holds one climate/grid scenario: the same 1000 sampled retrofit
designs evaluated for three objective functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise column names to snake_case-ish tokens."""
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return out


def _detect_scenario(df: pd.DataFrame) -> str:
    for scenario in config.SCENARIOS:
        if any(scenario in c for c in df.columns):
            return scenario
    raise ValueError(
        f"Cannot detect a known scenario in columns: {list(df.columns)}"
    )


@dataclass
class ScenarioData:
    """Container bundling inputs (X) and the three objectives per scenario."""

    name: str
    features: list[str]
    targets: list[str]
    X: pd.DataFrame
    Y: pd.DataFrame

    def __len__(self) -> int:  # number of simulated designs
        return len(self.X)

    def as_frame(self) -> pd.DataFrame:
        """Return inputs and objectives in a single DataFrame."""
        return pd.concat([self.X, self.Y], axis=1)


def load_scenarios(
    base_path: Path | str = config.BASE_CSV,
    future_path: Path | str = config.FUTURE_CSV,
) -> dict[str, ScenarioData]:
    """Load both scenario CSVs into ``{scenario_name: ScenarioData}``.

    Column cleaning mirrors the original Colab workflow: spaces and hyphens are
    replaced by underscores, scenario-suffixed targets are renamed to
    ``EUI``/``IDD``/``OCI``, and the remaining columns form the feature matrix.
    """
    raw = {
        "Baseline": _clean_columns(pd.read_csv(base_path)),
        "Mid_future": _clean_columns(pd.read_csv(future_path)),
    }

    # Rename scenario-suffixed objective columns to bare target names.
    for name, df in raw.items():
        df.rename(columns={f"{t}_{name}": t for t in config.TARGETS}, inplace=True)

    features = [c for c in raw["Baseline"].columns if c not in config.TARGETS]
    targets = list(config.TARGETS)

    out: dict[str, ScenarioData] = {}
    for name, df in raw.items():
        out[name] = ScenarioData(
            name=name,
            features=features,
            targets=targets,
            X=df[features].copy(),
            Y=df[targets].copy(),
        )

    # Sanity check: both files describe the identical set of design vectors.
    if not out["Baseline"].X.equals(out["Mid_future"].X):
        raise ValueError(
            "The two scenario files do not share identical input rows; "
            "cannot treat them as paired simulations."
        )
    return out
