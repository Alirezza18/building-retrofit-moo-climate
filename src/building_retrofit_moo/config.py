"""Central configuration for the building-retrofit multi-objective optimization pipeline.

All tunables (seeds, search sizes, figure resolution, colour palette, variable
labels) live here so that every experiment is reproducible from a single place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

BASE_CSV = DATA_DIR / "Base_Scenario_with_OCE.csv"
FUTURE_CSV = DATA_DIR / "2050_Scenario_with_OCE.csv"

# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
SCENARIOS = ("Baseline", "Mid_future")
SCENARIO_LABELS = {"Baseline": "Baseline scenario", "Mid_future": "Mid-future scenario"}
TARGETS = ("EUI", "IDD", "OCI")

# Display-friendly variable names (used on axes / heatmaps).
NICE_NAMES = {
    "Sensible_HR_Eff": "Sensible HR_Eff",
    "Cooling_SP": "Cooling SP",
    "Infiltration_rate": "Infiltration",
    "Window_SHGC": "Window SHGC",
    "Wall_U": "Wall U",
    "Window_U": "Window U",
    "Floor_U": "Floor U",
    "Roof_U": "Roof U",
}

TARGET_LABELS = {
    "EUI": r"EUI (kWh/(m$^2$$\cdot$y))",
    "IDD": r"IDD ($^\circ$C)",
    "OCI": r"OCI (kgCO$_2$-eq/(m$^2$$\cdot$y))",
}

# A conventional "initial" scheme used as the reference in Figure 10.
INITIAL_SCHEME = {
    "Cooling_SP": 24.0,
    "Wall_U": 0.60,
    "Window_U": 2.40,
    "Floor_U": 0.40,
    "Roof_U": 0.35,
    "Window_SHGC": 0.30,
    "Infiltration_rate": 0.50,
    "Sensible_HR_Eff": 0.75,
}

# --------------------------------------------------------------------------- #
# Colours / style
# --------------------------------------------------------------------------- #
TRAIN_C = "#9b59b6"
TEST_C = "#2ecc71"
CORR_CMAP_COLORS = ["#2f6ea5", "#f7f7f7", "#c0432b"]


@dataclass
class Settings:
    """Runtime settings controlling cost, quality and reproducibility."""

    seed: int = 42

    # Surrogate-model Bayesian optimization budget
    n_init: int = 10
    n_iter: int = 30
    kappa: float = 2.576
    cv_folds: int = 5

    # NSGA-II budget
    pop_size: int = 200
    n_gen: int = 200
    cross_prob: float = 0.9
    cross_eta: float = 15.0
    mut_eta: float = 20.0
    discrete: bool = True

    # How many Pareto solutions to display
    n_display_pareto: int = 50

    # Figure quality
    fig_dpi: int = 600
    preview_dpi: int = 300
    figure_formats: tuple[str, ...] = ("png", "pdf")

    # SHAP dependence configuration (x-feature + interaction colour feature)
    shap_dependence: tuple[dict, ...] = field(
        default_factory=lambda: (
            {
                "target": "EUI",
                "feature_x": "Cooling_SP",
                "label_x": "Cooling setpoint (°C)",
                "interaction_feature": "Infiltration_rate",
                "label_color": r"Infiltration rate (h$^{-1}$)",
            },
            {
                "target": "IDD",
                "feature_x": "Cooling_SP",
                "label_x": "Cooling setpoint (°C)",
                "interaction_feature": "Wall_U",
                "label_color": r"Wall U-value (W/m$^2$K)",
            },
            {
                "target": "OCI",
                "feature_x": "Cooling_SP",
                "label_x": "Cooling setpoint (°C)",
                "interaction_feature": "Roof_U",
                "label_color": r"Roof U-value (W/m$^2$K)",
            },
        )
    )

    def ensure_dirs(self) -> None:
        from .io_utils import ensure_dir

        ensure_dir(FIGURE_DIR)
        ensure_dir(TABLE_DIR)


def quick_settings() -> Settings:
    """A fast configuration for smoke-testing the full pipeline."""
    s = Settings()
    s.n_init, s.n_iter = 3, 5
    s.pop_size, s.n_gen = 40, 15
    s.fig_dpi = 150
    s.preview_dpi = 120
    return s
