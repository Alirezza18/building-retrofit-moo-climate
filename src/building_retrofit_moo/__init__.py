"""Data-driven multi-objective optimization of building energy consumption,
indoor discomfort, and CO2 emissions under future climate change and grid
decarbonization.

Public entry points are re-exported here for convenience.
"""
from .config import Settings, quick_settings
from .data import load_scenarios, ScenarioData
from .pipeline import run_pipeline

__all__ = [
    "Settings",
    "quick_settings",
    "load_scenarios",
    "ScenarioData",
    "run_pipeline",
]

__version__ = "1.0.0"
