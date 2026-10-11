# Data dictionary

The study uses two paired datasets describing the **same 1,000 sampled retrofit
designs**, each evaluated by a whole-building energy simulation under a different
climate- and grid-decarbonization scenario.

| File | Scenario | Climate / grid |
|---|---|---|
| `data/Base_Scenario_with_OCE.csv` | **Baseline** | present-day climate, carbon-intensive grid |
| `data/2050_Scenario_with_OCE.csv` | **Mid-future** | mid-century (≈2050) climate, decarbonized grid |

## Columns

The loader normalises header whitespace and replaces spaces/hyphens with underscores,
so `Sensible HR_Eff` becomes `Sensible_HR_Eff` and `EUI-Baseline` becomes `EUI_Baseline`
before being renamed to the bare objective names.

### Decision variables (inputs)

| Column | Unit | Typical range |
|---|---|---|
| `Cooling_SP` | °C | 20 – 26 |
| `Wall_U` | W/m²K | 0.15 – 0.90 |
| `Window_U` | W/m²K | 1.0 – 3.2 |
| `Floor_U` | W/m²K | 0.2 – 0.7 |
| `Roof_U` | W/m²K | 0.15 – 0.50 |
| `Window_SHGC` | – | 0.1 – 0.6 |
| `Infiltration_rate` | h⁻¹ | 0.3 – 1.0 |
| `Sensible_HR_Eff` | – | 0.5 – 0.9 |

### Objectives (outputs)

| Column | Name | Unit | Direction |
|---|---|---|---|
| `EUI` | Energy use intensity | kWh/m²·y | minimize |
| `IDD` | Indoor discomfort degree | °C | minimize |
| `OCI` | Operational carbon intensity | kgCO₂-eq/m²·y | minimize |

All three objectives are **cost-type** (lower is better), which is why the TOPSIS
ideal solution uses per-criterion minima.

## Quality checks performed by the pipeline

`building_retrofit_moo.data.load_scenarios` verifies that:

1. Both files expose a detectable scenario name in their column suffixes.
2. After cleaning, the two files share the **identical input design matrix** — they are
   paired simulations, not independent samples.
3. Every objective column is renamed consistently to `EUI`, `IDD`, `OCI`.
