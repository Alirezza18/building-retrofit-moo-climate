"""Global sensitivity (SHAP) and pairwise-correlation analytics.

These helpers are side-effect free: they return arrays/tables that the figure
module renders and the pipeline persists as CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for a fitted tree model as an ``(n_samples, n_features)`` array."""
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    return np.asarray(values)


def mean_abs_shap(sv: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature (global importance)."""
    return np.abs(sv).mean(axis=0)


def correlation_matrix(df: pd.DataFrame, method: str = "spearman") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return correlation coefficients and p-values for every column pair.

    Constant columns yield ``NaN`` (undefined correlation) rather than an error.
    """
    cols = list(df.columns)
    n = len(cols)
    rho = np.full((n, n), np.nan)
    pval = rho.copy()
    fn = spearmanr if method == "spearman" else pearsonr

    for i in range(n):
        for j in range(n):
            ok = df[cols[i]].nunique() > 1 and df[cols[j]].nunique() > 1
            if not ok:
                continue
            if i == j:
                rho[i, j], pval[i, j] = 1.0, 0.0
            else:
                r, p = fn(df[cols[i]].values, df[cols[j]].values)
                rho[i, j], pval[i, j] = r, p

    return (
        pd.DataFrame(rho, index=cols, columns=cols),
        pd.DataFrame(pval, index=cols, columns=cols),
    )


# Objective pairs reported in Table 3.
OBJECTIVE_PAIRS = (
    ("EUI", "OCI", "EUI-OCI"),
    ("EUI", "IDD", "EUI-IDD"),
    ("OCI", "IDD", "OCI-IDD"),
)


def pairwise_objective_table(
    pareto_by_scenario: dict[str, pd.DataFrame],
    scenario_labels: dict[str, str],
) -> pd.DataFrame:
    """Table 3: Pearson/Spearman correlation and regression slope per objective pair."""
    rows = []
    for scenario, df in pareto_by_scenario.items():
        for obj1, obj2, pair_label in OBJECTIVE_PAIRS:
            x = df[obj1].values.astype(float)
            y = df[obj2].values.astype(float)
            r_p, _ = pearsonr(x, y)
            r_s, _ = spearmanr(x, y)
            slope, _, _, _, _ = linregress(x, y)
            rows.append({
                "Scenario": scenario_labels.get(scenario, scenario),
                "Pair": pair_label,
                "Pearson (r)": f"{r_p:.3f}",
                "Spearman (rho)": f"{r_s:.3f}",
                "Regression slope": f"{slope:.4f}",
            })
    return pd.DataFrame(rows)
