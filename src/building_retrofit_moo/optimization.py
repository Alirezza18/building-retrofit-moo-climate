"""Multi-objective optimization: NSGA-II over surrogate models + entropy-TOPSIS.

Design variables are constrained to the discrete levels actually present in the
simulation dataset (``snap``), so every reported optimum is a physically
realisable retrofit package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings


def build_grids(X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: np.sort(X[c].dropna().unique().astype(float)) for c in X.columns}


def snap_to_grid(x: np.ndarray, features: list[str], grids: dict[str, np.ndarray],
                 fixed: dict[str, float] | None = None) -> np.ndarray:
    """Snap continuous candidates to the nearest realised value per variable."""
    x = np.asarray(x, dtype=float).copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    for j, c in enumerate(features):
        g = grids[c]
        x[:, j] = g[np.abs(x[:, j, None] - g[None, :]).argmin(axis=1)]
    for c, v in (fixed or {}).items():
        j = features.index(c)
        g = grids[c]
        x[:, j] = g[np.abs(g - v).argmin()]
    return x


def entropy_weights(F: np.ndarray) -> np.ndarray:
    """Objective weights from Shannon entropy (all objectives are cost-type)."""
    F = np.asarray(F, dtype=float)
    n_sol = F.shape[0]
    if n_sol < 2:
        raise ValueError("At least two Pareto solutions are required.")
    F_min, F_max = F.min(axis=0), F.max(axis=0)
    F_range = F_max - F_min
    F_shift = np.ones_like(F)
    for j in range(F.shape[1]):
        if F_range[j] > 1e-12:
            F_shift[:, j] = (F[:, j] - F_min[j]) / F_range[j]
    F_shift = F_shift + 1e-9
    P = F_shift / F_shift.sum(axis=0)
    entropy = -(P * np.log(P)).sum(axis=0) / np.log(n_sol)
    c_j = 1.0 - entropy
    if c_j.sum() <= 1e-12:
        return np.ones(F.shape[1]) / F.shape[1]
    return c_j / c_j.sum()


def topsis(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """TOPSIS closeness coefficient C* for cost-type objectives."""
    F = np.asarray(F, dtype=float)
    den = np.sqrt((F ** 2).sum(axis=0))
    den[den < 1e-12] = 1.0
    V = (F / den) * weights
    ideal, anti = V.min(axis=0), V.max(axis=0)
    d_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    s = d_plus + d_minus
    return np.where(s > 1e-12, d_minus / s, 0.5)


@dataclass
class OptimizationResult:
    scenario: str
    pareto: pd.DataFrame          # features + targets + C_star, ranked by C_star
    weights: np.ndarray
    best: pd.Series


def _make_problem(features, targets, models, xl, xu, grids, fixed, discrete):
    from pymoo.core.problem import Problem

    class RetrofitProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(features), n_obj=len(targets), xl=xl, xu=xu)
            self.models = [models[t] for t in targets]

        def _evaluate(self, x, out, *args, **kwargs):
            xe = snap_to_grid(x, features, grids, fixed) if discrete else x
            out["F"] = np.column_stack([m.predict(xe) for m in self.models])

    return RetrofitProblem()


def run_nsga2(
    scenario: str,
    features: list[str],
    targets: list[str],
    models: dict,
    X: pd.DataFrame,
    settings: Settings,
    fixed_variables: dict[str, float] | None = None,
) -> OptimizationResult:
    """Run NSGA-II for one scenario and rank the Pareto set with entropy-TOPSIS."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    grids = build_grids(X)
    xl = X.min().values.astype(float)
    xu = X.max().values.astype(float)

    algorithm = NSGA2(
        pop_size=settings.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=settings.cross_prob, eta=settings.cross_eta),
        mutation=PM(prob_var=1.0 / len(features), eta=settings.mut_eta),
        eliminate_duplicates=True,
    )
    problem = _make_problem(features, targets, models, xl, xu, grids,
                            fixed_variables or {}, settings.discrete)
    res = minimize(problem, algorithm, ("n_gen", settings.n_gen),
                   seed=settings.seed, verbose=False)

    X_par = snap_to_grid(res.X, features, grids, fixed_variables) if settings.discrete else res.X
    pareto = pd.DataFrame(X_par, columns=features)
    pareto[targets] = res.F
    pareto = pareto.drop_duplicates(subset=list(targets)).reset_index(drop=True)
    pareto["Scenario"] = scenario

    F = pareto[list(targets)].values.astype(float)
    weights = entropy_weights(F)
    pareto["C_star"] = topsis(F, weights)
    pareto = pareto.sort_values("C_star", ascending=False).reset_index(drop=True)

    return OptimizationResult(
        scenario=scenario, pareto=pareto, weights=weights, best=pareto.iloc[0],
    )


def predict_targets(models: dict, design: pd.DataFrame, targets: list[str]) -> dict:
    """Predict every objective for a single design (models keyed by target)."""
    return {t: float(np.ravel(models[t].predict(design))[0]) for t in targets}
