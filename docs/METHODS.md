# Methods

This document summarises the algorithms implemented in
`building_retrofit_moo`. Every parameter referenced here is configurable in
`src/building_retrofit_moo/config.py` (`Settings`).

## 1. Surrogate modelling

Because each whole-building simulation is expensive, the workflow learns a
**surrogate** `f: X → y` for each objective (`EUI`, `IDD`, `OCI`) under each scenario.

Five candidate learners are tuned and compared:

| Model | Search space (bounds) |
|---|---|
| **CatBoost** | iterations 200–800, depth 3–8, LR 0.01–0.3, L2 1–10, random-strength 0.1–5 |
| **XGBoost** | n_estimators 100–600, max_depth 2–8, LR 0.01–0.3, subsample 0.6–1.0, colsample 0.6–1.0, λ 0–10 |
| **Random Forest** | n_estimators 100–500, max_depth 3–20, min_split 2–10, min_leaf 1–5, max_features 0.3–1.0 |
| **KNN** | neighbours 1–20, weighted/uniform, L1/L2 |
| **SVR** | log₁₀C −1–3, log₁₀γ −3–0.5, ε 0.001–0.3 |

Hyper-parameters are chosen by **Bayesian optimization** maximizing the mean
**5-fold cross-validated R²** on the training split (80 / 20 split, seed 42). The
search uses an Upper-Confidence-Bound acquisition (`κ = 2.576`).

The **CatBoost** configuration is the one carried forward for sensitivity analysis and
optimization, since it is the strongest and most consistent learner across objectives.

## 2. Global sensitivity (SHAP)

For each fitted tree surrogate, exact TreeSHAP values are computed on the full design
matrix. Results are visualised two ways:

- **Beeswarm + mean |SHAP| bars** — global importance and directional effect.
- **Dependence plots** — SHAP value of `Cooling_SP` coloured by an interaction feature
  (infiltration for EUI, wall U-value for IDD, roof U-value for OCI).

Spearman and Pearson correlation matrices of the Pareto-optimal designs (upper triangle
= correlation coefficient with significance stars, lower triangle = magnitude circle)
further expose variable interactions.

## 3. Multi-objective optimization (NSGA-II)

A three-objective minimization problem is solved **independently for each scenario**:

- **Decision variables**: the 8 retrofit parameters.
- **Objectives**: `EUI`, `IDD`, `OCI`.
- **Discretisation**: candidates are *snapped* to the discrete levels actually present
  in the dataset, guaranteeing realisable designs.
- **Operator settings**: population 200, 200 generations, SBX crossover
  (p = 0.9, η = 15), polynomial mutation (p = 1/n_var, η = 20),
  duplicate elimination, seed 42.

### Entropy weighting

Objective weights are derived from the information entropy of the normalized objective
matrix. For each objective *j*, normalize the solutions to [0, 1], compute the
proportional matrix `P`, the entropy
`e_j = −(Σ p_ij ln p_ij) / ln n`, and the divergence `c_j = 1 − e_j`; the weight is
`w_j = c_j / Σ c_k`. Objectives with more dispersion carry more weight.

### TOPSIS ranking

Because all three objectives are **cost-type**, the weighted normalized matrix is
compared against the ideal (`min` per criterion) and anti-ideal (`max`). The closeness
coefficient `C* = d⁻ / (d⁺ + d⁻)` ranks the Pareto set; the highest `C*` is reported as
the **selected design**.

## 4. Moving from a mean to a design

The pipeline reports each selected design against the **mean of the sampled designs**
(a reference, *not* a measured building):

```
Change_% = (optimal − reference) / reference × 100
```

This isolates the improvement attributable to the optimization rather than to the
scenario itself.
