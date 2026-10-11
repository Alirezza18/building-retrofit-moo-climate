"""End-to-end pipeline: model comparison -> SHAP -> NSGA-II -> figures/tables.

Running :func:`run_pipeline` reproduces every figure and table reported in the
paper from the two input CSVs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, figures, optimization, sensitivity
from .config import SCENARIO_LABELS, Settings
from .data import load_scenarios
from .io_utils import write_csv
from .models import (
    MODEL_NAMES,
    SEARCH_SPACE,
    make_split,
    parameter_table,
    regression_metrics,
    tune_and_fit,
)


def _log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def _save_table(df: pd.DataFrame, name: str, settings: Settings, index: bool = False):
    settings.ensure_dirs()
    write_csv(df, config.TABLE_DIR / f"{name}.csv", index=index)


def _train_final_catboost(X_train, Y, targets, scenarios, kf, settings):
    """Train one CatBoost surrogate per (scenario, target) - used downstream."""
    models, best_params = {}, {}
    build = SEARCH_SPACE["CatBoost"]
    for scen in scenarios:
        for t in targets:
            y_tr = Y[scen].loc[X_train.index, t]
            model, params, cv, _ = tune_and_fit(
                build, X_train, y_tr, X_train, y_tr, kf, settings)
            models[(scen, t)] = model
            best_params[(scen, t)] = params
            _log(f"final surrogate {scen} | {t} | CV R2 = {cv:.4f}")
    return models, best_params


def run_pipeline(
    settings: Settings | None = None,
    base_csv=config.BASE_CSV,
    future_csv=config.FUTURE_CSV,
    save_figures: bool = True,
) -> dict:
    """Execute the complete analysis and persist figures + tables.

    Returns a dictionary with the fitted models, Pareto sets and metrics so the
    results can be inspected programmatically.
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    # 1. Data -------------------------------------------------------------- #
    scen = load_scenarios(base_csv, future_csv)
    features = scen["Baseline"].features
    targets = scen["Baseline"].targets
    X = scen["Baseline"].X
    Y = {s: scen[s].Y for s in config.SCENARIOS}
    scenarios = list(config.SCENARIOS)
    _log(f"loaded {len(X)} designs x {len(features)} features; targets = {targets}")

    X_train, X_test, idx_train, idx_test, kf = make_split(X, settings)

    # 2. Model comparison (Figure 4) -------------------------------------- #
    all_metrics, comparison_preds = {}, {}
    for scenario in scenarios:
        preds, rows = {}, []
        for t in targets:
            y_tr, y_te = Y[scenario].loc[idx_train, t], Y[scenario].loc[idx_test, t]
            for name in MODEL_NAMES:
                spec = SEARCH_SPACE[name]
                model, params, cv, metrics = tune_and_fit(
                    spec, X_train, y_tr, X_test, y_te, kf, settings)
                preds[(name, t)] = (
                    y_tr.values, np.ravel(model.predict(X_train)),
                    y_te.values, np.ravel(model.predict(X_test)))
                for split in ("Train", "Test"):
                    rows.append({"Target": t, "Model": name, "Split": split, **metrics[split]})
                _log(f"{scenario} | {t} | {name} | CV R2 = {cv:.4f}")
        metrics_df = pd.DataFrame(rows)
        all_metrics[scenario] = metrics_df
        comparison_preds[scenario] = preds
        _save_table(metrics_df, f"model_comparison_{scenario}", settings)
        if save_figures:
            figures.plot_model_comparison(scenario, preds, metrics_df, settings)

    _save_table(parameter_table().reset_index(), "Table_A1_all_models", settings)

    # 3. Final CatBoost surrogates ---------------------------------------- #
    models, best_params = _train_final_catboost(X_train, Y, targets, scenarios, kf, settings)
    _save_table(pd.DataFrame(best_params).T.round(4).reset_index(), "best_hyperparameters", settings)

    # 4. Final performance table + Figures 5/6 ---------------------------- #
    predictions_test, rows = {}, []
    for (scen_key, t), model in models.items():
        y_te = Y[scen_key].loc[idx_test, t]
        predictions_test[(scen_key, t)] = (y_te.values, np.ravel(model.predict(X_test)))
        for split, Xs in (("Train", X_train), ("Test", X_test)):
            preds = np.ravel(model.predict(Xs))
            m = regression_metrics(Y[scen_key].loc[Xs.index, t], preds)
            rows.append({"Scenario": scen_key, "Target": t, "Split": split,
                         "R2": m["R2"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                         "CV_RMSE_%": m["CV_RMSE"]})
    results_df = pd.DataFrame(rows)
    _save_table(results_df.round(4), "final_model_performance", settings)
    if save_figures:
        figures.plot_actual_vs_predicted(predictions_test, targets, settings)
        figures.plot_performance_metrics(models, X_train, X_test, Y, idx_train, idx_test,
                                         targets, settings)

    # 5. SHAP ------------------------------------------------------------- #
    if save_figures:
        figures.plot_shap_summary(models, X, targets, scenarios, settings)
        figures.plot_shap_dependence(models, X, settings)
        figures.plot_violin_distributions(X, Y["Baseline"], Y["Mid_future"], targets, settings)

    # 6. NSGA-II + TOPSIS per scenario ------------------------------------ #
    models_by_scenario = {s: {t: models[(s, t)] for t in targets} for s in scenarios}
    pareto_by_scenario, best_by_scenario, comparison_rows = {}, {}, []
    for scenario in scenarios:
        res = optimization.run_nsga2(scenario, features, targets, models_by_scenario[scenario],
                                     X, settings)
        pareto_by_scenario[scenario] = res.pareto
        best_by_scenario[scenario] = res.best
        _log(f"{scenario}: {len(res.pareto)} Pareto solutions | "
             f"weights = {np.round(res.weights, 4).tolist()}")
        write_csv(res.pareto, config.TABLE_DIR / f"pareto_solutions_{scenario}.csv")

        x_best = res.best[features].values.astype(float).reshape(1, -1)
        best_design = pd.DataFrame(x_best, columns=features)
        for t in targets:
            ref = float(Y[scenario][t].mean())
            opt = float(np.ravel(models[(scenario, t)].predict(best_design))[0])
            comparison_rows.append({
                "Scenario": scenario, "Target": t,
                "Reference (mean of sampled designs)": ref, "Optimal design": opt,
                "Change_%": (opt - ref) / ref * 100 if abs(ref) > 1e-12 else np.nan})

    write_csv(
        pd.concat([pareto_by_scenario[s] for s in scenarios], ignore_index=True),
        config.TABLE_DIR / "pareto_solutions_all_scenarios.csv")
    comparison = pd.DataFrame(comparison_rows)
    _save_table(comparison.round(3), "optimal_vs_reference", settings)

    selected = pd.DataFrame(best_by_scenario).T[features + targets + ["C_star"]].astype(float)
    _save_table(selected.round(4), "selected_TOPSIS_designs", settings, index=True)

    # 7. Pareto / parallel-coordinate / correlation figures --------------- #
    if save_figures:
        figures.plot_pareto_fronts(pareto_by_scenario, best_by_scenario, targets, settings)
        figures.plot_parallel_coordinates(
            X, features, targets, pareto_by_scenario, best_by_scenario, models,
            settings, config.INITIAL_SCHEME)

    # 8. Correlation tables + Table 3 ------------------------------------- #
    naming = {
        "spearman": ("spearman_rho", "spearman_pvalues"),
        "pearson": ("pearson_r", "pearson_pvalues"),
    }
    for method, (coef_prefix, pval_prefix) in naming.items():
        if save_figures:
            _, matrices = figures.plot_correlation_heatmap(
                pareto_by_scenario, features, settings, method)
        else:
            matrices = {s: sensitivity.correlation_matrix(
                pareto_by_scenario[s].head(settings.n_display_pareto), method)
                for s in scenarios}
        for s in scenarios:
            rho, pval = matrices[s]
            write_csv(rho.round(4), config.TABLE_DIR / f"{coef_prefix}_{s}_top50.csv", index=True)
            write_csv(pval, config.TABLE_DIR / f"{pval_prefix}_{s}_top50.csv", index=True)

    table3 = sensitivity.pairwise_objective_table(pareto_by_scenario, SCENARIO_LABELS)
    _save_table(table3, "Table3_Pairwise_Relationships", settings)

    _log("done - figures in results/figures, tables in results/tables")
    return {
        "settings": settings,
        "features": features,
        "targets": targets,
        "models": models,
        "metrics": all_metrics,
        "pareto": pareto_by_scenario,
        "best": best_by_scenario,
        "selected": selected,
        "optimal_vs_reference": comparison,
        "table3": table3,
    }
