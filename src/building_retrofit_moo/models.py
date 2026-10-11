"""Surrogate model zoo and Bayesian hyper-parameter optimization.

Five regressors (CatBoost, XGBoost, Random Forest, KNN, SVR) are tuned with a
Bayesian search maximizing 5-fold cross-validated R2. The best configuration per
(target, scenario) is then refit on the full training split and evaluated on the
held-out test split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:  # catboost / xgboost are optional at import time but required to run.
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

from .config import Settings


def _I(value: float) -> int:
    return int(round(float(value)))


def build_catboost(p: dict, seed: int):
    return CatBoostRegressor(
        iterations=_I(p["iterations"]), depth=_I(p["depth"]),
        learning_rate=float(p["learning_rate"]), l2_leaf_reg=float(p["l2_leaf_reg"]),
        random_strength=float(p["random_strength"]), random_seed=seed,
        verbose=0, thread_count=-1, allow_writing_files=False,
    )


def build_xgboost(p: dict, seed: int):
    return XGBRegressor(
        n_estimators=_I(p["n_estimators"]), max_depth=_I(p["max_depth"]),
        learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]), reg_lambda=float(p["reg_lambda"]),
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def build_rf(p: dict, seed: int):
    return RandomForestRegressor(
        n_estimators=_I(p["n_estimators"]), max_depth=_I(p["max_depth"]),
        min_samples_split=_I(p["min_samples_split"]), min_samples_leaf=_I(p["min_samples_leaf"]),
        max_features=float(p["max_features"]), random_state=seed, n_jobs=-1,
    )


def build_knn(p: dict, seed: int):
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=_I(p["n_neighbors"]),
            weights="distance" if float(p["distance_weight"]) > 0.5 else "uniform",
            p=1 if float(p["p"]) < 1.5 else 2,
        ),
    )


def build_svr(p: dict, seed: int):
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            SVR(C=10 ** float(p["log10_C"]), gamma=10 ** float(p["log10_gamma"]),
                epsilon=float(p["epsilon"])),
        ),
        transformer=StandardScaler(),
    )


@dataclass
class ModelSpec:
    name: str
    builder: object
    bounds: dict


# Search space: name -> (builder, hyper-parameter bounds).
SEARCH_SPACE: dict[str, ModelSpec] = {
    "CatBoost": ModelSpec(
        "CatBoost", build_catboost,
        {"iterations": (200, 800), "depth": (3, 8), "learning_rate": (0.01, 0.3),
         "l2_leaf_reg": (1, 10), "random_strength": (0.1, 5)},
    ),
    "XGBoost": ModelSpec(
        "XGBoost", build_xgboost,
        {"n_estimators": (100, 600), "max_depth": (2, 8), "learning_rate": (0.01, 0.3),
         "subsample": (0.6, 1.0), "colsample_bytree": (0.6, 1.0), "reg_lambda": (0, 10)},
    ),
    "RF": ModelSpec(
        "RF", build_rf,
        {"n_estimators": (100, 500), "max_depth": (3, 20), "min_samples_split": (2, 10),
         "min_samples_leaf": (1, 5), "max_features": (0.3, 1.0)},
    ),
    "KNN": ModelSpec(
        "KNN", build_knn,
        {"n_neighbors": (1, 20), "distance_weight": (0, 1), "p": (1, 2)},
    ),
    "SVR": ModelSpec(
        "SVR", build_svr,
        {"log10_C": (-1, 3), "log10_gamma": (-3, 0.5), "epsilon": (0.001, 0.3)},
    ),
}

MODEL_NAMES = list(SEARCH_SPACE)

FULL_NAMES = {
    "CatBoost": "Categorical Boosting\n(CatBoost)",
    "XGBoost": "eXtreme Gradient Boosting\n(XGBoost)",
    "RF": "Random Forest\n(RF)",
    "KNN": "K-Nearest Neighbors\n(KNN)",
    "SVR": "Support Vector Regression\n(SVR)",
}


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "CV_RMSE": (rmse / float(np.mean(y_true))) * 100.0,
    }


def cv_r2(builder, params: dict, X_tr, y_tr, kf: KFold, seed: int) -> float:
    """Mean 5-fold cross-validated R2 (lower-is-worse sentinel -1.0 on error)."""
    try:
        scores = []
        for tr_i, va_i in kf.split(X_tr):
            model = builder(params, seed).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
            scores.append(r2_score(y_tr.iloc[va_i], model.predict(X_tr.iloc[va_i])))
        return float(np.mean(scores))
    except Exception:
        return -1.0


def bayesian_optimize(objective, bounds: dict, settings: Settings):
    """Maximize ``objective`` over ``bounds`` with a UCB Bayesian optimizer."""
    from bayes_opt import BayesianOptimization

    try:
        from bayes_opt import acquisition
        bo = BayesianOptimization(
            f=objective, pbounds=bounds,
            acquisition_function=acquisition.UpperConfidenceBound(kappa=settings.kappa),
            random_state=settings.seed, verbose=0,
        )
        bo.maximize(init_points=settings.n_init, n_iter=settings.n_iter)
    except ImportError:  # older bayes_opt API
        from bayes_opt import UtilityFunction
        bo = BayesianOptimization(f=objective, pbounds=bounds,
                                  random_state=settings.seed, verbose=0)
        bo.maximize(init_points=settings.n_init, n_iter=settings.n_iter,
                    acquisition_function=UtilityFunction(kind="ucb", kappa=settings.kappa))
    return bo


@dataclass
class TrainedModel:
    scenario: str
    target: str
    model: object
    best_params: dict
    cv_r2: float


def tune_and_fit(
    spec: ModelSpec,
    X_train,
    y_train,
    X_test,
    y_test,
    kf: KFold,
    settings: Settings,
) -> tuple[object, dict, float, dict]:
    """Bayesian-tune a single model and return (fitted, params, cv_r2, metrics)."""
    bo = bayesian_optimize(
        lambda **p: cv_r2(spec.builder, p, X_train, y_train, kf, settings.seed),
        spec.bounds, settings,
    )
    model = spec.builder(bo.max["params"], settings.seed).fit(X_train, y_train)
    preds_train = np.ravel(model.predict(X_train))
    preds_test = np.ravel(model.predict(X_test))

    metrics = {}
    for split, y_true, y_pred in (
        ("Train", y_train.values, preds_train),
        ("Test", y_test.values, preds_test),
    ):
        metrics[split] = regression_metrics(y_true, y_pred)

    return model, bo.max["params"], float(bo.max["target"]), metrics


def make_split(X: pd.DataFrame, settings: Settings):
    idx_train, idx_test = train_test_split(
        X.index, test_size=0.2, random_state=settings.seed
    )
    kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed)
    return X.loc[idx_train], X.loc[idx_test], list(idx_train), list(idx_test), kf


def parameter_table() -> pd.DataFrame:
    """Table A1: lower/upper search bounds for every model."""
    frames = {
        name: pd.DataFrame(spec.bounds, index=["Lower", "Upper"]).T
        for name, spec in SEARCH_SPACE.items()
    }
    return pd.concat(frames).rename_axis(["Model", "Parameter"])
