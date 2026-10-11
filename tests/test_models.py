import numpy as np

from building_retrofit_moo.models import (
    MODEL_NAMES,
    SEARCH_SPACE,
    parameter_table,
    regression_metrics,
)


def test_search_space_has_five_models():
    assert MODEL_NAMES == ["CatBoost", "XGBoost", "RF", "KNN", "SVR"]
    for spec in SEARCH_SPACE.values():
        assert spec.bounds, f"{spec.name} has no search bounds"


def test_parameter_table_structure():
    table = parameter_table()
    assert list(table.columns) == ["Lower", "Upper"]
    assert set(table.index.get_level_values("Model")) == set(MODEL_NAMES)


def test_regression_metrics_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y.copy())
    assert m["R2"] == 1.0
    assert m["RMSE"] == 0.0
