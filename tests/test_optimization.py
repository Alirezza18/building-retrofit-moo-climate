import numpy as np
import pandas as pd
import pytest

from building_retrofit_moo.optimization import (
    build_grids,
    entropy_weights,
    snap_to_grid,
    topsis,
)


def test_entropy_weights_sum_to_one_and_are_positive():
    rng = np.random.default_rng(0)
    F = rng.random((40, 3))
    w = entropy_weights(F)
    assert w.shape == (3,)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_topsis_bounds_and_monotonicity():
    F = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [3.0, 3.0, 3.0]])
    w = np.ones(3) / 3
    c = topsis(F, w)
    assert c.shape == (3,)
    assert ((c >= 0) & (c <= 1)).all()
    # The smallest (all-cost) point should be closest to the ideal solution.
    assert c.argmax() == 0


def test_snap_to_grid_uses_realised_values():
    X = pd.DataFrame({
        "a": [0.0, 1.0, 2.0],
        "b": [10.0, 20.0, 30.0],
    })
    grids = build_grids(X)
    out = snap_to_grid(np.array([[0.4, 29.9], [1.6, 10.1]]), ["a", "b"], grids)
    assert out.tolist() == [[0.0, 30.0], [2.0, 10.0]]


def test_snap_respects_fixed_variables():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [10.0, 20.0, 30.0]})
    grids = build_grids(X)
    out = snap_to_grid(np.array([[1.0, 25.0]]), ["a", "b"], grids, {"a": 2.0})
    assert out[0, 0] == 2.0
