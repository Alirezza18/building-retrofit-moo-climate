import pytest

from building_retrofit_moo import config
from building_retrofit_moo.data import load_scenarios


@pytest.mark.skipif(not config.BASE_CSV.exists() or not config.FUTURE_CSV.exists(),
                    reason="input CSVs not available")
def test_load_scenarios_shapes_and_pairing():
    scen = load_scenarios()
    assert set(scen) == {"Baseline", "Mid_future"}

    base = scen["Baseline"]
    assert base.targets == ["EUI", "IDD", "OCI"]
    assert len(base.features) == 8
    assert len(base) > 0

    # Paired simulations: identical input designs across scenarios.
    assert scen["Baseline"].X.equals(scen["Mid_future"].X)


@pytest.mark.skipif(not config.BASE_CSV.exists(), reason="input CSV not available")
def test_objectives_are_numeric_and_finite():
    base = load_scenarios()["Baseline"]
    for t in base.targets:
        assert base.Y[t].notna().all()
        assert base.Y[t].map(float).apply(lambda v: v == v).all()
