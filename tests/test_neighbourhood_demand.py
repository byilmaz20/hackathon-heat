import numpy as np
import pandas as pd

from heattents.models.scenarios import N_SCENARIOS, neighbourhood_demand, scenario_table


def test_scenario_table_is_five_equally_likely_percentiles():
    samples = np.arange(1, 1001, dtype=float)  # uniform 1..1000
    S = scenario_table(samples, mapped_fraction=0.96)
    assert N_SCENARIOS == 5 and len(S) == 5
    assert S.percentile.tolist() == [10, 30, 50, 70, 90]
    assert np.allclose(S.probability, 0.2) and np.isclose(S.probability.sum(), 1.0)
    assert np.allclose(S.city_total, np.quantile(samples, [0.1, 0.3, 0.5, 0.7, 0.9]))
    assert np.allclose(S.located_total, 0.96 * S.city_total)


def test_neighbourhood_demand_is_daily_and_sums_to_city_total():
    nb = pd.DataFrame({"uhf": ["101", "102"], "share": [0.25, 0.75]})
    D = neighbourhood_demand(nb, np.array([10.0, 40.0]), mapped_fraction=0.96)
    assert list(D.columns) == ["uhf", "scenario", "probability", "expected_cases"] and len(D) == 2 * 2
    assert np.allclose(D.groupby("scenario").expected_cases.sum(), [9.6, 38.4])
    assert np.isclose(D[(D.scenario == 1) & (D.uhf == "102")].expected_cases.iloc[0], 40 * 0.96 * 0.75)
