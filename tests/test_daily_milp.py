"""Daily tent MILP on hand-checkable instances: a line of three neighbourhoods A - B - C (B borders both)."""
import numpy as np
import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy")
from heattents.deploy.daily import neighbourhood_request, solve  # noqa: E402

ADJ = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def demand(rows):
    """rows: list of (scenario, probability, {uhf: calls})."""
    return pd.DataFrame([dict(uhf=u, scenario=s, probability=p, expected_cases=v)
                         for s, p, d in rows for u, v in d.items()])


def test_one_tent_goes_to_the_hub_and_answers_everything():
    D = demand([(0, 1.0, {"A": 5, "B": 5, "C": 5})])
    plan = solve(neighbourhood_request(n_tents=1, n_workers=1, demand=D, adjacency=ADJ, patients_per_worker_day=20, visitors_per_call=1))
    assert plan.status == "optimal"
    assert plan.tents.site.tolist() == ["B"]  # only B covers all three
    assert plan.kpis["expected_unanswered"] == pytest.approx(0, abs=1e-4)


def test_capacity_binds_and_workers_follow_demand():
    D = demand([(0, 1.0, {"A": 30, "B": 0, "C": 5})])
    plan = solve(neighbourhood_request(n_tents=2, n_workers=2, demand=D, adjacency=ADJ, patients_per_worker_day=10, visitors_per_call=1))
    # 2 workers x 10 = 20 patients for 35 calls: 15 unanswered whichever tents open
    assert plan.kpis["expected_unanswered"] == pytest.approx(15, abs=1e-4)
    assert plan.kpis["workers_placed"] == 2


def test_expectation_over_scenarios_and_min_one_worker_per_tent():
    # quiet (p=0.5): 10 calls in A; busy (p=0.5): 30 calls in A; 3 workers of 10 -> capacity 30
    D = demand([(0, 0.5, {"A": 10, "B": 0, "C": 0}), (1, 0.5, {"A": 30, "B": 0, "C": 0})])
    plan = solve(neighbourhood_request(n_tents=3, n_workers=2, demand=D, adjacency=ADJ, patients_per_worker_day=10, visitors_per_call=1))
    # 2 workers -> 20 capacity: quiet 0 unanswered, busy 10 unanswered -> expected 5
    assert plan.kpis["expected_unanswered"] == pytest.approx(5, abs=1e-4)
    assert plan.tents.workers.sum() == 2 and (plan.tents.workers >= 1).all()
    assert np.allclose(plan.scenarios.unanswered, [0, 10], atol=1e-6)


def test_probabilities_must_sum_to_one():
    D = demand([(0, 0.4, {"A": 1, "B": 1, "C": 1})])
    with pytest.raises(ValueError):
        solve(neighbourhood_request(n_tents=1, n_workers=1, demand=D, adjacency=ADJ))


def test_visitors_scale_calls_and_own_neighbourhood_is_default():
    D = demand([(0, 1.0, {"A": 1, "B": 0, "C": 2})])  # 1 and 2 forecast calls
    plan = solve(neighbourhood_request(n_tents=1, n_workers=5, demand=D, visitors_per_call=20, patients_per_worker_day=10))
    # k = 20 -> 20 and 40 visitors; no adjacency -> one tent serves only its own neighbourhood: C (40 visitors)
    assert plan.tents.site.tolist() == ["C"] and plan.tents.workers.iloc[0] == 5  # all workers placed
    assert plan.kpis["expected_visitors"] == pytest.approx(60)
    assert plan.kpis["expected_unanswered"] == pytest.approx(20, abs=1e-4)


def test_ties_go_to_urgent_neighbourhoods_and_load_is_balanced():
    # every neighbourhood is overwhelmed (capacity 10 per worker, thousands of visitors): any 2 tents serve the
    # same; the plan must pick the two most urgent and split the 4 workers in proportion to need (300:200 -> 2:2
    # or 3:1; max-min capacity/need gives 2 at B (60/3000) and 2 at C (60/2000))
    D = demand([(0, 1.0, {"A": 100, "B": 300, "C": 200})])
    plan = solve(neighbourhood_request(n_tents=2, n_workers=4, demand=D, visitors_per_call=10, patients_per_worker_day=10))
    assert sorted(plan.tents.site) == ["B", "C"]
    assert plan.kpis["expected_answered"] == pytest.approx(40)
    assert plan.tents.set_index("site").workers.to_dict() == {"B": 2, "C": 2}
