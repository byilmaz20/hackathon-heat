"""Synthetic tests of the daily deployment MILP (small enough for Gurobi's size-limited licence)."""
import itertools

import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy")

from heattents.deploy import DeploymentRequest, Doctor, Shift, Site, solve, synthetic_request  # noqa: E402

SHIFTS = [Shift("early", 8, 12), Shift("late", 12, 16), Shift("long", 10, 16)]


@pytest.fixture(scope="module")
def env():
    try:
        e = gp.Env(empty=True)
        e.setParam("OutputFlag", 0)
        e.start()
    except gp.GurobiError as err:
        pytest.skip(f"Gurobi unavailable: {err}")
    yield e
    e.dispose()


def demand(rows):
    """rows: (area_id, lat, lon, hour, scenario, cases)."""
    return pd.DataFrame(rows, columns=["area_id", "lat", "lon", "hour", "scenario", "expected_cases"])


def request(dem, doctors, sites=None, n_tents=1, **kw):
    sites = sites or [Site("S0", "near", 40.70, -73.90)]
    kw.setdefault("distance_decay_km", None)
    return DeploymentRequest(n_tents=n_tents, sites=sites, doctors=doctors, shifts=SHIFTS, demand=dem, **kw)


def test_capacity_respected(env):
    # one doctor in 'early' (4 h) treats 4/h; demand 100 only at 10:00 -> 4 treated
    plan = solve(request(demand([("A", 40.70, -73.90, 10, 0, 100.0)]), [Doctor("d1", ["early"])]), env)
    assert plan.kpis["expected_served"] == pytest.approx(4.0)
    assert plan.served.expected_served.sum() == pytest.approx(4.0)


def test_doctor_availability_respected(env):
    dem = demand([("A", 40.70, -73.90, 9, 0, 10.0)])  # demand only in the morning
    plan = solve(request(dem, [Doctor("d1", ["late"])]), env)
    assert plan.kpis["expected_served"] == pytest.approx(0.0)
    assert set(plan.assignments["shift"]) <= {"late"}


def test_no_double_booking_and_overlaps(env):
    dem = demand([("A", 40.70, -73.90, h, 0, 50.0) for h in range(8, 16)] +
                 [("B", 40.75, -73.95, h, 0, 50.0) for h in range(8, 16)])
    sites = [Site("S0", "a", 40.70, -73.90), Site("S1", "b", 40.75, -73.95)]
    doctors = [Doctor(f"d{k}", ["early", "late", "long"], max_shifts=2) for k in range(3)]
    plan = solve(request(dem, doctors, sites, n_tents=2), env)
    asg = plan.assignments
    assert not asg.duplicated(["doctor_id", "shift"]).any()
    assert asg.groupby("doctor_id").size().max() <= 2
    by_doc = asg.groupby("doctor_id")["shift"].apply(list)
    sh = {s.name: s for s in SHIFTS}
    for shifts in by_doc:
        for s1, s2 in itertools.combinations(shifts, 2):
            assert not sh[s1].overlaps(sh[s2])
    assert plan.kpis["tents_open"] <= 2


def test_n_tents_and_useless_site_not_opened(env):
    dem = demand([("A", 40.70, -73.90, 10, 0, 5.0)])
    sites = [Site("far", "far", 40.85, -73.80), Site("near", "near", 40.701, -73.901)]
    plan = solve(request(dem, [Doctor(f"d{k}", ["early"]) for k in range(3)], sites, n_tents=2), env)
    assert list(plan.open_sites.site_id) == ["near"]
    assert plan.kpis["tents_open"] <= 2


def test_more_doctors_never_reduces_served(env):
    req2 = synthetic_request(n_areas=12, n_sites=6, n_doctors=2, n_scenarios=3, n_tents=3, total_daily_cases=200, seed=1)
    req4 = synthetic_request(n_areas=12, n_sites=6, n_doctors=2, n_scenarios=3, n_tents=3, total_daily_cases=200, seed=1)
    req4.doctors = req2.doctors + [Doctor("extra1", ["early", "late"], 2), Doctor("extra2", ["mid"], 1)]
    s2 = solve(req2, env).kpis["expected_served"]
    s4 = solve(req4, env).kpis["expected_served"]
    assert s4 >= s2 - 1e-6


def test_scenario_averaging(env):
    # capacity 4 at 10:00; scenario demands 2 and 10 -> served 2 and 4 -> mean 3
    dem = demand([("A", 40.70, -73.90, 10, 0, 2.0), ("A", 40.70, -73.90, 10, 1, 10.0)])
    plan = solve(request(dem, [Doctor("d1", ["early"])]), env)
    assert plan.kpis["expected_served"] == pytest.approx(3.0)
    assert plan.kpis["scenario_served_min"] == pytest.approx(2.0)
    assert plan.kpis["scenario_served_max"] == pytest.approx(4.0)
    assert plan.kpis["expected_cases"] == pytest.approx(6.0)


def test_min_doctors_per_open_tent(env):
    dem = demand([("A", 40.70, -73.90, 10, 0, 10.0)])
    plan = solve(request(dem, [Doctor("d1", ["early"])], min_doctors_per_open_tent=2), env)
    assert plan.kpis["tents_open"] == 0 and plan.kpis["expected_served"] == pytest.approx(0.0)


def test_distance_decay_and_hour_blocks(env):
    dem = demand([("A", 40.70, -73.90, 10, 0, 1.0)])
    site = Site("S0", "s", 40.70, -73.90 + 0.5 / (111.32 * 0.7581))  # ~0.5 km east
    plan = solve(request(dem, [Doctor("d1", ["early"])], [site], distance_decay_km=0.5), env)
    assert plan.kpis["expected_served"] == pytest.approx(0.3679, rel=0.02)  # exp(-0.5/0.5)
    blk = solve(request(dem, [Doctor("d1", ["early"])], hour_blocks=[(8, 12), (12, 16)]), env)
    assert blk.kpis["expected_served"] == pytest.approx(1.0)
