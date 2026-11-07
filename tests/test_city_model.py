"""Fast tests of the citywide model's numerical core (no data files needed)."""
import numpy as np
import pandas as pd
import pytest

from heattents.models.glm import cross_basis, fit_glm, fit_nb, lag_basis, nb_pmf, ns_basis
from heattents.models.city import Design, fit_city_model, randomized_pit, scenario_features


def test_glm_recovers_poisson_and_nb():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20_000), rng.normal(size=20_000)])
    y = rng.poisson(np.exp(X @ [0.5, 0.6]))
    assert np.allclose(fit_glm(X, y), [0.5, 0.6], atol=0.03)
    mu, alpha = np.exp(X @ [1.0, 0.4]), 0.5
    y = rng.poisson(rng.gamma(1 / alpha, mu * alpha))
    a, b = fit_nb(X, y)
    assert a == pytest.approx(alpha, abs=0.11) and np.allclose(b, [1.0, 0.4], atol=0.05)


def test_nb_pmf_normalised_and_moments():
    P = nb_pmf(np.array([3.0, 40.0]), 0.3, 600)
    k = np.arange(601)
    assert np.allclose(P.sum(1), 1)
    assert np.allclose((P * k).sum(1), [3, 40], rtol=1e-6)
    assert np.allclose((P * k ** 2).sum(1) - (P * k).sum(1) ** 2, [3 + 0.3 * 9, 40 + 0.3 * 1600], rtol=1e-4)


def test_spline_and_cross_basis_shapes():
    x = np.linspace(5, 50, 200)
    B, s = ns_basis(x, (26, 31, 35), (10, 44))
    assert B.shape == (200, 4)
    # natural spline: linear beyond the boundary knots
    hi = B[x > 44]
    d2 = np.diff(hi, 2, axis=0)
    assert np.allclose(d2, 0, atol=1e-10)
    lagged = np.column_stack([x, x - 1, x - 2, x - 3])
    cb, _ = cross_basis(lagged, (26, 31, 35), (10, 44), degree=2)
    assert cb.shape == (200, 4 * 3) and lag_basis(3, 2).shape == (4, 3)


def test_randomized_pit_uniform_when_calibrated():
    """PITs of correctly specified NB forecasts are ~uniform (the basis of the calibration checks)."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 60, 4000)
    y = rng.poisson(rng.gamma(1 / 0.5, mu * 0.5))
    F = np.cumsum(nb_pmf(mu, 0.5, 800), 1)
    h = np.histogram(randomized_pit(F, y, rng), bins=5, range=(0, 1))[0] / len(y)
    assert np.all(np.abs(h - 0.2) < 0.03)


def test_model_sample_and_predict_shapes():
    rng = np.random.default_rng(2)
    n = 400
    atmax = rng.uniform(15, 42, n)
    f = scenario_features(atmax, hot_days_before=rng.integers(0, 4, n), day_of_year=rng.integers(130, 260, n),
                          year=rng.integers(2005, 2026, n))
    f["date"] = pd.date_range("2010-05-01", periods=n)
    y = rng.poisson(np.exp(-4 + 0.18 * np.maximum(atmax - 20, 0) * 1.5))
    m = fit_city_model(f, y, with_gbm=False)
    s = scenario_features([30.0, 40.0])
    s["date"] = pd.to_datetime(["2026-07-01", "2026-07-02"])
    p = m.predict(s)
    assert list(p.columns[:2]) == ["date", "mean"]
    assert p["mean"].iloc[1] > p["mean"].iloc[0] and (p.q05 <= p.q50).all() and (p.q50 <= p.q95).all()
    assert m.sample(s, 7, 0).shape == (2, 7)
    assert isinstance(m.design, Design) and m.design.names[0] == "intercept"
