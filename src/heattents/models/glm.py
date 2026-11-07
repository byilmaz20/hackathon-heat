r"""Numerical core of the count models (numpy only).

GLM        log E[y] = offset + X b,  Var(y) = mu + alpha mu^2  (NB2; alpha = 0 is Poisson), fitted by penalised IRLS:
           b <- (X'WX + P)^{-1} X'Wz,  W = mu / (1 + alpha mu),  z = eta + (y - mu)/mu
NB pmf     computed by the stable recurrence  p(0) = (r/(r+mu))^r,  p(k) = p(k-1) (k-1+r)/k * mu/(r+mu),  r = 1/alpha
           (no log-gamma needed; vectorised over observations, looped over the count grid)
Splines    natural cubic spline basis (linear beyond the boundary knots, so heat extrapolates linearly in log-rate)
DLNM       cross-basis of a distributed-lag non-linear model (Gasparrini 2010): for exposure x_{t-l}, l = 0..L,
               cb_{t,(i,j)} = sum_l  B_i(x_{t-l}) C_j(l)
           B = exposure basis (natural spline), C = lag basis (polynomial in l). The fitted surface
           f(x, l) = sum_ij theta_ij B_i(x) C_j(l) gives the lag-specific and cumulative heat-response curves.
"""
from __future__ import annotations

import numpy as np

ALPHA_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5)


# --- GLM ------------------------------------------------------------------------------------------------------------
def fit_glm(X, y, alpha=0.0, penalty=None, offset=None, max_iter=200, tol=1e-8, b0=None):
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    P = np.diag(penalty) if penalty is not None else 0.0
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    if b0 is None:
        b[0] = np.log(max(np.mean(y), 1e-3))  # column 0 is assumed to be the intercept
    for _ in range(max_iter):
        eta = np.clip(off + X @ b, -20, 12)
        mu = np.exp(eta)
        W = mu / (1 + alpha * mu)
        z = (eta - off) + (y - mu) / np.maximum(mu, 1e-9)
        b_new = np.linalg.solve(X.T @ (W[:, None] * X) + P, X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < tol:
            return b_new
        b = b_new
    return b


def nb_pmf(mu, alpha, kmax):
    """pmf matrix (n, kmax+1) of NB2(mu, alpha) (Poisson if alpha ~ 0) via recurrence."""
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    P = np.empty((len(mu), kmax + 1))
    if alpha <= 1e-8:
        P[:, 0] = np.exp(-mu)
        for k in range(1, kmax + 1):
            P[:, k] = P[:, k - 1] * mu / k
    else:
        r = 1.0 / alpha
        q = mu / (r + mu)
        P[:, 0] = np.exp(r * np.log(r / (r + mu)))
        for k in range(1, kmax + 1):
            P[:, k] = P[:, k - 1] * (k - 1 + r) / k * q
    return P


def nb_loglik(y, mu, alpha):
    y = np.asarray(y, int)
    P = nb_pmf(mu, alpha, int(y.max()))
    return float(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)).sum())


def fit_nb(X, y, penalty=None, offset=None, grid=ALPHA_GRID):
    """Profile likelihood over the dispersion grid; returns (alpha, b)."""
    best, b = None, None
    for a in grid:
        b = fit_glm(X, y, a, penalty, offset, b0=b)
        mu = np.exp(np.clip((0 if offset is None else offset) + X @ b, -20, 12))
        ll = nb_loglik(y, mu, a)
        if best is None or ll > best[0]:
            best = (ll, a, b.copy())
    return best[1], best[2]


# --- splines --------------------------------------------------------------------------------------------------------
def ns_raw(x, knots, boundary):
    """Natural cubic spline basis in truncated-power form (Hastie, Tibshirani & Friedman 2009, eq. 5.4-5.5).

    K = all knots incl. the two boundary knots -> K-1 columns (x and K-2 non-linear terms), no intercept.
    The fit is linear beyond the boundary knots.
    """
    x = np.asarray(x, float)
    k = np.r_[boundary[0], np.sort(knots), boundary[1]].astype(float)
    K = len(k)
    d = lambda j: (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[K - 1], 0) ** 3) / (k[K - 1] - k[j])
    return np.column_stack([x - boundary[0]] + [d(j) - d(K - 2) for j in range(K - 2)])


def ns_basis(x, knots, boundary, scale=None):
    """Scaled natural spline basis; ``scale`` is learnt on training data (column max) and reused at prediction."""
    B = ns_raw(x, knots, boundary)
    if scale is None:
        scale = np.abs(B).max(0)
        scale = np.where(scale > 0, scale, 1.0)
    return B / scale, scale


def lag_basis(L, degree=2):
    """Polynomial lag basis C (L+1, degree+1) on l = 0..L, scaled to [0, 1]."""
    l = np.arange(L + 1) / max(L, 1)
    return np.column_stack([l ** j for j in range(degree + 1)])


def cross_basis(lagged_x, knots, boundary, degree=2, scale=None):
    """DLNM cross-basis. ``lagged_x``: (n, L+1) matrix of exposure at lags 0..L. Returns (cb, scale)."""
    n, L1 = lagged_x.shape
    C = lag_basis(L1 - 1, degree)
    if scale is None:
        _, scale = ns_basis(lagged_x.ravel(), knots, boundary)
    Bs = [ns_basis(lagged_x[:, l], knots, boundary, scale)[0] for l in range(L1)]  # each (n, nb)
    nb = Bs[0].shape[1]
    cb = np.zeros((n, nb * C.shape[1]))
    for i in range(nb):
        for j in range(C.shape[1]):
            cb[:, i * C.shape[1] + j] = sum(Bs[l][:, i] * C[l, j] for l in range(L1))
    return cb, scale
