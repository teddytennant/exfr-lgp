"""Closed forms for a finite-rank logistic density on [0, 1].

Supplement A of arXiv 2610.09452. The exponent is piecewise linear with value
delta[j] at knot u_j = j/N. On [u_j, u_{j+1}] it equals gamma_j + lambda_j y,
with lambda_j = (delta_{j+1} - delta_j) / Delta and gamma_j = delta_j - lambda_j u_j.
A cell with lambda_j = 0 contributes its limit.
"""

import jax
import jax.numpy as jnp


def knots(n):
    """Uniform knots u_j = j/N on [0, 1], j = 0, ..., N."""
    return jnp.linspace(0.0, 1.0, n + 1)


def hat(z, i, n):
    """Hat basis phi_i(z) = max(1 - |N z - i|, 0), equation (6)."""
    return jnp.maximum(1.0 - jnp.abs(n * z - i), 0.0)


def nodal_values(xi, w, s):
    """delta_i in (S1): the exponent at y-knot i, for a fixed second coordinate.

    xi has shape (N+1, N+1). At most two hat functions of w are nonzero, so
    each delta_i combines at most two coefficients. The location term of the
    paper is the same interpolation in s and is added by the caller.
    """
    n = xi.shape[0] - 1
    j = jnp.arange(n + 1)
    phi = jax.vmap(lambda jj: hat(w, jj, n))(j)
    return xi @ phi


def interpolate(delta, y):
    """Piecewise-linear exponent at y, value delta[j] at knot j/N."""
    n = delta.shape[0] - 1
    pos = jnp.clip(y, 0.0, 1.0) * n
    j = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n - 1)
    t = pos - j
    return (1.0 - t) * delta[j] + t * delta[j + 1]


def _cells(delta):
    """Per-cell (lambda, gamma, unnormalized integral), Supplement A.1.

    The max nodal value is subtracted before the exponential and restored by
    the caller, as the paper does to avoid overflow.
    """
    n = delta.shape[0] - 1
    delta_n = n
    shift = jnp.max(delta)
    d = delta - shift
    left = d[:-1]
    right = d[1:]
    lam = (right - left) * delta_n
    # gamma such that gamma + lam * u_j = left, with u_j = j/N.
    j = jnp.arange(n)
    gamma = left - lam * (j / delta_n)
    # integral of exp(gamma + lam y) on [u_j, u_{j+1}].
    # lam != 0: (exp(right) - exp(left)) / lam. lam == 0: Delta * exp(left).
    safe = jnp.where(jnp.abs(lam) < 1e-8, 1.0, lam)
    piece = (jnp.exp(right) - jnp.exp(left)) / safe
    piece = jnp.where(jnp.abs(lam) < 1e-8, jnp.exp(left) / delta_n, piece)
    return lam, gamma, piece, shift


def log_normalizer(delta):
    """psi = log Z, Supplement A.1 (S3), with the max-shift of the paper."""
    _, _, piece, shift = _cells(delta)
    return jnp.log(jnp.sum(piece)) + shift


def log_density(delta, y):
    """log f(y) = g(y) - psi."""
    return interpolate(delta, y) - log_normalizer(delta)


def mean(delta):
    """E[y], Supplement A.2.

    On a cell the exponent is gamma + lam y, so the unnormalized first moment
    is exp(gamma) times the integral of y exp(lam y). A flat cell contributes
    exp(delta) (u_{j+1}^2 - u_j^2) / 2, which is the paper's limit.
    """
    n = delta.shape[0] - 1
    lam, _, _, shift = _cells(delta)
    d = delta - shift
    j = jnp.arange(n)
    u = j / n
    v = (j + 1) / n
    safe = jnp.where(jnp.abs(lam) < 1e-8, 1.0, lam)
    # int_u^v y exp(lam y) dy, then multiply by exp(gamma) = exp(left - lam u).
    antideriv = lambda t: (t / safe - 1.0 / safe**2) * jnp.exp(lam * t)
    curved = jnp.exp(d[:-1] - lam * u) * (antideriv(v) - antideriv(u))
    flat = jnp.exp(d[:-1]) * (v**2 - u**2) / 2.0
    cell = jnp.where(jnp.abs(lam) < 1e-8, flat, curved)
    return jnp.sum(cell) * jnp.exp(shift - log_normalizer(delta))


def quantile(delta, eta):
    """eta-quantile, Supplement A.3.

    S_j is the sum of the first j cell integrals in (S3). j0 is the last cell
    whose cumulative mass is at most eta * Z, and t_eta solves the cell integral
    equation. The flat-cell branch is the paper's limit.
    """
    n = delta.shape[0] - 1
    lam, gamma, piece, shift = _cells(delta)
    z = jnp.sum(piece)
    target = eta * z
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(piece)])
    # S_{j0} <= target < S_{j0+1}, with S_0 = 0. cumulative[j] is S_j.
    j0 = jnp.sum(cumulative[:-1] <= target) - 1
    j0 = jnp.clip(j0, 0, n - 1)
    s_j0 = cumulative[j0]
    lam0 = lam[j0]
    gamma0 = gamma[j0]
    delta0 = (delta - shift)[j0]
    u = j0 / n
    # R_eta = lam (eta Z - S_j0) + exp(delta_j0), in the shifted scale. The
    # shift cancels because both the target and the cell integrals lost it.
    r = lam0 * (target - s_j0) + jnp.exp(delta0)
    curved = (jnp.log(jnp.maximum(r, 1e-30)) - gamma0) / jnp.where(
        jnp.abs(lam0) < 1e-8, 1.0, lam0
    )
    flat = u + jnp.exp(-delta0) * (target - s_j0)
    return jnp.where(jnp.abs(lam0) < 1e-8, flat, curved)


def sample_density(delta, key, n_samples):
    """Inverse-transform samples from the piecewise-linear logistic density."""
    eta = jax.random.uniform(key, (n_samples,), minval=1e-6, maxval=1.0 - 1e-6)
    return jax.vmap(lambda e: quantile(delta, e))(eta)
