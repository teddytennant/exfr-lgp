"""Closed forms against quadrature, and a short ESS recovery check."""

import jax
import jax.numpy as jnp
import numpy as np

from exfr_lgp.density import (
    hat,
    interpolate,
    log_density,
    log_normalizer,
    mean,
    nodal_values,
    quantile,
    sample_density,
)
from exfr_lgp.ess import fit_nodes


def _trap(values, y):
    return np.trapezoid(values, y)


def test_hat_basis_is_a_partition_of_unity():
    n = 4
    z = jnp.linspace(0.0, 1.0, 21)
    total = sum(hat(z, i, n) for i in range(n + 1))
    assert jnp.allclose(total, 1.0)


def test_normalizer_matches_quadrature():
    delta = jnp.array([0.2, -0.4, 0.7, 0.1, -0.3])
    psi = float(log_normalizer(delta))
    y = np.linspace(0.0, 1.0, 8001)
    g = np.asarray(jax.vmap(lambda yi: interpolate(delta, yi))(y))
    quad = np.log(_trap(np.exp(g), y))
    assert abs(psi - quad) < 1e-4


def test_flat_exponent_is_uniform():
    delta = jnp.zeros(5)
    assert abs(float(log_normalizer(delta))) < 1e-6
    assert abs(float(mean(delta)) - 0.5) < 1e-6
    for eta, expect in ((0.25, 0.25), (0.5, 0.5), (0.9, 0.9)):
        assert abs(float(quantile(delta, eta)) - expect) < 1e-6


def test_density_integrates_to_one():
    delta = jnp.array([1.0, -1.0, 0.5, 0.0, -0.5, 0.8])
    y = np.linspace(0.0, 1.0, 8001)
    f = np.exp(np.asarray(jax.vmap(lambda yi: log_density(delta, yi))(y)))
    assert abs(_trap(f, y) - 1.0) < 1e-4


def test_mean_matches_quadrature():
    delta = jnp.array([0.3, 1.2, -0.6, 0.4, -0.2])
    y = np.linspace(0.0, 1.0, 8001)
    log_f = np.asarray(jax.vmap(lambda yi: log_density(delta, yi))(y))
    f = np.exp(log_f)
    quad = _trap(y * f, y)
    assert abs(float(mean(delta)) - quad) < 1e-3


def test_quantile_inverts_the_cdf():
    delta = jnp.array([-0.5, 0.4, 1.1, -0.2, 0.3])
    y = np.linspace(0.0, 1.0, 8001)
    f = np.exp(np.asarray(jax.vmap(lambda yi: log_density(delta, yi))(y)))
    widths = np.diff(y)
    cdf = np.cumsum((f[:-1] + f[1:]) / 2.0 * widths)
    for eta in (0.1, 0.37, 0.5, 0.8):
        t = float(quantile(delta, eta))
        assert abs(np.interp(t, y[1:], cdf) - eta) < 2e-3


def test_nodal_interpolation_uses_two_hats():
    xi = jnp.arange(9.0).reshape(3, 3)
    # w = 0.5 sits on knot 1 of a N=2 grid, so only column 1 contributes.
    delta = nodal_values(xi, 0.5, 0.0)
    assert jnp.allclose(delta, xi[:, 1])


def test_inverse_transform_samples_match_mean():
    delta = jnp.array([0.0, 1.0, 0.0, -0.5])
    draws = sample_density(delta, jax.random.PRNGKey(0), 4000)
    assert abs(float(jnp.mean(draws)) - float(mean(delta))) < 0.03


def test_ess_recovers_a_shifted_density():
    """Samples from a known piecewise-linear density; ESS should move the
    posterior mean of the nodes toward that exponent, up to an additive
    constant the normalizer removes."""
    truth = jnp.array([-1.0, 0.6, 1.2, -0.4])
    draws = sample_density(truth, jax.random.PRNGKey(1), 600)
    chain = fit_nodes(draws, n=3, key=jax.random.PRNGKey(2), n_steps=40)
    posterior = jnp.mean(chain[20:], axis=0)
    # Compare centered exponents: the likelihood is invariant to adding a
    # constant, and the prior only weakly pins it.
    centered = posterior - jnp.mean(posterior)
    truth_c = truth - jnp.mean(truth)
    assert float(jnp.max(jnp.abs(centered - truth_c))) < 0.75
