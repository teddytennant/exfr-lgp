"""Elliptical slice sampling for the nodal coefficients.

Murray, Adams and MacKay, 2010, as used in Section 4.2 of arXiv 2610.09452.
The prior on the nodal vector is standard normal. The likelihood is the exact
finite-rank logistic likelihood: each observation contributes log f(y_i).
"""

import jax
import jax.numpy as jnp

from exfr_lgp.density import log_density


def _loglik(delta, y):
    return jnp.sum(jax.vmap(lambda yi: log_density(delta, yi))(y))


def elliptical_slice(delta, y, key, n_steps):
    """One chain of ESS updates of delta under a N(0, I) prior.

    Returns the state after n_steps. The bracket shrinks until the slice is
    met, which the construction guarantees, so the loop is bounded by a fixed
    number of shrinks rather than a rejection counter.
    """

    def step(state, key):
        delta, _ = state
        k_nu, k_theta, k_u = jax.random.split(key, 3)
        nu = jax.random.normal(k_nu, delta.shape)
        log_y = _loglik(delta, y) + jnp.log(jax.random.uniform(k_u, minval=1e-12))
        theta = jax.random.uniform(k_theta, minval=0.0, maxval=2.0 * jnp.pi)
        bracket = jnp.array([theta - 2.0 * jnp.pi, theta])

        def shrink(carry, _):
            delta, theta, bracket, accepted = carry
            proposal = delta * jnp.cos(theta) + nu * jnp.sin(theta)
            ok = _loglik(proposal, y) > log_y
            delta = jnp.where(ok, proposal, delta)
            # Shrink the bracket toward the side that contains 0.
            lo = jnp.where(theta < 0.0, theta, bracket[0])
            hi = jnp.where(theta < 0.0, bracket[1], theta)
            bracket = jnp.where(ok, bracket, jnp.array([lo, hi]))
            span = bracket[1] - bracket[0]
            theta = jnp.where(ok, theta, bracket[0] + jax.random.uniform(k_theta) * span)
            accepted = accepted | ok
            return (delta, theta, bracket, accepted), None

        (delta, _, _, accepted), _ = jax.lax.scan(
            shrink, (delta, theta, bracket, jnp.bool_(False)), None, length=16
        )
        return (delta, accepted), delta

    keys = jax.random.split(key, n_steps)
    (_, _), chain = jax.lax.scan(step, (delta, jnp.bool_(False)), keys)
    return chain


def fit_nodes(y, n, key, n_steps):
    """Posterior samples of an (N+1,)-node exponent, single location.

    This is the g2-only special case of Section 2: no covariate index, so the
    exponent is the nodal vector itself. Enough to check that ESS targets the
    exact posterior the closed form defines.
    """
    k_init, k_chain = jax.random.split(key)
    delta0 = jax.random.normal(k_init, (n + 1,)) * 0.1
    return elliptical_slice(delta0, y, k_chain, n_steps)
