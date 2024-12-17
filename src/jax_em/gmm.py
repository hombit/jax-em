from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.typing import ArrayLike

from jax_em.e_step import e_step
from jax_em.m_step import m_step
from jax_em.parameters import initialize_parameters


__all__ = ["em_gmm",]


def em_gmm(key: ArrayLike, n_components: int, x: ArrayLike, x_prob: ArrayLike | None, *, max_iter: int = 100, tol: float | None = 1e-6) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Fit a Gaussian mixture model using the EM algorithm.

    Parameters
    ----------
    key : ArrayLike
        Random key for algorithm initialization, usually output of
        `jax.random.PRNGKey`.
    n_components : int
        Number of Gaussian components.
    x : jax.Array
        Data points, shape (n_samples, n_features).
    x_prob : jax.Array or None
        If x is samples, counts should be None and it is equivalent to
        counts = jnp.ones(x.shape[0]). If x is histogram bin centers,
        than counts are probabilities in these bin centers, shape (n_bins,).
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    tol : float or None, optional
        Convergence tolerance for log-likelihood (default is 1e-6).
        If None, the algorithm will run for `max_iter` iterations.

    Returns
    -------
    tuple
        Fitted parameters (weights, means, covariances).
    """
    key = jnp.asarray(key)
    x = jnp.asarray(x)
    x_prob = None if x_prob is None else jnp.asarray(x_prob)

    n_samples, n_features = x.shape

    # Initialize parameters
    weights, means, covariances = initialize_parameters(key, n_components, n_features)

    if tol is None:
        weights, means, covariances = jax.lax.fori_loop(
            lower=0,
            upper=max_iter,
            body_fun=lambda _i, params: m_step(x, x_prob, e_step(x, x_prob, *params)),
            init_val=(weights, means, covariances),
        )
    else:
        prev_log_likelihood = -jnp.inf
        for iteration in range(max_iter):
            responsibilities = e_step(x, x_prob, weights, means, covariances)
            weights, means, covariances = m_step(x, x_prob, responsibilities)

            log_likelihood = jnp.sum(
                jnp.log(jnp.sum(
                    jnp.array([
                        weights[k] * multivariate_normal.pdf(x, mean=means[k], cov=covariances[k])
                        for k in range(n_components)
                    ]).T,
                    axis=1
                ))
            )

            # Check convergence
            if jnp.abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

    return weights, means, covariances
