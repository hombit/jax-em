import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.stats import multivariate_normal


@jit
def e_step(x: Array, x_prob: Array | None, weights: Array, means: Array, covariances: Array) -> Array:
    """
    Perform the E-step of the EM algorithm.

    Parameters
    ----------
    x : jax.Array
        Data points, shape (n_samples, n_features).
    x_prob : jax.Array or None
        If x is samples, counts should be None and it is equivalent to
        counts = jnp.ones(x.shape[0]). If x is histogram bin centers,
        than counts are probabilities in these bin centers, shape (n_bins,).
    weights : jax.Array
        Mixing weights, shape (n_components,).
    means : jax.Array
        Means of the Gaussians, shape (n_components, n_features).
    covariances : jax.Array
        Covariance matrices, shape (n_components, n_features, n_features).

    Returns
    -------
    jax.Array
        Responsibilities, shape (n_samples, n_components).
    """
    n_samples, n_components = x.shape[0], weights.shape[0]

    if x_prob is None:
        x_prob = jnp.array(1.0)

    # Compute responsibilities
    log_probs = jnp.stack([
        x_prob * (jnp.log(weights[k]) + multivariate_normal.logpdf(x, mean=means[k], cov=covariances[k]))
        for k in range(n_components)
    ], axis=1)

    log_probs_max = jnp.max(log_probs, axis=1, keepdims=True)
    log_probs = log_probs - log_probs_max  # For numerical stability
    responsibilities = jnp.exp(log_probs)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities
