import jax.numpy as jnp
from jax import Array, jit


@jit
def m_step(x: Array, x_prob: Array | None, responsibilities: Array) -> tuple[Array, Array, Array]:
    """
    Perform the M-step of the EM algorithm.

    Parameters
    ----------
    x : jax.Array
        Data points, shape (n_samples, n_features).
    x_prob : jax.Array or None
        If x is samples, counts should be None and it is equivalent to
        counts = jnp.ones(x.shape[0]). If x is histogram bin centers,
        than counts is probabilities in these bin centers, shape (n_bins,).
    responsibilities : jax.Array
        Responsibilities, shape (n_samples, n_components).

    Returns
    -------
    tuple
        Updated parameters (weights, means, covariances).
    """
    n_samples, n_features = x.shape
    n_components = responsibilities.shape[1]

    if x_prob is None:
        x_prob_sum = n_samples
        corrected_resp = responsibilities
    else:
        # Instead of having N samples we have x_prob.sum() "samples"
        x_prob_sum = x_prob.sum()
        # Each term for given "sample" (histogram bin) is "repeated" x_prob times
        responsibilities = x_prob[:, None] * responsibilities
    del x_prob

    # Sum probabilities over samples to get a (non-normalized) probability of each component
    resp_component = responsibilities.sum(axis=0)

    # Update weights - normalize the component probabilities
    weights = resp_component / x_prob_sum
    # Ensure weights sum to 1
    weights = weights / weights.sum()

    # Update means
    means = (responsibilities.T @ x) / resp_component[:, None]

    # Update covariances
    covariances = jnp.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        x_centered = x - means[k]
        covariances = covariances.at[k].set(
            ((responsibilities[:, k][:, None] * x_centered).T @ x_centered) / resp_component[k]
        )

    return weights, means, covariances
