import jax
import jax.numpy as jnp
from jax import Array


def initialize_parameters(key: Array, n_components: int, n_features: int) -> tuple[Array, Array, Array]:
    """
    Initialize parameters for the Gaussian mixture model.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components.
    n_features : int
        Dimensionality of the data.
    key : ArrayLike
        Random key for initialization.

    Returns
    -------
    tuple
        Tuple of initialized parameters (weights, means, covariances).
    """
    key1, key2 = jax.random.split(key, 2)
    weights = jnp.sort(jax.random.dirichlet(key1, jnp.ones(n_components)), axis=-1, descending=True)
    means = jax.random.normal(key2, shape=(n_components, n_features))
    covariances = jnp.tile(0.1*jnp.eye(n_features), (n_components, 1, 1))
    return weights, means, covariances
