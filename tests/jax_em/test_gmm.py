import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from jax_em.gmm import em_gmm
from jax_em.parameters import initialize_parameters


@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("n_samples", [10_000])
def test_em_gmm(n_samples, n_features, n_components):
    key = jax.random.PRNGKey(n_samples + (n_features << 3) + n_components)

    # Generate true parameters
    key, subkey = jax.random.split(key)
    true_weights, true_means, true_covariances = initialize_parameters(subkey, n_components=n_components, n_features=n_features)

    # Sample from the true model
    n_samples_component = jnp.round(n_samples * true_weights).astype(int)
    n_samples_component = n_samples_component.at[-1].add(n_samples - n_samples_component.sum())
    assert jnp.all(n_samples_component > 0)
    assert n_samples_component.sum() == n_samples
    samples = jnp.concatenate(
        [
            jax.random.multivariate_normal(subkey, mean=mean, cov=cov, shape=(n,))
            for n, weight, mean, cov in zip(n_samples_component, true_weights, true_means, true_covariances)
        ]
    )

    # Fit the model
    key, subkey = jax.random.split(key)
    # jax.jit doens't make it any faster
    fit_weights, fit_means, fit_covariances = em_gmm(subkey, n_components, samples, x_prob=None, max_iter=100, tol=None)

    # Sort by weights, true data is already sorted
    weight_sorter = jnp.argsort(fit_weights, descending=True)
    fit_weights, fit_means, fit_covariances = fit_weights[weight_sorter], fit_means[weight_sorter], fit_covariances[weight_sorter]

    assert_allclose(fit_weights, true_weights, atol=0.01, err_msg="weights")
    assert_allclose(fit_means, true_means, atol=0.01, err_msg="means")
    assert_allclose(fit_covariances, true_covariances, atol=0.01, err_msg="covariances")
