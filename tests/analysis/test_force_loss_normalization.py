import numpy as np
import jax.numpy as jnp

from training.trainer import valid_component_mse


def test_valid_component_mse_matches_hand_masked_average():
    predictions = jnp.array([[[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]], dtype=jnp.float32)
    targets = jnp.array([[[2.0, 4.0, 6.0], [100.0, 100.0, 100.0]]], dtype=jnp.float32)
    weights = jnp.array([[1.0, 0.0]], dtype=jnp.float32)

    loss = float(valid_component_mse(predictions, targets, weights=weights))
    expected = np.mean(np.square(np.asarray(targets[0, 0]) - np.asarray(predictions[0, 0])))
    np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)


def test_valid_component_mse_reduces_to_legacy_mean_without_weights():
    predictions = jnp.array([[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]], dtype=jnp.float32)
    targets = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=jnp.float32)

    loss = float(valid_component_mse(predictions, targets))
    expected = float(np.mean(np.square(np.asarray(targets) - np.asarray(predictions))))
    np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)


def test_valid_component_mse_is_finite_for_fully_masked_batch():
    predictions = jnp.ones((2, 3, 3), dtype=jnp.float32)
    targets = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    weights = jnp.zeros((2, 3), dtype=jnp.float32)

    loss = float(valid_component_mse(predictions, targets, weights=weights))
    assert np.isfinite(loss)
    assert loss == 0.0


def test_valid_component_mse_matches_per_structure_weighted_average():
    predictions = jnp.array([
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0]],
    ], dtype=jnp.float32)
    targets = jnp.array([
        [[1.0, 2.0, 3.0], [3.0, 5.0, 7.0], [13.0, 14.0, 15.0]],
    ], dtype=jnp.float32)
    # First two atoms belong to one structure, last atom to another.
    weights = jnp.array([[0.5, 0.5, 1.0]], dtype=jnp.float32)

    loss = float(valid_component_mse(predictions, targets, weights=weights))

    sq = np.square(np.asarray(targets) - np.asarray(predictions))
    struct0 = np.mean(sq[0, :2])
    struct1 = np.mean(sq[0, 2:3])
    expected = float((struct0 + struct1) / 2.0)
    np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)
