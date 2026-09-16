"""Unit tests for the optional heteroscedastic force weighting."""
import numpy as np
import pytest

# `training/__init__.py` imports .trainer -> chemtrain -> jax.tree_map, which JAX
# 0.10 removed. utils/jax_setup provides the shim (md/__init__.py applies it the
# same way), but there is no tests/conftest.py to do it at collection time.
# label_uncertainty itself only needs numpy.
from utils.jax_setup import apply_jax_compat_shims
apply_jax_compat_shims()

from training import label_uncertainty as lu


def _split(sigma, n=4, nb=6):
    mask = np.ones((n, nb), np.float32)
    return {"mask": mask, "sigma_label": np.asarray(sigma, np.float32)}


def _base(n=4, nb=6):
    return np.full((n, nb), 1.0 / nb, np.float32)


def test_disabled_by_default():
    assert not lu.is_enabled({})
    assert not lu.is_enabled({"enabled": False})


def test_mode_none_is_identity():
    b = _base()
    w = lu.build_weights(_split(np.ones((4, 6))), {"mode": "none"}, b)
    np.testing.assert_allclose(w, b)


def test_inverse_variance_ordering():
    """Smaller sigma must get MORE weight."""
    sig = np.tile(np.array([1.0, 2.0, 4.0, 8.0])[:, None], (1, 6))
    w = lu.build_weights(_split(sig), {"max_weight_ratio": 0}, _base())
    assert w[0, 0] > w[1, 0] > w[2, 0] > w[3, 0]
    # 1/sigma^2 -> doubling sigma quarters the weight
    np.testing.assert_allclose(w[0, 0] / w[1, 0], 4.0, rtol=1e-5)


def test_preserve_scale_keeps_total():
    b = _base()
    sig = np.tile(np.array([1.0, 2.0, 4.0, 8.0])[:, None], (1, 6))
    w = lu.build_weights(_split(sig), {"preserve_scale": True, "max_weight_ratio": 0}, b)
    # total weight unchanged => gradient scale unchanged => a weighted-vs-unweighted
    # comparison is not confounded by an effective learning-rate change
    np.testing.assert_allclose(w.sum(), b.sum(), rtol=1e-5)


def test_max_weight_ratio_caps_dynamic_range():
    sig = np.tile(np.array([0.01, 1.0, 1.0, 1.0])[:, None], (1, 6))
    w = lu.build_weights(_split(sig), {"max_weight_ratio": 10.0}, _base())
    r = w[w > 0]
    assert r.max() / r.min() <= 10.0 + 1e-4


def test_zero_base_weight_stays_zero():
    b = _base(); b[2, :] = 0.0
    w = lu.build_weights(_split(np.ones((4, 6))), {}, b)
    assert np.all(w[2] == 0.0)


def test_nan_sigma_does_not_poison():
    sig = np.ones((4, 6)); sig[1, :] = np.nan
    w = lu.build_weights(_split(sig), {}, _base())
    assert np.all(np.isfinite(w))
    assert np.all(w[1] == 0.0)          # unusable label -> no gradient contribution


def test_per_frame_sigma_broadcasts():
    w = lu.build_weights(_split(np.array([1.0, 2.0, 4.0, 8.0])), {"max_weight_ratio": 0}, _base())
    np.testing.assert_allclose(w[0, 0] / w[1, 0], 4.0, rtol=1e-5)


def test_missing_key_is_a_loud_error():
    with pytest.raises(KeyError, match="sigma_label"):
        lu.build_weights({"mask": np.ones((2, 6), np.float32)}, {"enabled": True}, _base(2))


def test_bad_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        lu.build_weights(_split(np.ones((4, 6))), {"mode": "bogus"}, _base())


def test_configure_roundtrip():
    lu.configure({"enabled": True, "key": "sigma_label"})
    assert lu.is_enabled(lu.active())
    lu.configure(None)
    assert not lu.is_enabled(lu.active())
