"""Tests for optional periodic dihedral MD bias."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from md.bias import PeriodicDihedralHarmonicBias, build_bias, wrap_degrees


def _bias(center=120.0, k=5.0):
    return PeriodicDihedralHarmonicBias((0, 1, 2, 3), center, k, 180.0)


def test_wrap_degrees_periodic():
    actual = np.asarray(wrap_degrees(jnp.asarray([-540.0, -180.0, 180.0, 540.0, 181.0])))
    np.testing.assert_allclose(actual, [-180.0, -180.0, -180.0, -180.0, -179.0])


def test_bias_config_validation():
    cfg = {
        "type": "periodic_dihedral_harmonic",
        "indices": [0, 1, 2, 3],
        "center_deg": 15,
        "shift_deg": 180,
        "k_kcal_per_mol_rad2": 7.5,
    }
    bias = build_bias(cfg)
    assert bias is not None
    assert bias.indices == (0, 1, 2, 3)
    assert bias.k_kcal_per_mol_rad2 == 7.5


def test_bias_zero_and_symmetry():
    R = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )
    center = float(_bias().cv_degrees(R))
    b0 = _bias(center=center, k=5.0)
    assert abs(float(b0.energy(R))) < 1.0e-7
    b_plus = _bias(center=center + 30.0, k=5.0)
    b_minus = _bias(center=center - 30.0, k=5.0)
    np.testing.assert_allclose(b_plus.energy(R), b_minus.energy(R), rtol=1e-6, atol=1e-6)


def test_bias_force_matches_finite_difference():
    R = jnp.asarray(
        [[-0.4, 0.8, 0.3], [0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [1.5, 0.9, 0.7]],
        dtype=jnp.float32,
    )
    bias = _bias(center=35.0, k=5.0)
    force = np.asarray(bias.force(R))
    assert np.all(np.isfinite(force))
    eps = 1.0e-3
    R_plus = np.asarray(R).copy()
    R_minus = np.asarray(R).copy()
    R_plus[0, 0] += eps
    R_minus[0, 0] -= eps
    fd_force = -(float(bias.energy(jnp.asarray(R_plus))) - float(bias.energy(jnp.asarray(R_minus)))) / (2 * eps)
    np.testing.assert_allclose(force[0, 0], fd_force, rtol=3e-3, atol=3e-3)


def test_jax_dihedral_matches_charron_dataset():
    workspace = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(workspace / "charron_fes_analysis"))
    from functions import compute_ramachandran_angles

    dataset = workspace / "cameo_cg/local_work/support_gate_force_matching/input_data/ala2_cg_data_paper.npz"
    with np.load(dataset) as data:
        R = np.asarray(data["R"][:100])
    phi_np, _ = compute_ramachandran_angles(R)
    phi_jax = np.asarray(jax.vmap(_bias().cv_degrees)(jnp.asarray(R)))
    delta = (phi_jax - phi_np + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(delta)) < 1.0e-4
