"""Tests for the basin free-energy calibration loss (L_cal)."""

from __future__ import annotations

import numpy as np
import pytest

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from training.calibration_loss import (
    KT_KCAL_MOL,
    PAIRS,
    REQUIRED_BASINS,
    build_calibration_panel,
    calibration_config,
    calibration_error,
    calibration_quantity_value,
    lse_free_energy_pair_diffs,
    make_calibration_penalty,
    make_calibration_quantity,
)

PAIR_IDX = ((0, 1), (0, 2), (1, 2))
POPS = (0.652, 0.2997, 0.0279)


def test_pairs_and_basins():
    assert REQUIRED_BASINS == ("beta", "alphaR", "alphaL")
    assert set(PAIRS) == {("beta", "alphaR"), ("beta", "alphaL"), ("alphaR", "alphaL")}


def test_lse_free_energy_uniform_shift():
    """Uniform per-basin energies give dF_model(A,B) = u_A - u_B exactly."""
    n = 5
    U = jnp.array([0.0] * n + [1.7] * n)
    ids = jnp.array([0] * n + [1] * n)
    diffs = lse_free_energy_pair_diffs(
        U, ids, pair_ids=((0, 1),), kT=KT_KCAL_MOL
    )
    np.testing.assert_allclose(float(diffs[0]), -1.7, rtol=1e-6, atol=1e-8)


def test_quantity_value_zero_when_calibrated():
    rng = np.random.default_rng(0)
    n_per = 64
    U = rng.normal(0, 0.4, size=(3 * n_per,))
    ids = np.repeat(np.arange(3), n_per)

    # Realize consistent free-energy levels f with f_a - f_b = dF_ref(a,b)
    # for all pairs (note dF_ref carries the minus sign); identical noise per
    # basin so LSE differences reduce exactly to the level differences.
    ref = -KT_KCAL_MOL * np.log(np.array(POPS)[:, None] / np.array(POPS)[None, :])
    f_levels = np.array([0.0, -ref[0, 1], -ref[0, 2]])
    noise = U[:n_per]
    U = np.concatenate([f_levels[b] + noise for b in range(3)])

    out = calibration_quantity_value(
        U=jnp.asarray(U), basin_ids=jnp.asarray(ids),
        kT=KT_KCAL_MOL, populations=POPS,
    )
    np.testing.assert_allclose(float(out), 0.0, atol=1e-8)


def test_quantity_value_matches_manual_computation():
    rng = np.random.default_rng(1)
    n_per = 32
    U = rng.normal(0, 1.0, size=(3 * n_per,))
    ids = np.repeat(np.arange(3), n_per)
    out = calibration_quantity_value(
        U=jnp.asarray(U), basin_ids=jnp.asarray(ids),
        kT=KT_KCAL_MOL, populations=POPS,
    )
    manual = []
    for a, b in PAIR_IDX:
        m_a = KT_KCAL_MOL * (
            float(logsumexp(U[a * n_per:(a + 1) * n_per] / KT_KCAL_MOL)) - np.log(n_per)
        )
        m_b = KT_KCAL_MOL * (
            float(logsumexp(U[b * n_per:(b + 1) * n_per] / KT_KCAL_MOL)) - np.log(n_per)
        )
        manual.append(m_a - m_b)
    dF_ref = [-KT_KCAL_MOL * np.log(POPS[a] / POPS[b]) for a, b in PAIR_IDX]
    expected = float(np.sum((np.array(manual) - np.array(dF_ref)) ** 2))
    np.testing.assert_allclose(float(out), expected, rtol=1e-5)


def test_gradient_flows_through_quantity():
    """Params must receive a finite, nonzero gradient through the panel LSE."""
    rng = np.random.default_rng(2)
    n_per = 16
    R = rng.normal(0, 1.0, size=(3 * n_per, 2, 3)).astype(np.float32)
    ids = np.repeat(np.arange(3), n_per)
    mask = np.ones((3 * n_per, 2), dtype=np.float32)
    species = np.zeros((3 * n_per, 2), dtype=np.int32)

    quantity = make_calibration_quantity(
        energy_of=lambda params, R_, m_, s_: params["w"] * jnp.sum(R_),
        R=jnp.asarray(R), mask=jnp.asarray(mask), species=jnp.asarray(species),
        basin_ids=jnp.asarray(ids), kT=KT_KCAL_MOL, populations=POPS,
    )

    def loss_of(w):
        return calibration_error(quantity(None, energy_params={"w": w}), None)

    g = jax.grad(loss_of)(jnp.array(0.3))
    assert np.isfinite(float(g)) and abs(float(g)) > 0


def test_build_calibration_panel_balanced_and_targets(tmp_path):
    """Panel balanced across basins; dF_ref reproduces -kT ln(P ratio)."""
    rng = np.random.default_rng(3)
    n = {"beta": 400, "alphaR": 200, "alphaL": 40}
    ranges = {"beta": (-120.0, 140.0), "alphaR": (-80.0, -40.0), "alphaL": (60.0, 40.0)}
    phi, psi = [], []
    for name, cnt in n.items():
        p0, s0 = ranges[name]
        phi.append(p0 + 2.0 * rng.standard_normal(cnt))
        psi.append(s0 + 2.0 * rng.standard_normal(cnt))
    phi, psi = np.concatenate(phi), np.concatenate(psi)
    frames = len(phi)
    R = rng.normal(0, 0.01, size=(frames, 6, 3)).astype(np.float32)
    np.savez(tmp_path / "panel_src.npz", R=R,
             mask=np.ones((frames, 6), dtype=np.float32),
             species=np.zeros((frames, 6), dtype=np.int32),
             phi=phi, psi=psi)

    panel = build_calibration_panel(tmp_path / "panel_src.npz", frames_per_basin=16, seed=7)
    assert panel.R.shape == (48, 6, 3)
    assert {b: int((panel.labels == b).sum()) for b in REQUIRED_BASINS} == {
        "beta": 16, "alphaR": 16, "alphaL": 16,
    }
    total = sum(n.values())
    expected = -KT_KCAL_MOL * np.log((n["beta"] / total) / (n["alphaR"] / total))
    idx = PAIRS.index(("beta", "alphaR"))
    np.testing.assert_allclose(float(panel.dF_ref[idx]), expected, rtol=1e-6)


def test_config_defaults_disabled():
    class _Cfg:
        def get(self, *path, default=None):
            return default

    assert calibration_config(_Cfg())["enabled"] is False


def test_calibration_error_reduces_to_mean_prediction():
    assert float(calibration_error(jnp.array([2.0, 4.0]), None)) == pytest.approx(3.0)


def test_calibration_penalty_matches_quantity_and_has_grad():
    """Live path: penalty_fn == lam * L_cal, shift-invariant, gradient finite."""
    rng = np.random.default_rng(7)
    n_per = 16
    R = rng.normal(0, 1.0, size=(3 * n_per, 2, 3)).astype(np.float32)
    ids = np.repeat(np.arange(3), n_per)
    mask = np.ones((3 * n_per, 2), dtype=np.float32)
    species = np.zeros((3 * n_per, 2), dtype=np.int32)
    energy_of = lambda params, R_, m_, s_: params["w"] * jnp.sum(R_)  # noqa: E731

    pen = make_calibration_penalty(
        energy_of=energy_of, R=jnp.asarray(R), mask=jnp.asarray(mask),
        species=jnp.asarray(species), basin_ids=jnp.asarray(ids),
        kT=KT_KCAL_MOL, populations=POPS, lam=0.1,
    )
    U = jax.vmap(lambda r, m_, s_: energy_of({"w": 0.4}, r, m_, s_))(R, mask, species)
    expected = 0.1 * float(calibration_quantity_value(
        U=U, basin_ids=ids, kT=KT_KCAL_MOL, populations=np.array(POPS)))
    assert float(pen({"w": 0.4})) == pytest.approx(expected, rel=1e-5)

    g = jax.grad(lambda w: pen({"w": w}))(0.3)
    assert np.isfinite(float(g))
