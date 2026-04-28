from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.prior_energy import (
    PriorEnergy,
    _build_charge_and_group_by_species,
    _stickiness_alpha_from_free,
    compute_dh_energy,
    compute_fene_energy,
    compute_leash_energy,
    compute_salt_bridge_energy,
)
from models.topology import TopologyBuilder


class _DummyConfig:
    def __init__(self, cfg):
        self._cfg = cfg
        self.config_path = Path(".")

    def get(self, *keys, default=None):
        value = self._cfg
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_prior_params(self):
        return self.get("model", "priors", default={})

    def get_prior_weights(self):
        weights = {
            "bond": 0.5,
            "angle": 0.1,
            "repulsive": 0.25,
            "dihedral": 0.15,
            "excluded_volume": 1.0,
            "wca": 0.0,
            "fene": 0.0,
            "leash": 0.0,
            "dh": 0.0,
            "stickiness": 0.0,
            "salt_bridge": 0.0,
        }
        weights.update(self.get("model", "priors", "weights", default={}))
        return weights


def _base_prior_cfg():
    return {
        "model": {
            "priors": {
                "use_spline_priors": False,
                "weights": {
                    "bond": 0.5,
                    "angle": 0.2,
                    "repulsive": 1.0,
                    "dihedral": 0.3,
                    "excluded_volume": 0.4,
                    "wca": 0.0,
                    "fene": 0.0,
                    "leash": 0.0,
                    "dh": 0.0,
                    "stickiness": 0.0,
                    "salt_bridge": 0.0,
                },
                "r0": 3.8,
                "kr": 120.0,
                "a": [0.0],
                "b": [0.0],
                "epsilon": 1.0,
                "sigma": 3.0,
                "epsilon_ex": 1.0,
                "sigma_ex": 3.5,
                "k_dih": [0.0],
                "gamma_dih": [0.0],
                "aa_typing": {
                    "source": "dataset_map",
                    "his_charge": 0.0,
                    "group_order": ["POSITIVE", "NEGATIVE", "POLAR_UNCHARGED", "NONPOLAR"],
                    "stickiness_reference_group": "POLAR_UNCHARGED",
                },
                "dh": {
                    "enabled": False,
                    "mode": "local_k",
                    "K": 2,
                    "k_DH": 1.0,
                    "lambda_D": 8.0,
                    "w_by_sep": [0.0, 1.0, 0.1],
                },
                "stickiness": {
                    "enabled": False,
                    "min_seq_sep": 3,
                    "r0": 3.8,
                    "sigma": 0.4,
                    "s_free_init": [0.0, 0.0, 0.0],
                },
                "salt_bridge": {
                    "enabled": False,
                    "min_seq_sep": 3,
                    "delta": -0.5,
                    "r0": 3.8,
                    "sigma": 0.3,
                },
            }
        }
    }


def _deep_update(dst, src):
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _make_prior(overrides=None, n_max=8, id_to_aa=None):
    cfg = _base_prior_cfg()
    if overrides:
        cfg = _deep_update(cfg, overrides)
    config = _DummyConfig(cfg)
    topology = TopologyBuilder(N_max=n_max, min_repulsive_sep=6)
    return PriorEnergy(config, topology, displacement=lambda a, b: a - b, id_to_aa=id_to_aa)


def test_dh_sign_channels():
    charge_by_species, _ = _build_charge_and_group_by_species(
        {0: "LYS", 1: "ARG", 2: "ASP"}, his_charge=0.0
    )
    R = jnp.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=jnp.float32)
    mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    seq_sep = jnp.array([1], dtype=jnp.int32)
    w = jnp.array([0.0, 1.0, 0.1], dtype=jnp.float32)

    e_pp = compute_dh_energy(
        R, mask, jnp.array([0, 1], dtype=jnp.int32), pairs, seq_sep, charge_by_species, 1.0, 8.0, w
    )
    e_mm = compute_dh_energy(
        R, mask, jnp.array([2, 2], dtype=jnp.int32), pairs, seq_sep, charge_by_species, 1.0, 8.0, w
    )
    e_pm = compute_dh_energy(
        R, mask, jnp.array([0, 2], dtype=jnp.int32), pairs, seq_sep, charge_by_species, 1.0, 8.0, w
    )

    assert float(e_pp) > 0.0
    assert float(e_mm) > 0.0
    assert float(e_pm) < 0.0


def test_fene_zero_force_at_r0_and_wall_growth():
    mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    bonds = jnp.array([[0, 1]], dtype=jnp.int32)

    R_eq = jnp.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=jnp.float32)

    def energy(R_):
        return compute_fene_energy(
            R_, mask, bonds, r0=3.8, R0=1.5, k=300.0, wall_energy=1.0e6, eps=1.0e-6
        )

    F_eq = -jax.grad(energy)(R_eq)
    np.testing.assert_allclose(np.asarray(F_eq), 0.0, atol=1e-6, rtol=0.0)

    R_wall = jnp.array([[0.0, 0.0, 0.0], [3.8 + 0.99 * 1.5, 0.0, 0.0]], dtype=jnp.float32)
    assert float(energy(R_wall)) > 100.0


def test_leash_zero_inside_flat_bottom():
    mask = jnp.array([1.0, 1.0], dtype=jnp.float32)
    pairs = jnp.array([[0, 1]], dtype=jnp.int32)
    R = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=jnp.float32)

    def energy(R_):
        return compute_leash_energy(R_, mask, pairs, d_safe=10.0, k_safe=0.2)

    np.testing.assert_allclose(np.asarray(energy(R)), 0.0, atol=0.0, rtol=0.0)
    F = -jax.grad(energy)(R)
    np.testing.assert_allclose(np.asarray(F), 0.0, atol=0.0, rtol=0.0)


def test_dh_sequence_gating_k1_vs_k2():
    id_to_aa = {0: "LYS"}
    R = jnp.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [11.4, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    mask = jnp.ones((4,), dtype=jnp.float32)
    species = jnp.zeros((4,), dtype=jnp.int32)

    p1 = _make_prior(
        overrides={
            "model": {"priors": {"dh": {"enabled": True, "K": 1, "w_by_sep": [0.0, 1.0]}}},
        },
        n_max=4,
        id_to_aa=id_to_aa,
    )
    p2 = _make_prior(
        overrides={
            "model": {"priors": {"dh": {"enabled": True, "K": 2, "w_by_sep": [0.0, 1.0, 0.5]}}},
        },
        n_max=4,
        id_to_aa=id_to_aa,
    )

    e1 = p1.compute_dh_energy(R, mask, species=species)
    e2 = p2.compute_dh_energy(R, mask, species=species)
    assert float(e2) > float(e1)


def test_stickiness_identifiability_reference_is_one():
    free = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    nonref = jnp.array([0, 1, 3], dtype=jnp.int32)
    alpha = _stickiness_alpha_from_free(
        free, nonref_group_indices=nonref, reference_group_idx=2, n_groups=4
    )
    np.testing.assert_allclose(float(alpha[2]), 1.0, rtol=0.0, atol=0.0)
    assert float(alpha[0]) > 0.0
    assert float(alpha[1]) > 0.0
    assert float(alpha[3]) > 0.0


def test_salt_bridge_only_opposite_charges_contribute():
    charge_by_species, _ = _build_charge_and_group_by_species(
        {0: "LYS", 1: "ASP", 2: "ALA"}, his_charge=0.0
    )
    R = jnp.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    mask = jnp.ones((3,), dtype=jnp.float32)
    species = jnp.array([0, 1, 2], dtype=jnp.int32)
    pairs = jnp.array([[0, 1], [0, 2], [1, 2]], dtype=jnp.int32)

    e = compute_salt_bridge_energy(
        R=R,
        mask=mask,
        species=species,
        pairs=pairs,
        charge_by_species=charge_by_species,
        delta_sb=jnp.array(-0.5, dtype=jnp.float32),
        r0_sb=jnp.array(3.8, dtype=jnp.float32),
        sigma_sb=jnp.array(0.3, dtype=jnp.float32),
    )
    assert float(e) < 0.0


def test_mask_safety_with_typed_terms_finite_gradients():
    id_to_aa = {0: "LYS", 1: "ASP", 2: "ALA"}
    prior = _make_prior(
        overrides={
            "model": {
                "priors": {
                    "weights": {
                        "bond": 0.0,
                        "angle": 0.0,
                        "repulsive": 0.0,
                        "dihedral": 0.0,
                        "excluded_volume": 0.0,
                        "dh": 1.0,
                        "stickiness": 1.0,
                        "salt_bridge": 1.0,
                    },
                    "dh": {"enabled": True},
                    "stickiness": {"enabled": True},
                    "salt_bridge": {"enabled": True},
                }
            }
        },
        n_max=6,
        id_to_aa=id_to_aa,
    )

    R = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.2, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    species = jnp.array([0, 1, 2, 0, 0, 0], dtype=jnp.int32)

    def energy_fn(R_):
        return prior.compute_total_energy(R_, mask, species=species)

    grad = jax.grad(energy_fn)(R)
    assert bool(jnp.all(jnp.isfinite(grad)))
    np.testing.assert_allclose(np.asarray(grad[3:]), 0.0, atol=1e-7, rtol=0.0)


def test_disabled_new_terms_regression_matches_old_component_sum():
    prior = _make_prior(n_max=8, id_to_aa=None)
    key = jax.random.PRNGKey(0)
    R = jax.random.normal(key, (8, 3), dtype=jnp.float32) * 2.0
    mask = jnp.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=jnp.float32)
    species = jnp.zeros((8,), dtype=jnp.int32)

    comps = prior.compute_energy(R, mask, species=species)
    old_sum = (
        comps["E_bond"]
        + comps["E_angle"]
        + comps["E_repulsive"]
        + comps["E_dihedral"]
        + comps["E_excluded_volume"]
        + comps["E_wca"]
        + comps["E_fene"]
        + comps["E_leash"]
    )
    np.testing.assert_allclose(np.asarray(comps["E_total"]), np.asarray(old_sum), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(np.asarray(comps["E_dh"]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(comps["E_stickiness"]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(comps["E_salt_bridge"]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(comps["E_fene"]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(comps["E_leash"]), 0.0, rtol=0.0, atol=0.0)


def test_typed_terms_enabled_without_mapping_raises():
    with pytest.raises(ValueError, match="id_to_aa"):
        _make_prior(
            overrides={"model": {"priors": {"dh": {"enabled": True}}}},
            n_max=4,
            id_to_aa=None,
        )
