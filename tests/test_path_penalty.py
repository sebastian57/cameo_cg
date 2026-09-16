"""Analytic tests for training/path_penalty.py."""
import numpy as np
from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()
import jax, jax.numpy as jnp
from training.path_penalty import make_path_penalty, full_graph_edges


def toy_energy(p, R, m, s, neighbor=None):
    d = R[:, None] - R[None]; r2 = jnp.sum(d * d, -1)
    return p["w"] * jnp.sum(r2 * (m[:, None] * m[None])) + p["b"] * jnp.sum(R[:, 0] * m)


def panel(rng, S=7):
    R = rng.standard_normal((S, 6, 3)); pairs = np.array([[0, k] for k in range(1, S)] + [[1, 2]])
    return {"R": R, "mask": np.ones((S, 6)), "species": np.zeros((S, 6), int), "pairs": pairs,
            "target": np.zeros(len(pairs)), "sigma": np.full(len(pairs), 0.5), "train": np.ones(len(pairs), bool)}


def test_perfect_model_gives_zero_and_offset_gives_exact_value():
    rng = np.random.default_rng(0); pn = panel(rng); p = {"w": jnp.array(0.3), "b": jnp.array(0.1)}
    U = np.array([toy_energy(p, jnp.asarray(r), jnp.ones(6), None) for r in pn["R"]])
    pn["target"] = U[pn["pairs"][:, 1]] - U[pn["pairs"][:, 0]]
    assert abs(float(make_path_penalty(energy_of=toy_energy, panel=pn, lam=2.0)(p))) < 1e-10
    pn["target"] = pn["target"] + 0.25                       # every pair off by 0.25 = 0.5 sigma
    assert np.isclose(float(make_path_penalty(energy_of=toy_energy, panel=pn, lam=2.0)(p)), 2.0 * 0.25)


def test_holdout_pairs_are_excluded():
    rng = np.random.default_rng(1); pn = panel(rng); p = {"w": jnp.array(0.3), "b": jnp.array(0.0)}
    U = np.array([toy_energy(p, jnp.asarray(r), jnp.ones(6), None) for r in pn["R"]])
    pn["target"] = U[pn["pairs"][:, 1]] - U[pn["pairs"][:, 0]]; pn["target"][-1] += 100.0; pn["train"][-1] = False
    assert abs(float(make_path_penalty(energy_of=toy_energy, panel=pn, lam=1.0)(p))) < 1e-10


def test_gradient_is_finite_and_descends():
    rng = np.random.default_rng(2); pn = panel(rng); p_true = {"w": jnp.array(0.3), "b": jnp.array(0.2)}
    U = np.array([toy_energy(p_true, jnp.asarray(r), jnp.ones(6), None) for r in pn["R"]])
    pn["target"] = U[pn["pairs"][:, 1]] - U[pn["pairs"][:, 0]]
    pen = make_path_penalty(energy_of=toy_energy, panel=pn, lam=1.0); p = {"w": jnp.array(0.1), "b": jnp.array(0.0)}
    g = jax.grad(pen)(p); gn = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values())))
    assert np.isfinite(gn) and gn > 0
    step = {k: p[k] - 1e-3 / gn * g[k] for k in p}
    assert float(pen(step)) < float(pen(p))


def test_basin_fep_mode_matches_numpy_and_offsets():
    rng = np.random.default_rng(3); S = 30; R = rng.standard_normal((S, 6, 3)); basin = np.repeat([0, 1, 2], 10)
    p_old = {"w": jnp.array(0.3), "b": jnp.array(0.0)}; kT = 0.6
    U0 = np.array([toy_energy(p_old, jnp.asarray(r), jnp.ones(6), None) for r in R])
    pn = {"R": R, "mask": np.ones((S, 6)), "species": np.zeros((S, 6), int), "pairs": np.array([[0, 1], [0, 2]]),
          "target": np.array([0.0, 0.0]), "sigma": np.array([0.5, 0.5]), "train": np.ones(2, bool), "U_old": U0, "basin": basin, "kT": kT}
    pen = make_path_penalty(energy_of=toy_energy, panel=pn, lam=1.0)
    assert abs(float(pen(p_old))) < 1e-10                                   # no change -> zero shift
    p_new = {"w": jnp.array(0.31), "b": jnp.array(0.05)}
    U1 = np.array([toy_energy(p_new, jnp.asarray(r), jnp.ones(6), None) for r in R])
    Sb = [-kT * np.log(np.mean(np.exp(-(U1 - U0)[basin == b] / kT))) for b in range(3)]
    np.testing.assert_allclose(np.asarray(pen.basin_shift(p_new)), Sb, rtol=1e-4, atol=1e-4)
    pn["target"] = np.array([Sb[1] - Sb[0], Sb[2] - Sb[0]])
    assert abs(float(make_path_penalty(energy_of=toy_energy, panel=pn, lam=1.0)(p_new))) < 1e-6


def test_full_graph_edges():
    e = full_graph_edges(6); assert e.shape == (2, 30) and not np.any(e[0] == e[1])
