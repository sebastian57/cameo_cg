"""Finite-difference check for end-to-end torque differentiability.

Settles whether the full orientation chain
    ω → O_pert = (I + skew(ω)) @ O  → gather to edges → AllegroEmbedding → E
is differentiable end-to-end, or whether a hidden stop_gradient / non-diff step
breaks it.

Method
------
For each (bead i, axis k), pick ω_i = ε * e_k (all other beads ω=0), compute

    FD_ik  = -(E(O_pert) - E(O)) / ε        (two forward passes)
    JAX_ik = -∂E/∂ω_i|_{ω=0} via jax.grad   (one autodiff pass)

If the full graph is differentiable, FD ≈ JAX to O(ε) + float32 noise.
If they disagree, the gather or something upstream is breaking the chain.

Usage
-----
With a trained export (recommended — non-trivial torques make the check more
sensitive):

    python analysis_tests/check_torque_fd.py /path/to/exports

With randomly-initialised weights (sanity baseline; torques may be ~zero):

    python analysis_tests/check_torque_fd.py --random

Options
-------
--eps FLOAT    FD step size (default 1e-3; ~1e-2 is safer for float32)
--atol FLOAT   Pass/fail threshold on max|FD - JAX| (default 5e-2)
--random       Use randomly initialised params instead of a trained export
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jax_setup import apply_jax_compat_shims
apply_jax_compat_shims()

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skew_perturb(O, omega):
    """Apply first-order rotation: O_pert[n] = (I + skew(omega[n])) @ O[n]."""
    delta = jnp.cross(omega[:, None, :], O.transpose(0, 2, 1)).transpose(0, 2, 1)
    return O + delta


def compute_analytic_torques(model, params, R, O, mask, species, box):
    """τ_i = -∂E/∂ω_i|_{ω=0}  via jax.grad through the full graph."""
    def energy_of_rotation(omega):
        O_pert = _skew_perturb(O, omega)
        return model.compute_energy(params, R, mask, species, box=box, O=O_pert)

    omega_zero = jnp.zeros((O.shape[0], 3), dtype=O.dtype)
    return -jax.grad(energy_of_rotation)(omega_zero)


def compute_dEdO_torques(model, params, R, O, mask, species, box):
    """Compute dE/dO directly, then project onto rotation tangent directions.

    This separates the model's orientation gradient (dE/dO) from the chain
    rule through _skew_perturb.  If this matches FD but compute_analytic_torques
    does not, the bug is in how _skew_perturb is traced.  If this also
    underestimates, the bug is inside model.compute_energy's gradient.

    τ_ik = -∑_{r,c} (dE/dO[i,r,c]) * (skew(e_k) @ O[i,:,c])[r]
         = -∑_{r,c} (dE/dO[i,r,c]) * (e_k × O[i,:,c])[r]
    """
    dE_dO = jax.grad(
        lambda O_: model.compute_energy(params, R, mask, species, box=box, O=O_)
    )(O)  # shape (N, 3, 3)

    N = O.shape[0]
    torques = np.zeros((N, 3), dtype=np.float32)
    O_np = np.asarray(O)
    dE_dO_np = np.asarray(dE_dO)
    for i in range(N):
        for k in range(3):
            e_k = np.zeros(3, dtype=np.float32)
            e_k[k] = 1.0
            # skew(e_k) @ O[i, :, c] for each column c
            for c in range(3):
                col = O_np[i, :, c]           # (3,)
                tangent = np.cross(e_k, col)   # (3,) = (skew(e_k) @ col)
                torques[i, k] -= np.dot(dE_dO_np[i, :, c], tangent)
    return torques


def compute_fd_torques(model, params, R, O, mask, species, box, eps):
    """τ_i ≈ -(E(O_pert_ik) - E0) / ε  for each bead i, axis k."""
    E0 = float(model.compute_energy(params, R, mask, species, box=box, O=O))
    N = O.shape[0]
    fd = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        if float(mask[i]) == 0:
            continue
        for k in range(3):
            omega_ik = np.zeros((N, 3), dtype=np.float32)
            omega_ik[i, k] = eps
            O_pert = _skew_perturb(O, jnp.asarray(omega_ik))
            E_pert = float(model.compute_energy(params, R, mask, species, box=box, O=O_pert))
            fd[i, k] = -(E_pert - E0) / eps
    return fd, E0


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def run_check(model, params, R, O, mask, species, box, eps, atol):
    R, O = jnp.asarray(R, dtype=jnp.float32), jnp.asarray(O, dtype=jnp.float32)
    mask     = jnp.asarray(mask,    dtype=jnp.float32)
    species  = jnp.asarray(species, dtype=jnp.int32)
    box      = jnp.asarray(box,     dtype=jnp.float32)

    print(f"  ε = {eps:.2e}   atol = {atol:.2e}")

    print("  Running analytic (jax.grad via omega) pass …", flush=True)
    analytic = np.asarray(compute_analytic_torques(model, params, R, O, mask, species, box))

    print("  Running dE/dO projected pass …", flush=True)
    dedo = compute_dEdO_torques(model, params, R, O, mask, species, box)

    print(f"  Running FD passes ({int(mask.sum()) * 3} total) …", flush=True)
    fd, E0 = compute_fd_torques(model, params, R, O, mask, species, box, eps)

    print(f"\n  E0 = {E0:.6g}")
    print(f"\n  {'Bead':>4}  {'Ax':>2}  {'FD':>12}  {'via-ω (JAX)':>14}  {'dE/dO·J':>12}  {'FD-ω':>10}  {'FD-dO':>10}")
    print("  " + "-" * 75)
    max_err_omega = 0.0
    max_err_dedo  = 0.0
    for i in range(analytic.shape[0]):
        if float(mask[i]) == 0:
            continue
        for k, ax in enumerate("xyz"):
            diff_omega = abs(float(fd[i, k]) - float(analytic[i, k]))
            diff_dedo  = abs(float(fd[i, k]) - float(dedo[i, k]))
            max_err_omega = max(max_err_omega, diff_omega)
            max_err_dedo  = max(max_err_dedo,  diff_dedo)
            print(f"  {i:>4}   {ax}  {fd[i,k]:>12.5f}  {analytic[i,k]:>14.5f}  {dedo[i,k]:>12.5f}  {diff_omega:>10.2e}  {diff_dedo:>10.2e}")

    print(f"\n  max |FD − via-ω|  = {max_err_omega:.3e}")
    print(f"  max |FD − dE/dO·J| = {max_err_dedo:.3e}   atol = {atol:.3e}")
    print()
    if max_err_dedo < atol:
        print("  dE/dO path is CORRECT — bug is in how jax.grad traces through _skew_perturb or energy_of_rotation")
    elif max_err_omega < atol:
        print("  via-ω path matches FD — dE/dO projection has a bug (chain rule error in compute_dEdO_torques)")
    else:
        print("  Both paths underestimate — bug is inside model.compute_energy gradient (blocked path in the network)")
    ok = max_err_omega < atol
    print("  " + ("PASS ✓" if ok else "FAIL ✗"))
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FD torque check")
    p.add_argument("export_dir", nargs="?", default=None,
                   help="Path to exports/ folder with *_config.yaml + *_params.pkl")
    p.add_argument("--random", action="store_true",
                   help="Use randomly initialised weights instead of a trained export")
    p.add_argument("--eps",  type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=5e-2)
    args = p.parse_args()

    if args.export_dir is None and not args.random:
        p.error("Provide export_dir or pass --random")

    print("=== Torque finite-difference check ===\n")

    if args.random:
        # Minimal model build with random params — tests the math, not specific weights.
        from analysis_tests.infer_torques import BOX, DEFAULT_MASK, DEFAULT_SPECIES
        from config.manager import ConfigManager
        from models.combined_model import CombinedModel

        # Construct a minimal config file in memory
        import tempfile, yaml, os
        cfg_dict = {
            "data": {},
            "model": {"ml_model": "allegro_cueq_fast", "use_priors": False},
            "training": {},
            "optimizer": {},
            "export": {"enabled": False},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg_dict, f)
            cfg_path = f.name

        R0 = np.array([[22., 22., 22.], [22., 22., 25.5]], dtype=np.float32)
        O0 = np.stack([np.eye(3, dtype=np.float32)] * 2)
        config = ConfigManager(cfg_path)
        os.unlink(cfg_path)

        model = CombinedModel(config=config, R0=R0, O0=O0, box=BOX,
                              species=DEFAULT_SPECIES, N_max=2)
        params = model.initialize_params(jax.random.PRNGKey(0))
        R = R0
        O = O0
        mask, species, box = DEFAULT_MASK, DEFAULT_SPECIES, BOX
        print("  Using randomly initialised weights.\n")

    else:
        from analysis_tests.infer_torques import (
            load_model, preprocess_positions,
            BOX, DEFAULT_MASK, DEFAULT_SPECIES,
        )
        model, params = load_model(args.export_dir)
        R_raw = np.array([[0., 0., 0.], [0., 0., 3.5]], dtype=np.float32)
        R = preprocess_positions(R_raw)
        O = np.stack([np.eye(3, dtype=np.float32)] * 2)
        # Tilt bead 0 slightly so torques are non-trivial
        theta = 0.3
        c, s = np.cos(theta), np.sin(theta)
        O[0] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        mask, species, box = DEFAULT_MASK, DEFAULT_SPECIES, BOX
        print(f"  Loaded trained params from {args.export_dir}\n")

    ok = run_check(model, params, R, O, mask, species, box,
                   eps=args.eps, atol=args.atol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
