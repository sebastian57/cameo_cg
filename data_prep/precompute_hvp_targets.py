#!/usr/bin/env python3
"""
Precompute HVP (Hessian-vector product) targets for CG force-field training.

Generates per-frame CG HVP arrays via central finite differences on the AA
force field:
  hvp_probe  (frames, K, N_cg, 3)  float32  -- unit-norm CG probe vectors
  HVP        (frames, K, N_cg, 3)  float32  -- projected AA HVP targets

Algorithm per frame f, per probe k:
  1. Generate deterministic unit probe v_k in CG space.
  2. Lift to AA:   u_aa = Xi_F^T  @ v_k   (shape N_aa x 3)
  3. Central FD:   H_aa u ≈ -(F_aa(r+εu) - F_aa(r-εu)) / (2ε)
  4. Project:      HVP_cg[k] = Xi_F @ H_aa_u   (shape N_cg x 3)

Mapping Xi_F (N_cg x N_aa) is recovered from the paired (F_aa, F_cg) data
via least-squares.  For the Charron ALA2 dataset it is an exact integer
{0,1} matrix (each CG bead = one heavy atom + its bonded H's).

Force evaluation: OpenMM AMBER14, vacuum, NoCutoff.

Usage:
  python data_prep/precompute_hvp_targets.py \\
      --aa-npz data_prep/datasets/ala2_fullMD_data.npz \\
      --cg-npz local_work/support_gate_force_matching/input_data/ala2_cg_data_paper.npz \\
      --out    local_work/runs_0306/input_data/ala2_cg_data_paper_hvp_targets.npz \\
      --num-probes 4 --epsilon 1e-3 --seed 42

Validation (first 10 frames, compare OpenMM to stored AA forces):
  python data_prep/precompute_hvp_targets.py ... --validate-only

"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("HVPPrecompute")
logger.propagate = False
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(_h)
logger.setLevel(logging.INFO)

KJ_NM_TO_KCAL_ANG = 1.0 / 41.84  # kJ/mol/nm → kcal/mol/Å


# ── Mapping ───────────────────────────────────────────────────────────────────

def fit_xi_f(
    F_aa: np.ndarray,  # (T, N_aa, 3) float32
    F_cg: np.ndarray,  # (T, N_cg, 3) float32
) -> np.ndarray:
    """
    Recover Xi_F (N_cg, N_aa) by least-squares on one Cartesian component.

    Xi_F is block-diagonal w.r.t. x/y/z (isotropy verified on dataset: residual <1e-6).
    """
    A = F_aa[:, :, 0].astype(np.float64)  # (T, N_aa)
    b = F_cg[:, :, 0].astype(np.float64)  # (T, N_cg)
    xi_T, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return xi_T.T  # (N_cg, N_aa)


# ── OpenMM context ────────────────────────────────────────────────────────────

def build_context(gro_ref: str, ff_name: str):
    """Build and return an OpenMM vacuum Context for ALA2."""
    import openmm as mm
    import openmm.app as app
    import mdtraj

    traj = mdtraj.load(gro_ref)
    tmp = "/tmp/_hvp_ref.pdb"
    traj.save(tmp)

    pdb = app.PDBFile(tmp)
    ff = app.ForceField(ff_name)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integrator)
    n = system.getNumParticles()
    logger.info(f"OpenMM: {n} atoms, FF={ff_name}, NoCutoff vacuum")
    return ctx


def eval_forces(ctx, r_ang: np.ndarray) -> np.ndarray:
    """
    Evaluate AA forces.

    Args:
        r_ang: (N_aa, 3) float64, Å
    Returns:
        (N_aa, 3) float64, kcal/mol/Å
    """
    import openmm.unit as unit

    ctx.setPositions((r_ang * 0.1) * unit.nanometer)
    state = ctx.getState(getForces=True)
    f_kj_nm = np.array(state.getForces(asNumpy=True), dtype=np.float64)
    return f_kj_nm * KJ_NM_TO_KCAL_ANG


# ── Probe generation ──────────────────────────────────────────────────────────

def make_probes(seed: int, frame_idx: int, N_cg: int, K: int) -> np.ndarray:
    """
    Generate K deterministic unit-norm probe vectors.

    Seed is derived from (global_seed, frame_idx) for reproducibility
    independent of processing order.

    Returns: (K, N_cg, 3) float32
    """
    rng = np.random.default_rng([seed, frame_idx])
    v = rng.standard_normal((K, N_cg, 3)).astype(np.float64)
    # Remove COM translation per probe so probes are pure deformation modes.
    v -= v.mean(axis=1, keepdims=True)
    # Normalise each probe to unit Frobenius norm.
    norms = np.linalg.norm(v.reshape(K, -1), axis=1)[:, None, None]
    v /= np.where(norms > 0, norms, 1.0)
    return v.astype(np.float32)


# ── HVP computation ───────────────────────────────────────────────────────────

def compute_hvp(
    r_aa: np.ndarray,       # (N_aa, 3) Å float64
    probes: np.ndarray,     # (K, N_cg, 3) float32
    xi_f: np.ndarray,       # (N_cg, N_aa) float64
    ctx,
    epsilon: float,
) -> np.ndarray:
    """
    Compute projected HVP for all K probes at one frame.

    Returns: (K, N_cg, 3) float32
    """
    K = probes.shape[0]
    N_cg, N_aa = xi_f.shape
    xi_f_T = xi_f.T  # (N_aa, N_cg)

    hvp_out = np.empty((K, N_cg, 3), dtype=np.float64)

    for k in range(K):
        v_k = probes[k].astype(np.float64)   # (N_cg, 3)

        # Lift probe to AA space: u_aa = Xi_F^T @ v_k
        u_aa = xi_f_T @ v_k                  # (N_aa, 3)

        # Central finite difference
        f_p = eval_forces(ctx, r_aa + epsilon * u_aa)   # (N_aa, 3)
        f_m = eval_forces(ctx, r_aa - epsilon * u_aa)   # (N_aa, 3)

        # H_aa u ≈ -(f_plus - f_minus) / (2ε)
        hvp_aa = -(f_p - f_m) / (2.0 * epsilon)          # (N_aa, 3)

        # Project to CG: HVP_cg = Xi_F @ hvp_aa
        hvp_out[k] = xi_f @ hvp_aa                        # (N_cg, 3)

    return hvp_out.astype(np.float32)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ff(ctx, R_aa: np.ndarray, F_aa_ref: np.ndarray, n_frames: int = 10):
    """
    Compare OpenMM forces to stored AA forces on the first n_frames.
    Logs per-frame RMSE and overall Pearson r.
    """
    logger.info(f"Validating FF on {n_frames} frames ...")
    all_pred, all_ref = [], []
    for f in range(min(n_frames, R_aa.shape[0])):
        f_pred = eval_forces(ctx, R_aa[f].astype(np.float64))
        f_ref = F_aa_ref[f].astype(np.float64)
        rmse = np.sqrt(np.mean((f_pred - f_ref) ** 2))
        logger.info(f"  frame {f}: RMSE={rmse:.3f} kcal/mol/Å")
        all_pred.append(f_pred.ravel())
        all_ref.append(f_ref.ravel())
    if not all_pred:
        return float("nan")
    r = np.corrcoef(np.concatenate(all_pred), np.concatenate(all_ref))[0, 1]
    logger.info(f"  Overall Pearson r = {r:.4f}")
    return r


# ── Main ──────────────────────────────────────────────────────────────────────

_DEFAULT_GRO = (
    "/e/project1/cameo/schmidt36/relative-entropy"
    "/examples/alanine_dipeptide/data/confs/heavy_2_7nm.gro"
)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute HVP targets for CG training."
    )
    parser.add_argument("--aa-npz", required=True,
                        help="AA trajectory NPZ (R, F, species)")
    parser.add_argument("--cg-npz", required=True,
                        help="CG NPZ to copy into output (must be frame-aligned with AA)")
    parser.add_argument("--out", required=True,
                        help="Output NPZ path")
    parser.add_argument("--gro-ref", default=_DEFAULT_GRO,
                        help="GRO reference structure for OpenMM topology")
    parser.add_argument("--force-field", default="amber14-all.xml",
                        help="OpenMM ForceField XML name")
    parser.add_argument("--num-probes", type=int, default=4,
                        help="K probe vectors per frame")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="Finite-difference step size in Å")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames processed")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Progress log interval (frames)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate FF agreement, do not compute HVP")
    parser.add_argument("--n-validate", type=int, default=20,
                        help="Frames to use for --validate-only")
    args = parser.parse_args()

    # ── Load datasets ──────────────────────────────────────────────────────
    logger.info(f"AA NPZ:  {args.aa_npz}")
    aa = np.load(args.aa_npz)
    R_aa = aa["R"]  # (T, N_aa, 3)
    F_aa = aa["F"]  # (T, N_aa, 3)

    logger.info(f"CG NPZ:  {args.cg_npz}")
    cg = np.load(args.cg_npz)
    F_cg = cg["F"]  # (T, N_cg, 3)

    T_aa, N_aa = R_aa.shape[:2]
    T_cg, N_cg = F_cg.shape[:2]
    if T_aa != T_cg:
        raise ValueError(f"Frame count mismatch: AA={T_aa}, CG={T_cg}")

    T = min(T_aa, args.max_frames) if args.max_frames else T_aa
    K = args.num_probes
    logger.info(
        f"Frames={T}  AA atoms={N_aa}  CG beads={N_cg}  K={K}  ε={args.epsilon} Å"
    )

    # ── Mapping ────────────────────────────────────────────────────────────
    logger.info("Fitting Xi_F from (F_aa, F_cg) ...")
    xi_f = fit_xi_f(F_aa, F_cg).astype(np.float64)  # (N_cg, N_aa)
    thresh = 0.1
    for i in range(N_cg):
        atoms = np.where(np.abs(xi_f[i]) > thresh)[0]
        weights = xi_f[i, atoms]
        logger.info(
            f"  bead {i}: AA atoms {atoms.tolist()} "
            f"weights {np.round(weights, 3).tolist()}"
        )
    covered = set(int(j) for i in range(N_cg) for j in np.where(np.abs(xi_f[i]) > thresh)[0])
    uncovered = sorted(set(range(N_aa)) - covered)
    logger.info(f"  {len(covered)}/{N_aa} AA atoms covered; uncovered: {uncovered}")

    # ── OpenMM context ─────────────────────────────────────────────────────
    ctx = build_context(args.gro_ref, args.force_field)

    # ── Validation ──────────────────────────────────────────────────────────
    r_val = validate_ff(ctx, R_aa, F_aa, n_frames=args.n_validate)
    if r_val < 0.95:
        logger.warning(
            f"FF validation: Pearson r={r_val:.4f} < 0.95 — "
            "force field mismatch may bias HVP targets"
        )
    if args.validate_only:
        return

    # ── HVP computation ─────────────────────────────────────────────────────
    hvp_probe_out = np.empty((T, K, N_cg, 3), dtype=np.float32)
    hvp_out = np.empty((T, K, N_cg, 3), dtype=np.float32)

    t0 = time.time()
    for f in range(T):
        probes = make_probes(args.seed, f, N_cg, K)
        r_f = R_aa[f].astype(np.float64)

        hvp_probe_out[f] = probes
        hvp_out[f] = compute_hvp(r_f, probes, xi_f, ctx, args.epsilon)

        if (f + 1) % args.log_every == 0 or f == T - 1:
            elapsed = time.time() - t0
            rate = (f + 1) / elapsed
            eta = (T - f - 1) / rate if rate > 0 else 0.0
            logger.info(
                f"  frame {f+1:>7}/{T}  {rate:.0f} fr/s  ETA {eta:.0f} s"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Done: {T} frames × {K} probes × 2 evals = {T*K*2} FF calls "
        f"in {elapsed:.1f} s  ({T*K*2/elapsed:.0f} calls/s)"
    )

    # ── Sanity check ──────────────────────────────────────────────────────
    n_nan = int(np.isnan(hvp_out).sum())
    if n_nan:
        logger.warning(f"{n_nan} NaN entries in HVP output!")
    hvp_rms = float(np.sqrt(np.mean(hvp_out ** 2)))
    f_rms = float(np.sqrt(np.mean(F_cg[:T] ** 2)))
    logger.info(
        f"HVP RMS = {hvp_rms:.3f}  CG force RMS = {f_rms:.3f}  "
        f"(ratio {hvp_rms/f_rms:.2f})"
    )

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save = {}
    for k, v in cg.items():
        # Truncate CG arrays if we processed fewer frames than the full dataset.
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T_cg:
            save[k] = v[:T]
        else:
            save[k] = v
    save["hvp_probe"] = hvp_probe_out
    save["HVP"] = hvp_out
    # Metadata (scalar 0-d arrays; ignored by the data loader but useful for auditing).
    save["hvp_epsilon"] = np.float32(args.epsilon)
    save["hvp_seed"] = np.int64(args.seed)
    save["hvp_num_probes"] = np.int64(K)
    save["hvp_force_field"] = np.bytes_(args.force_field)

    np.savez(out_path, **save)
    logger.info(f"Saved: {out_path}")
    logger.info(f"  Keys: {sorted(save.keys())}")
    logger.info(f"  hvp_probe: {hvp_probe_out.shape}  HVP: {hvp_out.shape}")


if __name__ == "__main__":
    main()
