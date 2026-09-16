"""Point-PMF energy-difference penalty from targeted AA paths (DESIGN/TARGETED_BASIN_PATHS.md).

WHY. Frozen-bead mean-force paths measure AA point-PMF differences A(x_j) - A(x_i) between chosen CG
configurations (basin hubs and model-ensemble frames). A CG energy model is itself a PMF up to a
constant, so U(x_j) - U(x_i) must equal those differences. Force matching constrains only local
gradients (93% of whose residual is random, LESSONS L64); these pairs pin the integrated offsets
between and within basins that set basin free energies (basin FEP: Delta F_AA = Delta F_model
- kT ln<e^{-beta D}>, D = A - U).

WHAT.  mode "pairs": L = lambda * mean_{train pairs} ((U(x_j) - U(x_i) - dA_ij) / sigma_ij)^2,
sigma_ij = sqrt(se_ij^2 + sigma_floor^2) (floor set when the panel is built).
       mode "basin_fep" (panel carries U_old + basin): L = lambda * mean_q ((S_Bq - S_Aq - c_q) / sigma_q)^2 with
S_b = -kT ln <exp(-(U - U_old)/kT)>_{panel frames in b} = exact FEP basin free-energy shift of the model being trained
relative to the model that generated the frames, and c_q the AA-path correction Delta F_AA - Delta F_old (basin FEP of
the path data). Constrains basin OFFSETS only; within-basin shape stays with force matching. Introduced 2026-09-11
because the pointwise mode fit 8 frames/basin below noise and over-shot the basin shift (+0.33 vs +0.16 supported).

HOW. chemtrain `penalty_fn(params)` on a FIXED panel (tens to a few hundred 6-bead structures), each evaluated
on its own with an explicit full intra-structure edge list (static neighbour list, no per-structure
`allocate`, cf. the 4,476-compilation trap in DESIGN/FORCE_BIAS_AND_ENSEMBLE_LOSSES.md). The panel is
tiny, so evaluating it on every device (params are replicated; L70) costs milliseconds.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def path_penalty_enabled(config) -> bool:
    return bool(config.get("training", "path_penalty", "enabled", default=False))


def path_penalty_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "path_penalty", default={}) or {}
    return {"enabled": bool(cfg.get("enabled", False)), "lambda": float(cfg.get("lambda", 1.0)),
            "panel_path": str(cfg.get("panel_path", ""))}


def load_path_panel(path: str) -> Dict[str, np.ndarray]:
    """pairs npz: R (S,n,3), mask (S,n), species (S,n), pairs (P,2) int, target (P,), sigma (P,), train (P,) bool.
    basin_fep npz additionally: U_old (S,), basin (S,) int; pairs then index BASINS (A,B), target = c_AB."""
    z = np.load(path)
    panel = {k: z[k] for k in ("R", "mask", "species", "pairs", "target", "sigma", "train")}
    if "U_old" in z.files:
        panel.update(U_old=z["U_old"], basin=z["basin"], kT=float(z["kT"]))
        if panel["pairs"].max() >= panel["basin"].max() + 1:
            raise ValueError(f"{path}: basin pair index out of range")
        if (panel["sigma"] <= 0).any():
            raise ValueError(f"{path}: sigma must be positive")
        return panel
    S = len(panel["R"])
    if panel["pairs"].min() < 0 or panel["pairs"].max() >= S or not panel["train"].any():
        raise ValueError(f"{path}: bad pair indices or no training pairs")
    if (panel["sigma"] <= 0).any():
        raise ValueError(f"{path}: sigma must be positive")
    return panel


def full_graph_edges(n: int) -> np.ndarray:
    """Directed edges i != j of one n-bead structure, JAX-MD Sparse (2, E) layout."""
    rec, sen = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    return np.array([rec, sen], np.int32)


def make_path_penalty(*, energy_of, panel: Dict[str, np.ndarray], lam: float, neighbors=None):
    """penalty_fn(params) -> lam * mean_train(((U_j - U_i - target) / sigma)^2)."""
    import jax
    import jax.numpy as jnp

    R, m, s = (jnp.asarray(panel[k]) for k in ("R", "mask", "species"))
    tr = np.asarray(panel["train"], bool)
    i, j = (jnp.asarray(panel["pairs"][tr, c]) for c in (0, 1))
    t, sg = jnp.asarray(panel["target"][tr]), jnp.asarray(panel["sigma"][tr])
    nb = None if neighbors is None else jax.tree.map(jnp.asarray, neighbors)

    def energies(params):
        if nb is None:
            return jax.vmap(lambda r, mm, ss: energy_of(params, r, mm, ss))(R, m, s)
        return jax.vmap(lambda r, mm, ss, n_: energy_of(params, r, mm, ss, neighbor=n_))(R, m, s, nb)

    if "U_old" in panel:
        U0 = jnp.asarray(panel["U_old"]); kT = float(panel["kT"]); nb_ = int(panel["basin"].max()) + 1
        memb = jnp.asarray(panel["basin"][None, :] == np.arange(nb_)[:, None])            # (B, S)
        lnN = jnp.log(memb.sum(1).astype(jnp.float32))

        def basin_shift(params):
            d = (energies(params) - U0) / kT
            z = jnp.where(memb, -d[None, :], -jnp.inf)
            return -kT * (jax.nn.logsumexp(z, axis=1) - lnN)                                 # (B,)

        def penalty(params):
            S = basin_shift(params)
            return float(lam) * jnp.mean(((S[j] - S[i] - t) / sg) ** 2)

        penalty.energies = energies; penalty.basin_shift = basin_shift
        return penalty

    def penalty(params):
        U = energies(params)
        return float(lam) * jnp.mean(((U[j] - U[i] - t) / sg) ** 2)

    penalty.energies = energies          # exposed for offline diagnostics
    return penalty
