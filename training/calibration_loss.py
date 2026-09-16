"""Basin free-energy calibration loss (L_cal, v5 pre-registration).

Supervises relative basin depths — the quantity force matching cannot pin.
Pre-registered in KNOWLEDGE_BASE/P_cameo_cg/DESIGN/V5_DATASET_DESIGN.md:

    L_cal = sum_pairs [dF_model - dF_ref]^2
    dF_ref   = -kT ln(P_A/P_B) from reference populations
    dF_model = kT [(LSE(U_A/kT) - ln N_A) - (LSE(U_B/kT) - ln N_B)]
               over TRAINING-split basin frames (restricted-ensemble free
               energies via E_A[exp(U/kT)] = V_A/Z_A; volume effects are
               absorbed into the pair-target convention on both sides)

Integration follows the supported chemtrain extension points only: the loss is a
quantity ("CAL", replicated per sample) + identity error fn + gamma weight, so it
rides the standard ForceMatching loss with gradients intact. The panel is FIXED
(balanced frames per basin from a dedicated dataset npz), evaluated with the same
direct per-frame energy path as training.basin_energy_monitor (no batch neighbor
lists; requires model.compute_energy to be callable frame-wise).

Assumptions/limits (see DESIGN/CALIBRATION_LOSS.md):
- dataset_path must be an equilibrium-representative, TRAINING-split export;
  dF_ref is estimated from its assigned-frame counts.
- meaningless for direct_force models (no scalar energy) — refused at init.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.scipy.special import logsumexp
from typing import Any, Callable, Tuple

import numpy as np

KT_KCAL_MOL = 0.5921868690749673  # kT at 298 K in kcal/mol (project constant)

REQUIRED_BASINS = ("beta", "alphaR", "alphaL")
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("beta", "alphaR"),
    ("beta", "alphaL"),
    ("alphaR", "alphaL"),
)
_BASIN_IDX = {b: i for i, b in enumerate(REQUIRED_BASINS)}
PAIR_IDS: Tuple[Tuple[int, int], ...] = tuple(
    (_BASIN_IDX[a], _BASIN_IDX[b]) for a, b in PAIRS
)


def calibration_enabled(config: Any) -> bool:
    return bool(config.get("training", "calibration", "enabled", default=False))


def calibration_config(config: Any) -> dict:
    cfg = config.get("training", "calibration", default={}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("training.calibration must be a mapping")
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lambda": float(cfg.get("lambda", 0.1)),
        "dataset_path": str(cfg.get("dataset_path", "")),
        "frames_per_basin": int(cfg.get("frames_per_basin", 256)),
        "seed": int(cfg.get("seed", 0)),
        "kT": float(cfg.get("kT", KT_KCAL_MOL)),
        "mapping": str(cfg.get("mapping", "ala2_backbone_cb_6")),
    }


@dataclass(frozen=True)
class CalibrationPanel:
    R: np.ndarray                 # (N, beads, 3)
    mask: np.ndarray              # (N, beads)
    species: np.ndarray           # (N, beads)
    labels: np.ndarray            # (N,) basin names, REQUIRED_BASINS only
    basin_ids: np.ndarray         # (N,) ints indexing REQUIRED_BASINS
    populations: Tuple[float, ...]
    dF_ref: np.ndarray            # (len(PAIRS),) kcal/mol


def build_calibration_panel(
    dataset_path: str,
    *,
    frames_per_basin: int,
    seed: int,
    mapping_name: str = "ala2_backbone_cb_6",
) -> CalibrationPanel:
    """Load a dataset npz, assign basins, estimate P, balance the panel.

    The npz must carry R/mask/species ((frames, beads, 3)/(frames, beads)).
    Basin assignment uses phi/psi keys when present, else evaluates the named
    mapping's CVS on R. Populations come from ALL assigned frames (before
    balancing), so the file must be equilibrium-representative and contain only
    training-split rows.
    """
    with np.load(dataset_path, allow_pickle=False) as loaded:
        R = np.asarray(loaded["R"], dtype=np.float32)
        mask = np.asarray(loaded["mask"], dtype=np.float32)
        species = np.asarray(loaded["species"], dtype=np.int32)
        if "phi" in loaded and "psi" in loaded:
            phi = np.asarray(loaded["phi"], dtype=np.float64)
            psi = np.asarray(loaded["psi"], dtype=np.float64)
        else:
            from sampling.mapping import get_mapping

            mapping = get_mapping(mapping_name)
            phi = mapping.cvs["phi"].evaluate(R)
            psi = mapping.cvs["psi"].evaluate(R)

    from .basin_energy_monitor import assign_ala2_basins

    labels_all = assign_ala2_basins(phi, psi)
    if not np.isin(labels_all, REQUIRED_BASINS).any():
        raise ValueError(f"no assignable basins in {dataset_path}")

    counts = {b: int((labels_all == b).sum()) for b in REQUIRED_BASINS}
    total = sum(counts.values())
    populations = tuple(counts[b] / total for b in REQUIRED_BASINS)
    dF_ref = np.array([
        -KT_KCAL_MOL * np.log(
            populations[_BASIN_IDX[a]] / populations[_BASIN_IDX[b]]
        )
        for a, b in PAIRS
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    keep = []
    for b in REQUIRED_BASINS:
        idx_b = np.flatnonzero(labels_all == b)
        if len(idx_b) < frames_per_basin:
            raise ValueError(
                f"{dataset_path}: only {len(idx_b)} '{b}' frames available "
                f"(wanted {frames_per_basin})"
            )
        keep.append(rng.choice(idx_b, size=frames_per_basin, replace=False))
    sel = np.sort(np.concatenate(keep))

    lab_sel = labels_all[sel]
    return CalibrationPanel(
        R=R[sel],
        mask=mask[sel],
        species=species[sel],
        labels=lab_sel,
        basin_ids=np.array([_BASIN_IDX[b] for b in lab_sel], dtype=np.int32),
        populations=populations,
        dF_ref=dF_ref,
    )


def _restricted_free_energies(U, basin_ids, *, kT: float):
    """Per-basin restricted F estimates: kT (LSE(U/kT) - ln N_b)."""
    ids = jnp.asarray(basin_ids)
    n_bas = len(REQUIRED_BASINS)
    onehot = jnp.equal(ids[:, None], jnp.arange(n_bas)[None, :])
    masked = jnp.where(onehot, U[:, None] / kT, -jnp.inf)
    lse = logsumexp(masked, axis=0)
    return kT * (lse - jnp.log(onehot.sum(axis=0)))


def lse_free_energy_pair_diffs(U, basin_ids, *, pair_ids=PAIR_IDS, kT: float):
    """dF_model(A, B) = F_A - F_B per pair, kcal/mol."""
    f_res = _restricted_free_energies(jnp.asarray(U), basin_ids, kT=kT)
    return jnp.stack([f_res[a] - f_res[b] for a, b in pair_ids])


def calibration_quantity_value(*, U, basin_ids, kT: float, populations):
    """Scalar L_cal from per-frame energies and reference populations."""
    diffs = lse_free_energy_pair_diffs(U, basin_ids, kT=kT)
    pops = jnp.asarray(populations, dtype=U.dtype)
    ref = jnp.stack([
        -kT * jnp.log(pops[a] / pops[b]) for a, b in PAIR_IDS
    ])
    return jnp.sum(jnp.square(diffs - ref))


def make_calibration_penalty(
    *,
    energy_of: Callable[[Any, Any, Any, Any], Any],
    R,
    mask,
    species,
    basin_ids,
    kT: float,
    populations,
    lam: float,
    neighbors=None,
):
    """Return ``penalty_fn(energy_params) -> lam * L_cal`` on a fixed panel.

    ``neighbors``: optional stacked per-frame neighbor-list pytree (leaf axes
    lead with n_frames). Needed when ``energy_of`` closes over a model whose
    internal buffers were sized for a larger batch (e.g. tiled training):
    without it, per-frame calls rebuild neighbor lists against mismatched
    training-capacity buffers and vmap fails. When given, ``energy_of`` must
    accept ``neighbor=``.

    This is the LIVE integration path (ForceMatching ``penalty_fn`` hook): L_cal
    depends only on parameters, never on the batch. The quantity/error-fn route
    below CANNOT work — chemtrain's ``_split_targets_inputs`` asserts every
    additional target key appears as an observation column, and no per-frame
    "CAL" column exists.
    """
    R_j = jnp.asarray(R)
    mask_j = jnp.asarray(mask)
    species_j = jnp.asarray(species)
    ids_j = jnp.asarray(basin_ids)
    nbrs_j = None if neighbors is None else jax.tree.map(jnp.asarray, neighbors)

    def calibration_penalty(energy_params):
        if nbrs_j is None:
            U = jax.vmap(lambda r, m, s: energy_of(energy_params, r, m, s))(
                R_j, mask_j, species_j
            )
        else:
            U = jax.vmap(
                lambda r, m, s, nb: energy_of(energy_params, r, m, s, neighbor=nb),
                in_axes=(0, 0, 0, 0),
            )(R_j, mask_j, species_j, nbrs_j)
        return lam * calibration_quantity_value(
            U=U, basin_ids=ids_j, kT=kT, populations=populations
        )

    return calibration_penalty


def make_calibration_quantity(
    *,
    energy_of: Callable[[Any, Any, Any, Any], Any],
    R,
    mask,
    species,
    basin_ids,
    kT: float,
    populations,
):
    """Return a chemtrain quantity computing replicated L_cal on a fixed panel.

    ``energy_of(params, R, mask, species)`` maps ONE frame to a scalar energy
    (e.g. ``model.compute_energy``). The returned callable ignores the batch
    state and evaluates the embedded panel instead; gradients flow to
    ``energy_params`` exactly like any other quantity.
    """
    R_j = jnp.asarray(R)
    mask_j = jnp.asarray(mask)
    species_j = jnp.asarray(species)
    ids_j = jnp.asarray(basin_ids)

    def calibration_quantity(state=None, energy_params=None, **kwargs):
        del kwargs
        if energy_params is None:
            raise ValueError("calibration quantity requires energy_params")
        U = jax.vmap(lambda r, m, s: energy_of(energy_params, r, m, s))(
            R_j, mask_j, species_j
        )
        err = calibration_quantity_value(
            U=U, basin_ids=ids_j, kT=kT, populations=populations
        )
        batch = state.position.shape[0] if state is not None else 1
        return jnp.full((batch,), err, dtype=U.dtype)

    return calibration_quantity


def calibration_error(predictions, targets, weights=None):
    """Identity reduction: the quantity already carries the full L_cal value."""
    del targets, weights
    return jnp.mean(predictions)
