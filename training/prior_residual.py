"""
Prior-force residual target utilities.

This module supports an optional training mode where prior forces are
precomputed on untiled data and force targets are transformed to residual
targets:

    F_residual = F_ref - F_prior
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from models.prior_energy import PriorEnergy
from models.topology import TopologyBuilder
from utils.logging import training_logger


def _stable_json(payload: Dict[str, Any]) -> str:
    """Serialize metadata deterministically for cache matching."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _params_signature(params: Optional[Dict[str, Any]]) -> str:
    """Compact SHA-256 prefix of prior parameter values for cache invalidation."""
    if params is None:
        return "none"
    pieces = []
    for k in sorted(params.keys()):
        v = np.asarray(params[k], dtype=np.float64)
        pieces.append(f"{k}:{v.tobytes().hex()}")
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()[:16]


def _resolve_cache_path(
    config,
    dataset_tag: str,
    project_root: Path,
) -> Path:
    """
    Resolve cache path for prior-force residual data.

    If training.prior_residual.cache_path is not set, defaults to:
        <checkpoint_path>/prior_residual_cache/<dataset_tag>_prior_residual.npz
    """
    configured = config.get_prior_residual_cache_path()
    if configured is None:
        checkpoint_path = Path(config.get_checkpoint_path())
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        return checkpoint_path / "prior_residual_cache" / f"{dataset_tag}_prior_residual.npz"

    path = Path(configured)
    if not path.is_absolute():
        path = project_root / path

    # Treat non-.npz path as a directory target.
    if path.suffix.lower() != ".npz":
        return path / f"{dataset_tag}_prior_residual.npz"
    return path


def _build_cache_metadata(
    config,
    dataset: Dict[str, np.ndarray],
    dataset_path: Union[str, Path],
    dataset_tag: str,
    seed: int,
    max_frames: Optional[int],
    cutoff: float,
    buffer_multiplier: float,
    park_multiplier: float,
    fitted_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build metadata fingerprint for cache validity checks."""
    prior_cfg = config.get("model", "priors", default={})
    return {
        "version": 1,
        "dataset_path": str(Path(dataset_path)),
        "dataset_tag": str(dataset_tag),
        "n_frames": int(dataset["R"].shape[0]),
        "n_atoms": int(dataset["R"].shape[1]),
        "seed": int(seed),
        "max_frames": None if max_frames is None else int(max_frames),
        "preprocessing": {
            "cutoff": float(cutoff),
            "buffer_multiplier": float(buffer_multiplier),
            "park_multiplier": float(park_multiplier),
        },
        "prior_signature": {
            "use_spline_priors": bool(config.use_spline_priors_enabled()),
            "prior_cfg": prior_cfg,
            "fitted_params_hash": _params_signature(fitted_params),
        },
    }


def _load_cached_prior_forces(
    cache_path: Path,
    expected_metadata: Dict[str, Any],
    expected_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    """Load cached prior forces if cache exists and metadata matches."""
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if "metadata_json" not in cached or "F_prior" not in cached:
                return None
            cached_meta = json.loads(str(cached["metadata_json"].item()))
            if _stable_json(cached_meta) != _stable_json(expected_metadata):
                return None
            f_prior = np.asarray(cached["F_prior"], dtype=np.float32)
    except Exception as exc:
        training_logger.warning(
            "[PriorResidual] Failed to load cache '%s': %s",
            cache_path,
            exc,
        )
        return None

    if tuple(f_prior.shape) != tuple(expected_shape):
        training_logger.warning(
            "[PriorResidual] Cache shape mismatch in '%s': got %s expected %s.",
            cache_path,
            tuple(f_prior.shape),
            tuple(expected_shape),
        )
        return None

    return f_prior


def _save_prior_cache(
    cache_path: Path,
    metadata: Dict[str, Any],
    f_prior: np.ndarray,
    f_residual: np.ndarray,
) -> None:
    """Save prior-force cache with metadata using atomic replace."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f".{cache_path.stem}.tmp.{os.getpid()}.npz"
    np.savez(
        tmp_path,
        metadata_json=np.asarray(_stable_json(metadata)),
        F_prior=np.asarray(f_prior, dtype=np.float32),
        F_residual=np.asarray(f_residual, dtype=np.float32),
    )
    os.replace(tmp_path, cache_path)


def _compute_prior_forces(
    config,
    R: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    id_to_aa: Optional[Dict[int, str]],
    compute_batch_size: int,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute prior forces for all frames via chunked JAX vmap+grad.

    If param_overrides is provided, the prior's parameters are replaced before
    computing forces (used after LBFGS pretraining).
    """
    n_atoms = int(R.shape[1])
    topology = TopologyBuilder(N_max=n_atoms, min_repulsive_sep=6)
    # Prior energy code path currently assumes free-space displacement.
    displacement = lambda Ra, Rb: Ra - Rb
    prior = PriorEnergy(config, topology, displacement, id_to_aa=id_to_aa)
    if param_overrides is not None:
        prior.params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in param_overrides.items()}

    def single_force(
        R_i: jax.Array,
        mask_i: jax.Array,
        species_i: jax.Array,
    ) -> jax.Array:
        def energy_of_R(R_var: jax.Array) -> jax.Array:
            R_detached = jax.lax.stop_gradient(R_var)
            R_masked = jnp.where(mask_i[:, None] > 0, R_var, R_detached)
            return prior.compute_total_energy(R_masked, mask_i, species=species_i)

        return -jax.grad(energy_of_R)(R_i)

    batched_force_fn = jax.jit(jax.vmap(single_force, in_axes=(0, 0, 0)))

    n_frames = int(R.shape[0])
    f_prior = np.zeros_like(R, dtype=np.float32)
    batch_size = int(compute_batch_size)

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        R_chunk = jnp.asarray(R[start:end], dtype=jnp.float32)
        mask_chunk = jnp.asarray(mask[start:end], dtype=jnp.float32)
        species_chunk = jnp.asarray(species[start:end], dtype=jnp.int32)
        f_chunk = batched_force_fn(R_chunk, mask_chunk, species_chunk)
        f_prior[start:end] = np.asarray(f_chunk, dtype=np.float32)

    return f_prior


def apply_prior_force_residual_targets(
    config,
    dataset: Dict[str, np.ndarray],
    dataset_path: Union[str, Path],
    dataset_tag: str,
    id_to_aa: Optional[Dict[int, str]],
    project_root: Union[str, Path],
    seed: int,
    max_frames: Optional[int],
    cutoff: float,
    buffer_multiplier: float,
    park_multiplier: float,
    fitted_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Replace dataset force targets with residual targets when enabled.

    Returns:
        Dictionary with cache and norm diagnostics.
    """
    if not config.prior_residual_enabled():
        return {
            "enabled": False,
            "cache_hit": False,
            "cache_path": None,
        }

    R = np.asarray(dataset["R"], dtype=np.float32)
    F_ref = np.asarray(dataset["F"], dtype=np.float32)
    mask = np.asarray(dataset["mask"], dtype=np.float32)
    species = np.asarray(dataset["species"], dtype=np.int32)

    metadata = _build_cache_metadata(
        config=config,
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_tag=dataset_tag,
        seed=seed,
        max_frames=max_frames,
        cutoff=cutoff,
        buffer_multiplier=buffer_multiplier,
        park_multiplier=park_multiplier,
        fitted_params=fitted_params,
    )

    project_root = Path(project_root)
    cache_path = _resolve_cache_path(config, dataset_tag, project_root=project_root)
    cache_hit = False

    f_prior: Optional[np.ndarray] = None
    if config.prior_residual_cache_enabled() and not config.prior_residual_force_recompute():
        f_prior = _load_cached_prior_forces(
            cache_path=cache_path,
            expected_metadata=metadata,
            expected_shape=tuple(F_ref.shape),
        )
        cache_hit = f_prior is not None

    if f_prior is None:
        training_logger.info(
            "[PriorResidual] Computing prior forces for dataset '%s' (frames=%d, atoms=%d, batch=%d).",
            dataset_tag,
            int(R.shape[0]),
            int(R.shape[1]),
            int(config.get_prior_residual_compute_batch_size()),
        )
        f_prior = _compute_prior_forces(
            config=config,
            R=R,
            mask=mask,
            species=species,
            id_to_aa=id_to_aa,
            compute_batch_size=config.get_prior_residual_compute_batch_size(),
            param_overrides=fitted_params,
        )

    f_residual = F_ref - f_prior
    dataset["F"] = np.asarray(f_residual, dtype=np.float32)

    if config.prior_residual_cache_enabled() and (not cache_hit or config.prior_residual_force_recompute()):
        _save_prior_cache(
            cache_path=cache_path,
            metadata=metadata,
            f_prior=f_prior,
            f_residual=f_residual,
        )

    return {
        "enabled": True,
        "cache_hit": bool(cache_hit),
        "cache_path": str(cache_path),
        "mean_prior_norm": float(np.mean(np.linalg.norm(f_prior, axis=-1))),
        "mean_residual_norm": float(np.mean(np.linalg.norm(f_residual, axis=-1))),
    }


def pretrain_prior_for_residual(
    config,
    dataset: Dict[str, np.ndarray],
    id_to_aa: Optional[Dict[int, str]],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Run standalone LBFGS prior pretraining against raw reference forces.

    This is the pre-residual pretraining step: it optimises the parametric
    prior parameters to best match F_ref BEFORE those parameters are used to
    compute the residual targets F_residual = F_ref - F_prior.

    Returns a dict of fitted numpy arrays (same keys as PriorEnergy.params),
    or None if pretraining is skipped.

    Skipped when:
    - training.pretrain_prior is false
    - training.prior_residual.enabled is false (nothing to prepare for)
    - model.priors.use_spline_priors is true (no parametric params to optimise)
    """
    if not config.pretrain_prior_enabled():
        return None
    if not config.prior_residual_enabled():
        return None
    if config.use_spline_priors_enabled():
        training_logger.warning(
            "[PriorResidual] pretrain_prior=true with use_spline_priors=true: "
            "LBFGS skipped (no parametric params). Residuals will use initial spline forces."
        )
        return None

    import optax

    R = np.asarray(dataset["R"], dtype=np.float32)
    F_ref = np.asarray(dataset["F"], dtype=np.float32)
    mask = np.asarray(dataset["mask"], dtype=np.float32)
    species = np.asarray(dataset["species"], dtype=np.int32)

    n_atoms = int(R.shape[1])
    topology = TopologyBuilder(N_max=n_atoms, min_repulsive_sep=6)
    displacement = lambda Ra, Rb: Ra - Rb
    prior = PriorEnergy(config, topology, displacement, id_to_aa=id_to_aa)

    R_jax = jnp.asarray(R, dtype=jnp.float32)
    F_jax = jnp.asarray(F_ref, dtype=jnp.float32)
    mask_jax = jnp.asarray(mask, dtype=jnp.float32)
    species_jax = jnp.asarray(species, dtype=jnp.int32)

    def force_matching_loss(params: Dict[str, jax.Array]) -> jax.Array:
        def single_force(R_i, mask_i, species_i):
            def energy_of_R(R_var):
                R_masked = jnp.where(mask_i[:, None] > 0, R_var, jax.lax.stop_gradient(R_var))
                return prior.compute_total_energy_from_params(params, R_masked, mask_i, species=species_i)
            return -jax.grad(energy_of_R)(R_i)

        F_pred = jax.vmap(single_force, in_axes=(0, 0, 0))(R_jax, mask_jax, species_jax)
        diff = F_pred - F_jax
        m3 = (mask_jax[:, :, None] > 0).astype(jnp.float32) * jnp.ones((1, 1, 3), dtype=jnp.float32)
        denom = jnp.maximum(jnp.sum(m3), 1.0)
        return jnp.sum(diff * diff) / denom

    max_steps = config.get_pretrain_prior_max_steps()
    tol_grad = config.get_pretrain_prior_tol_grad()
    min_steps = config.get_pretrain_prior_min_steps()

    training_logger.info(
        "[PriorResidual] LBFGS prior pretraining: max_steps=%d, tol_grad=%.2e",
        max_steps, tol_grad,
    )

    opt = optax.lbfgs(learning_rate=1.0)

    @jax.jit
    def step_fn(params, state):
        value_and_grad_fn = optax.value_and_grad_from_state(force_matching_loss)
        value, grad = value_and_grad_fn(params, state=state)
        updates, new_state = opt.update(
            grad, state, params,
            value=value, grad=grad, value_fn=force_matching_loss,
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, value, grad

    params = prior.params
    state = opt.init(params)

    for step in range(max_steps):
        params, state, loss, grad = step_fn(params, state)
        grad_norm = float(optax.tree.norm(grad))
        if step == 0 or (step + 1) % 10 == 0:
            training_logger.info(
                "[PriorResidual] LBFGS step %d/%d  loss=%.4e  grad_norm=%.4e",
                step + 1, max_steps, float(loss), grad_norm,
            )
        if step >= min_steps and grad_norm < tol_grad:
            training_logger.info(
                "[PriorResidual] LBFGS converged at step %d (grad_norm=%.4e < tol=%.4e)",
                step + 1, grad_norm, tol_grad,
            )
            break
    else:
        training_logger.info(
            "[PriorResidual] LBFGS reached max_steps=%d (grad_norm=%.4e, tol=%.4e)",
            max_steps, grad_norm, tol_grad,
        )

    return {k: np.array(v) for k, v in params.items()}
