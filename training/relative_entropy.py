"""Relative-entropy fine-tuning utilities and trainer loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import csv
import pickle
import sys

import jax
import jax.numpy as jnp
import optax
from jax_md import simulate

from mc.config import HMCConfig, MALAConfig
from utils.logging import training_logger


@dataclass(frozen=True)
class RelativeEntropyRolloutStage:
    """Piecewise-constant MD effort for a range of REM iterations."""

    start_iteration: int
    steps_per_iteration: int
    burn_in_steps: int
    sample_stride: int

    @property
    def retained_samples_per_replica(self) -> int:
        remaining = int(self.steps_per_iteration) - int(self.burn_in_steps)
        return max(remaining, 0) // int(self.sample_stride)


@dataclass(frozen=True)
class RelativeEntropyConfig:
    """Parsed configuration for RE fine-tuning."""

    enabled: bool = False
    reference_data_path: Optional[str] = None
    optimizer: str = "adam"
    sampler: str = "md_langevin"
    iterations: int = 100
    reference_batch_size: int = 16
    n_replicas: int = 8
    steps_per_iteration: int = 200
    burn_in_steps: int = 50
    sample_stride: int = 10
    dt: float = 0.02045
    kT: float = 0.636
    gamma: float = 0.000977
    mass: Union[float, tuple] = 12.011
    barrier_penalty_enabled: bool = False
    barrier_penalty_weight: float = 0.0
    barrier_penalty_v0: float = 1.3
    barrier_penalty_indices: tuple = (0, 1, 2, 3)
    start_frame_mode: str = "reference_random"
    persistent_chains: bool = False
    initial_state_data_path: Optional[str] = None
    initial_state_phi_targets_deg: tuple = ()
    initial_state_phi_indices: tuple = (0, 1, 2, 3)
    initial_state_phi_shift_deg: float = 180.0
    initial_state_cv_targets_deg: tuple = ()
    initial_state_cv_indices: tuple = ()
    initial_state_cv_shift_deg: tuple = ()
    rollout_schedule: tuple = ()
    checkpoint_freq: int = 10
    max_force: float = 1.0e4
    min_pair_distance: float = 1.5
    reject_on_instability: bool = True
    seed_offset: int = 9117
    output_dir: Optional[str] = None
    optimizer_gradient_scale: float = 1.0
    gradient_batch_size: int = 0
    diagnostics_interval: int = 1
    trainable_param_substring: Optional[str] = None
    chirality_diagnostics_enabled: bool = False
    chirality_diagnostics_indices: tuple = (0, 1, 2, 3)
    chirality_diagnostics_planar_threshold: float = 1.3
    mc_mala: Optional[MALAConfig] = None

    @property
    def beta(self) -> float:
        if self.kT <= 0.0:
            raise ValueError(f"training.relative_entropy.kT must be > 0, got {self.kT}.")
        return 1.0 / float(self.kT)

    @property
    def retained_samples_per_replica(self) -> int:
        remaining = int(self.steps_per_iteration) - int(self.burn_in_steps)
        if remaining <= 0:
            return 0
        return remaining // int(self.sample_stride)

    @property
    def total_model_samples(self) -> int:
        return int(self.n_replicas) * int(self.retained_samples_per_replica)

    @property
    def model_start_count(self) -> int:
        if self.sampler == "mc_mala" and self.mc_mala is not None:
            return int(self.mc_mala.n_chains)
        return int(self.n_replicas)

    def rollout_for_iteration(self, iteration: int) -> RelativeEntropyRolloutStage:
        """Return the active rollout settings for one zero-based iteration."""

        if not self.rollout_schedule:
            return RelativeEntropyRolloutStage(
                start_iteration=0,
                steps_per_iteration=int(self.steps_per_iteration),
                burn_in_steps=int(self.burn_in_steps),
                sample_stride=int(self.sample_stride),
            )
        active = self.rollout_schedule[0]
        for stage in self.rollout_schedule[1:]:
            if int(stage.start_iteration) > int(iteration):
                break
            active = stage
        return active


def _positive_int(name: str, value: Any) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"training.relative_entropy.{name} must be > 0, got {parsed}.")
    return parsed


def _nonnegative_int(name: str, value: Any) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"training.relative_entropy.{name} must be >= 0, got {parsed}.")
    return parsed


def _positive_float(name: str, value: Any) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"training.relative_entropy.{name} must be > 0, got {parsed}.")
    return parsed


def relative_entropy_config(config) -> RelativeEntropyConfig:
    """Parse and validate ``training.relative_entropy`` from a ConfigManager."""
    cfg = config.get("training", "relative_entropy", default={}) or {}
    if not isinstance(cfg, dict):
        raise ValueError("training.relative_entropy must be a mapping.")

    reference_data_path = cfg.get("reference_data_path", None)
    if reference_data_path is None or str(reference_data_path).strip() == "":
        reference_data_path = config.get_data_path()
    else:
        reference_data_path = str(reference_data_path)

    sampler = str(cfg.get("sampler", "md_langevin")).strip().lower()
    if sampler in {"mala", "blackjax_mala"}:
        sampler = "mc_mala"
    if sampler in {"hmc", "blackjax_hmc"}:
        sampler = "mc_hmc"

    parsed = RelativeEntropyConfig(
        enabled=bool(cfg.get("enabled", False)),
        reference_data_path=reference_data_path,
        optimizer=str(cfg.get("optimizer", "adam")).strip().lower(),
        sampler=sampler,
        iterations=_positive_int("iterations", cfg.get("iterations", 100)),
        reference_batch_size=_positive_int(
            "reference_batch_size", cfg.get("reference_batch_size", 16)
        ),
        n_replicas=_positive_int("n_replicas", cfg.get("n_replicas", 8)),
        steps_per_iteration=_positive_int(
            "steps_per_iteration", cfg.get("steps_per_iteration", 200)
        ),
        burn_in_steps=_nonnegative_int("burn_in_steps", cfg.get("burn_in_steps", 50)),
        sample_stride=_positive_int("sample_stride", cfg.get("sample_stride", 10)),
        dt=_positive_float("dt", cfg.get("dt", 0.02045)),
        kT=_positive_float("kT", cfg.get("kT", 0.636)),
        gamma=_positive_float("gamma", cfg.get("gamma", 0.000977)),
        mass=cfg.get("mass", 12.011),
        barrier_penalty_enabled=bool((cfg.get("barrier_penalty") or {}).get("enabled", False)),
        barrier_penalty_weight=float((cfg.get("barrier_penalty") or {}).get("weight", 0.0)),
        barrier_penalty_v0=float((cfg.get("barrier_penalty") or {}).get("v0", 1.3)),
        barrier_penalty_indices=tuple(
            int(i) for i in ((cfg.get("barrier_penalty") or {}).get("indices", [0, 1, 2, 3]))
        ),
        start_frame_mode=str(cfg.get("start_frame_mode", "reference_random")).strip().lower(),
        persistent_chains=bool(cfg.get("persistent_chains", False)),
        initial_state_data_path=(
            str(cfg["initial_state_data_path"])
            if cfg.get("initial_state_data_path") is not None
            else None
        ),
        initial_state_phi_targets_deg=tuple(
            float(x) for x in (cfg.get("initial_state_phi_targets_deg", []) or [])
        ),
        initial_state_phi_indices=tuple(
            int(x) for x in (cfg.get("initial_state_phi_indices", [0, 1, 2, 3]) or [])
        ),
        initial_state_phi_shift_deg=float(cfg.get("initial_state_phi_shift_deg", 180.0)),
        initial_state_cv_targets_deg=tuple(
            tuple(float(value) for value in target)
            for target in (cfg.get("initial_state_cv_targets_deg", []) or [])
        ),
        initial_state_cv_indices=tuple(
            tuple(int(value) for value in indices)
            for indices in (cfg.get("initial_state_cv_indices", []) or [])
        ),
        initial_state_cv_shift_deg=tuple(
            float(value) for value in (cfg.get("initial_state_cv_shift_deg", []) or [])
        ),
        checkpoint_freq=_nonnegative_int("checkpoint_freq", cfg.get("checkpoint_freq", 10)),
        max_force=_positive_float("max_force", cfg.get("max_force", 1.0e4)),
        min_pair_distance=_positive_float(
            "min_pair_distance", cfg.get("min_pair_distance", 1.5)
        ),
        reject_on_instability=bool(cfg.get("reject_on_instability", True)),
        seed_offset=int(cfg.get("seed_offset", 9117)),
        output_dir=str(cfg["output_dir"]) if cfg.get("output_dir") is not None else None,
        optimizer_gradient_scale=float(cfg.get("optimizer_gradient_scale", 1.0)),
        gradient_batch_size=_nonnegative_int(
            "gradient_batch_size", cfg.get("gradient_batch_size", 0)
        ),
        diagnostics_interval=_positive_int(
            "diagnostics_interval", cfg.get("diagnostics_interval", 1)
        ),
        trainable_param_substring=(
            str(cfg["trainable_param_substring"])
            if cfg.get("trainable_param_substring")
            else None
        ),
        chirality_diagnostics_enabled=bool(
            (cfg.get("chirality_diagnostics") or {}).get("enabled", False)
        ),
        chirality_diagnostics_indices=tuple(
            int(i)
            for i in (cfg.get("chirality_diagnostics") or {}).get(
                "indices", [0, 1, 2, 3]
            )
        ),
        chirality_diagnostics_planar_threshold=_positive_float(
            "chirality_diagnostics.planar_threshold",
            (cfg.get("chirality_diagnostics") or {}).get("planar_threshold", 1.3),
        ),
        mc_mala=None,
    )

    schedule_raw = cfg.get("rollout_schedule", []) or []
    if not isinstance(schedule_raw, list):
        raise ValueError("training.relative_entropy.rollout_schedule must be a list.")
    rollout_schedule = []
    for index, item in enumerate(schedule_raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"training.relative_entropy.rollout_schedule[{index}] must be a mapping."
            )
        rollout_schedule.append(
            RelativeEntropyRolloutStage(
                start_iteration=_nonnegative_int(
                    f"rollout_schedule[{index}].start_iteration",
                    item.get("start_iteration", 0),
                ),
                steps_per_iteration=_positive_int(
                    f"rollout_schedule[{index}].steps_per_iteration",
                    item.get("steps_per_iteration", parsed.steps_per_iteration),
                ),
                burn_in_steps=_nonnegative_int(
                    f"rollout_schedule[{index}].burn_in_steps",
                    item.get("burn_in_steps", parsed.burn_in_steps),
                ),
                sample_stride=_positive_int(
                    f"rollout_schedule[{index}].sample_stride",
                    item.get("sample_stride", parsed.sample_stride),
                ),
            )
        )
    if rollout_schedule:
        starts = [stage.start_iteration for stage in rollout_schedule]
        if starts[0] != 0 or starts != sorted(set(starts)):
            raise ValueError(
                "training.relative_entropy.rollout_schedule must start at iteration 0 "
                "and have unique, increasing start_iteration values."
            )
        if starts[-1] >= parsed.iterations:
            raise ValueError(
                "The final rollout_schedule start_iteration must be smaller than iterations."
            )
        for stage in rollout_schedule:
            if stage.retained_samples_per_replica <= 0:
                raise ValueError(
                    "Every rollout_schedule stage must retain at least one sample per replica."
                )
        parsed = replace(parsed, rollout_schedule=tuple(rollout_schedule))

    if parsed.sampler not in {"md_langevin", "mc_mala", "mc_hmc"}:
        raise ValueError(
            "training.relative_entropy.sampler must be 'md_langevin', 'mc_mala', or 'mc_hmc', "
            f"got {parsed.sampler!r}."
        )

    if parsed.sampler == "mc_mala":
        mc_cfg = cfg.get("mc", {}) or {}
        if not isinstance(mc_cfg, dict):
            raise ValueError("training.relative_entropy.mc must be a mapping.")
        mala_cfg = mc_cfg.get("mala", {}) or {}
        if not isinstance(mala_cfg, dict):
            raise ValueError("training.relative_entropy.mc.mala must be a mapping.")
        mc_mala = MALAConfig(
            n_chains=_positive_int("mc.mala.n_chains", mala_cfg.get("n_chains", parsed.n_replicas)),
            steps_per_iteration=_positive_int(
                "mc.mala.steps_per_iteration",
                mala_cfg.get("steps_per_iteration", parsed.steps_per_iteration),
            ),
            burn_in_steps=_nonnegative_int(
                "mc.mala.burn_in_steps",
                mala_cfg.get("burn_in_steps", parsed.burn_in_steps),
            ),
            sample_stride=_positive_int(
                "mc.mala.sample_stride",
                mala_cfg.get("sample_stride", parsed.sample_stride),
            ),
            step_size=_positive_float("mc.mala.step_size", mala_cfg.get("step_size", 1.0e-3)),
            beta=parsed.beta,
        )
        mc_mala.validate()
        parsed = RelativeEntropyConfig(**{**parsed.__dict__, "mc_mala": mc_mala})

    if parsed.sampler == "mc_hmc":
        mc_cfg = cfg.get("mc", {}) or {}
        if not isinstance(mc_cfg, dict):
            raise ValueError("training.relative_entropy.mc must be a mapping.")
        hmc_cfg = mc_cfg.get("hmc", {}) or {}
        if not isinstance(hmc_cfg, dict):
            raise ValueError("training.relative_entropy.mc.hmc must be a mapping.")
        mc_hmc = HMCConfig(
            n_chains=_positive_int("mc.hmc.n_chains", hmc_cfg.get("n_chains", parsed.n_replicas)),
            steps_per_iteration=_positive_int(
                "mc.hmc.steps_per_iteration",
                hmc_cfg.get("steps_per_iteration", parsed.steps_per_iteration),
            ),
            burn_in_steps=_nonnegative_int(
                "mc.hmc.burn_in_steps",
                hmc_cfg.get("burn_in_steps", parsed.burn_in_steps),
            ),
            sample_stride=_positive_int(
                "mc.hmc.sample_stride",
                hmc_cfg.get("sample_stride", parsed.sample_stride),
            ),
            step_size=_positive_float("mc.hmc.step_size", hmc_cfg.get("step_size", 1.0e-3)),
            num_integration_steps=_positive_int(
                "mc.hmc.num_integration_steps",
                hmc_cfg.get("num_integration_steps", 10),
            ),
            inverse_mass_matrix=hmc_cfg.get("inverse_mass_matrix", [1.0]),
            beta=parsed.beta,
        )
        mc_hmc.validate()
        parsed = RelativeEntropyConfig(**{**parsed.__dict__, "mc_hmc": mc_hmc})

    if parsed.retained_samples_per_replica <= 0:
        raise ValueError(
            "training.relative_entropy rollout settings must retain at least one model sample; "
            f"got steps_per_iteration={parsed.steps_per_iteration}, "
            f"burn_in_steps={parsed.burn_in_steps}, sample_stride={parsed.sample_stride}."
        )
    if parsed.start_frame_mode not in {
        "reference_random",
        "configured_phi_targets",
        "configured_cv_targets",
    }:
        raise ValueError(
            "training.relative_entropy.start_frame_mode must be 'reference_random', "
            "'configured_phi_targets', or 'configured_cv_targets'; "
            f"got {parsed.start_frame_mode!r}."
        )
    if parsed.start_frame_mode == "configured_phi_targets":
        if not parsed.initial_state_data_path:
            raise ValueError(
                "training.relative_entropy.initial_state_data_path is required for "
                "start_frame_mode='configured_phi_targets'."
            )
        if len(parsed.initial_state_phi_indices) != 4:
            raise ValueError(
                "training.relative_entropy.initial_state_phi_indices must contain four indices."
            )
        if len(parsed.initial_state_phi_targets_deg) != parsed.model_start_count:
            raise ValueError(
                "training.relative_entropy.initial_state_phi_targets_deg must contain exactly "
                f"model_start_count={parsed.model_start_count} targets, got "
                f"{len(parsed.initial_state_phi_targets_deg)}."
            )
    if parsed.start_frame_mode == "configured_cv_targets":
        if not parsed.initial_state_data_path:
            raise ValueError(
                "training.relative_entropy.initial_state_data_path is required for "
                "start_frame_mode='configured_cv_targets'."
            )
        n_cvs = len(parsed.initial_state_cv_indices)
        if n_cvs < 1 or any(len(indices) != 4 for indices in parsed.initial_state_cv_indices):
            raise ValueError(
                "training.relative_entropy.initial_state_cv_indices must contain one or more "
                "four-index dihedral definitions."
            )
        if len(parsed.initial_state_cv_shift_deg) not in {0, n_cvs}:
            raise ValueError(
                "training.relative_entropy.initial_state_cv_shift_deg must be empty or contain "
                "one shift per configured CV."
            )
        if len(parsed.initial_state_cv_targets_deg) != parsed.model_start_count:
            raise ValueError(
                "training.relative_entropy.initial_state_cv_targets_deg must contain exactly "
                f"model_start_count={parsed.model_start_count} targets."
            )
        if any(len(target) != n_cvs for target in parsed.initial_state_cv_targets_deg):
            raise ValueError(
                "Every initial_state_cv_targets_deg row must contain one value per configured CV."
            )
    if parsed.persistent_chains and parsed.sampler != "md_langevin":
        raise ValueError(
            "training.relative_entropy.persistent_chains currently requires sampler='md_langevin'."
        )
    if parsed.rollout_schedule and parsed.sampler != "md_langevin":
        raise ValueError(
            "training.relative_entropy.rollout_schedule currently requires sampler='md_langevin'."
        )
    if parsed.chirality_diagnostics_enabled and len(parsed.chirality_diagnostics_indices) != 4:
        raise ValueError(
            "training.relative_entropy.chirality_diagnostics.indices must contain four indices."
        )
    return parsed


def extract_params_from_checkpoint_payload(payload_or_path: Any) -> Dict[str, Any]:
    """Extract model params from supported CAMEO/Chemtrain checkpoint payloads."""
    payload = payload_or_path
    if isinstance(payload_or_path, (str, Path)):
        with Path(payload_or_path).open("rb") as handle:
            payload = pickle.load(handle)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("params"), dict):
            params = payload["params"]
        elif isinstance(payload.get("best_params"), dict):
            params = payload["best_params"]
        elif isinstance(payload.get("trainer_state"), dict) and isinstance(
            payload["trainer_state"].get("params"), dict
        ):
            params = payload["trainer_state"]["params"]

    if isinstance(params, dict) and "ml" not in params:
        params = {"ml": params}
    if not isinstance(params, dict) or "ml" not in params:
        raise TypeError(
            "Checkpoint payload must contain model params with an 'ml' subtree. "
            f"Got {type(params)!r}."
        )
    return jax.tree_util.tree_map(jnp.asarray, params)


def _parameter_path_tokens(path):
    return tuple(
        getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", str(entry))))
        for entry in path
    )


def merge_matching_parameter_trees(source: Any, initialized: Any):
    """Reuse same-path, same-shape checkpoint leaves and keep new initialization."""
    source_leaves = {
        _parameter_path_tokens(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(source)[0]
    }
    counts = {"reused_leaves": 0, "initialized_leaves": 0}

    def choose(path, target):
        source_leaf = source_leaves.get(_parameter_path_tokens(path))
        if source_leaf is not None and jnp.shape(source_leaf) == jnp.shape(target):
            counts["reused_leaves"] += 1
            return jnp.asarray(source_leaf, dtype=getattr(target, "dtype", None))
        counts["initialized_leaves"] += 1
        return target

    return jax.tree_util.tree_map_with_path(choose, initialized), counts


def mask_parameter_tree_by_substring(tree: Any, substring: str):
    """Keep leaves whose parameter path contains substring; zero all others."""
    token = str(substring)
    if not token:
        return tree

    def mask(path, leaf):
        path_text = "/".join(str(part) for part in _parameter_path_tokens(path))
        return leaf if token in path_text else jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(mask, tree)


def apply_ml_updates(params: Dict[str, Any], ml_updates: Any) -> Dict[str, Any]:
    """Apply Optax updates to params['ml'] while preserving all non-ML subtrees."""
    updated = dict(params)
    updated["ml"] = optax.apply_updates(params["ml"], ml_updates)
    return updated



def _mean_energy_and_grad(ml_params, R, mask, species, energy_fn, batch_size: int = 0):
    """Return mean energy and mean parameter gradient for a batch of samples."""
    R = jnp.asarray(R)
    mask = jnp.asarray(mask)
    species = jnp.asarray(species)

    def single_energy(params, R_i, mask_i, species_i):
        return energy_fn(params, R_i, mask_i, species_i)

    if int(batch_size) > 0 and int(batch_size) < int(R.shape[0]):
        n_frames = int(R.shape[0])
        batch_size = int(batch_size)
        n_batches = (n_frames + batch_size - 1) // batch_size
        padded = n_batches * batch_size
        pad_frames = padded - n_frames
        R_pad = jnp.pad(R, ((0, pad_frames), (0, 0), (0, 0)))
        mask_pad = jnp.pad(mask, ((0, pad_frames), (0, 0)))
        species_pad = jnp.pad(species, ((0, pad_frames), (0, 0)))
        weights = (jnp.arange(padded) < n_frames).astype(R.dtype)

        R_batches = R_pad.reshape((n_batches, batch_size) + R.shape[1:])
        mask_batches = mask_pad.reshape((n_batches, batch_size) + mask.shape[1:])
        species_batches = species_pad.reshape((n_batches, batch_size) + species.shape[1:])
        weight_batches = weights.reshape((n_batches, batch_size))

        def batch_sum_energy(params, R_b, mask_b, species_b, weight_b):
            energies = jax.vmap(
                lambda R_i, m_i, s_i: single_energy(params, R_i, m_i, s_i)
            )(R_b, mask_b, species_b)
            return jnp.sum(energies * weight_b), energies

        def batch_value_and_grad(batch):
            R_b, mask_b, species_b, weight_b = batch
            (energy_sum, _), grad = jax.value_and_grad(
                batch_sum_energy, has_aux=True
            )(ml_params, R_b, mask_b, species_b, weight_b)
            return energy_sum, grad

        energy_sums, grad_sums = jax.lax.map(
            batch_value_and_grad,
            (R_batches, mask_batches, species_batches, weight_batches),
        )
        total_energy = jnp.sum(energy_sums)
        grad_sum = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), grad_sums)
        denom = jnp.asarray(n_frames, dtype=total_energy.dtype)
        grad_mean = jax.tree_util.tree_map(lambda x: x / denom, grad_sum)
        return total_energy / denom, None, grad_mean

    def mean_energy(params):
        energies = jax.vmap(lambda R_i, m_i, s_i: single_energy(params, R_i, m_i, s_i))(
            R, mask, species
        )
        return jnp.mean(energies), energies

    (mean_value, energies), grad = jax.value_and_grad(mean_energy, has_aux=True)(ml_params)
    return mean_value, energies, grad


def relative_entropy_gradient(
    ml_params: Any,
    R_ref: jax.Array,
    mask_ref: jax.Array,
    species_ref: jax.Array,
    R_model: jax.Array,
    mask_model: jax.Array,
    species_model: jax.Array,
    energy_fn,
    beta: float,
    gradient_batch_size: int = 0,
):
    """Estimate beta * (E_ref[grad U] - E_model[grad U]) for ML params."""
    R_ref = jax.lax.stop_gradient(jnp.asarray(R_ref))
    mask_ref = jax.lax.stop_gradient(jnp.asarray(mask_ref))
    species_ref = jax.lax.stop_gradient(jnp.asarray(species_ref))
    R_model = jax.lax.stop_gradient(jnp.asarray(R_model))
    mask_model = jax.lax.stop_gradient(jnp.asarray(mask_model))
    species_model = jax.lax.stop_gradient(jnp.asarray(species_model))

    ref_mean, _, ref_grad = _mean_energy_and_grad(
        ml_params, R_ref, mask_ref, species_ref, energy_fn, gradient_batch_size
    )
    model_mean, _, model_grad = _mean_energy_and_grad(
        ml_params, R_model, mask_model, species_model, energy_fn, gradient_batch_size
    )
    beta_value = jnp.asarray(beta, dtype=ref_mean.dtype)
    grad = jax.tree_util.tree_map(lambda a, b: beta_value * (a - b), ref_grad, model_grad)
    grad_norm = optax.tree.norm(grad)
    metrics = {
        "ref_energy_mean": ref_mean,
        "model_energy_mean": model_mean,
        "re_energy_gap": ref_mean - model_mean,
        "grad_norm": grad_norm,
    }
    return grad, metrics


def signed_volume(R: jax.Array, indices: Sequence[int]) -> jax.Array:
    """Parity-ODD scalar: (r1-r0) x (r2-r1) . (r3-r2) for a bead quadruple.

    Flips sign under reflection, so |V| ~ 0 marks the planar transition state that
    separates a chiral conformation from its mirror image.
    """
    i0, i1, i2, i3 = (int(i) for i in indices)
    a = R[..., i1, :] - R[..., i0, :]
    b = R[..., i2, :] - R[..., i1, :]
    c = R[..., i3, :] - R[..., i2, :]
    return jnp.sum(jnp.cross(a, b) * c, axis=-1)


def chirality_population_metrics(
    R: jax.Array,
    indices: Sequence[int],
    planar_threshold: float,
) -> Dict[str, jax.Array]:
    """Summarize signed-volume branch populations for a coordinate batch."""
    volume = signed_volume(R, indices)
    return {
        "signed_volume_mean": jnp.mean(volume),
        "signed_volume_min": jnp.min(volume),
        "signed_volume_max": jnp.max(volume),
        "fraction_positive": jnp.mean((volume > 0.0).astype(jnp.float32)),
        "fraction_negative": jnp.mean((volume < 0.0).astype(jnp.float32)),
        "fraction_near_planar": jnp.mean(
            (jnp.abs(volume) < float(planar_threshold)).astype(jnp.float32)
        ),
    }


def barrier_penalty_weights(R: jax.Array, indices: Sequence[int], v0: float) -> jax.Array:
    """Per-sample penalty w = relu(v0 - |V|)^2 . Zero unless a sample is near-planar.

    |V| is parity-EVEN, so this is representable by a parity-invariant model: it raises
    the mirror-interconversion barrier without needing to tell the two basins apart
    (which an O(3)-invariant energy provably cannot do). The reference ensemble has zero
    density below v0, so the term does not distort the relative-entropy target.
    """
    v = jnp.abs(signed_volume(R, indices))
    return jnp.square(jax.nn.relu(float(v0) - v))


def barrier_penalty_gradient(
    ml_params,
    R_model,
    mask_model,
    species_model,
    energy_fn,
    weights: jax.Array,
    beta: float,
    weight_scale: float,
):
    """d/dtheta of lambda * <w>_model, via the covariance identity

        d<w>/dtheta = -beta * ( <w dU/dtheta> - <w><dU/dtheta> )

    computed as the gradient of a surrogate scalar with w held constant.
    """
    R_model = jax.lax.stop_gradient(jnp.asarray(R_model))
    mask_model = jax.lax.stop_gradient(jnp.asarray(mask_model))
    species_model = jax.lax.stop_gradient(jnp.asarray(species_model))
    w = jax.lax.stop_gradient(jnp.asarray(weights))
    w_centered = w - jnp.mean(w)

    def surrogate(params):
        energies = jax.vmap(
            lambda r, m, sp: energy_fn(params, r, m, sp)
        )(R_model, mask_model, species_model)
        # weight_scale / beta are traced under jit: never call float() on them here.
        scale = jnp.asarray(weight_scale, energies.dtype) * jnp.asarray(beta, energies.dtype)
        return -scale * jnp.mean(w_centered.astype(energies.dtype) * energies)

    return jax.grad(surrogate)(ml_params)


def compute_sample_diagnostics(R: jax.Array, forces: jax.Array, mask: jax.Array) -> Dict[str, float]:
    """Compute numerical and geometric safety diagnostics for sampled states."""
    R = jnp.asarray(R)
    forces = jnp.asarray(forces)
    mask = jnp.asarray(mask) > 0
    has_bad = jnp.logical_or(
        jnp.any(~jnp.isfinite(R)),
        jnp.any(~jnp.isfinite(forces)),
    )
    finite_force_abs = jnp.where(jnp.isfinite(forces), jnp.abs(forces), 0.0)
    max_force = jnp.max(finite_force_abs) if forces.size else jnp.asarray(0.0)

    def frame_min_distance(R_i, mask_i):
        dR = R_i[:, None, :] - R_i[None, :, :]
        dist = jnp.sqrt(jnp.sum(dR * dR, axis=-1) + 1.0e-12)
        valid = jnp.logical_and(mask_i[:, None], mask_i[None, :])
        valid = jnp.logical_and(valid, ~jnp.eye(R_i.shape[0], dtype=bool))
        dist = jnp.where(valid, dist, jnp.inf)
        return jnp.min(dist)

    min_dist = jnp.min(jax.vmap(frame_min_distance)(R, mask)) if R.shape[0] else jnp.inf
    return {
        "has_nan_or_inf": bool(jax.device_get(has_bad)),
        "max_force": float(jax.device_get(max_force)),
        "min_pair_distance": float(jax.device_get(min_dist)),
        "n_samples": int(R.shape[0]),
    }


def is_unstable(
    diagnostics: Dict[str, Any],
    max_force: float,
    min_pair_distance: float,
) -> bool:
    """Return True if diagnostics violate RE update safety thresholds."""
    if bool(diagnostics.get("has_nan_or_inf", False)):
        return True
    if float(diagnostics.get("max_force", 0.0)) > float(max_force):
        return True
    if float(diagnostics.get("min_pair_distance", float("inf"))) < float(min_pair_distance):
        return True
    return False



class InProcessLangevinSampler:
    """Generate model samples with short in-process JAX-MD Langevin rollouts."""

    def __init__(self, energy_fn, shift_fn, config: RelativeEntropyConfig):
        self.energy_fn = energy_fn
        self.shift_fn = shift_fn
        self.config = config

    def configure_rollout(self, stage: RelativeEntropyRolloutStage) -> None:
        """Update only rollout lengths while retaining the current chain states."""

        self.config = replace(
            self.config,
            steps_per_iteration=int(stage.steps_per_iteration),
            burn_in_steps=int(stage.burn_in_steps),
            sample_stride=int(stage.sample_stride),
        )

    def _mass_for_species(self, species: jax.Array) -> jax.Array:
        mass = self.config.mass
        if isinstance(mass, (list, tuple)):
            table = jnp.asarray(mass, dtype=jnp.float32)
            return table[jnp.asarray(species, dtype=jnp.int32)][:, None]
        return jnp.full((species.shape[0], 1), float(mass), dtype=jnp.float32)

    def _run_single(self, R0, mask, species, key):
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        R0 = jnp.asarray(R0, dtype=jnp.float32)

        def energy_for_md(R):
            R_masked = jnp.where(mask[:, None] > 0, R, jax.lax.stop_gradient(R0))
            return self.energy_fn(R_masked, mask=mask, species=species)

        init_fn, step_fn = simulate.nvt_langevin(
            energy_for_md,
            self.shift_fn,
            float(self.config.dt),
            float(self.config.kT),
            float(self.config.gamma),
        )
        state = init_fn(key, R0, mass=self._mass_for_species(species))
        n_retained = int(self.config.retained_samples_per_replica)
        retained0 = jnp.zeros((n_retained,) + R0.shape, dtype=R0.dtype)

        def scan_step(carry, step_idx):
            state, retained, retained_count = carry
            state = step_fn(state)
            position = jnp.where(mask[:, None] > 0, state.position, R0)
            state = state.set(position=position)
            after_burn = step_idx > int(self.config.burn_in_steps)
            on_stride = (step_idx - int(self.config.burn_in_steps)) % int(self.config.sample_stride) == 0
            should_retain = jnp.logical_and(after_burn, on_stride)
            safe_index = jnp.minimum(retained_count, n_retained - 1)
            updated_retained = jax.lax.dynamic_update_index_in_dim(
                retained,
                jax.lax.stop_gradient(position[None, ...]),
                safe_index,
                axis=0,
            )
            retained = jnp.where(should_retain, updated_retained, retained)
            retained_count = retained_count + should_retain.astype(jnp.int32)
            return (state, retained, retained_count), None

        steps = jnp.arange(1, int(self.config.steps_per_iteration) + 1)
        (final_state, retained, _), _ = jax.lax.scan(
            scan_step,
            (state, retained0, jnp.asarray(0, dtype=jnp.int32)),
            steps,
        )
        final_position = jnp.where(mask[:, None] > 0, final_state.position, R0)
        return retained, final_position

    def _forces_for_samples(self, R, mask, species):
        def force_single(R_i, mask_i, species_i):
            def energy_of_R(R_eval):
                return self.energy_fn(R_eval, mask=mask_i, species=species_i)

            return -jax.grad(energy_of_R)(R_i)

        return jax.vmap(force_single)(R, mask, species)

    def run(self, R0, mask, species, rng_key) -> Dict[str, Any]:
        R0 = jnp.asarray(R0, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        if R0.shape[0] != int(self.config.n_replicas):
            raise ValueError(
                f"RE sampler expected n_replicas={self.config.n_replicas}, got {R0.shape[0]}."
            )
        if mask.shape[0] != R0.shape[0] or species.shape[0] != R0.shape[0]:
            raise ValueError("R0, mask, and species must have matching replica axes.")

        keys = jax.random.split(jnp.asarray(rng_key), int(self.config.n_replicas))
        per_replica, final_R = jax.vmap(self._run_single)(R0, mask, species, keys)
        R_samples = per_replica.reshape((-1,) + R0.shape[1:])
        repeats = int(self.config.retained_samples_per_replica)
        mask_samples = jnp.repeat(mask, repeats, axis=0)
        species_samples = jnp.repeat(species, repeats, axis=0)
        diagnostics = {
            "has_nan_or_inf": bool(jax.device_get(jnp.any(~jnp.isfinite(R_samples)))),
            "n_samples": int(R_samples.shape[0]),
        }
        return {
            "R": R_samples,
            "mask": mask_samples,
            "species": species_samples,
            "final_R": final_R,
            "diagnostics": diagnostics,
        }

    def diagnostics_for_samples(self, R, mask, species) -> Dict[str, Any]:
        forces = self._forces_for_samples(R, mask, species)
        return compute_sample_diagnostics(R, forces, mask)


def _history_fieldnames(history: Sequence[Dict[str, Any]]) -> Sequence[str]:
    preferred = [
        "iteration",
        "rejected",
        "objective",
        "abs_re_energy_gap",
        "re_energy_gap",
        "ref_energy_mean",
        "model_energy_mean",
        "grad_norm",
        "update_norm",
        "ml_param_norm",
        "has_nan_or_inf",
        "max_force",
        "min_pair_distance",
        "n_samples",
    ]
    seen = set()
    fields = []
    for key in preferred:
        if any(key in row for row in history):
            fields.append(key)
            seen.add(key)
    for row in history:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def write_relative_entropy_history_artifacts(
    history: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    prefix: str = "relative_entropy",
) -> Dict[str, str]:
    """Write RE metrics to CSV/log and plot scalar history when matplotlib is available."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = list(history)
    csv_path = output_dir / f"{prefix}_history.csv"
    log_path = output_dir / f"{prefix}_loss.log"
    plot_path = output_dir / f"{prefix}_loss_curve.png"

    fields = list(_history_fieldnames(history)) if history else ["iteration"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    with log_path.open("w", encoding="utf-8") as handle:
        for row in history:
            parts = [f"iteration={row.get('iteration')}"]
            for key in ("objective", "re_energy_gap", "grad_norm", "update_norm", "rejected"):
                if key in row:
                    parts.append(f"{key}={row[key]}")
            handle.write("[RE] " + " ".join(parts) + "\n")

    readable_path = output_dir / f"{prefix}_loss_curve_readable.png"
    symlog_path = output_dir / f"{prefix}_loss_curve_symlog.png"
    readable_status = "not_written"
    symlog_status = "not_written"
    try:
        import math
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = np.asarray([int(row.get("iteration", idx)) for idx, row in enumerate(history)])

        def series(key):
            values = []
            for row in history:
                value = row.get(key)
                if value is None:
                    values.append(np.nan)
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = np.nan
                values.append(value if math.isfinite(value) else np.nan)
            return np.asarray(values, dtype=float)

        objective = series("objective")
        gap = series("re_energy_gap")
        grad_norm = series("grad_norm")
        update_norm = series("update_norm")
        rejected = np.asarray([bool(row.get("rejected", False)) for row in history])

        fig, ax = plt.subplots(figsize=(8, 5))
        for values, label in (
            (objective, "abs energy gap"),
            (grad_norm, "gradient norm"),
            (update_norm, "update norm"),
        ):
            if np.any(np.isfinite(values)):
                ax.plot(xs, values, marker="o", linewidth=1.5, markersize=3, label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_title("Relative Entropy Training")
        if ax.lines:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_status = str(plot_path)

        finite_objective = objective[np.isfinite(objective)]
        if finite_objective.size:
            clip_hi = float(np.percentile(finite_objective, 90))
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(
                xs,
                np.minimum(objective, clip_hi),
                marker="o",
                linewidth=1.2,
                markersize=3,
                label="objective clipped at p90",
            )
            axes[0].axhline(clip_hi, color="tab:red", linestyle="--", linewidth=1, label=f"p90={clip_hi:.1f}")
            axes[0].set_ylabel("abs gap")
            axes[0].legend(loc="upper right")
            if np.any(np.isfinite(gap)):
                axes[1].plot(xs, gap, marker="o", linewidth=1.2, markersize=3, label="signed gap")
                axes[1].axhline(0, color="black", linewidth=0.8)
                gap_lo = float(np.nanpercentile(gap, 5))
                gap_hi = float(np.nanpercentile(gap, 95))
                if math.isfinite(gap_lo) and math.isfinite(gap_hi) and gap_hi > gap_lo:
                    axes[1].set_ylim(gap_lo, gap_hi)
            axes[1].set_ylabel("gap p5-p95")
            axes[2].plot(xs, grad_norm, marker="o", linewidth=1.2, markersize=3, label="grad norm")
            axes[2].set_ylabel("grad norm")
            axes[2].set_xlabel("iteration")
            for ax_i in axes:
                ax_i.grid(True, alpha=0.25)
                if np.any(rejected):
                    ax_i.scatter(xs[rejected], np.full(np.sum(rejected), ax_i.get_ylim()[1]), marker="x", color="red")
            fig.suptitle("Relative Entropy Training - Robust View")
            fig.tight_layout()
            fig.savefig(readable_path, dpi=160)
            plt.close(fig)
            readable_status = str(readable_path)

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(xs, objective, marker="o", linewidth=1.2, markersize=3)
            axes[0].set_yscale("symlog", linthresh=10)
            axes[0].set_ylabel("abs gap symlog")
            axes[1].plot(xs, gap, marker="o", linewidth=1.2, markersize=3)
            axes[1].set_yscale("symlog", linthresh=10)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set_ylabel("signed gap symlog")
            axes[2].plot(xs, grad_norm, marker="o", linewidth=1.2, markersize=3, label="grad norm")
            axes[2].plot(xs, update_norm * 1.0e5, marker=".", linewidth=1.0, label="update norm x1e5")
            axes[2].set_ylabel("grad/update")
            axes[2].set_xlabel("iteration")
            axes[2].legend()
            for ax_i in axes:
                ax_i.grid(True, alpha=0.25, which="both")
                if np.any(rejected):
                    ax_i.scatter(xs[rejected], np.full(np.sum(rejected), ax_i.get_ylim()[1]), marker="x", color="red")
            fig.suptitle("Relative Entropy Training - Symlog View")
            fig.tight_layout()
            fig.savefig(symlog_path, dpi=160)
            plt.close(fig)
            symlog_status = str(symlog_path)
    except Exception as exc:
        plot_status = f"plot_failed: {exc}"
        readable_status = f"plot_failed: {exc}"
        symlog_status = f"plot_failed: {exc}"

    return {
        "csv": str(csv_path),
        "log": str(log_path),
        "plot": plot_status,
        "plot_readable": readable_status,
        "plot_symlog": symlog_status,
    }


class RelativeEntropyTrainer:
    """Small RE fine-tuning loop that updates only the ML parameter subtree."""

    def __init__(
        self,
        params: Dict[str, Any],
        reference_data: Dict[str, Any],
        sampler: Any,
        energy_fn,
        optimizer: optax.GradientTransformation,
        config: RelativeEntropyConfig,
        seed: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        initial_states: Optional[Dict[str, Any]] = None,
    ):
        self.params = params
        self.best_params = params
        self.reference_data = {
            "R": jnp.asarray(reference_data["R"], dtype=jnp.float32),
            "mask": jnp.asarray(reference_data["mask"], dtype=jnp.float32),
            "species": jnp.asarray(reference_data["species"], dtype=jnp.int32),
        }
        self.sampler = sampler
        self.energy_fn = energy_fn
        self.optimizer = optimizer
        self.config = config
        self.rng_key = jax.random.PRNGKey(int(seed) + int(config.seed_offset))
        self.opt_state = optimizer.init(self.params["ml"])
        self._relative_entropy_gradient = jax.jit(
            relative_entropy_gradient,
            static_argnames=("energy_fn", "gradient_batch_size"),
        )
        self._barrier_penalty_gradient = jax.jit(
            barrier_penalty_gradient,
            static_argnames=("energy_fn",),
        )
        self.history = []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.initial_states = None
        if initial_states is not None:
            self.initial_states = {
                "R": jnp.asarray(initial_states["R"], dtype=jnp.float32),
                "mask": jnp.asarray(initial_states["mask"], dtype=jnp.float32),
                "species": jnp.asarray(initial_states["species"], dtype=jnp.int32),
            }
            expected = int(config.model_start_count)
            if int(self.initial_states["R"].shape[0]) != expected:
                raise ValueError(
                    f"RE initial_states must contain {expected} replicas, got "
                    f"{self.initial_states['R'].shape[0]}."
                )
        self.chain_state = None
        self._initial_states_used = False
        # Live progress artifacts. The canonical history CSV/plots are only written after
        # train() returns, which gave zero visibility during multi-hour runs; these are
        # appended every iteration so a run can be monitored (and diagnosed if it dies).
        self._live_csv = (
            self.checkpoint_dir.parent / "relative_entropy_history_live.csv"
            if self.checkpoint_dir is not None else None
        )
        self._live_csv_fields = None

    def _sample_indices(self, count: int):
        n_ref = int(self.reference_data["R"].shape[0])
        if n_ref <= 0:
            raise ValueError("RE reference_data must contain at least one frame.")
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return jax.random.choice(subkey, n_ref, shape=(int(count),), replace=n_ref < int(count))

    def _reference_batch(self):
        idx = self._sample_indices(int(self.config.reference_batch_size))
        return {
            key: value[idx]
            for key, value in self.reference_data.items()
        }

    def _replica_starts(self):
        if self.config.persistent_chains and self.chain_state is not None:
            return self.chain_state, "persistent"
        if self.initial_states is not None and (
            self.config.persistent_chains or not self._initial_states_used
        ):
            self._initial_states_used = True
            return self.initial_states, "configured_initial"
        idx = self._sample_indices(int(self.config.model_start_count))
        return {
            key: value[idx]
            for key, value in self.reference_data.items()
        }, "reference_random"

    @staticmethod
    def _host_float_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in metrics.items():
            try:
                arr = jax.device_get(value)
                if getattr(arr, "shape", ()) == ():
                    out[key] = float(arr)
                else:
                    out[key] = arr
            except Exception:
                out[key] = value
        return out

    def train_step(self, iteration: int) -> Dict[str, Any]:
        rollout = self.config.rollout_for_iteration(iteration)
        if hasattr(self.sampler, "configure_rollout"):
            self.sampler.configure_rollout(rollout)
        elif self.config.rollout_schedule:
            raise TypeError(
                "Configured REM rollout_schedule requires a sampler with configure_rollout()."
            )
        if hasattr(self.sampler, "set_params"):
            self.sampler.set_params(self.params)
        ref = self._reference_batch()
        starts, start_source = self._replica_starts()
        self.rng_key, rollout_key = jax.random.split(self.rng_key)
        model_samples = self.sampler.run(
            starts["R"], starts["mask"], starts["species"], rollout_key
        )
        diagnostics = dict(model_samples.get("diagnostics", {}))
        should_diagnose = (
            int(self.config.diagnostics_interval) > 0
            and int(iteration) % int(self.config.diagnostics_interval) == 0
        )
        if should_diagnose and hasattr(self.sampler, "diagnostics_for_samples"):
            diagnostics.update(
                self.sampler.diagnostics_for_samples(
                    model_samples["R"], model_samples["mask"], model_samples["species"]
                )
            )
        if self.config.chirality_diagnostics_enabled:
            for prefix, coordinates in (
                ("reference_chirality_", ref["R"]),
                ("model_chirality_", model_samples["R"]),
            ):
                population = chirality_population_metrics(
                    coordinates,
                    self.config.chirality_diagnostics_indices,
                    self.config.chirality_diagnostics_planar_threshold,
                )
                diagnostics.update(
                    {
                        prefix + key: float(jax.device_get(value))
                        for key, value in population.items()
                    }
                )
        rejected = False
        if self.config.reject_on_instability and is_unstable(
            diagnostics,
            max_force=float(self.config.max_force),
            min_pair_distance=float(self.config.min_pair_distance),
        ):
            rejected = True
            metrics = {
                "iteration": int(iteration),
                "rejected": True,
                "chain_start_source": start_source,
                "persistent_chain_advanced": False,
                **diagnostics,
                "rollout_steps": int(rollout.steps_per_iteration),
                "rollout_burn_in_steps": int(rollout.burn_in_steps),
                "rollout_sample_stride": int(rollout.sample_stride),
            }
            self.history.append(metrics)
            return metrics

        if self.config.persistent_chains:
            if "final_R" not in model_samples:
                raise ValueError(
                    "Persistent REM chains require sampler.run() to return final_R."
                )
            self.chain_state = {
                "R": jax.lax.stop_gradient(jnp.asarray(model_samples["final_R"])),
                "mask": starts["mask"],
                "species": starts["species"],
            }

        grad, grad_metrics = self._relative_entropy_gradient(
            self.params["ml"],
            ref["R"],
            ref["mask"],
            ref["species"],
            model_samples["R"],
            model_samples["mask"],
            model_samples["species"],
            self.energy_fn,
            beta=self.config.beta,
            gradient_batch_size=int(self.config.gradient_batch_size),
        )
        penalty_metrics = {}
        if self.config.barrier_penalty_enabled and self.config.barrier_penalty_weight > 0.0:
            w = barrier_penalty_weights(
                model_samples["R"],
                self.config.barrier_penalty_indices,
                self.config.barrier_penalty_v0,
            )
            pen_grad = self._barrier_penalty_gradient(
                self.params["ml"],
                model_samples["R"],
                model_samples["mask"],
                model_samples["species"],
                self.energy_fn,
                w,
                beta=self.config.beta,
                weight_scale=float(self.config.barrier_penalty_weight),
            )
            grad = jax.tree_util.tree_map(lambda a, b: a + b, grad, pen_grad)
            abs_v = jnp.abs(signed_volume(model_samples["R"], self.config.barrier_penalty_indices))
            penalty_metrics = {
                "barrier_penalty_mean": float(jax.device_get(jnp.mean(w))),
                "barrier_frac_below_v0": float(
                    jax.device_get(jnp.mean((abs_v < float(self.config.barrier_penalty_v0)).astype(jnp.float32)))
                ),
                "min_abs_signed_volume": float(jax.device_get(jnp.min(abs_v))),
            }
        scaled_grad = jax.tree_util.tree_map(
            lambda x: float(self.config.optimizer_gradient_scale) * x,
            grad,
        )
        if self.config.trainable_param_substring:
            scaled_grad = mask_parameter_tree_by_substring(
                scaled_grad, self.config.trainable_param_substring
            )
        trainable_grad_norm = optax.tree.norm(scaled_grad)
        updates, new_opt_state = self.optimizer.update(
            scaled_grad, self.opt_state, self.params["ml"]
        )
        new_params = apply_ml_updates(self.params, updates)
        update_norm = optax.tree.norm(updates)
        param_norm = optax.tree.norm(new_params["ml"])
        trainable_params = new_params["ml"]
        if self.config.trainable_param_substring:
            trainable_params = mask_parameter_tree_by_substring(
                trainable_params, self.config.trainable_param_substring
            )
        trainable_param_norm = optax.tree.norm(trainable_params)

        self.params = new_params
        self.best_params = new_params
        self.opt_state = new_opt_state
        if hasattr(self.sampler, "set_params"):
            self.sampler.set_params(self.params)
        metrics = self._host_float_dict(grad_metrics)
        metrics["abs_re_energy_gap"] = abs(float(metrics["re_energy_gap"]))
        metrics["objective"] = metrics["abs_re_energy_gap"]
        metrics.update(
            {
                "iteration": int(iteration),
                "rejected": rejected,
                "chain_start_source": start_source,
                "persistent_chain_advanced": bool(self.config.persistent_chains),
                "update_norm": float(jax.device_get(update_norm)),
                "ml_param_norm": float(jax.device_get(param_norm)),
                "trainable_grad_norm": float(jax.device_get(trainable_grad_norm)),
                "trainable_param_norm": float(jax.device_get(trainable_param_norm)),
                **diagnostics,
                "rollout_steps": int(rollout.steps_per_iteration),
                "rollout_burn_in_steps": int(rollout.burn_in_steps),
                "rollout_sample_stride": int(rollout.sample_stride),
                **penalty_metrics,
            }
        )
        self.history.append(metrics)
        return metrics

    _LIVE_FIELDS = [
        "iteration", "rejected", "objective", "abs_re_energy_gap", "re_energy_gap",
        "ref_energy_mean", "model_energy_mean", "grad_norm", "trainable_grad_norm",
        "update_norm", "ml_param_norm", "trainable_param_norm", "has_nan_or_inf",
        "max_force", "min_pair_distance",
        "n_samples", "chain_start_source", "persistent_chain_advanced",
        "rollout_steps", "rollout_burn_in_steps", "rollout_sample_stride",
        "barrier_penalty_mean", "barrier_frac_below_v0", "min_abs_signed_volume",
        "reference_chirality_signed_volume_mean",
        "reference_chirality_signed_volume_min",
        "reference_chirality_signed_volume_max",
        "reference_chirality_fraction_positive",
        "reference_chirality_fraction_negative",
        "reference_chirality_fraction_near_planar",
        "model_chirality_signed_volume_mean",
        "model_chirality_signed_volume_min",
        "model_chirality_signed_volume_max",
        "model_chirality_fraction_positive",
        "model_chirality_fraction_negative",
        "model_chirality_fraction_near_planar",
    ]

    def _log_iteration(self, metrics: Dict[str, Any], n_rejected: int, n_done: int) -> None:
        """(A) One flushed stdout line per iteration so `tail -f` shows live progress."""
        parts = [
            "it=%4d/%d" % (int(metrics.get("iteration", -1)), int(self.config.iterations)),
            "rej=%s" % ("T" if metrics.get("rejected") else "F"),
            "rejrate=%.2f" % (n_rejected / max(n_done, 1)),
        ]
        for key, fmt in (("objective", "obj=%.4g"), ("grad_norm", "gnorm=%.3g"),
                         ("trainable_grad_norm", "tgnorm=%.3g"),
                         ("update_norm", "dupd=%.3g"), ("max_force", "maxF=%.3g"),
                         ("trainable_param_norm", "tpnorm=%.3g"),
                         ("min_pair_distance", "minpair=%.3f"), ("rollout_steps", "roll=%d"),
                         ("min_abs_signed_volume", "minV=%.2f"),
                         ("barrier_frac_below_v0", "fbar=%.4f"),
                         ("model_chirality_fraction_negative", "mirror=%.4f"),
                         ("model_chirality_fraction_near_planar", "planar=%.4f")):
            value = metrics.get(key)
            if value is not None:
                try:
                    parts.append(fmt % float(value))
                except (TypeError, ValueError):
                    pass
        training_logger.info("[RE] " + "  ".join(parts))
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def _append_live_row(self, metrics: Dict[str, Any]) -> None:
        """(B) Append this iteration to a live CSV instead of waiting for train() to end."""
        if self._live_csv is None:
            return
        try:
            self._live_csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = self._live_csv_fields is None
            if write_header:
                extra = [k for k in metrics if k not in self._LIVE_FIELDS]
                self._live_csv_fields = list(self._LIVE_FIELDS) + sorted(extra)
            with self._live_csv.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._live_csv_fields,
                                        extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow({k: metrics.get(k, "") for k in self._live_csv_fields})
        except OSError as exc:  # never let logging kill a training run
            training_logger.warning("[RE] live CSV append failed: %s", exc)

    def train(self) -> Dict[str, Any]:
        last = {}
        n_rejected = 0
        if self._live_csv is not None and self._live_csv.exists():
            self._live_csv.unlink()  # fresh file per run
        for iteration in range(int(self.config.iterations)):
            last = self.train_step(iteration)
            n_rejected += bool(last.get("rejected"))
            self._log_iteration(last, n_rejected, iteration + 1)
            self._append_live_row(last)
            if (
                self.checkpoint_dir is not None
                and int(self.config.checkpoint_freq) > 0
                and (iteration + 1) % int(self.config.checkpoint_freq) == 0
            ):
                self.save_checkpoint(self.checkpoint_dir / f"relative_entropy_iter{iteration + 1:06d}.pkl")
        return {"final": last, "history": list(self.history)}

    def save_checkpoint(self, output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.params,
            "best_params": self.best_params,
            "optimizer_state": self.opt_state,
            "rng_key": self.rng_key,
            "chain_state": self.chain_state,
            "metadata": {
                "relative_entropy": True,
                "prior_frozen": True,
                "history": list(self.history),
                **(metadata or {}),
            },
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
