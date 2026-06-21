"""Relative-entropy fine-tuning utilities and trainer loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import csv
import pickle

import jax
import jax.numpy as jnp
import optax
from jax_md import simulate

from mc.config import HMCConfig, MALAConfig


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
    start_frame_mode: str = "reference_random"
    checkpoint_freq: int = 10
    max_force: float = 1.0e4
    min_pair_distance: float = 1.5
    reject_on_instability: bool = True
    seed_offset: int = 9117
    output_dir: Optional[str] = None
    optimizer_gradient_scale: float = -1.0
    gradient_batch_size: int = 0
    diagnostics_interval: int = 1
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
        start_frame_mode=str(cfg.get("start_frame_mode", "reference_random")).strip().lower(),
        checkpoint_freq=_nonnegative_int("checkpoint_freq", cfg.get("checkpoint_freq", 10)),
        max_force=_positive_float("max_force", cfg.get("max_force", 1.0e4)),
        min_pair_distance=_positive_float(
            "min_pair_distance", cfg.get("min_pair_distance", 1.5)
        ),
        reject_on_instability=bool(cfg.get("reject_on_instability", True)),
        seed_offset=int(cfg.get("seed_offset", 9117)),
        output_dir=str(cfg["output_dir"]) if cfg.get("output_dir") is not None else None,
        optimizer_gradient_scale=float(cfg.get("optimizer_gradient_scale", -1.0)),
        gradient_batch_size=_nonnegative_int(
            "gradient_batch_size", cfg.get("gradient_batch_size", 0)
        ),
        diagnostics_interval=_positive_int(
            "diagnostics_interval", cfg.get("diagnostics_interval", 1)
        ),
        mc_mala=None,
    )

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
    if parsed.start_frame_mode != "reference_random":
        raise ValueError(
            "training.relative_entropy.start_frame_mode currently supports only "
            f"'reference_random', got {parsed.start_frame_mode!r}."
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
        (_, retained, _), _ = jax.lax.scan(
            scan_step,
            (state, retained0, jnp.asarray(0, dtype=jnp.int32)),
            steps,
        )
        return retained

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
        per_replica = jax.vmap(self._run_single)(R0, mask, species, keys)
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
        self.history = []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

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
        idx = self._sample_indices(int(self.config.model_start_count))
        return {
            key: value[idx]
            for key, value in self.reference_data.items()
        }

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
        if hasattr(self.sampler, "set_params"):
            self.sampler.set_params(self.params)
        ref = self._reference_batch()
        starts = self._replica_starts()
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
                **diagnostics,
            }
            self.history.append(metrics)
            return metrics

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
        scaled_grad = jax.tree_util.tree_map(
            lambda x: float(self.config.optimizer_gradient_scale) * x,
            grad,
        )
        updates, new_opt_state = self.optimizer.update(
            scaled_grad, self.opt_state, self.params["ml"]
        )
        new_params = apply_ml_updates(self.params, updates)
        update_norm = optax.tree.norm(updates)
        param_norm = optax.tree.norm(new_params["ml"])

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
                "update_norm": float(jax.device_get(update_norm)),
                "ml_param_norm": float(jax.device_get(param_norm)),
                **diagnostics,
            }
        )
        self.history.append(metrics)
        return metrics

    def train(self) -> Dict[str, Any]:
        last = {}
        for iteration in range(int(self.config.iterations)):
            last = self.train_step(iteration)
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
            "metadata": {
                "relative_entropy": True,
                "prior_frozen": True,
                "history": list(self.history),
                **(metadata or {}),
            },
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
