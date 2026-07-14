"""Hessian-vector product matching helpers for Chemtrain force matching."""

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp


def hvp_enabled(config) -> bool:
    return bool(config.get("training", "hvp", "enabled", default=False))


def hvp_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "hvp", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lambda": float(cfg.get("lambda", cfg.get("lambda_hvp", 0.01))),
        "mode": str(cfg.get("mode", "projected_aa_term1")),
        "target_key": str(cfg.get("target_key", "HVP")),
        "probe_key": str(cfg.get("probe_key", "hvp_probe")),
        "loss_mask_key": str(cfg.get("loss_mask_key", "hvp_loss_mask")),
        "energy_template": str(cfg.get("energy_template", "auto")),
        "require_targets": bool(cfg.get("require_targets", True)),
        "stop_gradient_target": bool(cfg.get("stop_gradient_target", True)),
    }


def hvp_error(predictions, targets, weights=None):
    """Mean squared HVP error over valid components."""
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets, dtype=predictions.dtype)
    sq = jnp.square(predictions - targets)
    if weights is None:
        return jnp.mean(sq)
    weights = jnp.asarray(weights, dtype=sq.dtype)
    if weights.ndim == sq.ndim - 1:
        weights = weights[..., None]
    weights = jnp.broadcast_to(weights, sq.shape)
    return jnp.sum(sq * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def make_hvp_quantity(
    energy_fn_template: Callable[[Any], Callable],
    probe_key: str = "hvp_probe",
) -> Callable:
    """Return a Chemtrain quantity computing model HVPs for batch probes."""

    def hvp_quantity(
        state,
        neighbor=None,
        energy_params=None,
        mask=None,
        species=None,
        segment_id=None,
        **kwargs,
    ):
        if probe_key not in kwargs:
            raise ValueError(f"HVP quantity requires {probe_key!r} in the batch.")
        if mask is None or species is None:
            raise ValueError("HVP quantity requires mask and species in the batch.")

        probes = jnp.asarray(kwargs[probe_key], dtype=state.position.dtype)
        energy_fn = energy_fn_template(energy_params)

        def energy_of_R(R_eval):
            return energy_fn(
                R_eval,
                neighbor=neighbor,
                mask=mask,
                species=species,
                segment_id=segment_id,
            )

        def force_dot_probe(R_eval, probe):
            forces = -jax.grad(energy_of_R)(R_eval)
            return jnp.sum(forces * probe)

        hvp = -jax.vmap(jax.grad(force_dot_probe), in_axes=(None, 0))(
            state.position,
            probes,
        )
        return jnp.where(mask[None, :, None] > 0, hvp, 0.0)

    return hvp_quantity
