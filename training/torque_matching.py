"""Torque-matching helpers for Chemtrain force matching.

Torque on a rigid bead is the derivative of the total energy with respect to
an infinitesimal rotation of that bead's body frame.  Given orientation matrix
O_i ∈ SO(3) for bead i, a small rotation δω around axis n̂ perturbs the frame
as O_i → (I + skew(δω)) O_i.  The torque is then:

    τ_i = -∂E / ∂(δω) |_{δω=0}

This is computed here by JAX autodiff through the energy function — exactly the
same idea as computing forces via -∂E/∂R, just with rotation parameters
instead of Cartesian coordinates.
"""

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp

from utils.logging import training_logger


def torque_enabled(config) -> bool:
    return bool(config.get("training", "torque_matching", "enabled", default=False))


def torque_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "torque_matching", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    lambda_val = float(cfg.get("lambda", cfg.get("lambda_torque", 1.0)))
    if lambda_val < 0.0:
        raise ValueError(
            f"training.torque_matching.lambda must be >= 0, got {lambda_val}."
        )
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lambda": lambda_val,
        "uncertainty_weighting": bool(cfg.get("uncertainty_weighting", False)),
        "orientation_key": str(cfg.get("orientation_key", "O")),
        "loss_mask_key": str(cfg.get("loss_mask_key", "")),
        "ref_key": str(cfg.get("ref_key", "T_ref")),
    }


def torque_error(predictions, targets, weights=None):
    """Mean squared torque error, optionally masked per bead."""
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


def torque_mse(predictions, targets, weights=None):
    """Average squared torque error over valid torque components only."""
    squared_differences = jnp.square(targets - predictions)
    if weights is None:
        return jnp.mean(squared_differences)

    weights = jnp.asarray(weights, dtype=squared_differences.dtype)
    if weights.ndim == squared_differences.ndim - 1:
        weights = weights[..., None]
    try:
        weights = jnp.broadcast_to(weights, squared_differences.shape)
    except ValueError as exc:
        raise ValueError(
            "torque_loss_mask must match torque target shape after broadcasting. "
            f"Got weights shape {weights.shape} and torque shape {squared_differences.shape}."
        ) from exc
    numerator = jnp.sum(squared_differences * weights)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    return numerator / denominator


def make_log_sigma_quantity(key: str) -> Callable:
    """Return a quantity fn that reads log_sigma[key] from energy_params.

    When included in chemtrain's additional_targets, the returned scalar is
    placed in predictions["log_sigma_{key}"] with shape (batch_size,) after
    vmap — all elements identical since energy_params is not batched.
    """
    pred_key = f"log_sigma_{key}"
    def log_sigma_quantity(state, neighbor=None, energy_params=None, **kwargs):
        return energy_params["log_sigma"][key]
    log_sigma_quantity.__name__ = pred_key
    return log_sigma_quantity


def init_log_sigma(lambda_val: float) -> "jnp.ndarray":
    """Compute initial log_sigma so the effective weight matches lambda_val.

    Kendall weighting: 0.5 * exp(-2*log_sigma) * loss + log_sigma
    Setting 0.5 * exp(-2*log_sigma) = lambda gives:
        log_sigma = -0.5 * log(2 * lambda)
    """
    import jax.numpy as _jnp
    return _jnp.array(-0.5 * _jnp.log(2.0 * lambda_val), dtype=_jnp.float32)


def wrap_uncertainty_loss_fn(base_loss_fn: Callable, loss_mask_key: str) -> Callable:
    """Wrap a chemtrain loss_fn to add uncertainty-weighted torque loss.

    The base_loss_fn must have been created WITHOUT "T" in gammas so that
    torque is not already included in the base loss.  This wrapper adds:

        L_T = 0.5 * exp(-2 * log_sigma_T) * mse_T + log_sigma_T

    where log_sigma_T comes from predictions["log_sigma_T"], a learnable
    scalar stored in energy_params["log_sigma"]["T"].

    Reference: Kendall, Gal & Cipolla, NeurIPS 2018.
    """
    def uncertainty_loss_fn(predictions, targets):
        base_loss, errors = base_loss_fn(predictions, targets)

        weights = targets.get(loss_mask_key) if loss_mask_key else None
        mse_T = torque_mse(predictions["T"], targets["T"], weights=weights)

        # predictions["log_sigma_T"] has shape (batch_size,); all values are
        # the same scalar since energy_params is not batched across the vmap.
        log_sigma_T = jnp.mean(predictions["log_sigma_T"])
        torque_loss = 0.5 * jnp.exp(-2.0 * log_sigma_T) * mse_T + log_sigma_T

        errors["T"] = mse_T
        return base_loss + torque_loss, errors

    return uncertainty_loss_fn


def make_torque_quantity(model, orientation_key: str = "O", ref_key: str = "T_ref") -> Callable:
    """Return a Chemtrain quantity that predicts torques via rotational autodiff.

    For each bead i with orientation matrix O_i, we define a perturbed frame:

        O_pert_i = (I + skew(ω_i)) O_i  ≈  R(ω_i) O_i   (first order in ω)

    where skew(ω) v = ω × v.  The torque on bead i is:

        τ_i = -∂E(O_pert) / ∂ω_i |_{ω=0}

    JAX computes this gradient exactly through the linear map ω → O_pert.
    The result has the same shape as the target T: (N_beads, 3).
    """
    def torque_quantity(
        state,
        neighbor=None,
        energy_params=None,
        mask=None,
        species=None,
        segment_id=None,
        **kwargs,
    ):
        O = kwargs.get(orientation_key)
        if O is None:
            raise ValueError(
                f"Torque quantity requires '{orientation_key}' in the batch. "
                "Make sure the dataset contains orientation matrices and that "
                f"orientation_key='{orientation_key}' matches the field name."
            )
        if mask is None or species is None:
            raise ValueError("Torque quantity requires mask and species in the batch.")

        energy_fn = model.energy_fn_template(energy_params)
        O = jnp.asarray(O, dtype=state.position.dtype)

        def energy_of_rotation(omega):
            # omega: (N, 3) — small rotation vectors, one per bead.
            # O_pert[n, :, c] = O[n, :, c] + omega[n] × O[n, :, c]
            #                  = (I + skew(omega[n])) @ O[n]  column-wise
            delta_O = jnp.cross(
                omega[:, None, :],          # (N, 1, 3)
                O.transpose(0, 2, 1),       # (N, 3_col, 3_row)
            ).transpose(0, 2, 1)            # → (N, 3_row, 3_col)
            O_pert = O + delta_O
            return energy_fn(
                state.position,
                neighbor,
                mask=mask,
                species=species,
                segment_id=segment_id,
                **{k: v for k, v in kwargs.items() if k != orientation_key},
                **{orientation_key: O_pert},
            )

        omega_zero = jnp.zeros((O.shape[0], 3), dtype=O.dtype)
        torques = -jax.grad(energy_of_rotation)(omega_zero)
        # def energy_of_rotation(_O):
        #  #_O: (N, 3, 3) — perturbed orientation matrices, one per bead.
        #  return energy_fn(
            #  state.position,
            #  neighbor,
            #  mask=mask,
            #  species=species,
            #  segment_id=segment_id,
            #  **{k: v for k, v in kwargs.items() if k != orientation_key},
            #  **{orientation_key: _O},
        #  )
        # dE_dO = -jax.grad(energy_of_rotation)(O)  # (N, 3, 3)
        # torques = jnp.cross(O[:,:,0], dE_dO[:,:,0]) + jnp.cross(O[:,:,1], dE_dO[:,:,1]) + jnp.cross(O[:,:,2], dE_dO[:,:,2])   # (N, 3)
        # torques = jnp.sum(jnp.cross(O, dE_dO), axis=2)
        torques = jnp.where(mask[:, None] > 0, torques, 0.0)

        return torques

    return torque_quantity
