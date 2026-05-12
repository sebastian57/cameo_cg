"""Denoising score-matching helpers for Chemtrain force matching."""

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

from utils.logging import training_logger


def dsm_enabled(config) -> bool:
    return bool(config.get("training", "dsm", "enabled", default=False))


def dsm_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "dsm", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    refresh_interval_steps = int(cfg.get("refresh_interval_steps", 0))
    if refresh_interval_steps < 0:
        raise ValueError(
            "training.dsm.refresh_interval_steps must be >= 0, "
            f"got {refresh_interval_steps}."
        )
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lambda": float(cfg.get("lambda", cfg.get("lambda_dsm", 1.0))),
        "sigma_min": float(cfg.get("sigma_min", 0.05)),
        "sigma_max": float(cfg.get("sigma_max", 2.0)),
        "kT": float(cfg.get("kT", 0.636)),
        "seed_offset": int(cfg.get("seed_offset", 1729)),
        "refresh_interval_steps": refresh_interval_steps,
    }


def add_dsm_noise_fields(split: Dict[str, np.ndarray], config, seed: int) -> Dict[str, np.ndarray]:
    """Attach fixed DSM noise targets to a split for Chemtrain additional_targets."""
    cfg = dsm_config(config)
    if not cfg["enabled"]:
        return split

    sigma_min = float(cfg["sigma_min"])
    sigma_max = float(cfg["sigma_max"])
    if sigma_min <= 0.0 or sigma_max <= 0.0 or sigma_max < sigma_min:
        raise ValueError(
            "training.dsm requires 0 < sigma_min <= sigma_max, got "
            f"sigma_min={sigma_min}, sigma_max={sigma_max}."
        )

    out = dict(split)
    R = np.asarray(out["R"], dtype=np.float32)
    mask = np.asarray(out["mask"], dtype=np.float32)
    rng = np.random.RandomState(int(seed) + int(cfg["seed_offset"]))

    log_sigma = rng.uniform(np.log(sigma_min), np.log(sigma_max), size=(R.shape[0],))
    sigma = np.exp(log_sigma).astype(np.float32)
    eps = rng.normal(size=R.shape).astype(np.float32)

    mask3 = mask[..., None]
    n_valid = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0).astype(np.float32)
    eps = eps * mask3
    eps_mean = np.sum(eps, axis=1, keepdims=True) / n_valid[:, :, None]
    eps = (eps - eps_mean) * mask3

    out["DSM"] = eps.astype(np.float32)
    out["dsm_eps"] = out["DSM"]
    out["dsm_sigma"] = sigma
    out["dsm_loss_mask"] = mask.astype(np.float32)
    training_logger.info(
        "[DSM] Attached fixed noise fields: n=%d sigma_min=%.4g sigma_max=%.4g kT=%.4g lambda=%.4g",
        int(R.shape[0]),
        sigma_min,
        sigma_max,
        float(cfg["kT"]),
        float(cfg["lambda"]),
    )
    return out


def dsm_error(predictions, targets, weights=None):
    """Mean squared DSM error over valid components."""
    sq = jnp.square(predictions - targets)
    if weights is None:
        return jnp.mean(sq)
    weights = jnp.asarray(weights, dtype=sq.dtype)
    if weights.ndim == sq.ndim - 1:
        weights = weights[..., None]
    weights = jnp.broadcast_to(weights, sq.shape)
    return jnp.sum(sq * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def make_dsm_quantity(
    energy_fn_template: Callable[[Any], Callable],
    kT: float,
) -> Callable:
    """Return a Chemtrain quantity computing eps_pred = -sigma * F(R+sigma eps) / kT."""
    kT_value = float(kT)

    def dsm_quantity(
        state,
        neighbor=None,
        energy_params=None,
        dsm_sigma=None,
        dsm_eps=None,
        mask=None,
        species=None,
        segment_id=None,
        **kwargs,
    ):
        if dsm_sigma is None or dsm_eps is None:
            raise ValueError("DSM quantity requires dsm_sigma and dsm_eps in the batch.")
        if mask is None or species is None:
            raise ValueError("DSM quantity requires mask and species in the batch.")

        sigma = jnp.asarray(dsm_sigma, dtype=state.position.dtype)
        eps = jnp.asarray(dsm_eps, dtype=state.position.dtype)
        R_tilde = state.position + sigma * eps
        energy_fn = energy_fn_template(energy_params)

        def energy_of_R(R_eval):
            return energy_fn(
                R_eval,
                neighbor=neighbor,
                mask=mask,
                species=species,
                segment_id=segment_id,
            )

        F_pred = -jax.grad(energy_of_R)(R_tilde)
        eps_pred = -sigma * F_pred / jnp.asarray(kT_value, dtype=state.position.dtype)
        return jnp.where(mask[:, None] > 0, eps_pred, 0.0)

    return dsm_quantity
