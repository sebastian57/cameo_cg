#!/usr/bin/env python3
"""
Standalone LBFGS fitting for new typed prior terms.

This script is intentionally separate from the main training pipeline so the
new prior terms can be tested in isolation:
  - Debye-Huckel (DH)
  - typed stickiness
  - salt-bridge correction

Workflow:
  1. Load regular training config YAML.
  2. Load and preprocess dataset (same path/splitting conventions as train.py).
  3. Build PriorEnergy directly.
  4. Randomly initialize NEW prior params only.
  5. Run LBFGS force matching on NEW params only (old params frozen).
  6. Report improvement in prior force prediction.
  7. Save full prior params (old + fitted new) to a .pkl.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Allow direct execution from data_prep/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.prior_energy import PriorEnergy
from models.topology import TopologyBuilder
from utils.logging import data_logger, training_logger


NEW_PRIOR_PARAM_KEYS = (
    "k_DH",
    "lambda_D",
    "dh_w_by_sep",
    "stick_r0",
    "stick_sigma",
    "stick_s_free",
    "salt_delta",
    "salt_r0",
    "salt_sigma",
)

OBJECTIVE_CHOICES = (
    "full_on_fref",
    "new_on_fref",
    "new_on_residual",
)

PREDICTION_MODE_BY_OBJECTIVE = {
    "full_on_fref": "full",
    "new_on_fref": "new_only",
    "new_on_residual": "new_only",
}

TERM_NAMES = ("dh", "stickiness", "salt_bridge")
KEYS_BY_TERM = {
    "dh": ("k_DH", "lambda_D", "dh_w_by_sep"),
    "stickiness": ("stick_r0", "stick_sigma", "stick_s_free"),
    "salt_bridge": ("salt_delta", "salt_r0", "salt_sigma"),
}


def _resolve_data_path(config: ConfigManager) -> Path:
    data_path = Path(config.get_data_path())
    if data_path.is_absolute():
        return data_path
    return config.config_path.parent / data_path


def _force_enable_new_terms_for_fit(
    config: ConfigManager,
    active_terms: Iterable[str],
    weight_overrides: Optional[Dict[str, float]] = None,
) -> None:
    """
    Configure typed-prior term enables/weights for standalone fitting.
    """
    active = set(active_terms)
    unknown = [t for t in active if t not in TERM_NAMES]
    if unknown:
        raise ValueError(f"Unknown active term(s): {unknown}")
    weight_overrides = weight_overrides or {}

    priors_cfg = config._config.setdefault("model", {}).setdefault("priors", {})
    weights = priors_cfg.setdefault("weights", {})

    for term in TERM_NAMES:
        term_cfg = priors_cfg.setdefault(term, {})
        if term in active:
            if not bool(term_cfg.get("enabled", False)):
                training_logger.info(f"Enabling model.priors.{term}.enabled for standalone fitting")
                term_cfg["enabled"] = True

            if term in weight_overrides:
                weights[term] = float(weight_overrides[term])
                training_logger.info(
                    f"Setting model.priors.weights.{term}: {weights[term]:.6g} (override)"
                )
            else:
                w0 = float(weights.get(term, 0.0))
                if w0 == 0.0:
                    training_logger.info(
                        f"Setting model.priors.weights.{term}: 0.0 -> 1.0 for standalone fitting"
                    )
                    weights[term] = 1.0
        else:
            term_cfg["enabled"] = False
            weights[term] = 0.0
            training_logger.info(f"Disabling model.priors.{term} (enabled=false, weight=0.0)")

    # Keep expected typed-source behavior explicit.
    aa_typing = priors_cfg.setdefault("aa_typing", {})
    aa_typing.setdefault("source", "dataset_map")
    aa_typing.setdefault("his_charge", 0.0)
    aa_typing.setdefault(
        "group_order",
        ["POSITIVE", "NEGATIVE", "POLAR_UNCHARGED", "NONPOLAR"],
    )
    aa_typing.setdefault("stickiness_reference_group", "POLAR_UNCHARGED")


def _apply_dh_overrides(
    config: ConfigManager,
    dh_k_override: Optional[int],
    dh_w_by_sep_override: Optional[Iterable[float]],
) -> None:
    priors_cfg = config._config.setdefault("model", {}).setdefault("priors", {})
    dh_cfg = priors_cfg.setdefault("dh", {})

    if dh_k_override is not None:
        dh_cfg["K"] = int(dh_k_override)
        training_logger.info(f"Setting model.priors.dh.K: {dh_cfg['K']}")

    if dh_w_by_sep_override is not None:
        w = [float(x) for x in dh_w_by_sep_override]
        if len(w) == 0:
            raise ValueError("--dh-w-by-sep must contain at least one value.")
        k = int(dh_cfg.get("K", 2))
        needed = k + 1
        if len(w) < needed:
            w = w + [w[-1]] * (needed - len(w))
            training_logger.info(
                f"Extended --dh-w-by-sep to length {needed} by repeating the last value."
            )
        elif len(w) > needed:
            w = w[:needed]
            training_logger.info(f"Truncated --dh-w-by-sep to length {needed} (K+1).")
        dh_cfg["w_by_sep"] = w
        training_logger.info(f"Setting model.priors.dh.w_by_sep: {w}")


def _prepare_dataset(
    config: ConfigManager,
    max_frames_override: Optional[int] = None,
    val_fraction_override: Optional[float] = None,
) -> Tuple[DatasetLoader, Dict[str, np.ndarray], Dict[str, jax.Array], Optional[Dict[str, jax.Array]]]:
    data_path = _resolve_data_path(config)
    max_frames = max_frames_override if max_frames_override is not None else config.get_max_frames()
    seed = config.get_seed()

    loader = DatasetLoader(str(data_path), max_frames=max_frames, seed=seed)
    data_logger.info(loader.summary())

    cutoff = config.get_cutoff()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = np.asarray(
        preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, shift),
        dtype=np.float32,
    )

    val_fraction = (
        float(val_fraction_override)
        if val_fraction_override is not None
        else float(config.get_val_fraction())
    )
    val_fraction = float(np.clip(val_fraction, 0.0, 0.95))

    n_total = int(dataset["R"].shape[0])
    n_train = int(np.round(n_total * (1.0 - val_fraction)))
    n_train = max(1, min(n_train, n_total))
    n_val = n_total - n_train

    train = {
        "R": jnp.asarray(dataset["R"][:n_train]),
        "F": jnp.asarray(dataset["F"][:n_train]),
        "mask": jnp.asarray(dataset["mask"][:n_train]),
        "species": jnp.asarray(dataset["species"][:n_train]),
    }
    val = None
    if n_val > 0:
        val = {
            "R": jnp.asarray(dataset["R"][n_train:]),
            "F": jnp.asarray(dataset["F"][n_train:]),
            "mask": jnp.asarray(dataset["mask"][n_train:]),
            "species": jnp.asarray(dataset["species"][n_train:]),
        }

    data_logger.info(
        f"[Split] Total={n_total}, train={n_train}, val={n_val}, val_fraction={val_fraction:.3f}"
    )
    return loader, dataset, train, val


def _sample_frame_indices(n_frames: int, n_keep: Optional[int], seed: int) -> Optional[np.ndarray]:
    if n_keep is None or n_keep <= 0 or n_keep >= n_frames:
        return None
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_frames, size=n_keep, replace=False))


def _subset_frames(data: Dict[str, jax.Array], idx: Optional[np.ndarray]) -> Dict[str, jax.Array]:
    if idx is None:
        return data
    return {k: v[idx] for k, v in data.items()}


def _uniform_like(key: jax.Array, ref: jax.Array, lo: float, hi: float) -> jax.Array:
    return jax.random.uniform(key, shape=ref.shape, dtype=ref.dtype, minval=lo, maxval=hi)


def _normal_like(key: jax.Array, ref: jax.Array, std: float) -> jax.Array:
    return std * jax.random.normal(key, shape=ref.shape, dtype=ref.dtype)


def _randomize_new_params(
    params: Dict[str, jax.Array],
    seed: int,
    keys_to_randomize: Optional[Iterable[str]] = None,
) -> Dict[str, jax.Array]:
    """
    Randomly initialize only NEW prior parameters.
    """
    missing = [k for k in NEW_PRIOR_PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"Missing expected new prior params in prior.params: {missing}")

    keyset = set(keys_to_randomize) if keys_to_randomize is not None else set(NEW_PRIOR_PARAM_KEYS)
    out = dict(params)
    rng = jax.random.PRNGKey(seed)
    keys = iter(jax.random.split(rng, 16))

    if "k_DH" in keyset:
        out["k_DH"] = _uniform_like(next(keys), out["k_DH"], 0.05, 2.5)
    if "lambda_D" in keyset:
        out["lambda_D"] = _uniform_like(next(keys), out["lambda_D"], 4.0, 14.0)
    if "dh_w_by_sep" in keyset:
        w = _uniform_like(next(keys), out["dh_w_by_sep"], 0.0, 1.5)
        # Keep sep-0 weight fixed at 0 if present.
        if w.ndim == 1 and w.shape[0] > 0:
            w = w.at[0].set(jnp.array(0.0, dtype=w.dtype))
        out["dh_w_by_sep"] = w

    if "stick_r0" in keyset:
        out["stick_r0"] = _uniform_like(next(keys), out["stick_r0"], 3.2, 4.8)
    if "stick_sigma" in keyset:
        out["stick_sigma"] = _uniform_like(next(keys), out["stick_sigma"], 0.20, 0.90)
    if "stick_s_free" in keyset:
        out["stick_s_free"] = _normal_like(next(keys), out["stick_s_free"], std=0.75)

    if "salt_delta" in keyset:
        out["salt_delta"] = _uniform_like(next(keys), out["salt_delta"], -2.0, -0.05)
    if "salt_r0" in keyset:
        out["salt_r0"] = _uniform_like(next(keys), out["salt_r0"], 3.2, 4.8)
    if "salt_sigma" in keyset:
        out["salt_sigma"] = _uniform_like(next(keys), out["salt_sigma"], 0.15, 0.80)
    return out


def _merge_params(frozen_params: Dict[str, jax.Array], fit_params: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    merged = dict(frozen_params)
    merged.update(fit_params)
    return merged


def _energy_mode(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    R: jax.Array,
    mask: jax.Array,
    species: jax.Array,
    mode: str,
) -> jax.Array:
    p = params
    if mode == "full":
        return prior.compute_total_energy_from_params(params=p, R=R, mask=mask, species=species)

    E_old = (
        prior.weights["bond"] * prior.compute_bond_energy(R, mask, params=p)
        + prior.weights["angle"] * prior.compute_angle_energy(R, mask, species=species, params=p)
        + prior.weights["repulsive"] * prior.compute_repulsive_energy(R, mask, params=p)
        + prior.weights["dihedral"] * prior.compute_dihedral_energy(R, mask, params=p)
        + prior.weights.get("excluded_volume", 1.0) * prior.compute_excluded_volume_energy(R, mask, params=p)
    )
    if mode == "old_only":
        return E_old

    E_new = (
        prior.weights.get("dh", 0.0) * prior.compute_dh_energy(R, mask, species=species, params=p)
        + prior.weights.get("stickiness", 0.0)
        * prior.compute_stickiness_energy(R, mask, species=species, params=p)
        + prior.weights.get("salt_bridge", 0.0)
        * prior.compute_salt_bridge_energy(R, mask, species=species, params=p)
    )
    if mode == "new_only":
        return E_new
    raise ValueError(f"Unsupported energy mode: {mode}")


def _force_batch_from_mode(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
    mode: str,
) -> jax.Array:
    def prior_forces(single_R, single_mask, single_species):
        def energy_of_R(R_):
            return _energy_mode(prior, params, R_, single_mask, single_species, mode=mode)

        return -jax.grad(energy_of_R)(single_R)

    return jax.vmap(prior_forces)(data["R"], data["mask"], data["species"])


def _build_target_forces(
    objective: str,
    prior: PriorEnergy,
    params_reference: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
) -> jax.Array:
    if objective in ("full_on_fref", "new_on_fref"):
        return data["F"]
    if objective == "new_on_residual":
        F_old = _force_batch_from_mode(prior, params_reference, data, mode="old_only")
        return data["F"] - F_old
    raise ValueError(f"Unsupported objective: {objective}")


def _force_array_metrics(F_pred: jax.Array, F_target: jax.Array, mask: jax.Array) -> Dict[str, float]:
    m3 = mask[..., None]
    pred = F_pred * m3
    target = F_target * m3
    diff = pred - target

    denom = jnp.maximum(jnp.sum(m3), 1.0)
    mse = jnp.sum(diff * diff) / denom
    rmse = jnp.sqrt(mse)
    mae = jnp.sum(jnp.abs(diff)) / denom

    p = pred.reshape(-1)
    t = target.reshape(-1)
    dot = jnp.sum(p * t)
    pnorm = jnp.linalg.norm(p)
    tnorm = jnp.linalg.norm(t)
    cosine = dot / jnp.maximum(pnorm * tnorm, 1e-12)
    norm_ratio = pnorm / jnp.maximum(tnorm, 1e-12)

    t_mean = jnp.mean(t)
    ss_res = jnp.sum((p - t) ** 2)
    ss_tot = jnp.sum((t - t_mean) ** 2)
    r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-12)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "cosine": float(cosine),
        "norm_ratio": float(norm_ratio),
        "r2": float(r2),
    }


def _objective_loss(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
    F_target: jax.Array,
    prediction_mode: str,
) -> jax.Array:
    F_pred = _force_batch_from_mode(prior, params, data, mode=prediction_mode)
    m3 = data["mask"][..., None]
    diff = (F_pred - F_target) * m3
    denom = jnp.maximum(jnp.sum(m3), 1.0)
    return jnp.sum(diff * diff) / denom


def _force_metrics(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
    F_target: jax.Array,
    prediction_mode: str,
) -> Dict[str, float]:
    F_pred = _force_batch_from_mode(prior, params, data, mode=prediction_mode)
    return _force_array_metrics(F_pred, F_target, data["mask"])


def _run_lbfgs_on_new_params(
    prior: PriorEnergy,
    params_full0: Dict[str, jax.Array],
    train_data: Dict[str, jax.Array],
    fit_keys: Iterable[str],
    F_target: jax.Array,
    prediction_mode: str,
    max_steps: int,
    tol_grad: float,
    min_steps: int,
) -> Tuple[Dict[str, jax.Array], Dict[str, Any]]:
    fit_keys = tuple(fit_keys)
    fit_set = set(fit_keys)

    frozen_params = {k: v for k, v in params_full0.items() if k not in fit_set}
    fit_params0 = {k: params_full0[k] for k in fit_keys}

    for key in fit_keys:
        if key not in params_full0:
            raise KeyError(f"Requested fit key '{key}' not found in prior params.")

    def loss_fit_only(fit_params):
        full = _merge_params(frozen_params, fit_params)
        return _objective_loss(
            prior=prior,
            params=full,
            data=train_data,
            F_target=F_target,
            prediction_mode=prediction_mode,
        )

    opt = optax.lbfgs(learning_rate=1.0)
    value_and_grad = optax.value_and_grad_from_state(loss_fit_only)

    class FitState(NamedTuple):
        params: Dict[str, jax.Array]
        opt_state: optax.OptState
        step: jax.Array
        loss: jax.Array
        loss_hist: jax.Array

    def init_state(p0):
        opt_state = opt.init(p0)
        value0, _ = value_and_grad(p0, state=opt_state)
        loss_hist = jnp.full((max_steps,), jnp.nan, dtype=jnp.float32)
        if max_steps > 0:
            loss_hist = loss_hist.at[0].set(value0.astype(jnp.float32))
        return FitState(
            params=p0,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
            loss=value0,
            loss_hist=loss_hist,
        )

    def cond_fn(st: FitState):
        not_done = st.step < max_steps
        grad = optax.tree.get(st.opt_state, "grad")
        grad_norm = optax.tree.norm(grad)
        not_converged_grad = jnp.logical_or(st.step < min_steps, grad_norm >= tol_grad)
        return jnp.logical_and(not_done, not_converged_grad)

    def body_fn(st: FitState):
        p, s, k = st.params, st.opt_state, st.step
        value, grad = value_and_grad(p, state=s)
        updates, s_new = opt.update(
            grad,
            s,
            p,
            value=value,
            grad=grad,
            value_fn=loss_fit_only,
        )
        p_new = optax.apply_updates(p, updates)
        value_new = loss_fit_only(p_new)
        loss_hist = st.loss_hist.at[k].set(value.astype(jnp.float32))
        return FitState(
            params=p_new,
            opt_state=s_new,
            step=k + 1,
            loss=value_new,
            loss_hist=loss_hist,
        )

    training_logger.info(f"[LBFGS] Starting optimization (max_steps={max_steps}, tol_grad={tol_grad:.3e})")
    st0 = init_state(fit_params0)
    stf = jax.lax.while_loop(cond_fn, body_fn, st0)

    grad_final = optax.tree.get(stf.opt_state, "grad")
    grad_norm_final = float(optax.tree.norm(grad_final))
    converged = bool(grad_norm_final < tol_grad)

    fitted_full = _merge_params(frozen_params, stf.params)
    steps = int(stf.step)
    loss_hist = np.asarray(stf.loss_hist[:steps]) if steps > 0 else np.asarray([], dtype=np.float32)

    stats = {
        "steps": steps,
        "final_loss": float(stf.loss),
        "grad_norm": grad_norm_final,
        "converged": converged,
        "loss_history": loss_hist,
    }
    return fitted_full, stats


def _log_metrics(tag: str, metrics: Dict[str, float]) -> None:
    training_logger.info(
        f"{tag}: mse={metrics['mse']:.6e}, rmse={metrics['rmse']:.6e}, mae={metrics['mae']:.6e}, "
        f"cos={metrics['cosine']:.4f}, norm_ratio={metrics['norm_ratio']:.4f}, r2={metrics['r2']:.4f}"
    )


def _component_energy(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    R: jax.Array,
    mask: jax.Array,
    species: jax.Array,
    component: str,
) -> jax.Array:
    p = params
    if component == "dh":
        return prior.weights.get("dh", 0.0) * prior.compute_dh_energy(R, mask, species=species, params=p)
    if component == "stickiness":
        return (
            prior.weights.get("stickiness", 0.0)
            * prior.compute_stickiness_energy(R, mask, species=species, params=p)
        )
    if component == "salt_bridge":
        return (
            prior.weights.get("salt_bridge", 0.0)
            * prior.compute_salt_bridge_energy(R, mask, species=species, params=p)
        )
    raise ValueError(f"Unknown component: {component}")


def _force_batch_component(
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
    component: str,
) -> jax.Array:
    def comp_forces(single_R, single_mask, single_species):
        def energy_of_R(R_):
            return _component_energy(
                prior=prior,
                params=params,
                R=R_,
                mask=single_mask,
                species=single_species,
                component=component,
            )

        return -jax.grad(energy_of_R)(single_R)

    return jax.vmap(comp_forces)(data["R"], data["mask"], data["species"])


def _log_new_term_force_diagnostics(
    tag: str,
    prior: PriorEnergy,
    params: Dict[str, jax.Array],
    data: Dict[str, jax.Array],
    max_frames: int = 64,
) -> None:
    n = int(data["R"].shape[0])
    n_use = max(1, min(int(max_frames), n))
    data_sub = {k: v[:n_use] for k, v in data.items()}
    m3 = data_sub["mask"][..., None]
    denom = jnp.maximum(jnp.sum(m3), 1.0)

    def rms_from_mode(mode: str) -> float:
        F = _force_batch_from_mode(prior, params, data_sub, mode=mode) * m3
        return float(jnp.sqrt(jnp.sum(F * F) / denom))

    def rms_from_component(component: str) -> float:
        F = _force_batch_component(prior, params, data_sub, component=component) * m3
        return float(jnp.sqrt(jnp.sum(F * F) / denom))

    training_logger.info(
        f"{tag} force RMS (first {n_use} frames): "
        f"old={rms_from_mode('old_only'):.6e}, "
        f"new={rms_from_mode('new_only'):.6e}, "
        f"full={rms_from_mode('full'):.6e}"
    )
    training_logger.info(
        f"{tag} new-term RMS (first {n_use} frames): "
        f"dh={rms_from_component('dh'):.6e}, "
        f"stick={rms_from_component('stickiness'):.6e}, "
        f"salt={rms_from_component('salt_bridge'):.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone LBFGS pretraining for DH + stickiness + salt-bridge prior parameters."
    )
    parser.add_argument("config", type=str, help="Path to training config YAML.")
    parser.add_argument(
        "--out-pkl",
        type=str,
        default="data_prep/fitted_new_prior_params.pkl",
        help="Output pickle file containing full prior params (old + new fitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for new-parameter initialization (default: config seed).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum LBFGS iterations (default: training.pretrain_prior_max_steps).",
    )
    parser.add_argument(
        "--tol-grad",
        type=float,
        default=None,
        help="Gradient-norm convergence threshold (default: training.pretrain_prior_tol_grad).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Minimum LBFGS steps before convergence check (default: training.pretrain_prior_min_steps).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override data.max_frames from config.",
    )
    parser.add_argument(
        "--fit-frames",
        type=int,
        default=None,
        help="Optional number of train frames used for LBFGS objective (random subset).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Override training.val_fraction for standalone evaluation split.",
    )
    parser.add_argument(
        "--fit-keys",
        nargs="+",
        default=list(NEW_PRIOR_PARAM_KEYS),
        help=f"Subset of keys to optimize (default: all new keys: {list(NEW_PRIOR_PARAM_KEYS)}).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=OBJECTIVE_CHOICES,
        default="full_on_fref",
        help=(
            "Fitting objective: "
            "full_on_fref (fit full prior to F_ref), "
            "new_on_fref (fit only new terms to F_ref), "
            "new_on_residual (fit only new terms to F_ref - F_old)."
        ),
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help="Number of random restarts (multi-start).",
    )
    parser.add_argument(
        "--restart-seed-stride",
        type=int,
        default=1009,
        help="Seed increment between restarts (seed_i = seed + i*stride).",
    )
    parser.add_argument(
        "--select-by",
        type=str,
        choices=("auto", "train", "val"),
        default="auto",
        help="Best-restart selection metric. auto=val if available else train.",
    )
    parser.add_argument(
        "--diag-frames",
        type=int,
        default=64,
        help="Number of frames used for diagnostics force-norm reporting.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable extra diagnostics logging.",
    )
    parser.add_argument(
        "--active-terms",
        nargs="+",
        choices=TERM_NAMES,
        default=list(TERM_NAMES),
        help="Typed-prior terms active during fitting.",
    )
    parser.add_argument(
        "--weight-dh",
        type=float,
        default=None,
        help="Override model.priors.weights.dh for this run.",
    )
    parser.add_argument(
        "--weight-stickiness",
        type=float,
        default=None,
        help="Override model.priors.weights.stickiness for this run.",
    )
    parser.add_argument(
        "--weight-salt-bridge",
        type=float,
        default=None,
        help="Override model.priors.weights.salt_bridge for this run.",
    )
    parser.add_argument(
        "--dh-k",
        type=int,
        default=None,
        help="Override model.priors.dh.K for this run.",
    )
    parser.add_argument(
        "--dh-w-by-sep",
        nargs="+",
        type=float,
        default=None,
        help="Override model.priors.dh.w_by_sep values for this run.",
    )
    args = parser.parse_args()

    config = ConfigManager(args.config)

    # Configure active typed terms and optional term-weight overrides.
    weight_overrides = {}
    if args.weight_dh is not None:
        weight_overrides["dh"] = float(args.weight_dh)
    if args.weight_stickiness is not None:
        weight_overrides["stickiness"] = float(args.weight_stickiness)
    if args.weight_salt_bridge is not None:
        weight_overrides["salt_bridge"] = float(args.weight_salt_bridge)

    _force_enable_new_terms_for_fit(
        config=config,
        active_terms=args.active_terms,
        weight_overrides=weight_overrides,
    )
    _apply_dh_overrides(
        config=config,
        dh_k_override=args.dh_k,
        dh_w_by_sep_override=args.dh_w_by_sep,
    )

    loader, _, train_data, val_data = _prepare_dataset(
        config=config,
        max_frames_override=args.max_frames,
        val_fraction_override=args.val_fraction,
    )

    seed = config.get_seed() if args.seed is None else int(args.seed)
    max_steps = config.get_pretrain_prior_max_steps() if args.max_steps is None else int(args.max_steps)
    tol_grad = config.get_pretrain_prior_tol_grad() if args.tol_grad is None else float(args.tol_grad)
    min_steps = config.get_pretrain_prior_min_steps() if args.min_steps is None else int(args.min_steps)

    if max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if min_steps < 0:
        raise ValueError("--min-steps must be >= 0")
    if tol_grad <= 0:
        raise ValueError("--tol-grad must be > 0")
    if args.n_restarts <= 0:
        raise ValueError("--n-restarts must be > 0")

    fit_keys = tuple(args.fit_keys)
    unknown = [k for k in fit_keys if k not in NEW_PRIOR_PARAM_KEYS]
    if unknown:
        raise ValueError(f"Unsupported --fit-keys entries: {unknown}")
    fit_key_term_set = {
        term for term, keys in KEYS_BY_TERM.items() if any(k in fit_keys for k in keys)
    }
    inactive_fit_terms = sorted(fit_key_term_set - set(args.active_terms))
    if inactive_fit_terms:
        raise ValueError(
            f"--fit-keys include parameters from inactive term(s): {inactive_fit_terms}. "
            f"Active terms: {args.active_terms}"
        )

    prediction_mode = PREDICTION_MODE_BY_OBJECTIVE[args.objective]

    topology = TopologyBuilder(N_max=loader.N_max, min_repulsive_sep=6)
    prior = PriorEnergy(
        config=config,
        topology=topology,
        displacement=lambda a, b: a - b,
        id_to_aa=loader.id_to_aa,
    )

    params_config = dict(prior.params)
    target_train = _build_target_forces(args.objective, prior, params_config, train_data)
    target_val = (
        _build_target_forces(args.objective, prior, params_config, val_data)
        if val_data is not None
        else None
    )

    fit_idx = _sample_frame_indices(int(train_data["R"].shape[0]), args.fit_frames, seed=seed)
    train_fit = _subset_frames(train_data, fit_idx)
    target_train_fit = target_train if fit_idx is None else target_train[fit_idx]
    if fit_idx is not None:
        training_logger.info(
            f"Using {train_fit['R'].shape[0]} train frames for LBFGS objective "
            f"(subset of {train_data['R'].shape[0]})."
        )

    training_logger.info("")
    training_logger.info("=" * 72)
    training_logger.info("Standalone LBFGS New-Prior Fitting")
    training_logger.info("=" * 72)
    training_logger.info(f"Config: {Path(args.config).resolve()}")
    training_logger.info(f"Objective: {args.objective} (prediction_mode={prediction_mode})")
    training_logger.info(f"Active terms: {tuple(args.active_terms)}")
    training_logger.info(f"Fit keys: {fit_keys}")
    training_logger.info(f"Seed: {seed} (restarts={args.n_restarts}, stride={args.restart_seed_stride})")
    training_logger.info(f"LBFGS: max_steps={max_steps}, min_steps={min_steps}, tol_grad={tol_grad:.3e}")
    training_logger.info("=" * 72)

    # Baseline metrics (config-initialized parameters)
    metrics_config_train = _force_metrics(
        prior=prior,
        params=params_config,
        data=train_data,
        F_target=target_train,
        prediction_mode=prediction_mode,
    )
    _log_metrics("[Train] Config-init", metrics_config_train)
    metrics_config_val = None
    if val_data is not None:
        metrics_config_val = _force_metrics(
            prior=prior,
            params=params_config,
            data=val_data,
            F_target=target_val,
            prediction_mode=prediction_mode,
        )
        _log_metrics("[Val]   Config-init", metrics_config_val)

    select_by = args.select_by
    if select_by == "auto":
        select_by = "val" if val_data is not None else "train"
    if select_by == "val" and val_data is None:
        raise ValueError("--select-by=val requested, but validation split is empty.")

    best_record = None
    for restart_idx in range(args.n_restarts):
        seed_i = int(seed + restart_idx * args.restart_seed_stride)
        training_logger.info("")
        training_logger.info(f"[Restart {restart_idx + 1}/{args.n_restarts}] seed={seed_i}")
        params_init = _randomize_new_params(
            params=params_config,
            seed=seed_i,
            keys_to_randomize=fit_keys,
        )

        metrics_init_train = _force_metrics(
            prior=prior,
            params=params_init,
            data=train_data,
            F_target=target_train,
            prediction_mode=prediction_mode,
        )
        _log_metrics("[Train] Random-init", metrics_init_train)
        metrics_init_val = None
        if val_data is not None:
            metrics_init_val = _force_metrics(
                prior=prior,
                params=params_init,
                data=val_data,
                F_target=target_val,
                prediction_mode=prediction_mode,
            )
            _log_metrics("[Val]   Random-init", metrics_init_val)

        fitted_params, lbfgs_stats = _run_lbfgs_on_new_params(
            prior=prior,
            params_full0=params_init,
            train_data=train_fit,
            fit_keys=fit_keys,
            F_target=target_train_fit,
            prediction_mode=prediction_mode,
            max_steps=max_steps,
            tol_grad=tol_grad,
            min_steps=min_steps,
        )

        metrics_fit_train = _force_metrics(
            prior=prior,
            params=fitted_params,
            data=train_data,
            F_target=target_train,
            prediction_mode=prediction_mode,
        )
        _log_metrics("[Train] Fitted", metrics_fit_train)
        metrics_fit_val = None
        if val_data is not None:
            metrics_fit_val = _force_metrics(
                prior=prior,
                params=fitted_params,
                data=val_data,
                F_target=target_val,
                prediction_mode=prediction_mode,
            )
            _log_metrics("[Val]   Fitted", metrics_fit_val)

        select_metric = (
            metrics_fit_val["mse"] if select_by == "val" else metrics_fit_train["mse"]
        )
        training_logger.info(
            "[Restart %d] steps=%d, final_loss=%.6e, grad_norm=%.6e, converged=%s, select_%s_mse=%.6e",
            restart_idx + 1,
            lbfgs_stats["steps"],
            lbfgs_stats["final_loss"],
            lbfgs_stats["grad_norm"],
            lbfgs_stats["converged"],
            select_by,
            select_metric,
        )

        record = {
            "restart_idx": restart_idx,
            "seed": seed_i,
            "params_init": params_init,
            "params_fit": fitted_params,
            "stats": lbfgs_stats,
            "metrics_init_train": metrics_init_train,
            "metrics_init_val": metrics_init_val,
            "metrics_fit_train": metrics_fit_train,
            "metrics_fit_val": metrics_fit_val,
            "select_metric": float(select_metric),
        }
        if best_record is None or record["select_metric"] < best_record["select_metric"]:
            best_record = record

    assert best_record is not None
    fitted_params = best_record["params_fit"]
    metrics_fit_train = best_record["metrics_fit_train"]
    metrics_fit_val = best_record["metrics_fit_val"]
    metrics_init_train = best_record["metrics_init_train"]
    metrics_init_val = best_record["metrics_init_val"]
    lbfgs_stats = best_record["stats"]

    training_logger.info("")
    training_logger.info(
        "[Best Restart] idx=%d, seed=%d, select_%s_mse=%.6e",
        best_record["restart_idx"] + 1,
        best_record["seed"],
        select_by,
        best_record["select_metric"],
    )
    training_logger.info(
        "[LBFGS] steps=%d, final_loss=%.6e, grad_norm=%.6e, converged=%s",
        lbfgs_stats["steps"],
        lbfgs_stats["final_loss"],
        lbfgs_stats["grad_norm"],
        lbfgs_stats["converged"],
    )

    training_logger.info(
        "[Train] Improvement vs best random-init: dMSE=%.6e, dRMSE=%.6e",
        metrics_init_train["mse"] - metrics_fit_train["mse"],
        metrics_init_train["rmse"] - metrics_fit_train["rmse"],
    )
    training_logger.info(
        "[Train] Improvement vs config-init: dMSE=%.6e, dRMSE=%.6e",
        metrics_config_train["mse"] - metrics_fit_train["mse"],
        metrics_config_train["rmse"] - metrics_fit_train["rmse"],
    )
    if val_data is not None and metrics_fit_val is not None:
        training_logger.info(
            "[Val]   Improvement vs best random-init: dMSE=%.6e, dRMSE=%.6e",
            metrics_init_val["mse"] - metrics_fit_val["mse"],
            metrics_init_val["rmse"] - metrics_fit_val["rmse"],
        )
        training_logger.info(
            "[Val]   Improvement vs config-init: dMSE=%.6e, dRMSE=%.6e",
            metrics_config_val["mse"] - metrics_fit_val["mse"],
            metrics_config_val["rmse"] - metrics_fit_val["rmse"],
        )

    if not args.no_diagnostics:
        _log_new_term_force_diagnostics(
            tag="[Diag][Train][Config-init]",
            prior=prior,
            params=params_config,
            data=train_data,
            max_frames=args.diag_frames,
        )
        _log_new_term_force_diagnostics(
            tag="[Diag][Train][Best-fit]",
            prior=prior,
            params=fitted_params,
            data=train_data,
            max_frames=args.diag_frames,
        )
        if val_data is not None:
            _log_new_term_force_diagnostics(
                tag="[Diag][Val][Config-init]",
                prior=prior,
                params=params_config,
                data=val_data,
                max_frames=args.diag_frames,
            )
            _log_new_term_force_diagnostics(
                tag="[Diag][Val][Best-fit]",
                prior=prior,
                params=fitted_params,
                data=val_data,
                max_frames=args.diag_frames,
            )

    out_path = Path(args.out_pkl)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_params = {k: np.asarray(v) for k, v in fitted_params.items()}
    with open(out_path, "wb") as f:
        pickle.dump(serializable_params, f)

    training_logger.info(f"Wrote fitted prior params: {out_path}")
    training_logger.info(
        "Saved keys: %s",
        sorted(serializable_params.keys()),
    )


if __name__ == "__main__":
    main()
