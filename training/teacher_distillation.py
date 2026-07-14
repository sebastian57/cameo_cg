"""Optional frozen-AA-teacher targets for CG force-matching experiments.

Teacher features are precomputed on paired AA frames and stored at the nine CG
anchor sites.  At training time this module only forwards the CG model.  The
feature target is intentionally an invariant scalar representation so the AA
and CG graphs need not share node identities or tensor bases.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp


FEATURE_TARGET = "TeacherFeature"
FORCE_TARGET = "TeacherForce"


def config_parsed(config) -> Dict[str, Any]:
    cfg = config.get("training", "teacher_distillation", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    feature = cfg.get("feature", {}) or {}
    force = cfg.get("force", {}) or {}
    return {
        "feature": {
            "enabled": bool(feature.get("enabled", False)),
            "lambda": float(feature.get("lambda", 0.0)),
            "target_key": str(feature.get("target_key", FEATURE_TARGET)),
            "mask_key": str(feature.get("mask_key", "teacher_feature_mask")),
            "feature_key": str(feature.get("feature_key", "edge_features")),
            "projection_hidden_dim": int(feature.get("projection_hidden_dim", 64)),
            "source_dim": int(feature.get("source_dim", 32)),
            "target_dim": int(feature.get("target_dim", 32)),
        },
        "force": {
            "enabled": bool(force.get("enabled", False)),
            "lambda": float(force.get("lambda", 0.0)),
            "target_key": str(force.get("target_key", FORCE_TARGET)),
            "mask_key": str(force.get("mask_key", "teacher_force_mask")),
        },
    }


def enabled(config) -> bool:
    cfg = config_parsed(config)
    return bool(cfg["feature"]["enabled"] or cfg["force"]["enabled"])


def weighted_mse(predictions, targets, weights=None):
    sq = jnp.square(jnp.asarray(predictions) - jnp.asarray(targets, dtype=jnp.asarray(predictions).dtype))
    if weights is None:
        return jnp.mean(sq)
    w = jnp.asarray(weights, dtype=sq.dtype)
    while w.ndim < sq.ndim:
        w = w[..., None]
    w = jnp.broadcast_to(w, sq.shape)
    return jnp.sum(sq * w) / jnp.maximum(jnp.sum(w), 1.0)


def _node_mean_edge_features(features, senders, valid_edges, n_nodes: int):
    valid = jnp.asarray(valid_edges, dtype=features.dtype)
    sender = jnp.where(jnp.asarray(valid_edges, dtype=bool), senders, 0)
    sums = jax.ops.segment_sum(features * valid[:, None], sender, num_segments=n_nodes)
    counts = jax.ops.segment_sum(valid, sender, num_segments=n_nodes)
    return sums / jnp.maximum(counts[:, None], 1.0)


def make_feature_quantity(model, cfg: Dict[str, Any]) -> Callable:
    feature_key = str(cfg["feature"]["feature_key"])

    def quantity(state, neighbor=None, energy_params=None, mask=None, species=None, segment_id=None, **kwargs):
        if mask is None or species is None:
            raise ValueError("TeacherFeature quantity requires mask and species.")
        node_features = model.compute_teacher_projected_features(
            energy_params,
            state.position,
            mask,
            species,
            neighbor,
            segment_id=segment_id,
        )
        return node_features * jnp.asarray(mask, dtype=node_features.dtype)[:, None]

    return quantity


def make_force_quantity(model) -> Callable:
    """Force prediction for a separately mapped frozen-teacher target."""

    def quantity(state, neighbor=None, energy_params=None, mask=None, species=None, segment_id=None, **kwargs):
        if mask is None or species is None:
            raise ValueError("TeacherForce quantity requires mask and species.")
        energy_fn = model.energy_fn_template(energy_params)

        def energy_of_R(R):
            return energy_fn(R, neighbor=neighbor, mask=mask, species=species, segment_id=segment_id)

        return -jax.grad(energy_of_R)(state.position)

    return quantity


def gammas(config) -> Dict[str, float]:
    cfg = config_parsed(config)
    out: Dict[str, float] = {}
    if cfg["feature"]["enabled"]:
        out[FEATURE_TARGET] = float(cfg["feature"]["lambda"])
    if cfg["force"]["enabled"]:
        out[FORCE_TARGET] = float(cfg["force"]["lambda"])
    return out


def error_fns(config) -> Dict[str, Callable]:
    return {key: weighted_mse for key in gammas(config)}


def weights_keys(config) -> Dict[str, str]:
    cfg = config_parsed(config)
    keys: Dict[str, str] = {}
    if cfg["feature"]["enabled"]:
        keys[FEATURE_TARGET] = str(cfg["feature"]["mask_key"])
    if cfg["force"]["enabled"]:
        keys[FORCE_TARGET] = str(cfg["force"]["mask_key"])
    return keys


def quantities(model, config) -> Dict[str, Callable]:
    cfg = config_parsed(config)
    out: Dict[str, Callable] = {}
    if cfg["feature"]["enabled"]:
        out[FEATURE_TARGET] = make_feature_quantity(model, cfg)
    if cfg["force"]["enabled"]:
        out[FORCE_TARGET] = make_force_quantity(model)
    return out


def required_fields(config) -> tuple[str, ...]:
    cfg = config_parsed(config)
    fields: list[str] = []
    if cfg["feature"]["enabled"]:
        fields += [str(cfg["feature"]["target_key"]), str(cfg["feature"]["mask_key"])]
    if cfg["force"]["enabled"]:
        fields += [str(cfg["force"]["target_key"]), str(cfg["force"]["mask_key"])]
    return tuple(fields)
