"""Optional heteroscedastic force weighting from per-label uncertainty.

WHY THIS IS NOT A NEW LOSS
--------------------------
`force_loss_weights` is already a first-class per-particle dataset key that the
trainer carries and applies (`scripts/train.py:_attach_batch_metadata`,
`training/trainer.py`). This module only supplies a different WEIGHT SOURCE for
that existing hook, so the loss path, normalisation options and diagnostics are
untouched.

WHEN IT HELPS
-------------
Only when label uncertainty actually varies between rows. Measured 2026-08-25:
stencil label noise is homogeneous to 1.4x across 288,000 states, so weighting by
per-configuration noise is a no-op there (ESS/N = 0.96-0.99). It has real dynamic
range for PSEUDO mean-force labels, where `sigma_label` depends on the partner
count K and K varies by orders of magnitude between common and rare environments.
See KB DESIGN/PSEUDO_MEANFORCE_LABELS.md.

CONFIG
------
    training:
      label_uncertainty:
        enabled: true
        key: sigma_label          # dataset key, per (frame, bead) or per frame
        mode: inverse_variance    # 1/sigma^2 (default) | inverse_sigma | none
        max_weight_ratio: 50.0    # cap max/min; a few tiny-sigma rows must not dominate
        preserve_scale: true      # renormalise so the total loss scale is unchanged

`preserve_scale` matters: without it the gradient magnitude changes with the
weighting, so a weighted-vs-unweighted comparison would confound the effect with
an effective learning-rate change.
"""
from __future__ import annotations

import numpy as np

__all__ = ["is_enabled", "config_from", "configure", "active", "build_weights", "describe"]

# Active config for this process. Set once from the trainer entry point so the
# per-split weight builder does not need a threaded-through argument (it is
# called from five places). Owned by this module rather than stashed on a
# function attribute, so the state has a single obvious home.
_ACTIVE: dict = {}


def configure(cfg: dict | None) -> dict:
    """Install the process-wide config; returns it for logging."""
    global _ACTIVE
    _ACTIVE = dict(cfg or {})
    return _ACTIVE


def active() -> dict:
    return _ACTIVE

_MODES = ("inverse_variance", "inverse_sigma", "none")


def config_from(config) -> dict:
    """Read `training.label_uncertainty` from a ConfigManager or plain dict."""
    if hasattr(config, "get"):
        try:
            raw = config.get("training", "label_uncertainty", default=None)
        except TypeError:
            raw = (config.get("training", {}) or {}).get("label_uncertainty")
    else:
        raw = None
    return dict(raw or {})


def is_enabled(cfg: dict) -> bool:
    return bool(cfg.get("enabled", False))


def build_weights(split: dict, cfg: dict, base_weights: np.ndarray) -> np.ndarray:
    """Scale `base_weights` (per-particle) by per-label precision.

    `base_weights` is whatever the default policy produced, so the per-structure
    normalisation it encodes is preserved and only modulated here.
    """
    key = str(cfg.get("key", "sigma_label"))
    mode = str(cfg.get("mode", "inverse_variance")).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"training.label_uncertainty.mode='{mode}' not in {_MODES}")
    if mode == "none":
        return base_weights
    if key not in split:
        raise KeyError(
            f"training.label_uncertainty.enabled=true but dataset has no '{key}'. "
            f"Available keys: {sorted(split)[:12]}... "
            f"Build one with analysis/labels/build_ala2_pseudo_dataset.py.")

    base = np.asarray(base_weights, dtype=np.float64)
    sigma = np.asarray(split[key], dtype=np.float64)
    if sigma.ndim == 1:                       # per frame -> broadcast over beads
        sigma = sigma[:, None]
    if sigma.shape[0] != base.shape[0]:
        raise ValueError(f"'{key}' has {sigma.shape[0]} rows, weights have {base.shape[0]}")
    sigma = np.broadcast_to(sigma, base.shape)

    valid = (base > 0) & np.isfinite(sigma) & (sigma > 0)
    if not valid.any():
        raise ValueError(f"'{key}' has no finite positive entries where weights are nonzero")

    prec = np.zeros_like(base)
    s = sigma[valid]
    prec[valid] = 1.0 / (s * s) if mode == "inverse_variance" else 1.0 / s

    # Cap the dynamic range: a handful of very small sigma must not dominate the
    # gradient. Applied on the precision BEFORE renormalisation.
    ratio = float(cfg.get("max_weight_ratio", 50.0))
    if ratio and ratio > 1.0:
        lo = np.min(prec[valid])
        prec[valid] = np.minimum(prec[valid], lo * ratio)

    w = base * prec
    if bool(cfg.get("preserve_scale", True)):
        tot_base = float(base[valid].sum())
        tot_w = float(w[valid].sum())
        if tot_w > 0:
            w *= tot_base / tot_w
    return np.asarray(w, dtype=np.float32)


def describe(split: dict, cfg: dict, base_weights: np.ndarray, weights: np.ndarray) -> str:
    """One-line summary for the training log, incl. the effective sample size."""
    b = np.asarray(base_weights, np.float64); w = np.asarray(weights, np.float64)
    v = b > 0
    if not v.any():
        return "label_uncertainty: no valid weights"
    r = w[v] / np.maximum(b[v], 1e-30)
    ess = (w[v].sum() ** 2) / max(float((w[v] ** 2).sum()), 1e-30)
    ess_b = (b[v].sum() ** 2) / max(float((b[v] ** 2).sum()), 1e-30)
    return (f"label_uncertainty[{cfg.get('mode','inverse_variance')}] "
            f"key={cfg.get('key','sigma_label')} "
            f"weight ratio p5/p95 = {np.percentile(r,5):.3g}/{np.percentile(r,95):.3g} "
            f"(max/min {r.max()/max(r.min(),1e-30):.1f}x)  "
            f"ESS/ESS_uniform = {ess/max(ess_b,1e-30):.3f}")
