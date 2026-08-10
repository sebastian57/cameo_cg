"""Force-label selection and teacher blending for energy-student training."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from utils.logging import training_logger


def apply_force_label_mode(config, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow dataset copy with the configured effective ``F`` label.

    For a common prediction and MSE, a weighted raw+teacher target is exactly
    equivalent (up to a parameter-independent constant) to two weighted loss
    terms.  Mixing once here therefore avoids a second energy-force VJP.
    """
    cfg = config.get_force_label_config()
    mode = str(cfg["mode"])
    if mode == "raw":
        return dataset

    teacher_key = str(cfg["teacher_key"])
    if teacher_key not in dataset:
        raise ValueError(
            f"training.force_labels.mode={mode!r} requires dataset field "
            f"{teacher_key!r}; available keys: {sorted(dataset)}"
        )

    raw = np.asarray(dataset["F"], dtype=np.float32)
    teacher = np.asarray(dataset[teacher_key], dtype=np.float32)
    if teacher.shape != raw.shape:
        raise ValueError(
            f"Teacher force shape {teacher.shape} does not match raw F shape {raw.shape}."
        )
    if not np.isfinite(teacher).all():
        raise ValueError(f"Dataset field {teacher_key!r} contains NaN or Inf values.")

    result = dict(dataset)
    result["RawForce"] = raw.copy()
    if mode == "teacher":
        effective = teacher
    else:
        raw_weight = float(cfg["raw_weight"])
        teacher_weight = float(cfg["teacher_weight"])
        effective = (
            raw_weight * raw + teacher_weight * teacher
        ) / (raw_weight + teacher_weight)
    mask = np.asarray(dataset["mask"] > 0, dtype=np.float32)
    effective = np.asarray(effective, dtype=np.float32) * mask[..., None]
    result["F"] = effective

    delta = (teacher - raw) * mask[..., None]
    denom = max(float(np.sum(mask) * 3.0), 1.0)
    teacher_raw_rmse = float(np.sqrt(np.sum(delta * delta) / denom))
    training_logger.info(
        "[ForceLabels] mode=%s teacher_key=%s raw_weight=%.6g "
        "teacher_weight=%.6g teacher_vs_raw_rmse=%.6g",
        mode,
        teacher_key,
        float(cfg["raw_weight"]),
        float(cfg["teacher_weight"]),
        teacher_raw_rmse,
    )
    return result

