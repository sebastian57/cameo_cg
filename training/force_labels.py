"""Force-label selection and teacher blending for energy-student training."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from utils.logging import training_logger


def _apply_label_mode(
    result: Dict[str, Any],
    dataset: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    target_key: str,
    raw_result_key: str,
    teacher_key: str,
    log_prefix: str,
) -> None:
    raw = np.asarray(dataset[target_key], dtype=np.float32)
    teacher = np.asarray(dataset[teacher_key], dtype=np.float32)
    if teacher.shape != raw.shape:
        raise ValueError(
            f"Teacher {target_key.lower()} shape {teacher.shape} does not match raw {target_key} shape {raw.shape}."
        )
    if not np.isfinite(teacher).all():
        raise ValueError(f"Dataset field {teacher_key!r} contains NaN or Inf values.")

    result[raw_result_key] = raw.copy()
    if cfg["mode"] == "teacher":
        effective = teacher
    else:
        raw_weight = float(cfg["raw_weight"])
        teacher_weight = float(cfg["teacher_weight"])
        effective = (
            raw_weight * raw + teacher_weight * teacher
        ) / (raw_weight + teacher_weight)

    mask = np.asarray(dataset["mask"] > 0, dtype=np.float32)
    effective = np.asarray(effective, dtype=np.float32) * mask[..., None]
    result[target_key] = effective

    delta = (teacher - raw) * mask[..., None]
    denom = max(float(np.sum(mask) * 3.0), 1.0)
    teacher_raw_rmse = float(np.sqrt(np.sum(delta * delta) / denom))
    training_logger.info(
        "[%s] mode=%s teacher_key=%s raw_weight=%.6g "
        "teacher_weight=%.6g teacher_vs_raw_rmse=%.6g",
        log_prefix,
        cfg["mode"],
        teacher_key,
        float(cfg["raw_weight"]),
        float(cfg["teacher_weight"]),
        teacher_raw_rmse,
    )


def apply_force_label_mode(config, dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow dataset copy with the configured effective force and torque labels.

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

    result = dict(dataset)
    _apply_label_mode(
        result,
        dataset,
        cfg,
        target_key="F",
        raw_result_key="RawForce",
        teacher_key=teacher_key,
        log_prefix="ForceLabels",
    )

    torque_raw = dataset.get("T")
    if torque_raw is not None and np.isfinite(np.asarray(torque_raw, dtype=np.float32)).all():
        torque_teacher_key = str(cfg.get("torque_teacher_key", "TeacherTorque"))
        if torque_teacher_key not in dataset:
            raise ValueError(
                f"training.force_labels.mode={mode!r} requires dataset field "
                f"{torque_teacher_key!r} for torque labels; available keys: {sorted(dataset)}"
            )
        _apply_label_mode(
            result,
            dataset,
            cfg,
            target_key="T",
            raw_result_key="RawTorque",
            teacher_key=torque_teacher_key,
            log_prefix="TorqueLabels",
        )
    return result

