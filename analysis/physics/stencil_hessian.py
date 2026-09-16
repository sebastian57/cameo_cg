"""Shared utilities for Hessian-vector diagnostics from force stencils."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class StencilPairs:
    """Row indices for complete central-difference stencils."""

    anchor_ids: np.ndarray
    anchor_rows: np.ndarray
    directions: np.ndarray
    positive_rows: np.ndarray
    negative_rows: np.ndarray
    layer: float

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_ids.size)

    @property
    def n_directions(self) -> int:
        return int(self.directions.size)


@dataclass(frozen=True)
class MeasuredHVP:
    """Measured Hessian-vector products for selected stencil anchors."""

    anchor_ids: np.ndarray
    anchor_rows: np.ndarray
    r0: np.ndarray
    v: np.ndarray
    hvp: np.ndarray
    realized_displacement: np.ndarray
    noise: np.ndarray | None


def _require_array(mapping: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    try:
        return np.asarray(mapping[key])
    except KeyError as exc:
        raise ValueError(f"input data is missing required array {key!r}") from exc


def build_stencil_pairs(
    meanforce: Mapping[str, np.ndarray],
    stencil: Mapping[str, np.ndarray],
    layer: float = 1.0,
) -> StencilPairs:
    """Group complete central-difference rows using stencil ``anchor`` metadata.

    The stencil must contain one row with ``direction < 0`` per anchor and one
    ``+layer``/``-layer`` pair for every non-negative direction label.  Rows
    outside the requested layer are ignored.  Incomplete or ambiguous groups
    raise a descriptive error instead of being silently discarded.
    """

    if not np.isfinite(layer) or layer <= 0:
        raise ValueError(f"layer must be a positive finite number, got {layer!r}")

    coordinates = _require_array(meanforce, "R")
    forces = _require_array(meanforce, "F")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"R must have shape (rows, beads, 3), got {coordinates.shape}")
    if forces.shape != coordinates.shape:
        raise ValueError(f"F shape {forces.shape} does not match R shape {coordinates.shape}")

    anchor = _require_array(stencil, "anchor")
    direction = _require_array(stencil, "direction")
    multiplier = _require_array(stencil, "multiplier")
    n_rows = coordinates.shape[0]
    for name, values in (("anchor", anchor), ("direction", direction), ("multiplier", multiplier)):
        if values.ndim != 1 or values.size != n_rows:
            raise ValueError(f"stencil {name!r} must have shape ({n_rows},), got {values.shape}")

    directions = np.unique(direction[direction >= 0]).astype(int)
    if directions.size == 0:
        raise ValueError("stencil contains no non-negative direction labels")

    anchor_ids = np.unique(anchor)
    anchor_rows = []
    positive_rows = []
    negative_rows = []
    for anchor_id in anchor_ids:
        rows = np.flatnonzero(anchor == anchor_id)
        center = rows[direction[rows] < 0]
        if center.size != 1:
            raise ValueError(
                f"anchor {anchor_id!r} has {center.size} center rows; expected exactly one"
            )
        anchor_rows.append(int(center[0]))

        plus_for_anchor = []
        minus_for_anchor = []
        for direction_id in directions:
            direction_rows = rows[direction[rows] == direction_id]
            plus = direction_rows[np.isclose(multiplier[direction_rows], layer)]
            minus = direction_rows[np.isclose(multiplier[direction_rows], -layer)]
            if plus.size != 1 or minus.size != 1:
                raise ValueError(
                    f"anchor {anchor_id!r}, direction {direction_id}: "
                    f"found {plus.size} +{layer:g} and {minus.size} -{layer:g} rows; expected one each"
                )
            plus_for_anchor.append(int(plus[0]))
            minus_for_anchor.append(int(minus[0]))
        positive_rows.append(plus_for_anchor)
        negative_rows.append(minus_for_anchor)

    return StencilPairs(
        anchor_ids=np.asarray(anchor_ids),
        anchor_rows=np.asarray(anchor_rows, dtype=np.int64),
        directions=directions,
        positive_rows=np.asarray(positive_rows, dtype=np.int64),
        negative_rows=np.asarray(negative_rows, dtype=np.int64),
        layer=float(layer),
    )


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ file into ordinary arrays and close the archive."""

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def choose_anchor_indices(n_available: int, n_requested: int, seed: int) -> np.ndarray:
    """Choose reproducible anchor positions without replacement."""

    if n_requested <= 0:
        raise ValueError(f"n_anchors must be positive, got {n_requested}")
    if n_requested > n_available:
        raise ValueError(f"requested {n_requested} anchors, but only {n_available} are available")
    return np.random.default_rng(seed).choice(n_available, size=n_requested, replace=False)


def compute_measured_hvp(
    meanforce: Mapping[str, np.ndarray],
    pairs: StencilPairs,
    anchor_indices: np.ndarray | list[int] | None = None,
) -> MeasuredHVP:
    """Compute central-difference HVPs using the realized displacement norm."""

    coordinates = np.asarray(_require_array(meanforce, "R"), dtype=np.float64)
    forces = np.asarray(_require_array(meanforce, "F"), dtype=np.float64)
    if anchor_indices is None:
        selected = np.arange(pairs.n_anchors, dtype=np.int64)
    else:
        selected = np.asarray(anchor_indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= pairs.n_anchors):
        raise ValueError("anchor_indices must be a one-dimensional selection of valid anchor positions")

    plus = pairs.positive_rows[selected]
    minus = pairs.negative_rows[selected]
    displacement = coordinates[plus] - coordinates[minus]
    realized = np.linalg.norm(displacement, axis=(-2, -1))
    if np.any(~np.isfinite(realized)) or np.any(realized <= 0):
        raise ValueError("central-difference stencil contains a non-finite or zero realized displacement")

    denominator = realized[..., None, None]
    v = displacement / denominator
    hvp = -(forces[plus] - forces[minus]) / denominator

    noise = None
    if "SE" in meanforce:
        standard_error = np.asarray(meanforce["SE"], dtype=np.float64)
        if standard_error.shape != coordinates.shape:
            raise ValueError(f"SE shape {standard_error.shape} does not match R shape {coordinates.shape}")
        noise = np.sqrt(standard_error[plus] ** 2 + standard_error[minus] ** 2) / denominator

    return MeasuredHVP(
        anchor_ids=pairs.anchor_ids[selected],
        anchor_rows=pairs.anchor_rows[selected],
        r0=coordinates[pairs.anchor_rows[selected]],
        v=v,
        hvp=hvp,
        realized_displacement=realized,
        noise=noise,
    )


def evaluate_model_hvp(model, params, r0, v, mask, species) -> np.ndarray:
    """Evaluate ``d grad(U) / dR`` along each supplied direction with JAX."""

    import jax
    import jax.numpy as jnp

    r0_j = jnp.asarray(r0)
    v_j = jnp.asarray(v)
    mask_j = jnp.asarray(mask)
    species_j = jnp.asarray(species)

    def energy(coordinates):
        return model.compute_energy(params, coordinates, mask_j, species_j)

    gradient = jax.grad(energy)

    def anchor_hvp(coordinates, directions):
        return jax.vmap(lambda direction: jax.jvp(gradient, (coordinates,), (direction,))[1])(directions)

    return np.asarray(jax.jit(jax.vmap(anchor_hvp))(r0_j, v_j))


def summarize_hvp(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Return alignment, norm-ratio, and relative-error medians."""

    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.shape != predicted.shape or reference.ndim < 2 or reference.shape[-1] != 3:
        raise ValueError(
            "HVP arrays must have matching shapes ending in 3, "
            f"got {reference.shape} and {predicted.shape}"
        )

    axes = tuple(range(reference.ndim - 2, reference.ndim))
    reference_norm = np.linalg.norm(reference, axis=axes)
    predicted_norm = np.linalg.norm(predicted, axis=axes)
    dot = np.sum(reference * predicted, axis=axes)
    denominator = reference_norm * predicted_norm
    valid = (
        np.isfinite(reference_norm)
        & np.isfinite(predicted_norm)
        & np.isfinite(dot)
        & (reference_norm > 0)
        & (predicted_norm > 0)
    )
    if not np.any(valid):
        raise ValueError("no non-zero finite HVP vectors available for summary")

    cosine = dot[valid] / denominator[valid]
    norm_ratio = predicted_norm[valid] / reference_norm[valid]
    relative_error = np.linalg.norm(predicted - reference, axis=axes)[valid] / reference_norm[valid]
    return {
        "count": int(valid.sum()),
        "cosine_median": float(np.median(cosine)),
        "norm_ratio_median": float(np.median(norm_ratio)),
        "relative_error_median": float(np.median(relative_error)),
    }


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    """Parse ``LABEL=config.yaml:params.pkl`` without resolving paths."""

    if "=" not in spec:
        raise ValueError(f"model specification must be LABEL=config:params, got {spec!r}")
    label, paths = spec.split("=", 1)
    if not label.strip() or ":" not in paths:
        raise ValueError(f"model specification must be LABEL=config:params, got {spec!r}")
    config, params = paths.split(":", 1)
    if not config.strip() or not params.strip():
        raise ValueError(f"model specification must be LABEL=config:params, got {spec!r}")
    return label.strip(), config.strip(), params.strip()


def output_key(label: str) -> str:
    """Make a model label safe for use as an NPZ field name."""

    key = "".join(character if character.isalnum() else "_" for character in label).strip("_")
    return key or "model"
