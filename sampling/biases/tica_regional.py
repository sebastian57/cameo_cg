"""TICA-guided regional acquisition bias.

Generalised from SAMPLING/tica_regional_weighting/tica_bias.py, which hardcoded five
beads in two places (the shape check and the Jacobian allocation). Bead count is now
taken from the artifact's pair indices.

The bias is

    V(z) = -kT log( pi_target(z) / p_ref(z) ) + C

with both densities smooth KDEs over occupied reference-cell centres, plus a weak flat
outer wall beyond the frozen grid. Low energy where the target/reference ratio is large,
i.e. regions the acquisition target wants more of.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.special import logsumexp

from .base import BiasTerm, register_bias

__all__ = ["TICAProjection", "SmoothTICABias", "TICARegionalBias"]


@dataclass(frozen=True)
class TICAProjection:
    """Frozen pair-distance TICA projection, valid for any bead count."""

    pairs: np.ndarray            # (n_pairs, 2) bead indices
    mean: np.ndarray             # (n_pairs,)
    coefficients: np.ndarray     # (n_pairs, n_tics)
    # Bead count of the mapping this projection was built for. Persisted in the
    # artifact rather than inferred: `max(pairs)+1` undercounts whenever the feature
    # set omits the last bead (any non-all-pairs selection), and the projection then
    # silently claims a smaller mapping than it was fitted on.
    declared_n_beads: int | None = None

    @property
    def n_beads(self) -> int:
        if self.declared_n_beads is not None:
            return int(self.declared_n_beads)
        return int(self.pairs.max()) + 1

    def validate(self) -> None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError(f"pair_indices must be (n_pairs, 2), got {self.pairs.shape}")
        if self.pairs.min() < 0:
            raise ValueError(f"negative bead index in pair_indices: {self.pairs.min()}")
        if len(self.mean) != len(self.pairs):
            raise ValueError(
                f"tica_mean has {len(self.mean)} entries for {len(self.pairs)} pairs"
            )
        if self.coefficients.shape[0] != len(self.pairs):
            raise ValueError(
                f"tica_coefficients has {self.coefficients.shape[0]} rows for "
                f"{len(self.pairs)} pairs"
            )
        if self.declared_n_beads is not None:
            if int(self.pairs.max()) >= int(self.declared_n_beads):
                raise ValueError(
                    f"artifact declares {self.declared_n_beads} beads but pair_indices "
                    f"reference bead {int(self.pairs.max())}"
                )

    def features(self, positions_A: np.ndarray) -> np.ndarray:
        p = np.asarray(positions_A, dtype=np.float64)
        delta = p[..., self.pairs[:, 0], :] - p[..., self.pairs[:, 1], :]
        return np.linalg.norm(delta, axis=-1)

    def transform(self, positions_A: np.ndarray) -> np.ndarray:
        return (self.features(positions_A) - self.mean) @ self.coefficients

    def value_and_jacobian(self, positions_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(positions_A, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"expected one (n_beads, 3) structure, got {p.shape}")
        n_beads = p.shape[0]
        if n_beads <= int(self.pairs.max()):
            raise ValueError(
                f"projection references bead {int(self.pairs.max())} but structure has {n_beads}"
            )
        delta = p[self.pairs[:, 0]] - p[self.pairs[:, 1]]
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances < 1.0e-10):
            raise ValueError("coincident TICA pair")
        unit = delta / distances[:, None]
        value = (distances - self.mean) @ self.coefficients
        jacobian = np.zeros((self.coefficients.shape[1], n_beads, 3), dtype=np.float64)
        for k, (i, j) in enumerate(self.pairs):
            contribution = self.coefficients[k, :, None] * unit[k]
            jacobian[:, i] += contribution
            jacobian[:, j] -= contribution
        return value, jacobian


@dataclass(frozen=True)
class SmoothTICABias:
    projection: TICAProjection
    centers: np.ndarray
    reference_weights: np.ndarray
    target_weights: np.ndarray
    bandwidth: np.ndarray
    kbt_kcal_mol: float
    bounds: np.ndarray
    wall_k_kcal_mol: np.ndarray
    energy_offset_kcal_mol: float = 0.0

    @staticmethod
    def _log_density_and_gradient(z, centers, weights, bandwidth):
        delta = z[None, :] - centers
        exponents = np.log(weights) - 0.5 * np.sum((delta / bandwidth) ** 2, axis=1)
        log_density = float(logsumexp(exponents))
        responsibilities = np.exp(exponents - log_density)
        gradient = np.sum(responsibilities[:, None] * (-delta / bandwidth**2), axis=0)
        return log_density, gradient

    def tica_energy_gradient(self, z: np.ndarray) -> Tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        if getattr(self, "attractor_weights", None) is not None:
            return self._attractor_energy_gradient(z)
        log_ref, grad_ref = self._log_density_and_gradient(
            z, self.centers, self.reference_weights, self.bandwidth
        )
        log_target, grad_target = self._log_density_and_gradient(
            z, self.centers, self.target_weights, self.bandwidth
        )
        energy = -self.kbt_kcal_mol * (log_target - log_ref)
        gradient = -self.kbt_kcal_mol * (grad_target - grad_ref)
        for axis in range(len(self.bounds)):
            lower, upper = self.bounds[axis]
            if z[axis] < lower:
                d = z[axis] - lower
            elif z[axis] > upper:
                d = z[axis] - upper
            else:
                continue
            energy += 0.5 * self.wall_k_kcal_mol[axis] * d**2
            gradient[axis] += self.wall_k_kcal_mol[axis] * d
        return energy + self.energy_offset_kcal_mol, gradient

    def _attractor_energy_gradient(self, z: np.ndarray) -> Tuple[float, np.ndarray]:
        """V(z) = A * (1 - rho_t(z)/rho_max):  wells ON transition cells, nothing else.

        The default log-ratio form is anti-density BY CONSTRUCTION -- it contains
        log(target/p_ref), and wherever p_ref -> 0 that ratio diverges no matter what the
        numerator holds. Reweighting `exploration` therefore cannot redirect it: measured
        2026-08-05, an 85%-transition rebuild correlated +0.9907 with the original map and
        still pulled hardest into alphaL, a genuine but sparsely populated basin.

        This form has no p_ref in it at all. Basins are neither pushed nor pulled; only
        cells with high committor/transition relevance attract, with depth `A`. Enriching
        transitions is safe because the mean force there points outward into the basins,
        so the basin RATIOS -- the part of the FES the model must get right -- are left to
        the reference data.
        """
        log_t, grad_log_t = self._log_density_and_gradient(
            z, self.centers, self.attractor_weights, self.bandwidth
        )
        rho = float(np.exp(log_t)) / self.attractor_norm
        energy = self.attractor_depth * (1.0 - rho)
        # d/dz [ -A*rho ] = -A * rho * dlog(rho)/dz
        gradient = -self.attractor_depth * rho * np.asarray(grad_log_t, dtype=np.float64)
        for axis in range(len(self.bounds)):
            lower, upper = self.bounds[axis]
            if z[axis] < lower:
                d = z[axis] - lower
            elif z[axis] > upper:
                d = z[axis] - upper
            else:
                continue
            energy += 0.5 * self.wall_k_kcal_mol[axis] * d**2
            gradient[axis] += self.wall_k_kcal_mol[axis] * d
        return energy, gradient

    def evaluate_A(self, positions_A: np.ndarray):
        z, jacobian = self.projection.value_and_jacobian(positions_A)
        energy, gradient_z = self.tica_energy_gradient(z)
        gradient_R = np.einsum("k,kij->ij", gradient_z, jacobian)
        return energy, -gradient_R, z

    @classmethod
    def load(cls, path) -> "SmoothTICABias":
        with np.load(path) as data:
            # Artifacts built before n_beads was persisted fall back to inference;
            # the value is only trustworthy for an all-pairs feature set.
            declared = int(data["n_beads"]) if "n_beads" in data.files else None
            projection = TICAProjection(
                np.asarray(data["pair_indices"], dtype=np.int64),
                np.asarray(data["tica_mean"], dtype=np.float64),
                np.asarray(data["tica_coefficients"], dtype=np.float64),
                declared_n_beads=declared,
            )
            projection.validate()
            obj = cls(
                projection=projection,
                centers=np.asarray(data["centers"], dtype=np.float64),
                reference_weights=np.asarray(data["reference_weights"], dtype=np.float64),
                target_weights=np.asarray(data["target_weights"], dtype=np.float64),
                bandwidth=np.asarray(data["bandwidth"], dtype=np.float64),
                kbt_kcal_mol=float(data["kbt_kcal_mol"]),
                bounds=np.asarray(data["bounds"], dtype=np.float64),
                wall_k_kcal_mol=np.asarray(data["wall_k_kcal_mol"], dtype=np.float64),
                energy_offset_kcal_mol=float(data["energy_offset_kcal_mol"]),
            )
            # frozen dataclass -> set through object.__setattr__
            if "attractor_weights" in data.files:
                object.__setattr__(obj, "attractor_weights",
                                   np.asarray(data["attractor_weights"], dtype=np.float64))
                object.__setattr__(obj, "attractor_depth", float(data["attractor_depth"]))
                object.__setattr__(obj, "attractor_norm", float(data["attractor_norm"]))
            else:
                object.__setattr__(obj, "attractor_weights", None)
            return obj


@register_bias("tica_regional")
class TICARegionalBias(BiasTerm):
    """Acquisition bias toward an under-sampled TICA target distribution.

    Config keys: bias_npz, scale (default 1.0), name, enabled.
    """

    def __init__(self, bias_npz: str, scale: float = 1.0, name: str | None = None,
                 enabled: bool = True):
        super().__init__(name=name or "tica_regional", enabled=enabled)
        self.bias_npz = str(bias_npz)
        self.scale = float(scale)
        self._bias = SmoothTICABias.load(Path(self.bias_npz))
        self._last_z = np.zeros(self._bias.projection.coefficients.shape[1])
        self._last_energy = 0.0

    def n_beads_expected(self) -> int:
        return self._bias.projection.n_beads

    def evaluate(self, positions_A: np.ndarray, step: int):
        energy, forces, z = self._bias.evaluate_A(positions_A)
        self._last_z = np.asarray(z, dtype=np.float64)
        self._last_energy = float(energy) * self.scale
        return self._last_energy, forces * self.scale

    def diagnostics(self) -> Dict[str, Any]:
        d = {"%s_energy_kcal" % self.name: self._last_energy}
        for i, v in enumerate(self._last_z[:2]):
            d["%s_tic%d" % (self.name, i + 1)] = float(v)
        return d

    def describe(self) -> str:
        return (
            f"{self.name} (tica_regional): {self._bias.projection.n_beads} beads, "
            f"{len(self._bias.projection.pairs)} pairs, scale={self.scale}, "
            f"artifact={Path(self.bias_npz).name}"
        )
