"""Optional differentiable sampling biases for direct JAX MD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from training.edge_distance_gate import _dihedral_degrees


def wrap_degrees(angle: jax.Array) -> jax.Array:
    """Wrap degrees to [-180, 180)."""
    return jnp.mod(angle + 180.0, 360.0) - 180.0


@dataclass(frozen=True)
class PeriodicDihedralHarmonicBias:
    """Periodic harmonic bias ``0.5 * k * delta(phi, center)^2``."""

    indices: tuple[int, int, int, int]
    center_deg: float
    k_kcal_per_mol_rad2: float
    shift_deg: float = 180.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PeriodicDihedralHarmonicBias":
        indices_raw: Sequence[int] = cfg.get("indices", ())
        if len(indices_raw) != 4:
            raise ValueError("periodic_dihedral_harmonic bias requires four indices")
        indices = tuple(int(i) for i in indices_raw)
        if min(indices) < 0:
            raise ValueError("bias indices must be non-negative")
        k = float(cfg.get("k_kcal_per_mol_rad2", 0.0))
        if k < 0.0:
            raise ValueError("bias force constant must be non-negative")
        return cls(
            indices=indices,
            center_deg=float(cfg.get("center_deg", 0.0)),
            k_kcal_per_mol_rad2=k,
            shift_deg=float(cfg.get("shift_deg", 180.0)),
        )

    @property
    def indices_array(self) -> jax.Array:
        return jnp.asarray(self.indices, dtype=jnp.int32)

    def cv_degrees(self, R: jax.Array) -> jax.Array:
        # The shared JAX helper uses the Charron-shifted convention. Convert
        # back to conventional, then apply the explicit configured shift.
        charron_deg = _dihedral_degrees(R, self.indices_array)
        conventional_deg = wrap_degrees(charron_deg - 180.0)
        return wrap_degrees(conventional_deg + self.shift_deg)

    def delta_degrees(self, R: jax.Array) -> jax.Array:
        return wrap_degrees(self.cv_degrees(R) - self.center_deg)

    def energy(self, R: jax.Array) -> jax.Array:
        delta_rad = jnp.deg2rad(self.delta_degrees(R))
        return 0.5 * self.k_kcal_per_mol_rad2 * delta_rad * delta_rad

    def force(self, R: jax.Array) -> jax.Array:
        return -jax.grad(self.energy)(R)

    def metadata(self) -> dict[str, Any]:
        return {
            "bias_type": "periodic_dihedral_harmonic",
            "bias_indices": list(self.indices),
            "bias_center_deg": self.center_deg,
            "bias_shift_deg": self.shift_deg,
            "bias_k_kcal_per_mol_rad2": self.k_kcal_per_mol_rad2,
        }


class BiasedEnergyModel:
    """Duck-typed CombinedModel wrapper adding a fixed sampling bias."""

    def __init__(self, base_model: Any, bias: PeriodicDihedralHarmonicBias):
        self.base_model = base_model
        self.bias = bias

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def energy_fn_template(self, params: dict[str, Any]):
        base_energy_fn = self.base_model.energy_fn_template(params)
        bias = self.bias
        # Preserve the original computation graph exactly for zero-bias
        # deterministic regression tests.
        if bias.k_kcal_per_mol_rad2 == 0.0:
            return base_energy_fn

        def energy_fn(R: jax.Array, neighbor: Any, **kwargs) -> jax.Array:
            return base_energy_fn(R, neighbor, **kwargs) + bias.energy(R)

        return energy_fn

    def compute_components(self, params: dict[str, Any], R: jax.Array, *args, **kwargs):
        components = dict(self.base_model.compute_components(params, R, *args, **kwargs))
        e_bias = self.bias.energy(R)
        components["E_bias"] = e_bias
        components["E_total"] = components["E_total"] + e_bias
        return components


def build_bias(cfg: Mapping[str, Any] | None) -> PeriodicDihedralHarmonicBias | None:
    """Build an optional MD bias from the ``md.bias`` mapping."""
    if not cfg or not bool(cfg.get("enabled", True)):
        return None
    bias_type = str(cfg.get("type", "")).strip().lower()
    if bias_type != "periodic_dihedral_harmonic":
        raise ValueError(
            f"Unknown md.bias.type {bias_type!r}; expected 'periodic_dihedral_harmonic'."
        )
    return PeriodicDihedralHarmonicBias.from_config(cfg)
