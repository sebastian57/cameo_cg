"""Bias-term interface and registry.

A bias term maps CG bead positions to an energy and forces on those beads. The server
sums whatever terms a run declares, so "teacher only", "TICA only", "both", or any
future term is a configuration choice rather than a code change.

Adding a new bias means writing one class with `evaluate()` and decorating it with
`@register_bias("name")`. Nothing in the plugin, the wire protocol or the case
generator needs to change.

Units: positions in Angstrom, energy in kcal/mol, forces in kcal/mol/A. The protocol
layer converts to PLUMED's nm / kJ.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Tuple

import numpy as np

__all__ = ["BiasTerm", "register_bias", "build_bias", "build_biases", "BIAS_REGISTRY"]

BIAS_REGISTRY: Dict[str, type] = {}


def register_bias(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        key = str(name).strip().lower()
        if key in BIAS_REGISTRY:
            raise ValueError(f"bias type {key!r} already registered")
        BIAS_REGISTRY[key] = cls
        cls.type_name = key
        return cls

    return decorator


class BiasTerm(abc.ABC):
    """One additive contribution to the sampling bias."""

    type_name: str = "abstract"

    def __init__(self, name: str | None = None, enabled: bool = True):
        self.name = name or self.type_name
        self.enabled = bool(enabled)

    @abc.abstractmethod
    def evaluate(self, positions_A: np.ndarray, step: int) -> Tuple[float, np.ndarray]:
        """Return (energy_kcal_per_mol, forces_kcal_per_mol_A) with forces shaped (n_beads, 3).

        `step` is the MD step, so terms can implement ramps or schedules.
        """

    def n_beads_expected(self) -> int | None:
        """Bead count this term requires, or None if it is shape-agnostic."""
        return None

    def diagnostics(self) -> Dict[str, Any]:
        """Per-term scalars worth logging each report interval."""
        return {}

    def describe(self) -> str:
        return f"{self.name} ({self.type_name})"


def build_bias(spec: Dict[str, Any]) -> BiasTerm:
    """Instantiate one bias from a config mapping with a `type` key."""
    if "type" not in spec:
        raise ValueError(f"bias spec needs a 'type' key, got {sorted(spec)}")
    spec = dict(spec)
    key = str(spec.pop("type")).strip().lower()
    if key not in BIAS_REGISTRY:
        raise KeyError(
            f"unknown bias type {key!r}; registered: {sorted(BIAS_REGISTRY)}"
        )
    return BIAS_REGISTRY[key](**spec)


def build_biases(specs, n_beads: int | None = None):
    """Instantiate a list of biases and check they agree on bead count."""
    terms = [build_bias(s) for s in (specs or [])]
    terms = [t for t in terms if t.enabled]
    for t in terms:
        want = t.n_beads_expected()
        if want is not None and n_beads is not None and want != n_beads:
            raise ValueError(
                f"bias {t.name!r} expects {want} beads but the mapping has {n_beads}"
            )
    return terms


def evaluate_all(terms, positions_A: np.ndarray, step: int):
    """Sum energies and forces over terms; also return the per-term breakdown."""
    positions_A = np.asarray(positions_A, dtype=np.float64)
    total_e = 0.0
    total_f = np.zeros_like(positions_A)
    per_term: Dict[str, float] = {}
    for t in terms:
        e, f = t.evaluate(positions_A, step)
        f = np.asarray(f, dtype=np.float64).reshape(positions_A.shape)
        total_e += float(e)
        total_f += f
        per_term[t.name] = float(e)
    return total_e, total_f, per_term
