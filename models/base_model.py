"""
Abstract base class for ML energy model wrappers.

All ML backends (Allegro, MACE, PaiNN, cuEq variants) implement this
interface so they can be used interchangeably by CombinedModel.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def resolve_compute_dtype(config) -> Tuple[str, jnp.dtype]:
    """Resolve model compute dtype from config."""
    name = str(config.get_compute_dtype()).lower()
    if name == "bfloat16":
        return name, jnp.bfloat16
    return "float32", jnp.float32


# ---------------------------------------------------------------------------
# Model registry: ml_model_type string -> model class
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, type] = {}


def register_ml_model(*names: str):
    """Class decorator that registers an ML model under one or more names.

    Usage::

        @register_ml_model("mace")
        class MACEModel(BaseMLModel):
            ...
    """
    def wrapper(cls):
        for n in names:
            _MODEL_REGISTRY[n] = cls
        return cls
    return wrapper


def get_ml_model_class(name: str) -> type:
    """Look up a registered ML model class by config name.

    Raises ValueError with available names if not found.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown ml_model type: '{name}'. "
            f"Registered: {', '.join(sorted(_MODEL_REGISTRY))}"
        )
    return _MODEL_REGISTRY[name]


def get_available_ml_models():
    """Return sorted list of registered ML model type names."""
    return sorted(_MODEL_REGISTRY)


class BaseMLModel(ABC):
    """Interface contract for ML energy model wrappers.

    Subclasses must implement:
    - ``__init__`` (model-specific setup)
    - ``initialize_params``
    - ``compute_energy``

    Provided by base class:
    - ``compute_energy_and_forces`` (autodiff of compute_energy)
    - Common property accessors
    """

    # Subclasses must set these in __init__
    config: Any
    N_max: int
    cutoff: float
    n_species: int
    displacement: Any
    nneigh_fn: Any
    nbrs_init: Any

    @abstractmethod
    def initialize_params(self, rng_key: jax.random.PRNGKey) -> Any:
        """Initialize model parameters from a random key."""
        ...

    @abstractmethod
    def compute_energy(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute scalar energy for coordinates *R*."""
        ...

    def compute_energy_and_forces(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute energy and forces via ``-jax.grad(E)(R)``."""
        def energy_fn(R_):
            return self.compute_energy(
                params, R_, mask, species, neighbor, segment_id=segment_id
            )
        E = energy_fn(R)
        F = -jax.grad(energy_fn)(R)
        return E, F

    def compute_energy_for_export(
        self,
        params: Any,
        R: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        neighbor: Optional[Any] = None,
        segment_id: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Export-time energy hook; defaults to the runtime energy path."""
        return self.compute_energy(
            params, R, mask, species, neighbor, segment_id=segment_id
        )

    @property
    @abstractmethod
    def model_apply_fn(self):
        """Raw Haiku apply function (for exporter compatibility)."""
        ...

    @property
    def initial_neighbors(self) -> Any:
        """Initial neighbor list allocated during setup."""
        return self.nbrs_init
