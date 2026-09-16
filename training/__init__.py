"""Training infrastructure for force matching."""

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from .optimizers import (
    create_optimizer,
    create_optimizer_from_config,
    get_available_optimizers,
    register_optimizer,
)
from .trainer import Trainer

__all__ = [
    "create_optimizer",
    "create_optimizer_from_config",
    "get_available_optimizers",
    "register_optimizer",
    "Trainer",
]
