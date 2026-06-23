"""Stochastic weight averaging helpers for force-matching training."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp


def _copy_tree(params: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x).copy(), params)


@dataclass
class SWAState:
    """Arithmetic running average over matching parameter pytrees."""

    stage: str
    start_epoch: int
    sample_freq_epochs: int
    use_best_params: bool = False
    averaged_params: Optional[Any] = None
    n_samples: int = 0
    sample_epochs: List[int] = field(default_factory=list)
    _tree_def: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.start_epoch < 0:
            raise ValueError(f"SWA start_epoch must be >= 0, got {self.start_epoch}.")
        if self.sample_freq_epochs < 1:
            raise ValueError(
                "SWA sample_freq_epochs must be >= 1, "
                f"got {self.sample_freq_epochs}."
            )

    def should_sample(self, epoch: int) -> bool:
        """Return True when a completed stage-local epoch should be averaged."""
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.sample_freq_epochs == 0

    def update(self, params: Any, epoch: int) -> None:
        """Add one parameter sample to the arithmetic running average."""
        tree_def = jax.tree_util.tree_structure(params)
        if self._tree_def is None:
            self._tree_def = tree_def
            self.averaged_params = _copy_tree(params)
            self.n_samples = 1
            self.sample_epochs.append(int(epoch))
            return
        if tree_def != self._tree_def:
            raise ValueError("SWA parameter tree changed between samples.")

        next_count = self.n_samples + 1
        self.averaged_params = jax.tree_util.tree_map(
            lambda avg, value: avg + (jnp.asarray(value) - avg) / next_count,
            self.averaged_params,
            params,
        )
        self.n_samples = next_count
        self.sample_epochs.append(int(epoch))

    def metadata(self) -> Dict[str, Any]:
        """Return serializable SWA sampling metadata."""
        return {
            "stage": self.stage,
            "start_epoch": int(self.start_epoch),
            "sample_freq_epochs": int(self.sample_freq_epochs),
            "use_best_params": bool(self.use_best_params),
            "sample_count": int(self.n_samples),
            "sample_epochs": list(self.sample_epochs),
        }


def save_swa_checkpoint(
    output_path: Path,
    state: SWAState,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save averaged params in the simple CAMEO checkpoint payload format."""
    if state.averaged_params is None or state.n_samples <= 0:
        raise ValueError("Cannot save SWA checkpoint before collecting samples.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata = dict(metadata or {})
    merged_metadata.update(state.metadata())
    merged_metadata.setdefault("timestamp", time.time())

    payload = {
        "params": state.averaged_params,
        "best_params": state.averaged_params,
        "metadata": merged_metadata,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    meta_path = output_path.with_suffix(".meta.pkl")
    with meta_path.open("wb") as handle:
        pickle.dump(merged_metadata, handle)
