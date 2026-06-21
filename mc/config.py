"""Configuration objects for Monte Carlo ensemble samplers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class HMCConfig:
    """Settings for Hamiltonian Monte Carlo ensemble sampling.

    HMC uses leapfrog integration with Metropolis acceptance, combining the
    continuous exploration of MD with MC safety via acceptance checks.
    """

    n_chains: int = 8
    steps_per_iteration: int = 200
    burn_in_steps: int = 50
    sample_stride: int = 10
    step_size: float = 1.0e-3
    num_integration_steps: int = 10
    inverse_mass_matrix: List[float] | float = field(default_factory=lambda: [1.0])
    beta: float = 1.0

    @property
    def retained_samples_per_chain(self) -> int:
        remaining = int(self.steps_per_iteration) - int(self.burn_in_steps)
        if remaining <= 0:
            return 0
        return remaining // int(self.sample_stride)

    @property
    def total_samples(self) -> int:
        return int(self.n_chains) * int(self.retained_samples_per_chain)

    def validate(self) -> None:
        if int(self.n_chains) <= 0:
            raise ValueError(f"mc.hmc.n_chains must be > 0, got {self.n_chains}.")
        if int(self.steps_per_iteration) <= 0:
            raise ValueError("mc.hmc.steps_per_iteration must be > 0.")
        if int(self.burn_in_steps) < 0:
            raise ValueError("mc.hmc.burn_in_steps must be >= 0.")
        if int(self.sample_stride) <= 0:
            raise ValueError("mc.hmc.sample_stride must be > 0.")
        if float(self.step_size) <= 0.0:
            raise ValueError("mc.hmc.step_size must be > 0.")
        if int(self.num_integration_steps) <= 0:
            raise ValueError("mc.hmc.num_integration_steps must be > 0.")
        if float(self.beta) <= 0.0:
            raise ValueError("mc.hmc.beta must be > 0.")
        if self.retained_samples_per_chain <= 0:
            raise ValueError(
                "HMC rollout settings retain zero samples; increase steps_per_iteration "
                "or reduce burn_in_steps/sample_stride."
            )


@dataclass(frozen=True)
class MALAConfig:
    """Settings for Metropolis-adjusted Langevin ensemble sampling.

    The target distribution is defined by the caller's log-density. For RE this
    should be ``log p(R) = -beta * U_total(R)``, where ``U_total`` includes both
    fixed priors and the ML energy.
    """

    n_chains: int = 8
    steps_per_iteration: int = 200
    burn_in_steps: int = 50
    sample_stride: int = 10
    step_size: float = 1.0e-3
    beta: float = 1.0

    @property
    def retained_samples_per_chain(self) -> int:
        remaining = int(self.steps_per_iteration) - int(self.burn_in_steps)
        if remaining <= 0:
            return 0
        return remaining // int(self.sample_stride)

    @property
    def total_samples(self) -> int:
        return int(self.n_chains) * int(self.retained_samples_per_chain)

    def validate(self) -> None:
        if int(self.n_chains) <= 0:
            raise ValueError(f"mc.mala.n_chains must be > 0, got {self.n_chains}.")
        if int(self.steps_per_iteration) <= 0:
            raise ValueError("mc.mala.steps_per_iteration must be > 0.")
        if int(self.burn_in_steps) < 0:
            raise ValueError("mc.mala.burn_in_steps must be >= 0.")
        if int(self.sample_stride) <= 0:
            raise ValueError("mc.mala.sample_stride must be > 0.")
        if float(self.step_size) <= 0.0:
            raise ValueError("mc.mala.step_size must be > 0.")
        if float(self.beta) <= 0.0:
            raise ValueError("mc.mala.beta must be > 0.")
        if self.retained_samples_per_chain <= 0:
            raise ValueError(
                "MALA rollout settings retain zero samples; increase steps_per_iteration "
                "or reduce burn_in_steps/sample_stride."
            )
