"""BlackJAX-backed MCMC samplers with the RE sampler interface."""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp

from .blackjax_compat import import_blackjax
from .config import HMCConfig, MALAConfig


class MALAChainResult(NamedTuple):
    samples: jax.Array
    final_position: jax.Array
    acceptance_rate: jax.Array
    logdensity_mean: jax.Array
    has_nan_or_inf: jax.Array


class HMCChainResult(NamedTuple):
    samples: jax.Array
    final_position: jax.Array
    acceptance_rate: jax.Array
    logdensity_mean: jax.Array
    has_nan_or_inf: jax.Array
    num_divergent: jax.Array


def run_mala_chain(
    initial_position: jax.Array,
    logdensity_fn: Callable[[jax.Array], jax.Array],
    *,
    rng_key: jax.Array,
    steps: int,
    burn_in: int,
    sample_stride: int,
    step_size: float,
    project_position_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> MALAChainResult:
    """Run one MALA chain and retain strict post-burn-in stride samples.

    ``logdensity_fn`` must describe the exact target distribution. For relative
    entropy this is ``-beta * U_total(R)``. ``project_position_fn`` is only for
    nonphysical padded coordinates; using it for physical constraints would
    change the sampled ensemble unless those constraints are part of the model.
    """
    blackjax = import_blackjax()
    if project_position_fn is None:
        project_position_fn = lambda x: x

    n_retained = (int(steps) - int(burn_in)) // int(sample_stride)
    if n_retained <= 0:
        raise ValueError("MALA chain retained zero samples; check rollout settings.")

    initial_position = project_position_fn(jnp.asarray(initial_position, dtype=jnp.float32))
    algorithm = blackjax.mala(logdensity_fn, float(step_size))
    state = algorithm.init(initial_position)
    retained0 = jnp.zeros((n_retained,) + initial_position.shape, dtype=initial_position.dtype)

    def scan_step(carry, step_idx):
        state, retained, retained_count, accept_sum, logdensity_sum, bad = carry
        key = jax.random.fold_in(rng_key, step_idx)
        state, info = algorithm.step(key, state)
        position = jax.lax.stop_gradient(project_position_fn(state.position))
        after_burn = step_idx > int(burn_in)
        on_stride = (step_idx - int(burn_in)) % int(sample_stride) == 0
        should_retain = jnp.logical_and(after_burn, on_stride)
        safe_index = jnp.minimum(retained_count, n_retained - 1)
        updated = jax.lax.dynamic_update_index_in_dim(retained, position[None, ...], safe_index, axis=0)
        retained = jnp.where(should_retain, updated, retained)
        retained_count = retained_count + should_retain.astype(jnp.int32)
        accept_sum = accept_sum + info.is_accepted.astype(jnp.float32)
        logdensity_sum = logdensity_sum + jnp.asarray(state.logdensity, dtype=jnp.float32)
        bad = jnp.logical_or(bad, jnp.any(~jnp.isfinite(position)))
        bad = jnp.logical_or(bad, ~jnp.isfinite(state.logdensity))
        return (state, retained, retained_count, accept_sum, logdensity_sum, bad), None

    step_ids = jnp.arange(1, int(steps) + 1)
    (state, retained, _, accept_sum, logdensity_sum, bad), _ = jax.lax.scan(
        scan_step,
        (
            state,
            retained0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
        ),
        step_ids,
    )
    denom = jnp.asarray(float(steps), dtype=jnp.float32)
    return MALAChainResult(
        samples=retained,
        final_position=project_position_fn(state.position),
        acceptance_rate=accept_sum / denom,
        logdensity_mean=logdensity_sum / denom,
        has_nan_or_inf=bad,
    )


def run_hmc_chain(
    initial_position: jax.Array,
    logdensity_fn: Callable[[jax.Array], jax.Array],
    *,
    rng_key: jax.Array,
    steps: int,
    burn_in: int,
    sample_stride: int,
    step_size: float,
    num_integration_steps: int,
    inverse_mass_matrix: jax.Array,
    project_position_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> HMCChainResult:
    """Run one HMC chain and retain strict post-burn-in stride samples.

    HMC uses leapfrog integration which explores conformational space more
    efficiently than MALA's Gaussian proposals, especially for collective
    motions and barrier crossing.
    """
    blackjax = import_blackjax()
    if project_position_fn is None:
        project_position_fn = lambda x: x

    n_retained = (int(steps) - int(burn_in)) // int(sample_stride)
    if n_retained <= 0:
        raise ValueError("HMC chain retained zero samples; check rollout settings.")

    initial_position = project_position_fn(jnp.asarray(initial_position, dtype=jnp.float32))
    state = blackjax.hmc.init(initial_position, logdensity_fn)
    retained0 = jnp.zeros((n_retained,) + initial_position.shape, dtype=initial_position.dtype)

    kernel = blackjax.hmc(logdensity_fn, step_size=float(step_size), inverse_mass_matrix=inverse_mass_matrix, num_integration_steps=int(num_integration_steps))

    def scan_step(carry, step_idx):
        state, retained, retained_count, accept_sum, logdensity_sum, divergent_sum, bad = carry
        key = jax.random.fold_in(rng_key, step_idx)
        state, info = kernel.step(key, state)
        position = jax.lax.stop_gradient(project_position_fn(state.position))
        after_burn = step_idx > int(burn_in)
        on_stride = (step_idx - int(burn_in)) % int(sample_stride) == 0
        should_retain = jnp.logical_and(after_burn, on_stride)
        safe_index = jnp.minimum(retained_count, n_retained - 1)
        updated = jax.lax.dynamic_update_index_in_dim(retained, position[None, ...], safe_index, axis=0)
        retained = jnp.where(should_retain, updated, retained)
        retained_count = retained_count + should_retain.astype(jnp.int32)
        accept_sum = accept_sum + info.is_accepted.astype(jnp.float32)
        divergent_sum = divergent_sum + info.is_divergent.astype(jnp.int32)
        logdensity_sum = logdensity_sum + jnp.asarray(state.logdensity, dtype=jnp.float32)
        bad = jnp.logical_or(bad, jnp.any(~jnp.isfinite(position)))
        bad = jnp.logical_or(bad, ~jnp.isfinite(state.logdensity))
        return (state, retained, retained_count, accept_sum, logdensity_sum, divergent_sum, bad), None

    step_ids = jnp.arange(1, int(steps) + 1)
    (state, retained, _, accept_sum, logdensity_sum, divergent_sum, bad), _ = jax.lax.scan(
        scan_step,
        (
            state,
            retained0,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        ),
        step_ids,
    )
    denom = jnp.asarray(float(steps), dtype=jnp.float32)
    return HMCChainResult(
        samples=retained,
        final_position=project_position_fn(state.position),
        acceptance_rate=accept_sum / denom,
        logdensity_mean=logdensity_sum / denom,
        has_nan_or_inf=bad,
        num_divergent=divergent_sum,
    )


class BlackJaxMALASampler:
    """Generate model samples from ``exp(-beta * U_total)`` using MALA.

    This intentionally mirrors ``training.relative_entropy.InProcessLangevinSampler``
    so RE training can swap samplers without changing the gradient estimator.
    """

    def __init__(self, energy_fn: Callable[..., jax.Array], config: MALAConfig):
        config.validate()
        self.energy_fn = energy_fn
        self.config = config
        self._run_chains = jax.jit(jax.vmap(self._run_single))

    def _run_single(self, R0, mask, species, key) -> MALAChainResult:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        R0 = jnp.asarray(R0, dtype=jnp.float32)

        def project_position(R):
            return jnp.where(mask[:, None] > 0, R, R0)

        def logdensity_fn(R):
            R_projected = project_position(R)
            energy = self.energy_fn(R_projected, mask=mask, species=species)
            return -float(self.config.beta) * energy

        return run_mala_chain(
            R0,
            logdensity_fn,
            rng_key=key,
            steps=int(self.config.steps_per_iteration),
            burn_in=int(self.config.burn_in_steps),
            sample_stride=int(self.config.sample_stride),
            step_size=float(self.config.step_size),
            project_position_fn=project_position,
        )

    def run(self, R0, mask, species, rng_key) -> Dict[str, Any]:
        R0 = jnp.asarray(R0, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        if R0.shape[0] != int(self.config.n_chains):
            raise ValueError(f"MALA sampler expected n_chains={self.config.n_chains}, got {R0.shape[0]}.")
        if mask.shape[0] != R0.shape[0] or species.shape[0] != R0.shape[0]:
            raise ValueError("R0, mask, and species must have matching chain axes.")

        keys = jax.random.split(jnp.asarray(rng_key), int(self.config.n_chains))
        chain_results = self._run_chains(R0, mask, species, keys)
        R_samples = chain_results.samples.reshape((-1,) + R0.shape[1:])
        repeats = int(self.config.retained_samples_per_chain)
        mask_samples = jnp.repeat(mask, repeats, axis=0)
        species_samples = jnp.repeat(species, repeats, axis=0)
        diagnostics = {
            "has_nan_or_inf": bool(jax.device_get(jnp.any(chain_results.has_nan_or_inf))),
            "n_samples": int(R_samples.shape[0]),
            "acceptance_rate_mean": float(jax.device_get(jnp.mean(chain_results.acceptance_rate))),
            "acceptance_rate_min": float(jax.device_get(jnp.min(chain_results.acceptance_rate))),
            "acceptance_rate_max": float(jax.device_get(jnp.max(chain_results.acceptance_rate))),
            "logdensity_mean": float(jax.device_get(jnp.mean(chain_results.logdensity_mean))),
        }
        return {
            "R": R_samples,
            "mask": mask_samples,
            "species": species_samples,
            "diagnostics": diagnostics,
        }


class BlackJaxHMCSampler:
    """Generate model samples from ``exp(-beta * U_total)`` using HMC.

    HMC uses leapfrog integration which explores conformational space more
    efficiently than MALA, especially for collective motions. The Metropolis
    acceptance check provides safety against numerical instability.
    """

    def __init__(self, energy_fn: Callable[..., jax.Array], config: HMCConfig):
        config.validate()
        self.energy_fn = energy_fn
        self.config = config
        self._run_chains = jax.jit(jax.vmap(self._run_single))

    def _run_single(self, R0, mask, species, key) -> HMCChainResult:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        R0 = jnp.asarray(R0, dtype=jnp.float32)

        def project_position(R):
            return jnp.where(mask[:, None] > 0, R, R0)

        def logdensity_fn(R):
            R_projected = project_position(R)
            energy = self.energy_fn(R_projected, mask=mask, species=species)
            return -float(self.config.beta) * energy

        inv_mass = jnp.asarray(self.config.inverse_mass_matrix, dtype=jnp.float32)
        if inv_mass.ndim == 1:
            inv_mass = jnp.broadcast_to(inv_mass, R0.shape)

        return run_hmc_chain(
            R0,
            logdensity_fn,
            rng_key=key,
            steps=int(self.config.steps_per_iteration),
            burn_in=int(self.config.burn_in_steps),
            sample_stride=int(self.config.sample_stride),
            step_size=float(self.config.step_size),
            num_integration_steps=int(self.config.num_integration_steps),
            inverse_mass_matrix=inv_mass,
            project_position_fn=project_position,
        )

    def run(self, R0, mask, species, rng_key) -> Dict[str, Any]:
        R0 = jnp.asarray(R0, dtype=jnp.float32)
        mask = jnp.asarray(mask, dtype=jnp.float32)
        species = jnp.asarray(species, dtype=jnp.int32)
        if R0.shape[0] != int(self.config.n_chains):
            raise ValueError(f"HMC sampler expected n_chains={self.config.n_chains}, got {R0.shape[0]}.")
        if mask.shape[0] != R0.shape[0] or species.shape[0] != R0.shape[0]:
            raise ValueError("R0, mask, and species must have matching chain axes.")

        keys = jax.random.split(jnp.asarray(rng_key), int(self.config.n_chains))
        chain_results = self._run_chains(R0, mask, species, keys)
        R_samples = chain_results.samples.reshape((-1,) + R0.shape[1:])
        repeats = int(self.config.retained_samples_per_chain)
        mask_samples = jnp.repeat(mask, repeats, axis=0)
        species_samples = jnp.repeat(species, repeats, axis=0)
        diagnostics = {
            "has_nan_or_inf": bool(jax.device_get(jnp.any(chain_results.has_nan_or_inf))),
            "n_samples": int(R_samples.shape[0]),
            "acceptance_rate_mean": float(jax.device_get(jnp.mean(chain_results.acceptance_rate))),
            "acceptance_rate_min": float(jax.device_get(jnp.min(chain_results.acceptance_rate))),
            "acceptance_rate_max": float(jax.device_get(jnp.max(chain_results.acceptance_rate))),
            "logdensity_mean": float(jax.device_get(jnp.mean(chain_results.logdensity_mean))),
            "num_divergent_mean": float(jax.device_get(jnp.mean(chain_results.num_divergent))),
        }
        return {
            "R": R_samples,
            "mask": mask_samples,
            "species": species_samples,
            "diagnostics": diagnostics,
        }
