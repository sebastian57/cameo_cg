"""Micro-batch SAM helpers for force-matching training."""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import optax
from jax import lax, value_and_grad
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from chemtrain.learn import max_likelihood


def tree_l2_norm(tree: Any) -> jax.Array:
    """Return the global L2 norm of a pytree."""
    return optax.global_norm(tree)


def sam_perturbation(params: Any, grad: Any, rho: float, epsilon: float) -> Any:
    """Return the SAM perturbation tree for one gradient estimate."""
    del params
    scale = jnp.asarray(rho) / (tree_l2_norm(grad) + jnp.asarray(epsilon))
    return jax.tree_util.tree_map(lambda g: scale * g, grad)


def apply_perturbation(params: Any, perturbation: Any) -> Any:
    """Add a perturbation pytree to params."""
    return jax.tree_util.tree_map(lambda p, e: p + e, params, perturbation)


def microbatch_sam_gradient(
    param_loss_fn: Callable[[Any, Any], Any],
    params: Any,
    batch: Any,
    rho: float,
    epsilon: float,
) -> tuple[Any, Any, Any]:
    """Return unperturbed loss/targets and SAM gradient for one micro-batch."""
    (loss, per_target_loss), grad = value_and_grad(param_loss_fn, has_aux=True)(
        params, batch
    )
    perturbation = sam_perturbation(params, grad, rho=rho, epsilon=epsilon)
    perturbed_params = apply_perturbation(params, perturbation)
    (_, _), sam_grad = value_and_grad(param_loss_fn, has_aux=True)(
        perturbed_params, batch
    )
    return loss, per_target_loss, sam_grad


def _tree_add(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return jax.tree_util.tree_map(jnp.add, a, b)


def _tree_scale(tree: Any, scale: Any) -> Any:
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def _slice_microbatch(batch: Any, index: int, microbatch_size: int) -> Any:
    start = index * microbatch_size
    return jax.tree_util.tree_map(
        lambda arr: lax.dynamic_slice_in_dim(arr, start, microbatch_size, axis=0),
        batch,
    )


def _msam_accumulated_local_grad(
    param_loss_fn: Callable[[Any, Any], Any],
    params: Any,
    batch: Any,
    microbatch_count: int,
    accum_mode: str,
    rho: float,
    epsilon: float,
) -> tuple[tuple[Any, Any], Any]:
    if microbatch_count < 1:
        raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")

    if microbatch_count == 1:
        loss, per_target_loss, sam_grad = microbatch_sam_gradient(
            param_loss_fn, params, batch, rho=rho, epsilon=epsilon
        )
        return (loss, per_target_loss), sam_grad

    if accum_mode == "concat_slice":
        local_batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        if local_batch_size % microbatch_count != 0:
            raise ValueError(
                "Local batch size must be divisible by microbatch_count. "
                f"Got local_batch_size={local_batch_size}, "
                f"microbatch_count={microbatch_count}."
            )
        microbatch_size = local_batch_size // microbatch_count
        first_batch = _slice_microbatch(batch, 0, microbatch_size)
    else:
        stacked_microbatch_count = jax.tree_util.tree_leaves(batch)[0].shape[0]
        if stacked_microbatch_count != microbatch_count:
            raise ValueError(
                "Stacked microbatch axis mismatch for stack_scan mode. "
                f"Got stacked_microbatch_count={stacked_microbatch_count}, "
                f"microbatch_count={microbatch_count}."
            )
        first_batch = jax.tree_util.tree_map(lambda arr: arr[0], batch)

    loss_0, per_target_0, grad_0 = microbatch_sam_gradient(
        param_loss_fn, params, first_batch, rho=rho, epsilon=epsilon
    )

    if accum_mode == "concat_slice":

        def _accumulate(i, carry):
            grad_sum, loss_sum, per_target_sum = carry
            micro_batch = _slice_microbatch(batch, i, microbatch_size)
            loss_i, per_target_i, grad_i = microbatch_sam_gradient(
                param_loss_fn, params, micro_batch, rho=rho, epsilon=epsilon
            )
            grad_sum = jax.tree_util.tree_map(jnp.add, grad_sum, grad_i)
            loss_sum = loss_sum + loss_i
            per_target_sum = _tree_add(per_target_sum, per_target_i)
            return grad_sum, loss_sum, per_target_sum

        grad_sum, loss_sum, per_target_sum = lax.fori_loop(
            1, microbatch_count, _accumulate, (grad_0, loss_0, per_target_0)
        )
    else:

        def _scan_accumulate(carry, micro_batch):
            grad_sum, loss_sum, per_target_sum = carry
            loss_i, per_target_i, grad_i = microbatch_sam_gradient(
                param_loss_fn, params, micro_batch, rho=rho, epsilon=epsilon
            )
            grad_sum = jax.tree_util.tree_map(jnp.add, grad_sum, grad_i)
            loss_sum = loss_sum + loss_i
            per_target_sum = _tree_add(per_target_sum, per_target_i)
            return (grad_sum, loss_sum, per_target_sum), None

        rest_batches = jax.tree_util.tree_map(lambda arr: arr[1:], batch)
        (grad_sum, loss_sum, per_target_sum), _ = lax.scan(
            _scan_accumulate,
            (grad_0, loss_0, per_target_0),
            rest_batches,
            length=microbatch_count - 1,
        )

    inv = jnp.asarray(1.0 / microbatch_count, dtype=loss_sum.dtype)
    sam_grad = jax.tree_util.tree_map(lambda x: x * inv, grad_sum)
    loss = loss_sum * inv
    per_target_loss = _tree_scale(per_target_sum, inv)
    return (loss, per_target_loss), sam_grad


def shmap_msam_update_fn(
    batched_model: Callable[[Any, Any], Any],
    loss_fn: Callable[[Any, Any], Any],
    optimizer: optax.GradientTransformation,
    penalty_fn: Callable[[Any], Any] | None = None,
    *,
    rho: float,
    epsilon: float,
) -> Callable[..., Any]:
    """Build a Chemtrain-compatible shmap update function using mSAM gradients."""
    mesh = Mesh(jax.devices(), axis_names=("batch",))
    reduce_dtype = max_likelihood._dtype_from_name(
        os.environ.get("CHEMTRAIN_REDUCE_DTYPE", "float32")
    )
    enable_buffer_donation = str(
        os.environ.get("CHEMTRAIN_ENABLE_BUFFER_DONATION", "0")
    ).strip().lower() in ("1", "true", "yes", "on")
    donate_mode = str(os.environ.get("CHEMTRAIN_DONATE_MODE", "state_only")).strip().lower()
    if donate_mode not in ("state_only", "state_and_batch"):
        raise ValueError(
            f"Unsupported CHEMTRAIN_DONATE_MODE='{donate_mode}'. "
            "Expected one of: state_only, state_and_batch."
        )
    grad_accum_mode_default = str(
        os.environ.get("CHEMTRAIN_GRAD_ACCUM_MODE", "stack_scan")
    ).strip().lower()
    if grad_accum_mode_default not in ("concat_slice", "stack_scan"):
        raise ValueError(
            "Unsupported CHEMTRAIN_GRAD_ACCUM_MODE="
            f"'{grad_accum_mode_default}'. Expected one of: "
            "concat_slice, stack_scan."
        )

    param_loss_fn = max_likelihood._get_param_loss_fn(
        loss_fn, batched_model, penalty_fn
    )
    batch_update_fns: Dict[tuple[int, str], Callable[..., Any]] = {}

    def _resolve_accum_mode(accum_mode):
        if accum_mode is None:
            accum_mode = grad_accum_mode_default
        else:
            accum_mode = str(accum_mode).strip().lower()
        if accum_mode not in ("concat_slice", "stack_scan"):
            raise ValueError(
                f"Unsupported accum_mode='{accum_mode}'. "
                "Expected one of: concat_slice, stack_scan."
            )
        return accum_mode

    def _batch_in_spec(accum_mode, microbatch_count):
        if accum_mode == "stack_scan" and microbatch_count > 1:
            return PartitionSpec(None, "batch")
        return PartitionSpec("batch")

    def _build_batch_update_fn(microbatch_count: int, accum_mode: str):
        if microbatch_count < 1:
            raise ValueError(f"microbatch_count must be >= 1, got {microbatch_count}")
        batch_in_spec = _batch_in_spec(accum_mode, microbatch_count)

        def batch_update(params, opt_state, data):
            if mesh.size > 1:

                @partial(
                    shard_map,
                    mesh=mesh,
                    in_specs=batch_in_spec,
                    out_specs=PartitionSpec(),
                    check_rep=False,
                )
                def _inner(batch):
                    (loss, per_target_loss), grad = _msam_accumulated_local_grad(
                        param_loss_fn,
                        params,
                        batch,
                        microbatch_count,
                        accum_mode,
                        rho,
                        epsilon,
                    )
                    grad = max_likelihood._cast_tree_floating(grad, reduce_dtype)
                    loss = jnp.asarray(loss, dtype=reduce_dtype)
                    per_target_loss = max_likelihood._cast_tree_floating(
                        per_target_loss, reduce_dtype
                    )
                    grad = lax.pmean(grad, axis_name="batch")
                    loss = lax.pmean(loss, axis_name="batch")
                    per_target_loss = lax.pmean(per_target_loss, axis_name="batch")
                    grad = max_likelihood._cast_grad_like_params(grad, params)
                    new_params, new_opt_state = max_likelihood.step_optimizer(
                        params, opt_state, grad, optimizer
                    )
                    return new_params, new_opt_state, loss, grad, per_target_loss

            else:

                def _inner(batch):
                    (loss, per_target_loss), grad = _msam_accumulated_local_grad(
                        param_loss_fn,
                        params,
                        batch,
                        microbatch_count,
                        accum_mode,
                        rho,
                        epsilon,
                    )
                    new_params, new_opt_state = max_likelihood.step_optimizer(
                        params, opt_state, grad, optimizer
                    )
                    return new_params, new_opt_state, loss, grad, per_target_loss

            return _inner(data)

        if enable_buffer_donation:
            donate_argnums = (0, 1, 2) if donate_mode == "state_and_batch" else (0, 1)
            return jax.jit(batch_update, donate_argnums=donate_argnums)
        return jax.jit(batch_update)

    def _get_batch_update_fn(microbatch_count: int, accum_mode: str):
        key = (int(microbatch_count), str(accum_mode))
        batch_update = batch_update_fns.get(key)
        if batch_update is None:
            batch_update = _build_batch_update_fn(*key)
            batch_update_fns[key] = batch_update
        return batch_update

    def batch_update(
        params,
        opt_state,
        batch,
        per_target=False,
        microbatch_count=1,
        accum_mode=None,
        **kwargs,
    ):
        if "per_target_loss" in kwargs:
            per_target = kwargs["per_target_loss"]
        resolved_accum_mode = _resolve_accum_mode(accum_mode)
        result = _get_batch_update_fn(microbatch_count, resolved_accum_mode)(
            params,
            opt_state,
            batch,
        )
        new_params, new_opt_state, loss, grad, per_target_loss = result
        if per_target:
            return new_params, new_opt_state, loss, grad, per_target_loss
        return new_params, new_opt_state, loss, grad

    return batch_update

