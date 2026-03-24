"""
Optimizer Factory for Training

Registry-based optimizer creation with learning rate schedules, clipping,
and weight decay. New optimizers are added by decorating a factory function
with @register_optimizer.
"""

import optax
from typing import Callable, Dict, Any, List

# ---------------------------------------------------------------------------
# Registry: name -> (factory_fn, handles_weight_decay)
# factory_fn signature: (schedule, config) -> optax.GradientTransformation
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, tuple] = {}


def register_optimizer(name: str, handles_weight_decay: bool = False):
    """Decorator that registers an optimizer factory under *name*.

    Args:
        name: Lowercase optimizer name used in config YAML.
        handles_weight_decay: If True, the optimizer applies weight decay
            internally (e.g. adamw, lamb), so the outer chain skips it.
    """
    def wrapper(fn: Callable):
        _REGISTRY[name] = (fn, handles_weight_decay)
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Built-in optimizers
# ---------------------------------------------------------------------------

@register_optimizer("adabelief")
def _adabelief(schedule, cfg):
    return optax.adabelief(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
    )

@register_optimizer("yogi")
def _yogi(schedule, cfg):
    return optax.yogi(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-6),
    )

@register_optimizer("adam")
def _adam(schedule, cfg):
    return optax.adam(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
    )

@register_optimizer("adamw", handles_weight_decay=True)
def _adamw(schedule, cfg):
    return optax.adamw(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

@register_optimizer("lamb", handles_weight_decay=True)
def _lamb(schedule, cfg):
    return optax.lamb(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-6),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

@register_optimizer("lion")
def _lion(schedule, cfg):
    return optax.lion(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.99),
    )

@register_optimizer("sgd_nesterov")
def _sgd_nesterov(schedule, cfg):
    return optax.sgd(
        learning_rate=schedule,
        momentum=cfg.get("momentum", 0.9),
        nesterov=True,
    )

@register_optimizer("polyak_sgd")
def _polyak_sgd(schedule, cfg):
    return optax.sgd(learning_rate=schedule, momentum=0.9)

@register_optimizer("fromage")
def _fromage(schedule, cfg):
    return optax.sgd(learning_rate=cfg.get("lr", 2e-4))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_optimizer(
    name: str,
    config: Dict[str, Any],
    global_grad_clip: float = None,
) -> optax.GradientTransformation:
    """Create an optax optimizer from a name string and config dict.

    Args:
        name: Optimizer name (case-insensitive). Must be registered.
        config: Hyperparameter dict (lr, peak_lr, decay_steps, beta1, …).
        global_grad_clip: If provided, overrides config["grad_clip"].

    Returns:
        Composed ``optax.GradientTransformation`` (clip + [weight_decay] + base).

    Raises:
        ValueError: If *name* is not in the registry.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Registered: {', '.join(sorted(_REGISTRY))}"
        )
    factory_fn, handles_wd = _REGISTRY[key]

    lr = config.get("lr", 0.001)
    peak_lr = config.get("peak_lr", lr)
    end_lr = config.get("end_lr", lr / 10)
    # Both 'warmup_steps' and legacy 'warmup_epochs' accepted
    warmup_steps = config.get("warmup_steps", config.get("warmup_epochs", 0))
    decay_steps = config.get("decay_steps", 100)
    weight_decay = config.get("weight_decay", 0.0)
    grad_clip = (
        global_grad_clip if global_grad_clip is not None
        else config.get("grad_clip", 1.0)
    )

    # decay_steps * 2 matches the original implementation convention
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps * 2,
        end_value=end_lr,
        exponent=1.0,
    )

    base_optimizer = factory_fn(schedule, config)

    chain_ops = [optax.clip_by_global_norm(grad_clip)]
    if not handles_wd:
        chain_ops.append(optax.add_decayed_weights(weight_decay=weight_decay))
    chain_ops.append(base_optimizer)

    return optax.chain(*chain_ops)


def create_optimizer_from_config(
    config_manager,
    optimizer_name: str,
) -> optax.GradientTransformation:
    """Create optimizer using a ``ConfigManager`` instance."""
    optimizer_config = config_manager.get_optimizer_config(optimizer_name)
    global_grad_clip = config_manager.get_grad_clip()
    return create_optimizer(optimizer_name, optimizer_config, global_grad_clip)


def get_available_optimizers() -> List[str]:
    """Return sorted list of registered optimizer names."""
    return sorted(_REGISTRY)
