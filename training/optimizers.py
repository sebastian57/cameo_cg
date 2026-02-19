"""
Optimizer Factory for Training

Creates optax optimizers with learning rate schedules, clipping, and weight decay.
Supports multiple optimizers configured from YAML.

Extracted from:
- train_fm_multiple_proteins.py
"""

import optax
from typing import Dict, Any


def create_optimizer(name: str, config: Dict[str, Any], global_grad_clip: float = None) -> optax.GradientTransformation:
    """
    Create an optax optimizer from configuration.

    Args:
        name: Optimizer name ("adabelief", "yogi", "adam", "lion", "polyak_sgd", "fromage")
        config: Optimizer configuration dictionary with hyperparameters
        global_grad_clip: Global gradient clipping value (overrides config if provided)

    Returns:
        optax.GradientTransformation (optimizer)

    Raises:
        ValueError: If optimizer name is not recognized

    Example:
        >>> config = {
        ...     "lr": 0.01,
        ...     "peak_lr": 0.05,
        ...     "end_lr": 0.001,
        ...     "warmup_epochs": 10,
        ...     "decay_steps": 100,
        ...     "beta1": 0.9,
        ...     "beta2": 0.999,
        ...     "weight_decay": 1e-4,
        ...     "grad_clip": 1.0,
        ... }
        >>> optimizer = create_optimizer("adabelief", config)
    """
    # Extract common parameters
    lr = config.get("lr", 0.001)
    peak_lr = config.get("peak_lr", lr)
    end_lr = config.get("end_lr", lr / 10)
    # warmup_steps is the number of gradient update steps for warmup (not epochs).
    # The old key was 'warmup_epochs' which was misleading; both keys are accepted
    # for backward compatibility with existing config files.
    warmup_steps = config.get("warmup_steps", config.get("warmup_epochs", 0))
    decay_steps = config.get("decay_steps", 100)
    weight_decay = config.get("weight_decay", 0.0)

    # Gradient clipping (use global if provided, else from config)
    grad_clip = global_grad_clip if global_grad_clip is not None else config.get("grad_clip", 1.0)

    # Create learning rate schedule.
    # decay_steps * 2 matches the original implementation convention.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps * 2,
        end_value=end_lr,
        exponent=1.0
    )

    # Create optimizer based on name
    if name.lower() == "adabelief":
        base_optimizer = optax.adabelief(
            learning_rate=schedule,
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            eps=config.get("eps", 1e-8)
        )

    elif name.lower() == "yogi":
        base_optimizer = optax.yogi(
            learning_rate=schedule,
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            eps=config.get("eps", 1e-6)
        )

    elif name.lower() == "adam":
        base_optimizer = optax.adam(
            learning_rate=schedule,
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.999),
            eps=config.get("eps", 1e-8)
        )

    elif name.lower() == "lion":
        base_optimizer = optax.lion(
            learning_rate=schedule,
            b1=config.get("beta1", 0.9),
            b2=config.get("beta2", 0.99)
        )

    elif name.lower() == "sgd_nesterov":
        base_optimizer = optax.sgd(
            learning_rate=schedule,
            momentum=config.get("momentum", 0.9),
            nesterov=True
        )

    elif name.lower() == "polyak_sgd":
        # Special case: uses different parameters
        f_star = config.get("f_star", 0.0)
        eps = config.get("eps", 1e-8)
        base_optimizer = optax.sgd(learning_rate=schedule, momentum=0.9)

    elif name.lower() == "fromage":
        # Fromage optimizer (simple case)
        lr_fromage = config.get("lr", 2e-4)
        base_optimizer = optax.sgd(learning_rate=lr_fromage)

    else:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Supported: adabelief, yogi, adam, lion, sgd_nesterov, polyak_sgd, fromage"
        )

    # Compose optimizer with gradient clipping and weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.add_decayed_weights(weight_decay=weight_decay),
        base_optimizer
    )

    return optimizer


def create_optimizer_from_config(config_manager, optimizer_name: str) -> optax.GradientTransformation:
    """
    Create optimizer from ConfigManager.

    Args:
        config_manager: ConfigManager instance
        optimizer_name: Name of optimizer to create

    Returns:
        optax.GradientTransformation

    Example:
        >>> config = ConfigManager("config.yaml")
        >>> optimizer = create_optimizer_from_config(config, "adabelief")
    """
    optimizer_config = config_manager.get_optimizer_config(optimizer_name)
    global_grad_clip = config_manager.get_grad_clip()

    return create_optimizer(optimizer_name, optimizer_config, global_grad_clip)


def get_available_optimizers() -> list:
    """
    Get list of supported optimizer names.

    Returns:
        List of optimizer names
    """
    return ["adabelief", "yogi", "adam", "lion", "sgd_nesterov", "polyak_sgd", "fromage"]
