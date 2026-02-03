"""
Logging utilities for chemtrain clean code base.

Provides consistent logging across all modules with standard formatting.
"""

import logging
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting.

    Args:
        name: Logger name (typically module category like "Data", "Model", etc.)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("MyModule")
        >>> logger.info("Starting processing...")
        [MyModule] Starting processing...
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Format: [Module] Message
        formatter = logging.Formatter('[%(name)s] %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)
        logger.propagate = False  # Don't propagate to root logger

    return logger


# Module-level loggers for different components
data_logger = setup_logger("Data")
model_logger = setup_logger("Model")
training_logger = setup_logger("Training")
export_logger = setup_logger("Export")
eval_logger = setup_logger("Eval")


def set_log_level(level: int):
    """
    Set logging level for all module loggers.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> import logging
        >>> from utils.logging import set_log_level
        >>> set_log_level(logging.DEBUG)  # Enable debug output
    """
    for logger in [data_logger, model_logger, training_logger, export_logger, eval_logger]:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
