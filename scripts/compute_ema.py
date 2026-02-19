#!/usr/bin/env python3
"""
Post-hoc EMA (Exponential Moving Average) of Checkpoint Parameters

Computes an EMA over a sequence of saved checkpoints (epoch*.pkl) and
saves the resulting averaged parameters.  This is EMA Option B: no changes
to the training loop are needed — just run this script after training.

With checkpoint_freq=10 and 80 training epochs, 8 checkpoints are available
(epochs 10, 20, …, 80). The EMA is computed in chronological order.

EMA update rule (applied to each leaf of the parameter pytree):
    ema ← decay * ema + (1 - decay) * params_at_epoch_k

Usage:
    python scripts/compute_ema.py \\
        --checkpoint_dir ./checkpoints_allegro \\
        --output ./exported_models/ema_params.pkl \\
        [--decay 0.999] [--verbose]

The output file is a plain params dict (same format as <model>_params.pkl)
and can be passed directly to evaluate_forces.py or used for LAMMPS export.
"""

import sys
import argparse
import pickle
import re
import numpy as np
from pathlib import Path


def extract_epoch(path: Path) -> int:
    """Extract epoch number from checkpoint filename (e.g. epoch00040.pkl → 40)."""
    m = re.search(r'epoch0*(\d+)', path.stem)
    return int(m.group(1)) if m else -1


def load_params_from_checkpoint(path: Path):
    """
    Extract the params dict from a chemtrain checkpoint file.

    Handles:
    - Plain dict with 'trainer_state' key (chemtrain's save_trainer format)
    - Plain dict with 'params' key (our save_checkpoint format)
    - Direct params dict
    """
    with open(path, 'rb') as f:
        saved = pickle.load(f)

    if isinstance(saved, dict):
        if 'trainer_state' in saved:
            ts = saved['trainer_state']
            if isinstance(ts, dict) and 'params' in ts:
                return ts['params']
            elif hasattr(ts, 'params'):
                return ts.params
        if 'best_params' in saved:
            return saved['best_params']
        if 'params' in saved:
            return saved['params']
        # Treat the dict itself as params if it has expected keys
        if 'allegro' in saved or 'prior' in saved:
            return saved

    raise ValueError(f"Cannot extract params from {path}. "
                     f"Keys: {list(saved.keys()) if isinstance(saved, dict) else type(saved)}")


def ema_update(ema, params, decay: float):
    """Apply one EMA update: ema ← decay * ema + (1 - decay) * params."""
    if ema is None:
        # First update: initialise EMA to the first params
        return {k: np.array(v, copy=True) for k, v in _flatten(params)}

    result = {}
    for key, ema_val in _iter_leaves(ema):
        p_val = _get_leaf(params, key)
        result_val = decay * ema_val + (1.0 - decay) * np.asarray(p_val)
        _set_leaf(result, key, result_val)
    return result


# ---------------------------------------------------------------------------
# Pytree-style helpers for nested dicts of numpy arrays
# (avoids a JAX dependency at post-processing time)
# ---------------------------------------------------------------------------

def _flatten(tree, prefix=()):
    """Yield (key_tuple, leaf) for all leaves in a nested dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, prefix + (k,))
    else:
        yield prefix, tree


def _iter_leaves(tree, prefix=()):
    """Same as _flatten but yields mutable references."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _iter_leaves(v, prefix + (k,))
    else:
        yield prefix, tree


def _get_leaf(tree, key_tuple):
    node = tree
    for k in key_tuple:
        node = node[k]
    return node


def _set_leaf(tree, key_tuple, value):
    node = tree
    for k in key_tuple[:-1]:
        if k not in node:
            node[k] = {}
        node = node[k]
    node[key_tuple[-1]] = value


def ema_update_tree(ema, params, decay: float):
    """
    EMA update over a nested dict of arrays.

    ema is a nested dict mirroring the structure of params, where each leaf
    is a numpy array. On the first call (ema is None) a copy of params is
    returned as the initial EMA.
    """
    if ema is None:
        # Deep copy: nested dicts with numpy arrays
        def deep_copy(node):
            if isinstance(node, dict):
                return {k: deep_copy(v) for k, v in node.items()}
            return np.array(node, copy=True, dtype=np.float32)
        return deep_copy(params)

    def update(e, p):
        if isinstance(e, dict):
            return {k: update(e[k], p[k]) for k in e}
        e_arr = np.asarray(e, dtype=np.float32)
        p_arr = np.asarray(p, dtype=np.float32)
        return decay * e_arr + (1.0 - decay) * p_arr

    return update(ema, params)


def main():
    parser = argparse.ArgumentParser(
        description="Compute post-hoc EMA over saved checkpoints."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing epoch*.pkl checkpoint files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for EMA params pickle file."
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.999,
        help="EMA decay factor (default: 0.999). Higher = smoother / slower."
    )
    parser.add_argument(
        "--pattern",
        default="epoch*.pkl",
        help="Glob pattern for checkpoint files (default: epoch*.pkl)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-checkpoint progress."
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint_dir not found: {checkpoint_dir}")
        sys.exit(1)

    # Find and sort checkpoints by epoch number
    ckpt_paths = sorted(
        [p for p in checkpoint_dir.glob(args.pattern)
         if not p.name.endswith(".meta.pkl")],
        key=extract_epoch
    )

    if not ckpt_paths:
        print(f"ERROR: No checkpoints matching '{args.pattern}' in {checkpoint_dir}")
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoints (decay={args.decay})")
    for p in ckpt_paths:
        print(f"  epoch {extract_epoch(p):4d}  {p.name}")

    # Compute EMA in chronological order
    ema_params = None
    for i, path in enumerate(ckpt_paths):
        try:
            params = load_params_from_checkpoint(path)
        except Exception as e:
            print(f"  WARNING: skipping {path.name} — {e}")
            continue

        ema_params = ema_update_tree(ema_params, params, args.decay)

        if args.verbose:
            print(f"  [{i+1}/{len(ckpt_paths)}] Updated EMA from {path.name}")

    if ema_params is None:
        print("ERROR: no checkpoints could be loaded.")
        sys.exit(1)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(ema_params, f)

    print(f"\nEMA params saved: {out_path}")
    print(f"  Checkpoints used: {len(ckpt_paths)}")
    print(f"  Decay: {args.decay}")
    print(f"  Top-level keys: {list(ema_params.keys()) if isinstance(ema_params, dict) else type(ema_params)}")


if __name__ == "__main__":
    main()
