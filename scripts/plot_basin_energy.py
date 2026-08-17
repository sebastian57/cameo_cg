#!/usr/bin/env python3
"""Regenerate a basin-energy learning curve from persisted training history."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from training.basin_energy_monitor import plot_basin_energy_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = plot_basin_energy_history(
        args.history, args.provenance, args.output
    )
    print(output)


if __name__ == "__main__":
    main()
