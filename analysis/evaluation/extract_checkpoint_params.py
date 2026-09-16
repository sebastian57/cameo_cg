#!/usr/bin/env python3
"""
Extract best_params from an epoch checkpoint and write them as a params.pkl
to the exports/ directory, enabling analyze_training_testing_suite.py to run
on a partially-trained model.

Usage:
    python extract_checkpoint_params.py <checkpoint_pkl> <exports_dir> <model_name>

Example:
    python extract_checkpoint_params.py \
        outputs/20260323_.../phase4_largerdata/checkpoints/epoch00040.pkl \
        outputs/20260323_.../phase4_largerdata/exports \
        training_exp_configs_phase4_largerdataset
"""

import sys
import pickle
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    exports_dir = Path(sys.argv[2])
    model_name = sys.argv[3]

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)

    # Epoch checkpoints are saved by chemtrain's save_trainer() and contain
    # a dict with keys: trainer_state (params + opt_state), best_params, etc.
    if isinstance(payload, dict):
        if "best_params" in payload:
            params = payload["best_params"]
            print("  Found best_params in checkpoint dict")
        elif "trainer_state" in payload and "params" in payload["trainer_state"]:
            params = payload["trainer_state"]["params"]
            print("  Found params in trainer_state (no best_params key)")
        elif "params" in payload:
            params = payload["params"]
            print("  Found params directly in checkpoint dict")
        else:
            print(f"ERROR: unexpected checkpoint structure. Keys: {list(payload.keys())}")
            sys.exit(1)
    elif hasattr(payload, "best_inference_params"):
        params = payload.best_inference_params
        print("  Found best_inference_params on trainer object")
    elif hasattr(payload, "params"):
        params = payload.params
        print("  Found params on trainer object")
    else:
        print(f"ERROR: cannot extract params from checkpoint of type {type(payload)}")
        sys.exit(1)

    exports_dir.mkdir(parents=True, exist_ok=True)
    params_path = exports_dir / f"{model_name}_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(params, f)

    print(f"Saved params to: {params_path}")


if __name__ == "__main__":
    main()
