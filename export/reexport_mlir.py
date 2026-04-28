#!/usr/bin/env python3
"""
Post-hoc MLIR export for saved params.pkl files.

This script loads an existing trained parameter pickle, rebuilds the matching
CombinedModel from a config file, and exports a fresh MLIR artifact. It is
especially useful for ML-only training runs where priors should be added back
only at export time.

Examples:
    python export/reexport_mlir.py         /path/to/model_params.pkl         /path/to/model_config.yaml

    python export/reexport_mlir.py         /path/to/model_params.pkl         /path/to/model_config.yaml         --mode combined         --prior-source config         --output-name training_testing_large_model_lr_with_priors
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import pickle
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported lazily in _load_runtime_modules() so --help works without the
# full training/export environment.
yaml = None
ConfigManager = None
DatasetLoader = None
CoordinatePreprocessor = None
ModelExporter = None
CombinedModel = None


def _load_runtime_modules() -> None:
    """Import heavy runtime dependencies only when we actually need them."""
    global yaml, ConfigManager, DatasetLoader, CoordinatePreprocessor, ModelExporter, CombinedModel

    if ConfigManager is not None:
        return

    try:
        from utils.jax_setup import apply_jax_compat_shims

        apply_jax_compat_shims()

        import yaml as _yaml
        from config.manager import ConfigManager as _ConfigManager
        from data.loader import DatasetLoader as _DatasetLoader
        from data.preprocessor import CoordinatePreprocessor as _CoordinatePreprocessor
        from export.exporter import ModelExporter as _ModelExporter
        from models.combined_model import CombinedModel as _CombinedModel
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "unknown module")
        raise ModuleNotFoundError(
            "Missing runtime dependency '"
            f"{missing}"
            "'. Load the project environment before running reexport_mlir.py."
        ) from exc

    yaml = _yaml
    ConfigManager = _ConfigManager
    DatasetLoader = _DatasetLoader
    CoordinatePreprocessor = _CoordinatePreprocessor
    ModelExporter = _ModelExporter
    CombinedModel = _CombinedModel


def _resolve_existing_path(raw_path: str, config_path: Path, project_root: Path) -> Path:
    """Resolve repo-relative paths from saved runtime/export config copies."""
    path = Path(raw_path)
    if path.is_absolute():
        return path

    candidates = [
        config_path.parent / path,
        project_root / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (config_path.parent / path).resolve()


def _resolve_config_paths(config: ConfigManager) -> None:
    """Rewrite relative dataset/spline paths to absolute paths in-place."""
    data_path = config.get_data_path()
    resolved_data = _resolve_existing_path(data_path, config.config_path, PROJECT_ROOT)
    config.set("data", "path", str(resolved_data))

    if config.use_spline_priors_enabled():
        spline_path = config.get_spline_file_path()
        if spline_path:
            resolved_spline = _resolve_existing_path(
                spline_path, config.config_path, PROJECT_ROOT
            )
            config.set("model", "priors", "spline_file", str(resolved_spline))


def _extract_nested_params(obj: Any) -> Dict[str, Any]:
    """Extract model params from direct params, checkpoints, or trainer payloads."""
    if isinstance(obj, dict):
        if "best_params" in obj and isinstance(obj["best_params"], dict):
            return obj["best_params"]

        if "trainer_state" in obj:
            trainer_state = obj["trainer_state"]
            if isinstance(trainer_state, dict) and "params" in trainer_state:
                return trainer_state["params"]
            if hasattr(trainer_state, "params"):
                return trainer_state.params

        if "params" in obj and isinstance(obj["params"], dict):
            return obj["params"]

        return obj

    if hasattr(obj, "best_inference_params"):
        return obj.best_inference_params

    if hasattr(obj, "trainer_state"):
        trainer_state = obj.trainer_state
        if isinstance(trainer_state, dict) and "params" in trainer_state:
            return trainer_state["params"]
        if hasattr(trainer_state, "params"):
            return trainer_state.params

    if hasattr(obj, "params"):
        return obj.params

    raise TypeError(f"Unsupported parameter payload type: {type(obj)}")


def _coerce_combined_params(params: Any) -> Dict[str, Any]:
    """Ensure params use the {'ml': ..., 'prior': ...} CombinedModel layout."""
    if not isinstance(params, dict):
        raise TypeError(f"Expected params dict, got {type(params)}")

    if "ml" in params or "prior" in params:
        return copy.deepcopy(params)

    # Some legacy exports wrap the raw Haiku tree in a single backend key like
    # {'allegro': <haiku_params>}. The current model wrappers expect the raw
    # Haiku pytree itself, so unwrap those known compatibility containers.
    legacy_backend_keys = {
        "allegro",
        "allegro_cueq",
        "allegro_cueq_fast",
        "mace",
        "painn",
    }
    if len(params) == 1:
        only_key = next(iter(params.keys()))
        if only_key in legacy_backend_keys and isinstance(params[only_key], dict):
            return {"ml": copy.deepcopy(params[only_key])}

    # Otherwise assume this is already the raw backend pytree.
    return {"ml": copy.deepcopy(params)}


def _build_export_config(config: ConfigManager, mode: str) -> ConfigManager:
    """Clone config and apply export-time prior mode."""
    export_config = ConfigManager(config.config_path)
    export_config._config = copy.deepcopy(config._config)

    if mode == "combined":
        has_prior_config = export_config.get("model", "priors", default=None) is not None
        if not has_prior_config:
            raise ValueError(
                "Combined export requested, but model.priors is missing from the config."
            )
        export_config.set("model", "use_priors", True)
        export_config.set("model", "train_priors", False)
        export_config.set("training", "export_combined_ml_priors", True)
    else:
        export_config.set("model", "use_priors", False)
        export_config.set("model", "train_priors", False)
        export_config.set("training", "export_combined_ml_priors", False)

    _resolve_config_paths(export_config)
    return export_config


def _prepare_dataset_context(config: ConfigManager):
    """Load dataset and build a representative centered structure for export."""
    loader = DatasetLoader(str(config.get_data_path()), max_frames=None, seed=config.get_seed())

    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, shift)
    return loader, dataset, extent


def _derive_output_name(params_path: Path, mode: str) -> str:
    stem = params_path.stem
    if stem.endswith("_params"):
        stem = stem[:-7]
    suffix = "combined_priors" if mode == "combined" else "ml_only"
    return f"{stem}_{suffix}"


def _materialize_export_params(
    params: Dict[str, Any],
    model: CombinedModel,
    mode: str,
    prior_source: str,
) -> Dict[str, Any]:
    """Create the params dict to export/save."""
    export_params = copy.deepcopy(params)

    if mode != "combined":
        export_params.pop("prior", None)
        return export_params

    if prior_source == "params" and "prior" in export_params:
        return export_params

    if not model.use_priors or model.prior is None:
        raise ValueError("Requested combined export, but the constructed model has no priors.")

    export_params["prior"] = copy.deepcopy(model.prior.params)
    return export_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export a saved params.pkl to MLIR, optionally adding fixed priors."
    )
    parser.add_argument("params", help="Path to params.pkl or a checkpoint-like pickle")
    parser.add_argument("config", help="Path to config YAML used to rebuild the model")
    parser.add_argument(
        "--mode",
        choices=("combined", "ml-only"),
        default="combined",
        help="combined: add/use priors for export; ml-only: export without priors",
    )
    parser.add_argument(
        "--prior-source",
        choices=("config", "params"),
        default="config",
        help="For combined export: use fixed prior params from config or preserve an existing prior block from the pickle",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for exported artifacts (default: ../saved_models relative to export/)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for exported artifacts (default: derived from params filename)",
    )
    parser.add_argument(
        "--export-mode",
        choices=("auto", "symbolic", "fixed_size"),
        default=None,
        help=(
            "Override export.mode during re-export. "
            "Use 'symbolic' for connector-compatible dynamic/signature export."
        ),
    )
    parser.add_argument(
        "--naive-equivalence-atol",
        type=float,
        default=None,
        help=(
            "Override the absolute energy tolerance for cuEq fast -> symbolic/naive "
            "export validation. By default the exporter uses max(1e-4, 1e-6 * n_atoms)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _load_runtime_modules()

    params_path = Path(args.params).resolve()
    config_path = Path(args.config).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {params_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ConfigManager(config_path)
    export_config = _build_export_config(config, args.mode)
    if args.export_mode is not None:
        export_config.set("export", "mode", args.export_mode)
    if args.naive_equivalence_atol is not None:
        export_config.set("export", "naive_equivalence_atol", float(args.naive_equivalence_atol))

    with open(params_path, "rb") as handle:
        payload = pickle.load(handle)
    raw_params = _extract_nested_params(payload)
    params = _coerce_combined_params(raw_params)

    loader, dataset, box = _prepare_dataset_context(export_config)
    r0 = dataset["R"][0]
    species0 = dataset["species"][0]

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = CombinedModel(
            config=export_config,
            R0=r0,
            box=box,
            species=species0,
            N_max=loader.N_max,
            id_to_aa=loader.id_to_aa,
            prior_only=False,
        )

    export_params = _materialize_export_params(
        params=params,
        model=model,
        mode=args.mode,
        prior_source=args.prior_source,
    )

    default_output_dir = (SCRIPT_DIR.parent / "saved_models").resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or _derive_output_name(params_path, args.mode)

    mlir_path = output_dir / f"{output_name}.mlir"
    params_out = output_dir / f"{output_name}_params.pkl"
    config_out = output_dir / f"{output_name}_config.yaml"

    exporter = ModelExporter.from_combined_model(
        model=model,
        params=export_params,
        box=box,
        species=species0,
    )
    exporter.export_to_file(mlir_path)

    with open(params_out, "wb") as handle:
        pickle.dump(export_params, handle)

    config_out.write_text(yaml.safe_dump(export_config._config, sort_keys=False))

    print("=" * 60)
    print("Post-hoc MLIR export complete")
    print("=" * 60)
    print(f"Input params : {params_path}")
    print(f"Input config : {config_path}")
    print(f"Mode        : {args.mode}")
    print(f"Prior source: {args.prior_source if args.mode == 'combined' else 'n/a'}")
    print(f"MLIR        : {mlir_path}")
    print(f"Params      : {params_out}")
    print(f"Config copy : {config_out}")


if __name__ == "__main__":
    main()
