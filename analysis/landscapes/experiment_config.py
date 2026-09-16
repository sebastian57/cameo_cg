"""Strict manifest loading for the MD-less landscape experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelSpec:
    label: str
    config: Path
    checkpoint: Path
    checkpoint_key: str
    cache_key: str | None
    md_ensemble: Path | None
    md_replicas: int | None
    lineage: str = ""
    optimization_dataset: str = ""
    stability_ood: str = "not evaluated"


@dataclass
class PairSpec:
    label: str
    pre: str
    post: str



@dataclass
class Profile:
    name: str
    scientific_result: bool
    backend: str
    direct_frames: int
    map_frames: int
    bins: int
    min_count: int
    cache_atol: float
    cache_rtol: float
    reference_frames: int | None
    cache_validation_frames: int
    energy_batch_size: int
    bootstrap_replicates: int


@dataclass
class ExperimentConfig:
    manifest: Path
    root: Path
    cameo_root: Path
    reference: Path
    energy_cache: Path
    mapping: str
    temperature: float
    seed: int
    models: list[ModelSpec]
    pairs: list[PairSpec]
    profile: Profile
    results_dir: Path


def _required(mapping: dict[str, Any], key: str, context: str = "manifest") -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_experiment(path: str | Path, profile_name: str) -> ExperimentConfig:
    manifest = Path(path).expanduser().resolve()
    raw = yaml.safe_load(manifest.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a mapping")
    root = manifest.parent
    profiles = _required(raw, "profiles")
    if profile_name not in profiles:
        raise ValueError(f"unknown profile {profile_name!r}; available: {sorted(profiles)}")
    p = profiles[profile_name]
    reference_frames_raw = p.get("reference_frames", p.get("map_frames"))
    if reference_frames_raw == "all":
        reference_frames = None
    else:
        reference_frames = int(reference_frames_raw)
    profile = Profile(
        name=profile_name,
        scientific_result=bool(_required(p, "scientific_result", profile_name)),
        backend=str(_required(p, "backend", profile_name)),
        direct_frames=int(_required(p, "direct_frames", profile_name)),
        map_frames=int(_required(p, "map_frames", profile_name)),
        bins=int(_required(p, "bins", profile_name)),
        min_count=int(_required(p, "min_count", profile_name)),
        cache_atol=float(_required(p, "cache_atol", profile_name)),
        cache_rtol=float(_required(p, "cache_rtol", profile_name)),
        reference_frames=reference_frames,
        cache_validation_frames=int(p.get("cache_validation_frames", p["direct_frames"])),
        energy_batch_size=int(p.get("energy_batch_size", 1)),
        bootstrap_replicates=int(p.get("bootstrap_replicates", 0)),
    )
    for name in ("direct_frames", "map_frames", "bins", "min_count"):
        if getattr(profile, name) <= 0:
            raise ValueError(f"profile {profile_name}.{name} must be positive")
    for name in ("cache_validation_frames", "energy_batch_size"):
        if getattr(profile, name) <= 0:
            raise ValueError(f"profile {profile_name}.{name} must be positive")
    if profile.reference_frames is not None and profile.reference_frames <= 0:
        raise ValueError(f"profile {profile_name}.reference_frames must be positive or 'all'")
    if profile.bootstrap_replicates < 0:
        raise ValueError(f"profile {profile_name}.bootstrap_replicates must be nonnegative")
    if profile.cache_validation_frames % 4:
        raise ValueError("cache_validation_frames must be divisible by four for basin-stratified validation")
    if profile.cache_atol < 0 or profile.cache_rtol < 0:
        raise ValueError("cache tolerances must be nonnegative")
    rows = _required(raw, "models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("models must be a nonempty list")
    models = []
    for row in rows:
        label = str(_required(row, "label", "model"))
        cache_value = row.get("cache_key")
        if cache_value is not None and (
            not isinstance(cache_value, str) or not cache_value.strip()
        ):
            raise ValueError(f"{label} cache_key must be null or a non-empty string")
        md_ensemble_raw = row.get("md_ensemble")
        md_replicas_raw = row.get("md_replicas")
        if (md_ensemble_raw is None) != (md_replicas_raw is None):
            raise ValueError(f"{label} md_ensemble and md_replicas must both be set or both be null")
        md_replicas = None if md_replicas_raw is None else int(md_replicas_raw)
        if md_replicas is not None and md_replicas <= 0:
            raise ValueError("model md_replicas must be positive")
        models.append(ModelSpec(
            label=label,
            config=_resolve(root, _required(row, "config", "model")),
            checkpoint=_resolve(root, _required(row, "checkpoint", "model")),
            checkpoint_key=str(_required(row, "checkpoint_key", "model")),
            cache_key=None if cache_value is None else str(cache_value),
            md_ensemble=None if md_ensemble_raw is None else _resolve(root, md_ensemble_raw),
            md_replicas=md_replicas,
            lineage=str(row.get("lineage", "")),
            optimization_dataset=str(row.get("optimization_dataset", "")),
            stability_ood=str(row.get("stability_ood", "not evaluated")),
        ))
    labels = [model.label for model in models]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate model label in {labels}")
    pair_rows = raw.get("pairs", [])
    pairs = [PairSpec(label=str(_required(row, "label", "pair")),
                      pre=str(_required(row, "pre", "pair")),
                      post=str(_required(row, "post", "pair")))
             for row in pair_rows]
    pair_labels = [pair.label for pair in pairs]
    if len(pair_labels) != len(set(pair_labels)):
        raise ValueError(f"duplicate pair label in {pair_labels}")
    known_labels = set(labels)
    for pair in pairs:
        if pair.pre not in known_labels or pair.post not in known_labels:
            raise ValueError(f"pair {pair.label} references unknown model")
        if pair.pre == pair.post:
            raise ValueError(f"pair {pair.label} pre and post must differ")
    keys = [model.cache_key for model in models if model.cache_key is not None]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate model cache_key in {keys}")
    results_root = _resolve(root, _required(raw, "results_root"))
    return ExperimentConfig(
        manifest=manifest,
        root=root,
        cameo_root=_resolve(root, _required(raw, "cameo_root")),
        reference=_resolve(root, _required(raw, "reference")),
        energy_cache=_resolve(root, _required(raw, "energy_cache")),
        mapping=str(_required(raw, "mapping")),
        temperature=float(_required(raw, "temperature_K")),
        seed=int(_required(raw, "seed")),
        models=models,
        pairs=pairs,
        profile=profile,
        results_dir=(results_root / profile_name).resolve(),
    )


def validate_input_paths(config: ExperimentConfig) -> None:
    paths = [("cameo_root", config.cameo_root), ("reference", config.reference),
             ("energy_cache", config.energy_cache)]
    for model in config.models:
        paths.extend([(f"{model.label}.config", model.config),
                      (f"{model.label}.checkpoint", model.checkpoint)])
        if model.md_ensemble is not None:
            paths.append((f"{model.label}.md_ensemble", model.md_ensemble))
    for label, input_path in paths:
        if not input_path.exists():
            raise FileNotFoundError(f"missing input {label}: {input_path}")
