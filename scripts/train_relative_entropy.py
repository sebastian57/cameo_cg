#!/usr/bin/env python3
"""Standalone relative-entropy fine-tuning entry point."""

import argparse
import copy
import logging
import os
import pickle
import sys
from pathlib import Path

# Avoid CUDA preallocation/OOM noise for --help and login-node smoke tests.
# Override manually with JAX_PLATFORMS=cuda inside an allocation.
if "JAX_PLATFORMS" not in os.environ and not os.environ.get("SLURM_JOB_ID"):
    os.environ["JAX_PLATFORMS"] = "cpu"

# Match train.py: make package-local absolute imports work from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import numpy as np

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from export.exporter import ModelExporter
from models.combined_model import CombinedModel
from training.optimizers import create_optimizer_from_config
from training.path_utils import repo_root_from_file, resolve_from_config_or_repo
from training.relative_entropy import (
    InProcessLangevinSampler,
    RelativeEntropyTrainer,
    extract_params_from_checkpoint_payload,
    relative_entropy_config,
    write_relative_entropy_history_artifacts,
)
from training.support_gate import build_support_gate_bank, support_gate_config, support_gate_enabled
from utils.logging import training_logger
from mc.samplers import BlackJaxHMCSampler, BlackJaxMALASampler


def _wrap_into_box(R, box):
    box = np.asarray(box, dtype=np.float32)
    return np.mod(np.asarray(R, dtype=np.float32), box[None, None, :])


def _resolve_reference_path(config, re_cfg):
    raw = re_cfg.reference_data_path or config.get_data_path()
    return resolve_from_config_or_repo(raw, config.config_path, repo_root_from_file(__file__))


def _load_reference_dataset(config, re_cfg):
    data_path = _resolve_reference_path(config, re_cfg)
    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    dataset = loader.get_all()
    use_pbc = config.use_pbc_enabled()
    if use_pbc:
        if loader.box is None:
            raise ValueError(
                "model.pbc=true requires a box in the RE reference dataset."
            )
        box = jnp.asarray(loader.box, dtype=jnp.float32)
        dataset["R"] = _wrap_into_box(dataset["R"], np.asarray(box))
    else:
        preprocessor = CoordinatePreprocessor(
            cutoff=config.get_cutoff(),
            buffer_multiplier=config.get_buffer_multiplier(),
            park_multiplier=config.get_park_multiplier(),
        )
        box, shift = preprocessor.compute_box_extent(loader.R, loader.mask)
        dataset["R"] = preprocessor.center_and_park(
            dataset["R"], dataset["mask"], box, shift
        )

    if dataset.get("species") is None:
        dataset["species"] = np.zeros(dataset["mask"].shape, dtype=np.int32)

    return loader, dataset, box, data_path


def _checkpoint_path(config):
    cfg = config.get("training", "init_from_checkpoint", default={}) or {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        raise ValueError(
            "Relative-entropy fine-tuning requires "
            "training.init_from_checkpoint.enabled=true."
        )
    raw = cfg.get("path")
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            "Relative-entropy fine-tuning requires training.init_from_checkpoint.path."
        )
    return resolve_from_config_or_repo(raw, config.config_path, repo_root_from_file(__file__))


def _build_support_gate_bank_for_re(config, dataset, data_path):
    if not support_gate_enabled(config):
        return None

    cfg = support_gate_config(config)
    descriptor = str(cfg.get("descriptor", "pairwise_distances")).strip().lower()
    if descriptor != "pairwise_distances":
        raise ValueError(
            "training.support_gate.descriptor currently supports only 'pairwise_distances'."
        )
    bank = build_support_gate_bank(
        R=np.asarray(dataset["R"], dtype=np.float32),
        mask=np.asarray(dataset["mask"], dtype=np.float32),
        max_centers=int(cfg.get("max_centers", 512)),
        sigma_multiplier=float(cfg.get("sigma_multiplier", 1.0)),
        seed=int(cfg.get("seed", config.get_seed())),
        floor=float(cfg.get("floor", 0.0)),
        stop_gradient=bool(cfg.get("stop_gradient", False)),
    )
    training_logger.info(
        "[SupportGate] RE runtime bank rebuilt from %s: centers=%d sigma=%.6g floor=%.3g stop_gradient=%s",
        data_path,
        int(bank.centers.shape[0]),
        float(bank.sigma),
        float(bank.floor),
        bool(bank.stop_gradient),
    )
    return bank


def _build_model(config, loader, dataset, box, support_gate_bank=None):
    n_species_global = int(np.max(dataset["species"])) + 1
    config_n_species = config.get("model", "allegro", "num_types", default=None)
    if config_n_species is not None:
        n_species_global = max(n_species_global, int(config_n_species))

    return CombinedModel(
        config=config,
        R0=jnp.asarray(dataset["R"][0], dtype=jnp.float32),
        box=jnp.asarray(box, dtype=jnp.float32),
        species=jnp.asarray(dataset["species"][0], dtype=jnp.int32),
        N_max=int(dataset["R"].shape[1]),
        init_mask=jnp.asarray(dataset["mask"][0], dtype=jnp.float32),
        n_species_override=n_species_global,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
        support_gate_bank=support_gate_bank,
    )


def _make_param_bound_sampler(model, params, re_cfg):
    holder = {"params": params}

    def total_energy(R, mask=None, species=None):
        species_safe = jnp.where(mask > 0, species, 0).astype(jnp.int32)
        return model.compute_energy(holder["params"], R, mask, species_safe)

    if re_cfg.sampler == "mc_mala":
        if re_cfg.mc_mala is None:
            raise ValueError("training.relative_entropy.sampler='mc_mala' requires parsed mc.mala config.")
        sampler = BlackJaxMALASampler(total_energy, re_cfg.mc_mala)
    elif re_cfg.sampler == "mc_hmc":
        if re_cfg.mc_hmc is None:
            raise ValueError("training.relative_entropy.sampler='mc_hmc' requires parsed mc.hmc config.")
        sampler = BlackJaxHMCSampler(total_energy, re_cfg.mc_hmc)
    else:
        sampler = InProcessLangevinSampler(total_energy, model.ml_model.shift, re_cfg)

    def set_params(new_params):
        holder["params"] = new_params

    sampler.set_params = set_params

    def ml_energy(ml_params, R, mask, species):
        full_params = dict(holder["params"])
        full_params["ml"] = ml_params
        species_safe = jnp.where(mask > 0, species, 0).astype(jnp.int32)
        return model.compute_energy(full_params, R, mask, species_safe)

    return sampler, ml_energy


def run_relative_entropy(config_file: str):
    config = ConfigManager(config_file)
    re_cfg = relative_entropy_config(config)
    if not re_cfg.enabled:
        raise ValueError("Set training.relative_entropy.enabled=true to run this script.")
    if config.get_batch_mode() == "tiled":
        raise ValueError("Relative-entropy V1 requires data.batch_mode='standard'.")

    ckpt_path = _checkpoint_path(config)
    loader, dataset, box, data_path = _load_reference_dataset(config, re_cfg)
    support_gate_bank = _build_support_gate_bank_for_re(config, dataset, data_path)
    model = _build_model(config, loader, dataset, box, support_gate_bank=support_gate_bank)
    params = extract_params_from_checkpoint_payload(ckpt_path)
    if model.use_priors and "prior" not in params:
        params = dict(params)
        params["prior"] = model.prior.params

    optimizer = create_optimizer_from_config(config, re_cfg.optimizer)
    sampler, ml_energy = _make_param_bound_sampler(model, params, re_cfg)
    reference_data = {
        "R": jnp.asarray(dataset["R"], dtype=jnp.float32),
        "mask": jnp.asarray(dataset["mask"], dtype=jnp.float32),
        "species": jnp.asarray(dataset["species"], dtype=jnp.int32),
    }

    output_dir = Path(re_cfg.output_dir) if re_cfg.output_dir else Path(config.get_output_dir()) / "relative_entropy"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = RelativeEntropyTrainer(
        params=params,
        reference_data=reference_data,
        sampler=sampler,
        energy_fn=ml_energy,
        optimizer=optimizer,
        config=re_cfg,
        seed=config.get_seed(),
        checkpoint_dir=output_dir / "checkpoints",
    )

    training_logger.info("[RE] Reference data: %s (%d frames)", data_path, dataset["R"].shape[0])
    training_logger.info("[RE] Warm-start checkpoint: %s", ckpt_path)
    training_logger.info("[RE] Sampler: %s", re_cfg.sampler)
    results = trainer.train()
    artifact_paths = write_relative_entropy_history_artifacts(
        trainer.history, output_dir, prefix="relative_entropy"
    )
    for artifact_name, artifact_path in artifact_paths.items():
        training_logger.info("[RE] History %s: %s", artifact_name, artifact_path)
    final_path = output_dir / "relative_entropy_final_checkpoint.pkl"
    trainer.save_checkpoint(
        final_path,
        metadata={
            "config_file": str(config.config_path),
            "reference_data_path": str(data_path),
            "source_checkpoint": str(ckpt_path),
            "results": results,
            "history_artifacts": artifact_paths,
        },
    )
    training_logger.info("[RE] Final checkpoint: %s", final_path)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run standalone relative-entropy fine-tuning for a CAMEO CG model."
    )
    parser.add_argument("config_file", nargs="?", help="Training YAML with training.relative_entropy enabled.")
    args = parser.parse_args(argv)
    if args.config_file is None:
        parser.print_help()
        return 0
    try:
        run_relative_entropy(args.config_file)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
