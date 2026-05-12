"""JAX distributed setup helpers for training entry points."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass

import jax
from jax.experimental import multihost_utils

from utils.jax_setup import apply_jax_compat_shims


@dataclass(frozen=True)
class DistributedState:
    is_distributed: bool
    rank: int
    world_size: int


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _slurm_first_host(nodelist: str | None) -> str | None:
    if not nodelist:
        return None
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostname", nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        host = result.stdout.strip().splitlines()[0]
        return host or None
    except Exception:
        match = re.match(r"([a-zA-Z-]+)(\d+)", nodelist.replace("[", "").replace("]", ""))
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return nodelist.split(",")[0].split("[")[0]


def _coordinator_host(slurm_nodelist: str | None) -> tuple[str | None, str]:
    explicit = os.environ.get("CHEMTRAIN_COORDINATOR_HOST")
    if explicit:
        return explicit, "CHEMTRAIN_COORDINATOR_HOST"

    host = _slurm_first_host(slurm_nodelist)
    suffix = os.environ.get("CHEMTRAIN_COORDINATOR_HOST_SUFFIX", "").strip()
    if host and suffix and not host.endswith(suffix):
        host = f"{host}{suffix}"
    return host, "SLURM_NODELIST"


def _local_device_ids() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return list(range(len([item for item in visible.split(",") if item.strip()])))
    return list(range(jax.local_device_count()))


def initialize_jax_distributed() -> DistributedState:
    apply_jax_compat_shims()

    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    distributed = bool(slurm_job_id and slurm_ntasks and int(slurm_ntasks) > 1)

    if distributed:
        num_processes = int(slurm_ntasks)
        process_id = int(slurm_procid or 0)
        coordinator_host, coordinator_source = _coordinator_host(slurm_nodelist)
        if coordinator_host is None:
            raise RuntimeError("Could not determine JAX coordinator host from SLURM environment.")

        coordinator_port = int(
            os.environ.get("CHEMTRAIN_COORDINATOR_PORT")
            or (29400 + (int(slurm_job_id) % 1000))
        )
        local_ids = _local_device_ids()

        logging.info(
            "[Distributed] processes=%d process_id=%d coordinator=%s:%d source=%s local_device_ids=%s",
            num_processes,
            process_id,
            coordinator_host,
            coordinator_port,
            coordinator_source,
            local_ids,
        )
        try:
            jax.distributed.initialize(
                coordinator_address=f"{coordinator_host}:{coordinator_port}",
                num_processes=num_processes,
                process_id=process_id,
                local_device_ids=local_ids,
                initialization_timeout=int(os.environ.get("JAX_INIT_TIMEOUT", "1800")),
            )
        except Exception as exc:
            logging.error("[Distributed] jax.distributed.initialize failed: %s", exc)
            traceback.print_exc()
            sys.exit(1)

    rank = jax.process_index()
    world_size = jax.process_count()
    state = DistributedState(distributed, rank, world_size)

    expected_local = os.environ.get("CAMEO_EXPECTED_LOCAL_DEVICES")
    if expected_local:
        expected = int(expected_local)
        observed = jax.local_device_count()
        if observed != expected:
            logging.warning(
                "[Distributed] expected %d local devices but observed %d.",
                expected,
                observed,
            )

    logging.info(
        "[Distributed] rank=%d world_size=%d local_devices=%d global_devices=%d",
        rank,
        world_size,
        jax.local_device_count(),
        jax.device_count(),
    )

    jax.config.update("jax_enable_x64", _truthy(os.environ.get("JAX_ENABLE_X64")))
    return state


def sync_all_ranks(state: DistributedState, tag: str) -> None:
    if state.world_size > 1:
        multihost_utils.sync_global_devices(tag)
