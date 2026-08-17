from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from training.basin_energy_monitor import BasinEnergyMonitor
import training.trainer as trainer_module
from training.trainer import Trainer


class Recorder:
    def __init__(self, stride: int = 1):
        self.stride = stride
        self.calls: list[dict] = []

    def should_record(self, step: int, *, final_step: int) -> bool:
        return step == 0 or step == final_step or step % self.stride == 0

    def record(self, params, **metadata):
        self.calls.append({"params": params, **metadata})


class FakeChemtrainTrainer:
    def __init__(self):
        self.params = {"value": "start"}
        self._epoch = 0
        self.tasks: dict[str, list] = {}

    def add_task(self, name: str, callback) -> None:
        self.tasks.setdefault(name, []).append(callback)


def test_fm_monitor_records_stage_start_and_post_update_global_epoch() -> None:
    owner = Trainer.__new__(Trainer)
    owner._basin_energy_monitor = Recorder()
    chemtrain = FakeChemtrainTrainer()

    owner._install_basin_energy_monitor(
        chemtrain,
        stage_name="adabelief",
        stage_start_epoch=4,
        stage_end_epoch=8,
    )
    chemtrain.params = {"value": "updated"}
    chemtrain.tasks["post_epoch"][0](chemtrain)

    assert owner._basin_energy_monitor.calls == [
        {
            "params": {"value": "start"},
            "mode": "fm",
            "stage": "adabelief",
            "step": 4,
        },
        {
            "params": {"value": "updated"},
            "mode": "fm",
            "stage": "adabelief",
            "step": 5,
        },
    ]


def test_fm_monitor_respects_stride_and_final_epoch() -> None:
    owner = Trainer.__new__(Trainer)
    owner._basin_energy_monitor = Recorder(stride=3)
    chemtrain = FakeChemtrainTrainer()
    owner._install_basin_energy_monitor(
        chemtrain,
        stage_name="adabelief",
        stage_start_epoch=1,
        stage_end_epoch=5,
    )
    callback = chemtrain.tasks["post_epoch"][0]
    for local_epoch in range(4):
        chemtrain._epoch = local_epoch
        callback(chemtrain)

    assert [call["step"] for call in owner._basin_energy_monitor.calls] == [3, 5]


def test_disabled_fm_monitor_installs_no_task() -> None:
    owner = Trainer.__new__(Trainer)
    owner._basin_energy_monitor = None
    chemtrain = FakeChemtrainTrainer()

    owner._install_basin_energy_monitor(
        chemtrain,
        stage_name="adabelief",
        stage_start_epoch=0,
        stage_end_epoch=2,
    )

    assert chemtrain.tasks == {}


def _persisted_monitor(output_dir: Path) -> BasinEnergyMonitor:
    labels = np.array(["beta", "beta", "alphaR", "alphaR", "alphaL", "alphaL"])
    positions = np.arange(18, dtype=np.float32).reshape(6, 1, 3)
    arrays = {
        "R": positions,
        "mask": np.ones((6, 1), dtype=np.float32),
        "species": np.zeros((6, 1), dtype=np.int32),
    }

    def energy_fn(params, R, mask, species):
        del species
        return params["scale"] * np.sum(R * mask[..., None], axis=(1, 2))

    return BasinEnergyMonitor(
        arrays=arrays,
        labels=labels,
        energy_fn=energy_fn,
        output_dir=output_dir,
        dataset_path=output_dir / "independent_reference.npz",
        mapping="ala2_backbone_cb_6",
        temperature_K=298.0,
        stride=1,
        seed=3,
        frames_per_basin=1,
        batch_size=3,
    )


def test_fm_resume_does_not_duplicate_persisted_stage_start(tmp_path: Path) -> None:
    first_owner = Trainer.__new__(Trainer)
    first_owner._basin_energy_monitor = _persisted_monitor(tmp_path)
    first_trainer = FakeChemtrainTrainer()
    first_trainer.params = {"scale": 1.0}
    first_owner._install_basin_energy_monitor(
        first_trainer,
        stage_name="adabelief",
        stage_start_epoch=0,
        stage_end_epoch=2,
    )

    resumed_owner = Trainer.__new__(Trainer)
    resumed_owner._basin_energy_monitor = _persisted_monitor(tmp_path)
    resumed_trainer = FakeChemtrainTrainer()
    resumed_trainer.params = {"scale": 1.0}
    resumed_owner._install_basin_energy_monitor(
        resumed_trainer,
        stage_name="adabelief",
        stage_start_epoch=0,
        stage_end_epoch=2,
    )

    with (tmp_path / "basin_energy_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["mode"], row["stage"], int(row["step"])) for row in rows] == [
        ("fm", "adabelief", 0)
    ]


def test_nonzero_rank_does_not_construct_monitor(monkeypatch, tmp_path: Path) -> None:
    sentinel = object()
    calls = []

    def fake_build(config, model, *, default_output_dir):
        calls.append((config, model, default_output_dir))
        return sentinel

