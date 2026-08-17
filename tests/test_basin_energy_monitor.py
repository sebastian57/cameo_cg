from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from config.manager import ConfigManager
from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from training.basin_energy_monitor import (
    BasinEnergyMonitor,
    basin_energy_metrics,
    build_basin_energy_monitor,
    build_balanced_panel,
    dataset_fingerprint,
    plot_basin_energy_history,
)


def _config(tmp_path: Path, *, monitor: dict, direct_force: bool = False) -> ConfigManager:
    model = {"ml_model": "allegro"}
    training: dict = {"basin_energy_monitor": monitor}
    export = {"enabled": True}
    if direct_force:
        model = {
            "ml_model": "allegro_cueq_fast",
            "output_mode": "direct_force",
            "use_priors": False,
        }
        training["gammas"] = {"F": 1.0, "U": 0.0}
        export = {"enabled": False}
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {},
                "model": model,
                "training": training,
                "optimizer": {},
                "export": export,
            }
        )
    )
    return ConfigManager(path)


def test_basin_monitor_config_has_reproducible_defaults(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        monitor={"enabled": True, "dataset_path": "/tmp/reference.npz"},
    )

    assert config.get_basin_energy_monitor_config() == {
        "enabled": True,
        "dataset_path": "/tmp/reference.npz",
        "stride": 1,
        "seed": 0,
        "frames_per_basin": 512,
        "batch_size": 256,
        "mapping": "ala2_backbone_cb_6",
        "temperature_K": 298.0,
        "output_dir": None,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [("stride", 0), ("frames_per_basin", 0), ("batch_size", 0), ("temperature_K", 0.0)],
)
def test_basin_monitor_config_rejects_nonpositive_values(
    tmp_path: Path, key: str, value: float
) -> None:
    monitor = {"enabled": True, "dataset_path": "/tmp/reference.npz", key: value}
    config = _config(tmp_path, monitor=monitor)

    with pytest.raises(ValueError, match=key):
        config.get_basin_energy_monitor_config()


def test_basin_monitor_rejects_string_boolean(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        _config(tmp_path, monitor={"enabled": "false"}).get_basin_energy_monitor_config()


def test_direct_force_mode_rejects_basin_energy_monitor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="basin_energy_monitor"):
        _config(
            tmp_path,
            monitor={"enabled": True, "dataset_path": "/tmp/reference.npz"},
            direct_force=True,
        )


def test_basin_energy_metrics_are_offset_invariant() -> None:
    labels = np.array(["beta", "beta", "alphaR", "alphaR", "alphaL", "alphaL"])
    energies = np.array([1.0, 3.0, 4.0, 6.0, 8.0, 10.0])

    first = basin_energy_metrics(energies, labels)
    shifted = basin_energy_metrics(energies + 1234.5, labels)

    assert first["dU_alphaR_minus_beta"] == pytest.approx(3.0)
    assert first["dU_alphaL_minus_beta"] == pytest.approx(7.0)
    assert shifted["dU_alphaR_minus_beta"] == pytest.approx(
        first["dU_alphaR_minus_beta"]
    )
    assert shifted["dU_alphaL_minus_beta"] == pytest.approx(
        first["dU_alphaL_minus_beta"]
    )


def test_balanced_panel_is_deterministic_and_equal_per_basin() -> None:
    labels = np.repeat(["beta", "alphaR", "alphaL"], 10)

    first = build_balanced_panel(labels, frames_per_basin=4, seed=7)
    second = build_balanced_panel(labels, frames_per_basin=4, seed=7)

    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert {
        name: int(np.sum(first.labels == name))
        for name in ("beta", "alphaR", "alphaL")
    } == {"beta": 4, "alphaR": 4, "alphaL": 4}


def test_balanced_panel_rejects_an_undersized_basin() -> None:
    labels = np.array(["beta"] * 4 + ["alphaR"] * 4 + ["alphaL"] * 3)

    with pytest.raises(ValueError, match="alphaL.*3.*4"):
        build_balanced_panel(labels, frames_per_basin=4, seed=7)


def test_dataset_fingerprint_changes_with_array_content() -> None:
    arrays = {
        "R": np.arange(18, dtype=np.float32).reshape(1, 6, 3),
        "mask": np.ones((1, 6), dtype=np.float32),
        "species": np.arange(6, dtype=np.int32)[None, :],
    }
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["R"][0, 0, 0] += 1.0

    assert dataset_fingerprint(arrays) == dataset_fingerprint(arrays)
    assert dataset_fingerprint(arrays) != dataset_fingerprint(changed)


def test_basin_energy_metrics_reject_nonfinite_values() -> None:
    labels = np.array(["beta", "alphaR", "alphaL"])
    with pytest.raises(ValueError, match="non-finite"):
        basin_energy_metrics(np.array([0.0, np.nan, 1.0]), labels)

def _runtime_monitor(
    tmp_path: Path,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    stride: int = 3,
) -> BasinEnergyMonitor:
    if labels is None:
        labels = np.repeat(["beta", "alphaR", "alphaL"], 4)
    if arrays is None:
        R = np.zeros((len(labels), 1, 3), dtype=np.float32)
        R[:, 0, 0] = np.arange(len(labels), dtype=np.float32)
        arrays = {
            "R": R,
            "mask": np.ones((len(labels), 1), dtype=np.float32),
            "species": np.zeros((len(labels), 1), dtype=np.int32),
        }

    def quadratic_energy(params, R, mask, species):
        del species
        return params["scale"] * np.sum(R * R * mask[..., None], axis=(1, 2))

    return BasinEnergyMonitor(
        arrays=arrays,
        labels=labels,
        energy_fn=quadratic_energy,
        output_dir=tmp_path,
        dataset_path=tmp_path / "reference.npz",
        mapping="ala2_backbone_cb_6",
        temperature_K=298.0,
        stride=stride,
        seed=9,
        frames_per_basin=2,
        batch_size=4,
    )


def test_monitor_records_zero_stride_and_final_without_duplicates(tmp_path: Path) -> None:
    monitor = _runtime_monitor(tmp_path, stride=3)
    for step in range(5):
        if monitor.should_record(step, final_step=4):
            monitor.record(
                {"scale": 1.0}, mode="fm", stage="adabelief", step=step
            )
    monitor.record({"scale": 1.0}, mode="fm", stage="adabelief", step=3)

    import csv

    with (tmp_path / "basin_energy_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [0, 3, 4]
    assert all(row["dataset_path"].endswith("reference.npz") for row in rows)


def test_monitor_resume_reuses_panel_and_rejects_changed_dataset(tmp_path: Path) -> None:
    first = _runtime_monitor(tmp_path)
    first.record({"scale": 1.0}, mode="fm", stage="adabelief", step=0)
    first_indices = first.panel.indices.copy()

    resumed = _runtime_monitor(tmp_path)
    np.testing.assert_array_equal(resumed.panel.indices, first_indices)
    resumed.record({"scale": 1.0}, mode="fm", stage="adabelief", step=0)

    with pytest.raises(ValueError, match="stride changed"):
        _runtime_monitor(tmp_path, stride=4)

    changed_arrays = {key: value.copy() for key, value in first.arrays.items()}
    changed_arrays["R"][0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="fingerprint"):
        _runtime_monitor(tmp_path, arrays=changed_arrays)


def test_monitor_provenance_uses_full_dataset_counts_for_reference_df(tmp_path: Path) -> None:
    labels = np.array(["beta"] * 4 + ["alphaR"] * 2 + ["alphaL"] * 2)
    monitor = _runtime_monitor(tmp_path, labels=labels)

    assert monitor.provenance["reference_basin_counts"] == {
        "beta": 4,
        "alphaR": 2,
        "alphaL": 2,
    }
    assert monitor.provenance["reference_dF_kcal_mol"][
        "alphaR_minus_beta"
    ] > 0.0


def test_monitor_marks_rejected_rem_iteration(tmp_path: Path) -> None:
    import csv

    monitor = _runtime_monitor(tmp_path)
    monitor.record(
        {"scale": 1.0}, mode="rem", stage="relative_entropy", step=1, rejected=True
    )
    with (tmp_path / "basin_energy_history.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["rejected"] == "True"


def test_reference_df_guides_use_one_log_ratio_per_basin(tmp_path: Path) -> None:
    monitor = _runtime_monitor(tmp_path, stride=1)
    provenance = monitor.provenance
    assert provenance["reference_dF_kcal_mol"]["alphaR_minus_beta"] == pytest.approx(0.0)
    assert provenance["reference_dF_kcal_mol"]["alphaL_minus_beta"] == pytest.approx(0.0)


def test_monitor_finalize_and_standalone_plot_write_png(tmp_path: Path) -> None:
    monitor = _runtime_monitor(tmp_path)
    monitor.record({"scale": 1.0}, mode="fm", stage="adabelief", step=0)
    output = monitor.finalize()
    assert output.is_file() and output.stat().st_size > 1000

    second = tmp_path / "second.png"
    plot_basin_energy_history(
        tmp_path / "basin_energy_history.csv",
        tmp_path / "basin_energy_provenance.json",
        second,
    )
    assert second.is_file() and second.stat().st_size > 1000


def test_plot_cli_regenerates_history_figure(tmp_path: Path) -> None:
    import subprocess
    import sys

    monitor = _runtime_monitor(tmp_path)
    monitor.record({"scale": 1.0}, mode="fm", stage="adabelief", step=0)
    output = tmp_path / "cli.png"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "plot_basin_energy.py"),
            "--history",
            str(monitor.history_path),
            "--provenance",
            str(monitor.provenance_path),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file() and output.stat().st_size > 1000


def test_builder_loads_independent_npz_and_evaluates_model(tmp_path: Path, monkeypatch) -> None:
    import jax.numpy as jnp
    from types import SimpleNamespace
    import sampling.mapping as mapping_module

    labels = (
        [(-100.0, 150.0)] * 2
        + [(-60.0, -50.0)] * 2
        + [(60.0, 40.0)] * 2
    )
    R = np.zeros((6, 1, 3), dtype=np.float32)
    R[:, 0, :2] = np.asarray(labels)
    dataset_path = tmp_path / "independent_reference.npz"
    np.savez(
        dataset_path,
        R=R,
        mask=np.ones((6, 1), dtype=np.float32),
        species=np.zeros((6, 1), dtype=np.int32),
    )

    class CV:
        def __init__(self, component: int):
            self.component = component

        def evaluate(self, positions):
            return np.asarray(positions)[:, 0, self.component]

    monkeypatch.setattr(
        mapping_module,
        "get_mapping",
        lambda name: SimpleNamespace(
            name=name,
            n_beads=1,
            cvs={"phi": CV(0), "psi": CV(1)},
        ),
    )

    class Config:
        def get_basin_energy_monitor_config(self):
            return {
                "enabled": True,
                "dataset_path": str(dataset_path),
                "stride": 1,
                "seed": 4,
                "frames_per_basin": 1,
                "batch_size": 2,
                "mapping": "ala2_backbone_cb_6",
                "temperature_K": 298.0,
                "output_dir": None,
            }

    class Model:
        def compute_energy(self, params, positions, mask, species):
            del species
            return params["scale"] * jnp.sum(positions * positions * mask[:, None])

    monitor = build_basin_energy_monitor(
        Config(), Model(), default_output_dir=tmp_path / "output"
    )
    assert monitor is not None
    row = monitor.record(
        {"scale": 1.0}, mode="fm", stage="adabelief", step=0
    )
    assert row is not None
    assert row["dataset_path"] == str(dataset_path.resolve())
    assert row["n_beta"] == row["n_alphaR"] == row["n_alphaL"] == 1

