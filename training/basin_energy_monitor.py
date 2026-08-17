"""Relative Ala2 basin-energy diagnostics on an independent reference panel."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


REQUIRED_BASINS = ("beta", "alphaR", "alphaL")


@dataclass(frozen=True)
class BalancedPanel:
    """Indices and labels for an equal-size sample from each required basin."""

    indices: np.ndarray
    labels: np.ndarray


def dataset_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    """Return a content digest that also binds keys, shapes, and dtypes."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(int(n) for n in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def build_balanced_panel(
    labels: np.ndarray, *, frames_per_basin: int, seed: int
) -> BalancedPanel:
    """Select a deterministic, without-replacement panel within each basin."""

    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1:
        raise ValueError("basin labels must be one-dimensional")
    if frames_per_basin <= 0:
        raise ValueError("frames_per_basin must be positive")
    rng = np.random.default_rng(int(seed))
    pieces: list[np.ndarray] = []
    for basin in REQUIRED_BASINS:
        candidates = np.flatnonzero(labels == basin)
        if len(candidates) < frames_per_basin:
            raise ValueError(
                f"{basin} has {len(candidates)} frames, fewer than requested "
                f"{frames_per_basin}"
            )
        pieces.append(
            np.sort(rng.choice(candidates, size=frames_per_basin, replace=False))
        )
    indices = np.concatenate(pieces).astype(np.int64, copy=False)
    return BalancedPanel(indices=indices, labels=labels[indices])


def basin_energy_metrics(
    energies: np.ndarray, labels: np.ndarray
) -> dict[str, float | int]:
    """Compute mean energies and relative alpha/beta basin gaps."""

    energies = np.asarray(energies, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    if energies.ndim != 1 or labels.ndim != 1 or energies.shape != labels.shape:
        raise ValueError("energies and labels must be matching one-dimensional arrays")
    if not np.isfinite(energies).all():
        raise ValueError("basin energies contain non-finite values")
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for basin in REQUIRED_BASINS:
        selected = energies[labels == basin]
        if selected.size == 0:
            raise ValueError(f"basin-energy panel contains no {basin} frames")
        means[basin] = float(np.mean(selected))
        counts[basin] = int(selected.size)
    return {
        "n_beta": counts["beta"],
        "n_alphaR": counts["alphaR"],
        "n_alphaL": counts["alphaL"],
        "mean_U_beta": means["beta"],
        "mean_U_alphaR": means["alphaR"],
        "mean_U_alphaL": means["alphaL"],
        "dU_alphaR_minus_beta": means["alphaR"] - means["beta"],
        "dU_alphaL_minus_beta": means["alphaL"] - means["beta"],
    }


def assign_ala2_basins(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Assign the project-standard non-overlapping bb6 Ramachandran basins."""

    phi = (np.asarray(phi, dtype=np.float64) + 180.0) % 360.0 - 180.0
    psi = (np.asarray(psi, dtype=np.float64) + 180.0) % 360.0 - 180.0
    if phi.shape != psi.shape:
        raise ValueError("phi and psi must have matching shapes")
    labels = np.full(phi.shape, "other", dtype="U6")
    labels[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    labels[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    labels[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    return labels


def parse_basin_energy_monitor_config(config: Any) -> dict[str, Any]:
    """Normalize and validate ``training.basin_energy_monitor``."""

    raw = config.get("training", "basin_energy_monitor", default={}) or {}
    if not isinstance(raw, dict):
        raise ValueError("training.basin_energy_monitor must be a mapping")
    enabled_raw = raw.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        raise ValueError("training.basin_energy_monitor.enabled must be a boolean")
    enabled = enabled_raw
    dataset_path = raw.get("dataset_path")
    if enabled and (dataset_path is None or not str(dataset_path).strip()):
        raise ValueError(
            "training.basin_energy_monitor.dataset_path is required when enabled"
        )
    parsed = {
        "enabled": enabled,
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "stride": int(raw.get("stride", 1)),
        "seed": int(raw.get("seed", 0)),
        "frames_per_basin": int(raw.get("frames_per_basin", 512)),
        "batch_size": int(raw.get("batch_size", 256)),
        "mapping": str(raw.get("mapping", "ala2_backbone_cb_6")),
        "temperature_K": float(raw.get("temperature_K", 298.0)),
        "output_dir": (
            None if raw.get("output_dir") is None else str(raw.get("output_dir"))
        ),
    }
    for key in ("stride", "frames_per_basin", "batch_size", "temperature_K"):
        if parsed[key] <= 0:
            raise ValueError(f"training.basin_energy_monitor.{key} must be positive")
    if enabled and parsed["mapping"] != "ala2_backbone_cb_6":
        raise ValueError(
            "training.basin_energy_monitor.mapping currently supports only "
            "ala2_backbone_cb_6"
        )
    return parsed

KB_KCAL = 0.0019872042586


HISTORY_FIELDS = (
    "mode",
    "stage",
    "step",
    "rejected",
    "dataset_path",
    "dataset_fingerprint",
    "mapping",
    "n_beta",
    "n_alphaR",
    "n_alphaL",
    "mean_U_beta",
    "mean_U_alphaR",
    "mean_U_alphaL",
    "dU_alphaR_minus_beta",
    "dU_alphaL_minus_beta",
)


class BasinEnergyMonitor:
    """Evaluate and persist relative basin energies on one fixed frame panel."""

    def __init__(
        self,
        *,
        arrays: Mapping[str, np.ndarray],
        labels: np.ndarray,
        energy_fn: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        output_dir: str | Path,
        dataset_path: str | Path,
        mapping: str,
        temperature_K: float,
        stride: int,
        seed: int,
        frames_per_basin: int,
        batch_size: int,
    ):
        self.arrays = {
            key: np.asarray(arrays[key]) for key in ("R", "mask", "species")
        }
        n_frames = int(self.arrays["R"].shape[0])
        if any(value.shape[0] != n_frames for value in self.arrays.values()):
            raise ValueError("monitor dataset arrays must have the same frame count")
        self.labels = np.asarray(labels).astype(str)
        if self.labels.shape != (n_frames,):
            raise ValueError("monitor basin labels must match the dataset frame count")
        if stride <= 0 or batch_size <= 0 or temperature_K <= 0:
            raise ValueError("stride, batch_size, and temperature_K must be positive")
        self.energy_fn = energy_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.mapping = str(mapping)
        self.temperature_K = float(temperature_K)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.panel_path = self.output_dir / "basin_energy_panel.npz"
        self.provenance_path = self.output_dir / "basin_energy_provenance.json"
        self.history_path = self.output_dir / "basin_energy_history.csv"
        self.plot_path = self.output_dir / "basin_energy_learning_curve.png"
        fingerprint = dataset_fingerprint(self.arrays)

        if self.provenance_path.exists() or self.panel_path.exists():
            if not (self.provenance_path.exists() and self.panel_path.exists()):
                raise ValueError("incomplete basin-energy resume artifacts")
            self.provenance = json.loads(self.provenance_path.read_text())
            if self.provenance.get("dataset_fingerprint") != fingerprint:
                raise ValueError("basin-energy dataset fingerprint changed on resume")
            expected_contract = {
                "dataset_path": str(self.dataset_path), "temperature_K": self.temperature_K,
                "seed": int(seed), "frames_per_basin": int(frames_per_basin),
                "batch_size": self.batch_size, "stride": self.stride,
            }
            for field, expected in expected_contract.items():
                if self.provenance.get(field) != expected:
                    raise ValueError(f"basin-energy {field} changed on resume")
            if self.provenance.get("mapping") != self.mapping:
                raise ValueError("basin-energy mapping changed on resume")
            with np.load(self.panel_path, allow_pickle=False) as saved:
                indices = np.asarray(saved["indices"], dtype=np.int64)
                panel_labels = np.asarray(saved["labels"]).astype(str)
            if np.any(indices < 0) or np.any(indices >= n_frames):
                raise ValueError("saved basin-energy panel indices are invalid")
            if not np.array_equal(panel_labels, self.labels[indices]):
                raise ValueError("saved basin-energy panel labels changed on resume")
            self.panel = BalancedPanel(indices=indices, labels=panel_labels)
        else:
            self.panel = build_balanced_panel(
                self.labels,
                frames_per_basin=int(frames_per_basin),
                seed=int(seed),
            )
            np.savez_compressed(
                self.panel_path, indices=self.panel.indices, labels=self.panel.labels
            )
            counts = {
                basin: int(np.sum(self.labels == basin)) for basin in REQUIRED_BASINS
            }
            if any(count == 0 for count in counts.values()):
                raise ValueError("full monitor dataset has an empty required basin")
            kT = KB_KCAL * self.temperature_K
            self.provenance = {
                "dataset_path": str(self.dataset_path),
                "dataset_fingerprint": fingerprint,
                "mapping": self.mapping,
                "temperature_K": self.temperature_K,
                "kT_kcal_mol": kT,
                "seed": int(seed),
                "frames_per_basin": int(frames_per_basin),
                "batch_size": self.batch_size,
                "stride": self.stride,
                "reference_basin_counts": counts,
                "reference_dF_kcal_mol": {
                    "alphaR_minus_beta": float(
                        -kT * np.log(counts["alphaR"] / counts["beta"])
                    ),
                    "alphaL_minus_beta": float(
                        -kT * np.log(counts["alphaL"] / counts["beta"])
                    ),
                },
            }
            self.provenance_path.write_text(
                json.dumps(self.provenance, indent=2) + "\n"
            )
        self._recorded = self._load_recorded_keys()

    def _load_recorded_keys(self) -> set[tuple[str, str, int]]:
        if not self.history_path.exists():
            return set()
        with self.history_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
                raise ValueError("basin-energy history schema changed on resume")
            return {
                (row["mode"], row["stage"], int(row["step"])) for row in reader
            }

    def should_record(self, step: int, *, final_step: int) -> bool:
        step = int(step)
        return step == 0 or step == int(final_step) or step % self.stride == 0

    def _evaluate(self, params: Any) -> np.ndarray:
        indices = self.panel.indices
        pieces: list[np.ndarray] = []
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start : start + self.batch_size]
            values = np.asarray(
                self.energy_fn(
                    params,
                    self.arrays["R"][selected],
                    self.arrays["mask"][selected],
                    self.arrays["species"][selected],
                ),
                dtype=np.float64,
            )
            if values.shape != (len(selected),):
                raise ValueError(
                    "basin-energy function must return one scalar per frame"
                )
            pieces.append(values)
        energies = np.concatenate(pieces)
        if not np.isfinite(energies).all():
            raise ValueError("basin-energy function returned non-finite values")
        return energies

    def record(
        self,
        params: Any,
        *,
        mode: str,
        stage: str,
        step: int,
        rejected: bool = False,
    ) -> dict[str, Any] | None:
        key = (str(mode), str(stage), int(step))
        if key in self._recorded:
            return None
        row: dict[str, Any] = {
            "mode": key[0],
            "stage": key[1],
            "step": key[2],
            "rejected": bool(rejected),
            "dataset_path": str(self.dataset_path),
            "dataset_fingerprint": self.provenance["dataset_fingerprint"],
            "mapping": self.mapping,
            **basin_energy_metrics(self._evaluate(params), self.panel.labels),
        }
        write_header = not self.history_path.exists()
        with self.history_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        self._recorded.add(key)
        return row

    def finalize(self) -> Path:
        plot_basin_energy_history(
            self.history_path, self.provenance_path, self.plot_path
        )
        return self.plot_path


def plot_basin_energy_history(
    history_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render alphaR/beta and alphaL/beta learning curves from persisted data."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    provenance = json.loads(Path(provenance_path).read_text())
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("cannot plot an empty basin-energy history")
    steps = np.asarray([int(row["step"]) for row in rows])
    rejected = np.asarray([row["rejected"].lower() == "true" for row in rows])
    specs = (
        ("dU_alphaR_minus_beta", "alphaR - beta", "alphaR_minus_beta"),
        ("dU_alphaL_minus_beta", "alphaL - beta", "alphaL_minus_beta"),
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True
    )
    for ax, (column, title, target_key) in zip(axes, specs):
        values = np.asarray([float(row[column]) for row in rows])
        ax.plot(steps, values, marker="o", markersize=3, label="model dU")
        target = float(provenance["reference_dF_kcal_mol"][target_key])
        ax.axhline(
            target, color="black", linestyle="--", label="reference dF guide"
        )
        if np.any(rejected):
            ax.scatter(
                steps[rejected],
                values[rejected],
                marker="x",
                color="red",
                label="rejected REM",
            )
        ax.set(title=title, ylabel="kcal/mol")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("FM epoch or REM iteration")
    fig.suptitle("Relative basin-energy learning (potential diagnostic; dU != dF)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def build_basin_energy_monitor(
    config: Any,
    model: Any,
    *,
    default_output_dir: str | Path,
) -> BasinEnergyMonitor | None:
    """Construct the shared monitor from its explicitly configured dataset."""

    settings = config.get_basin_energy_monitor_config()
    if not settings["enabled"]:
        return None
    dataset_path = Path(settings["dataset_path"]).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"basin-energy monitor dataset not found: {dataset_path}"
        )
    with np.load(dataset_path, allow_pickle=False) as loaded:
        missing = [key for key in ("R", "mask", "species") if key not in loaded]
        if missing:
            raise ValueError(
                f"basin-energy monitor dataset is missing keys: {missing}"
            )
        arrays = {
            "R": np.asarray(loaded["R"], dtype=np.float32),
            "mask": np.asarray(loaded["mask"], dtype=np.float32),
            "species": np.asarray(loaded["species"], dtype=np.int32),
        }
    if not np.isfinite(arrays["R"]).all():
        raise ValueError("basin-energy monitor R contains non-finite coordinates")
    expected_aux_shape = arrays["R"].shape[:2]
    if arrays["mask"].shape != expected_aux_shape or arrays["species"].shape != expected_aux_shape:
        raise ValueError("basin-energy monitor mask/species must have shape (frames, beads)")
    if arrays["R"].ndim != 3 or arrays["R"].shape[-1] != 3:
        raise ValueError("basin-energy monitor R must have shape (frames, beads, 3)")

    from sampling.mapping import get_mapping

    mapping = get_mapping(settings["mapping"])
    if arrays["R"].shape[1] != mapping.n_beads:
        raise ValueError(
            "basin-energy monitor bead count does not match its configured mapping"
        )
    phi = mapping.cvs["phi"].evaluate(arrays["R"])
    psi = mapping.cvs["psi"].evaluate(arrays["R"])
    labels = assign_ala2_basins(phi, psi)

    import jax

    @jax.jit
    def batched_energy(params, positions, masks, species):
        return jax.vmap(
            lambda R, mask, atom_types: model.compute_energy(
                params, R, mask, atom_types
            )
        )(positions, masks, species)

    output_dir = (
        Path(settings["output_dir"])
        if settings["output_dir"] is not None
        else Path(default_output_dir)
    )
    return BasinEnergyMonitor(
        arrays=arrays,
        labels=labels,
        energy_fn=batched_energy,
        output_dir=output_dir,
        dataset_path=dataset_path,
        mapping=settings["mapping"],
        temperature_K=settings["temperature_K"],
        stride=settings["stride"],
        seed=settings["seed"],
        frames_per_basin=settings["frames_per_basin"],
        batch_size=settings["batch_size"],
    )

