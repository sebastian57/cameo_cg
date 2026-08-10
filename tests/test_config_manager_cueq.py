"""Regression tests for cuEq-specific Allegro configuration overlays."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.manager import ConfigManager


def _write_config(path: Path, model: dict) -> ConfigManager:
    path.write_text(
        yaml.safe_dump(
            {
                "data": {},
                "model": model,
                "training": {},
                "optimizer": {},
            }
        )
    )
    return ConfigManager(path)


def test_nested_cueq_overlay_warns_and_is_not_applied(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "nested.yaml",
        {
            "ml_model": "allegro_cueq_fast",
            "allegro": {
                "avg_num_neighbors": 4.0,
                "allegro_cueq": {"tp_backend": "fused_sp"},
            },
        },
    )

    with pytest.warns(UserWarning, match="nested too deeply"):
        resolved = config.get_allegro_config()

    assert "tp_backend" not in resolved


def test_top_level_cueq_overlay_is_applied(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "overlay.yaml",
        {
            "ml_model": "allegro_cueq_fast",
            "allegro": {"avg_num_neighbors": 4.0},
            "allegro_cueq": {"tp_backend": "fused_sp"},
        },
    )

    resolved = config.get_allegro_config()

    assert resolved["avg_num_neighbors"] == 4.0
    assert resolved["tp_backend"] == "fused_sp"


def test_direct_force_mode_config(tmp_path: Path) -> None:
    path = tmp_path / "direct.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {},
                "model": {
                    "ml_model": "allegro_cueq_fast",
                    "output_mode": "direct_force",
                    "use_priors": False,
                    "direct_force": {"head_hidden": 96, "head_layers": 3},
                },
                "training": {"gammas": {"F": 1.0, "U": 0.0}},
                "optimizer": {},
                "export": {"enabled": False},
            }
        )
    )
    config = ConfigManager(path)
    assert config.get_model_output_mode() == "direct_force"
    assert config.get_direct_force_config()["hidden"] == 96
    assert config.get_direct_force_config()["layers"] == 3


def test_direct_force_mode_rejects_energy_export(tmp_path: Path) -> None:
    path = tmp_path / "invalid_direct.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {},
                "model": {"ml_model": "allegro_cueq_fast", "output_mode": "direct_force"},
                "training": {"gammas": {"F": 1.0, "U": 0.0}},
                "optimizer": {},
            }
        )
    )
    with pytest.raises(ValueError, match="export.enabled=false"):
        ConfigManager(path)
