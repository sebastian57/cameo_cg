import numpy as np
import pytest

from training.noised_residual import (
    _parse_noise_levels,
    noised_residual_config_parsed,
    noised_residual_enabled,
    attach_noised_residual_fields,
    noised_residual_tiled_split_extension,
)


class FakeConfig:
    def __init__(self, d):
        self._d = d

    def get(self, *path, default=None):
        result = self._d
        for k in path:
            if isinstance(result, dict):
                result = result.get(k, default)
            else:
                return default
        return result

    def get_min_repulsive_sep(self):
        return self._d.get("model", {}).get("priors", {}).get("min_repulsive_sep", 6)

    def get_prior_weights(self):
        user = self._d.get("model", {}).get("priors", {}).get("weights", {})
        defaults = {
            "bond": 0.5, "angle": 0.1, "repulsive": 0.25, "dihedral": 0.15,
            "excluded_volume": 1.0, "wca": 0.0, "fene": 0.0, "leash": 0.0,
            "local_in": 0.0, "local_bond_in": 0.0, "dh": 0.0,
            "stickiness": 0.0, "salt_bridge": 0.0,
        }
        merged = dict(defaults)
        merged.update(user)
        return merged

    def get_prior_params(self):
        model_priors = self._d.get("model", {}).get("priors", None)
        if model_priors is not None:
            return model_priors
        return self._d.get("priors", {})

    def prior_residual_enabled(self):
        return self._d.get("training", {}).get("prior_residual", {}).get("enabled", False)


def _make_split(n_frames=10, n_atoms=20):
    rng = np.random.default_rng(42)
    return {
        "R": rng.uniform(-10, 10, (n_frames, n_atoms, 3)).astype(np.float32),
        "F": rng.uniform(-1, 1, (n_frames, n_atoms, 3)).astype(np.float32),
        "mask": np.ones((n_frames, n_atoms), dtype=np.float32),
        "species": np.zeros((n_frames, n_atoms), dtype=np.int32),
        "force_loss_mask": np.ones((n_frames, n_atoms), dtype=np.float32),
    }


class TestParseNoiseLevels:
    def test_float_entries(self):
        cfg = {"noise_levels": [0.5, 1.0]}
        levels = _parse_noise_levels(cfg)
        assert len(levels) == 2
        assert levels[0]["sigma"] == 0.5
        assert levels[1]["sigma"] == 1.0
        assert levels[0]["attenuation"] == 0.0
        assert levels[0]["weight"] == 2.0

    def test_dict_entries(self):
        cfg = {"noise_levels": [{"sigma": 0.5, "attenuation": 0.3, "weight": 3.0}]}
        levels = _parse_noise_levels(cfg)
        assert len(levels) == 1
        assert levels[0]["sigma"] == 0.5
        assert levels[0]["attenuation"] == 0.3
        assert levels[0]["weight"] == 3.0

    def test_name_override(self):
        cfg = {"noise_levels": [{"sigma": 1.0, "name": "my_level"}]}
        levels = _parse_noise_levels(cfg)
        assert levels[0]["name"] == "my_level"


class TestNoisedResidualEnabled:
    def test_disabled(self):
        cfg = FakeConfig({"training": {"noised_residual_training": {"enabled": False}}})
        assert noised_residual_enabled(cfg) is False

    def test_enabled(self):
        cfg = FakeConfig({"training": {"noised_residual_training": {"enabled": True, "noise_levels": [1.0]}}})
        assert noised_residual_enabled(cfg) is True


class TestDuplicateEvery:
    def _cfg(self, duplicate_every=None, duplicate_offset=0):
        d = {
            "model": {"priors": {"min_repulsive_sep": 6}},
            "training": {
                "prior_residual": {"enabled": True},
                "noised_residual_training": {
                    "enabled": True,
                    "noise_levels": [{"sigma": 0.5, "attenuation": 0.0, "weight": 1.0}],
                    "duplicate_every": duplicate_every,
                    "duplicate_offset": duplicate_offset,
                }
            }
        }
        return FakeConfig(d)

    def test_null_duplicates_all(self):
        cfg = self._cfg(duplicate_every=None)
        split = _make_split(n_frames=6)
        out = attach_noised_residual_fields(split, cfg, id_to_aa=None, seed=0, split_seed=0)
        n_clean = 6
        n_total = out["R"].shape[0]
        n_noised = n_total - n_clean
        assert n_noised == n_clean, f"Expected {n_clean} noised, got {n_noised}"
        assert np.sum(out["is_noised_frame"] == 0) == n_clean
        assert np.sum(out["is_noised_frame"] == 1) == n_noised

    def test_every_2_duplicates_half(self):
        cfg = self._cfg(duplicate_every=2)
        split = _make_split(n_frames=6)
        out = attach_noised_residual_fields(split, cfg, id_to_aa=None, seed=0, split_seed=0)
        n_clean = 6
        n_dup = 3
        n_total = out["R"].shape[0]
        n_noised = n_total - n_clean
        assert n_noised == n_dup, f"Expected {n_dup} noised, got {n_noised}"
        assert np.sum(out["is_noised_frame"] == 0) == n_clean
        assert np.sum(out["is_noised_frame"] == 1) == n_noised

    def test_every_3_duplicates_third(self):
        cfg = self._cfg(duplicate_every=3)
        split = _make_split(n_frames=9)
        out = attach_noised_residual_fields(split, cfg, id_to_aa=None, seed=0, split_seed=0)
        n_clean = 9
        n_dup = 3
        n_total = out["R"].shape[0]
        n_noised = n_total - n_clean
        assert n_noised == n_dup, f"Expected {n_dup} noised, got {n_noised}"

    def test_offset_shifts_selection(self):
        cfg_offset = self._cfg(duplicate_every=2, duplicate_offset=1)
        split = _make_split(n_frames=6)
        out = attach_noised_residual_fields(split, cfg_offset, id_to_aa=None, seed=0, split_seed=0)
        n_clean = 6
        n_noised = out["R"].shape[0] - n_clean
        assert n_noised == 3, f"With offset=1, every 2nd starting at 1 (indices 1,3,5) should give 3 duplicates, got {n_noised}"

    def test_weight_repeats(self):
        cfg = FakeConfig({
            "model": {"priors": {"min_repulsive_sep": 6}},
            "training": {
                "prior_residual": {"enabled": True},
                "noised_residual_training": {
                    "enabled": True,
                    "noise_levels": [{"sigma": 0.5, "attenuation": 0.0, "weight": 3.0}],
                    "duplicate_every": None,
                }
            }
        })
        split = _make_split(n_frames=4)
        out = attach_noised_residual_fields(split, cfg, id_to_aa=None, seed=0, split_seed=0)
        n_clean = 4
        n_total = out["R"].shape[0]
        n_noised = n_total - n_clean
        assert n_noised == 12, f"Weight=3 on 4 frames should give 12 noised, got {n_noised}"


class TestNoisedResidualConfigParsed:
    def test_duplicate_every_validation(self):
        d = {
            "model": {"priors": {"min_repulsive_sep": 6}},
            "training": {
                "prior_residual": {"enabled": True},
                "noised_residual_training": {
                    "enabled": True,
                    "noise_levels": [1.0],
                    "duplicate_every": 1,
                }
            }
        }
        cfg = FakeConfig(d)
        with pytest.raises(ValueError, match="duplicate_every must be >= 2"):
            noised_residual_config_parsed(cfg)

    def test_duplicate_offset_validation(self):
        d = {
            "model": {"priors": {"min_repulsive_sep": 6}},
            "training": {
                "prior_residual": {"enabled": True},
                "noised_residual_training": {
                    "enabled": True,
                    "noise_levels": [1.0],
                    "duplicate_offset": -1,
                }
            }
        }
        cfg = FakeConfig(d)
        with pytest.raises(ValueError, match="duplicate_offset must be >= 0"):
            noised_residual_config_parsed(cfg)


class TestTiledExtension:
    def test_tiled_clean_frames_regenerated(self):
        cfg = FakeConfig({
            "model": {"priors": {"min_repulsive_sep": 6}},
            "training": {
                "prior_residual": {"enabled": True},
                "noised_residual_training": {
                    "enabled": True,
                    "noise_levels": [{"sigma": 0.5, "attenuation": 0.0, "weight": 1.0}],
                    "duplicate_every": None,
                }
            }
        })
        tiled = {
            "R": np.random.rand(5, 10, 3).astype(np.float32),
            "F": np.random.rand(5, 10, 3).astype(np.float32),
            "mask": np.ones((5, 10), dtype=np.float32),
            "species": np.zeros((5, 10), dtype=np.int32),
            "is_noised_frame": np.zeros(5, dtype=np.int32),
            "force_loss_mask": np.ones((5, 10), dtype=np.float32),
            "n_valid": np.full(5, 10, dtype=np.int32),
        }
        out = noised_residual_tiled_split_extension(
            tiled, cfg, id_to_aa=None, epoch_seed=0, fitted_params=None
        )
        assert out["R"].shape[0] > 5
        assert "is_noised_frame" in out

    def test_tiled_skips_when_disabled(self):
        cfg = FakeConfig({
            "training": {
                "noised_residual_training": {"enabled": False},
            }
        })
        tiled = {
            "R": np.random.rand(3, 5, 3).astype(np.float32),
            "F": np.random.rand(3, 5, 3).astype(np.float32),
            "mask": np.ones((3, 5), dtype=np.float32),
            "species": np.zeros((3, 5), dtype=np.int32),
            "is_noised_frame": np.zeros(3, dtype=np.int32),
        }
        out = noised_residual_tiled_split_extension(tiled, cfg, id_to_aa=None, epoch_seed=0)
        assert out["R"].shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])