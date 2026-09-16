import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gradient_recovery_main_keeps_diagnostic_config_for_summary(monkeypatch, tmp_path):
    import jax.numpy as jnp

    from analysis.sampling import gradient_recovery_error as module
    from analysis.sampling.diagnostics_common import DiagnosticConfig

    reference = tmp_path / "reference.npz"
    bias = tmp_path / "bias.npz"
    reference.touch()
    bias.touch()
    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    mapping = SimpleNamespace(n_beads=2)
    n_frames = 80
    latent = {
        "R": np.zeros((n_frames, 2, 3), dtype=np.float32),
        "inside": np.ones(n_frames, dtype=bool),
        "flat": np.arange(n_frames) % 16,
        "bins": 4,
        "centers": np.linspace(-1.0, 1.0, 4),
        "u": np.zeros((n_frames, 2), dtype=np.float64),
        "edges": np.linspace(-2.0, 2.0, 5),
        "mapping": mapping,
    }
    diagnostic_cfg = DiagnosticConfig(reference, bias, flow_dir, temperature_K=310.0)

    class FakeModel:
        def compute_energy(self, params, coordinates, mask, species):
            del params, mask, species
            return jnp.sum(coordinates**2)

    monkeypatch.setattr(module.dc, "config_from_args", lambda args: diagnostic_cfg)
    monkeypatch.setattr(module, "load_latent", lambda *args, **kwargs: latent)
    monkeypatch.setattr(module, "knn_indicator", lambda u, F, k: (np.ones(len(u)), None))

    import analysis.md.analyze_model_residuals_by_region as model_module

    monkeypatch.setattr(
        model_module,
        "_load_model",
        lambda *args, **kwargs: (FakeModel(), None, np.ones(2), np.zeros(2, dtype=np.int32)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gradient_recovery_error",
            "--outdir",
            str(tmp_path / "out"),
            "--frames",
            str(reference),
            "--bias-npz",
            str(bias),
            "--flow-dir",
            str(flow_dir),
            "--model",
            "arm=config.yaml:params.pkl",
        ],
    )

    module.main()

    summary = (tmp_path / "out" / "gradient_recovery.json").read_text()
    assert '"kT_kcal_mol": 0.616033320166' in summary
