import importlib
import json

import numpy as np
import pytest


covariance = importlib.import_module("06_restraint_covariance")


KBT = 0.5921868690749673


def test_deconvolution_recovers_known_internal_hessian():
    hessian = np.diag([2.0, 5.0, 11.0])
    restraint = np.diag([20.0, 20.0, 20.0])
    observed = KBT * np.linalg.inv(hessian + restraint)

    recovered, info = covariance.deconvolve_covariance(
        observed, restraint, KBT, eigen_floor=1.0e-10
    )

    np.testing.assert_allclose(recovered, hessian, atol=1.0e-8)
    assert info["condition_number"] > 1.0
    assert info["valid"] is True


def test_rigid_projector_removes_translation_and_rotation():
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    projector = covariance.rigid_projector(anchor)
    translation = np.tile([1.0, 2.0, 3.0], len(anchor))
    rotation = np.cross(
        np.array([0.0, 0.0, 1.0]), anchor - anchor.mean(axis=0)
    ).reshape(-1)

    assert np.linalg.norm(projector @ translation) < 1.0e-10
    assert np.linalg.norm(projector @ rotation) < 1.0e-10
    np.testing.assert_allclose(projector, projector.T, atol=1.0e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-12)


def test_alignment_removes_global_translation_and_rotation():
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotated = np.array([[0.0, 3.0, 0.0], [-1.0, 3.0, 0.0], [0.0, 3.0, 2.0]])
    frames = np.stack([anchor, rotated])

    displacements, info = covariance.align_internal_displacements(frames, anchor)

    assert displacements.shape == (2, anchor.size)
    assert info["translation_rms_A"] > 0.0
    assert info["rotation_rms_A"] > 0.0
    assert np.linalg.norm(displacements[0]) < 1.0e-10
    assert np.linalg.norm(displacements[1]) < 1.0e-10


def test_estimate_covariance_reports_effective_rank_and_conditioning():
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(400, 3)) @ np.diag([3.0, 1.0, 0.2])

    observed, info = covariance.estimate_covariance(samples)

    np.testing.assert_allclose(observed, observed.T, atol=1.0e-12)
    assert info["effective_rank"] == 3
    assert info["condition_number"] > 10.0
    assert info["n_frames"] == 400


def test_bootstrap_covariance_is_deterministic_for_fixed_seed():
    samples = np.arange(60.0).reshape(10, 2, 3)

    first = covariance.bootstrap_covariance(samples, block_length=2, n_boot=20, seed=7)
    second = covariance.bootstrap_covariance(samples, block_length=2, n_boot=20, seed=7)

    np.testing.assert_array_equal(first["eigenvalues"], second["eigenvalues"])
    np.testing.assert_array_equal(first["samples"], second["samples"])


def test_mode_comparison_reports_soft_mode_overlap():
    reference = {
        "eigenvalues": np.array([1.0, 4.0]),
        "eigenvectors": np.eye(2),
    }
    restrained = {
        "eigenvalues": np.array([1.2, 3.8]),
        "eigenvectors": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    comparison = covariance.compare_mode_sets(reference, restrained)

    assert comparison["overlap_matrix"].shape == (2, 2)
    np.testing.assert_allclose(np.diag(comparison["overlap_matrix"]), 0.0)
    np.testing.assert_allclose(
        np.diag(comparison["overlap_matrix"][:, ::-1]), np.ones(2)
    )


def test_deconvolution_rejects_non_positive_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        covariance.deconvolve_covariance(
            np.diag([1.0, 0.0]), np.eye(2), KBT, eigen_floor=1.0e-10
        )


def _write_window(path, n_frames=8, n_beads=3):
    path.mkdir(parents=True)
    frames = np.zeros((n_frames, n_beads, 3), dtype=float)
    frames[:, :, 0] = np.arange(n_frames)[:, None]
    frames[:, 1, 1] = 1.0
    frames[:, 2, 2] = 1.0
    np.savez(path / "cg_coords.npz", R=frames, time_ps=np.arange(n_frames, dtype=float))


def test_reference_loader_preserves_optional_time_and_shape(tmp_path):
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, R=np.zeros((5, 3, 3)), time_ps=np.arange(5.0))

    loaded = covariance.load_reference_frames(reference_path)

    assert loaded["R"].shape == (5, 3, 3)
    np.testing.assert_array_equal(loaded["time_ps"], np.arange(5.0))


def test_v3_adapter_loads_finite_and_width_derived_windows_and_audits_bad_ones(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_window(campaign / "state_000000")
    _write_window(campaign / "state_000001")
    (campaign / "state_000002").mkdir()
    manifest = {
        "kappa_kcal_mol_A2": 50.0,
        "states": [
            {"state": 0, "target": np.zeros((3, 3)).tolist()},
            {"state": 1, "restraint_width_A": 0.2, "target": np.zeros((3, 3)).tolist()},
            {"state": 2, "frozen": True},
            {"state": 3},
        ],
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest))

    windows, audit = covariance.load_v3_windows(
        campaign, max_frames=4, discard_ps=2.0, kT=KBT
    )

    assert [window["state"] for window in windows] == [0, 1]
    assert all(len(window["frames"]) == 4 for window in windows)
    assert windows[0]["kappa_kcal_mol_A2"] == pytest.approx(50.0)
    assert windows[1]["kappa_kcal_mol_A2"] == pytest.approx(KBT / 0.2**2)
    assert audit["usable_windows"] == 2
    assert audit["unusable_windows"] == 2
    reasons = " ".join(str(entry.get("reason", "")) for entry in audit["entries"])
    assert "frozen" in reasons
    assert "not found" in reasons


def test_v3_adapter_rejects_duplicate_state_ids(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(
        json.dumps({"kappa_kcal_mol_A2": 50.0, "states": [{"state": 4}, {"state": 4}]})
    )

    with pytest.raises(ValueError, match="duplicate state ID"):
        covariance.load_v3_windows(campaign)


def test_v3_adapter_marks_aggregated_files_as_audit_only(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(
        json.dumps({"states": [{"state": 0, "kappa_kcal_mol_A2": 50.0}]})
    )
    np.savez(campaign / "meanforce_dataset.npz", state=np.array([0]), R=np.zeros((1, 3, 3)))

    windows, audit = covariance.load_v3_windows(campaign)

    assert windows == []
    assert audit["aggregated_only"] is True


def test_v3_adapter_audits_aggregated_files_without_manifest(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    np.savez(campaign / "stencil_states.npz", R=np.zeros((2, 3, 3)))

    windows, audit = covariance.load_v3_windows(campaign)

    assert windows == []
    assert audit["aggregated_only"] is True
    assert "missing manifest" in audit["entries"][0]["reason"]


def test_cli_writes_proof_of_concept_reports(tmp_path):
    rng = np.random.default_rng(9)
    anchor = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    reference = anchor[None, :, :] + rng.normal(scale=0.03, size=(80, 4, 3))
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, R=reference, time_ps=np.arange(len(reference), dtype=float))

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_window(campaign / "state_000000", n_frames=80, n_beads=4)
    np.savez(
        campaign / "state_000000" / "cg_coords.npz",
        R=anchor[None, :, :] + rng.normal(scale=0.02, size=(80, 4, 3)),
        time_ps=np.arange(80, dtype=float),
    )
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "states": [
                    {
                        "state": 0,
                        "reference_index": 0,
                        "target": anchor.tolist(),
                        "kappa_kcal_mol_A2": 50.0,
                    }
                ]
            }
        )
    )
    outdir = tmp_path / "out"

    assert covariance.main(
        [
            "--reference",
            str(reference_path),
            "--v3-campaign",
            str(campaign),
            "--outdir",
            str(outdir),
            "--n-anchors",
            "1",
            "--max-frames",
            "50",
            "--discard-ps",
            "5",
            "--min-reference-frames",
            "20",
            "--reference-radius-A",
            "0.5",
            "--block-length",
            "5",
            "--n-boot",
            "10",
            "--seed",
            "4",
        ]
    ) == 0

    for name in (
        "input_audit.json",
        "covariance_summary.json",
        "per_anchor.csv",
        "covariance_raw.npz",
    ):
        assert (outdir / name).exists(), name
    summary = json.loads((outdir / "covariance_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["n_analyzed_anchors"] == 1


def test_audit_plot_visualizes_available_and_missing_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    outdir = tmp_path / "audit"
    outdir.mkdir()
    report = {
        "reference_frames": 204000,
        "usable_windows": 0,
        "unusable_windows": 0,
        "aggregated_only": True,
        "entries": [
            {
                "status": "audit_only",
                "reason": "available aggregated files do not contain raw covariance frames",
            }
        ],
    }

    covariance._write_audit_plot(report, outdir)

    assert (outdir / "audit_status.png").exists()
