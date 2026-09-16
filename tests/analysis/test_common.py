from __future__ import annotations

import json
from pathlib import Path

from analysis.common.paths import AnalysisPathError, repo_root, resolve_glob, resolve_input, resolve_output
from analysis.common.provenance import write_manifest


def test_repo_root_is_derived_from_package() -> None:
    root = repo_root()
    assert (root / "analysis").is_dir()
    assert (root / "README.md").is_file()


def test_relative_input_and_glob_resolve_against_explicit_base(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "b.npz").write_bytes(b"b")
    (tmp_path / "data" / "a.npz").write_bytes(b"a")
    assert resolve_input("data/a.npz", base=tmp_path).name == "a.npz"
    assert [path.name for path in resolve_glob("data/*.npz", base=tmp_path)] == ["a.npz", "b.npz"]


def test_missing_input_is_explicit(tmp_path: Path) -> None:
    try:
        resolve_input("missing.npz", base=tmp_path, label="reference")
    except FileNotFoundError as exc:
        assert "reference does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing input unexpectedly resolved")


def test_invalid_project_root_is_rejected(tmp_path: Path) -> None:
    try:
        repo_root(tmp_path)
    except AnalysisPathError as exc:
        assert "Not a cameo_cg checkout" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid project root unexpectedly accepted")


def test_manifest_records_resolved_inputs_and_outputs(tmp_path: Path) -> None:
    output = resolve_output(tmp_path / "results")
    input_path = tmp_path / "input.txt"
    input_path.write_text("analysis input\n")
    result = output / "summary.json"
    result.write_text(json.dumps({"ok": True}) + "\n")
    manifest_path = write_manifest(
        output,
        inputs={"reference": input_path},
        parameters={"temperature_K": 300.0},
        module="analysis.tests.example",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["inputs"]["reference"]["path"] == str(input_path.resolve())
    assert manifest["parameters"]["temperature_K"] == 300.0
    assert any(item["path"] == str(result.resolve()) for item in manifest["outputs"])
    assert manifest["git"]["commit"]
