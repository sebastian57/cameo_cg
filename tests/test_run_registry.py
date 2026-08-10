from pathlib import Path

import pytest

from runs.registry import Registry, read_run_metadata, slurm_identity


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(
        tmp_path / "registry.sqlite3",
        tmp_path / "REGISTRY.md",
        tmp_path / "registry.lock",
    )


def test_array_identity_uses_parent_and_task():
    assert slurm_identity(
        {
            "SLURM_JOB_ID": "90123",
            "SLURM_ARRAY_JOB_ID": "90000",
            "SLURM_ARRAY_TASK_ID": "7",
        }
    ) == ("90000_7", "90123", "7", "90000")


def test_plain_identity_uses_job_id():
    assert slurm_identity({"SLURM_JOB_ID": "123"}) == (
        "123",
        "123",
        None,
        None,
    )


def test_metadata_is_optional_and_normalized(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  description: compare models\n  tags: [ala2, baseline]\n"
    )
    assert read_run_metadata(config) == (
        "compare models",
        ["ala2", "baseline"],
    )
    assert read_run_metadata(None) == (None, [])


def test_metadata_rejects_non_list_tags(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("run:\n  tags: ala2\n")
    with pytest.raises(ValueError, match="run.tags"):
        read_run_metadata(config)


def test_finish_preserves_metadata_and_sets_terminal_state(registry: Registry):
    registry.start(
        {
            "identity": "123",
            "job_id": "123",
            "state": "RUNNING",
            "run_type": "training",
            "description": "baseline",
            "outputs": ["/work/run"],
            "tags": ["ala2"],
            "source": "hook",
        }
    )
    registry.finish(
        "123", exit_code=2, finished_at="2026-08-10T12:00:00+00:00"
    )

    record = registry.get("123")
    assert record is not None
    assert record["state"] == "FAILED"
    assert record["exit_code"] == 2
    assert record["finished_at"] == "2026-08-10T12:00:00+00:00"
    assert record["description"] == "baseline"
    assert record["outputs"] == ["/work/run"]
    assert record["tags"] == ["ala2"]


def test_start_is_idempotent_and_does_not_erase_launcher_fields(
    registry: Registry,
):
    registry.start(
        {
            "identity": "123",
            "job_id": "123",
            "state": "RUNNING",
            "description": "authoritative",
            "outputs": ["/work/run"],
            "source": "hook",
        }
    )
    registry.start(
        {
            "identity": "123",
            "job_id": "123",
            "state": "RUNNING",
            "description": None,
            "outputs": [],
            "source": "discovered",
            "partition": "booster",
        }
    )

    record = registry.get("123")
    assert record is not None
    assert record["description"] == "authoritative"
    assert record["outputs"] == ["/work/run"]
    assert record["source"] == "hook"
    assert record["partition"] == "booster"
