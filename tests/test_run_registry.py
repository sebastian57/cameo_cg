from pathlib import Path
import subprocess

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


def test_render_orders_active_before_recent_and_escapes_tables(
    registry: Registry,
):
    registry.start(
        {
            "identity": "11",
            "job_id": "11",
            "state": "RUNNING",
            "run_type": "md",
            "description": "A | B",
            "outputs": ["/work/md", "/work/log"],
            "source": "hook",
        }
    )
    registry.start(
        {
            "identity": "10",
            "job_id": "10",
            "state": "RUNNING",
            "run_type": "training",
            "description": "done",
            "outputs": [],
            "source": "hook",
        }
    )
    registry.finish("10", 0, "2026-08-10T12:00:00+00:00")

    rendered = registry.render()

    assert rendered.index("## Active runs") < rendered.index("## Recent runs")
    assert "A \\| B" in rendered
    assert "/work/md<br>/work/log" in rendered
    assert "| 10 | COMPLETED |" in rendered
    assert registry.markdown_path.read_text() == rendered


def test_status_summarizes_states_and_show_returns_decoded_record(
    registry: Registry,
):
    registry.start(
        {
            "identity": "21",
            "job_id": "21",
            "state": "PENDING",
            "tags": ["queued"],
            "source": "discovered",
        }
    )

    assert registry.status() == "1 active, 0 completed, 0 failed/cancelled"
    assert registry.show("21")["tags"] == ["queued"]
    assert registry.show("missing") is None


def fake_slurm(responses: dict[str, str]):
    def run(command, **_kwargs):
        if command[0] == "squeue":
            key = "squeue"
        elif command[0] == "scontrol":
            key = f"scontrol {command[-1]}"
        elif command[0] == "sacct":
            key = "sacct"
        else:
            raise AssertionError(f"unexpected command: {command}")
        if key not in responses:
            raise AssertionError(f"unexpected Slurm call: {key}")
        return subprocess.CompletedProcess(command, 0, responses[key], "")

    return run


def test_sync_discovers_only_current_jobs_associated_with_checkout(
    registry: Registry, tmp_path: Path
):
    root = tmp_path / "cameo_cg"
    root.mkdir()
    responses = {
        "squeue": (
            "101|alice|train|RUNNING|booster|jwb001|"
            "2026-08-10T10:00:00|2026-08-10T10:01:00\n"
            "102|bob|other|RUNNING|booster|jwb002|"
            "2026-08-10T10:00:00|2026-08-10T10:01:00\n"
        ),
        "scontrol 101": (
            f"JobId=101 WorkDir={root} "
            f"Command={root}/scripts/run_training.sh\n"
        ),
        "scontrol 102": (
            "JobId=102 WorkDir=/elsewhere Command=/elsewhere/job.sh\n"
        ),
    }

    registry.sync(root, run_command=fake_slurm(responses))

    assert registry.get("101")["source"] == "discovered"
    assert registry.get("101")["state"] == "RUNNING"
    assert registry.get("102") is None


def test_sync_uses_sacct_only_for_known_job_that_disappeared(
    registry: Registry, tmp_path: Path
):
    registry.start(
        {
            "identity": "201",
            "job_id": "201",
            "state": "RUNNING",
            "description": "keep me",
            "source": "hook",
        }
    )
    responses = {
        "squeue": "",
        "sacct": (
            "201|OUT_OF_MEMORY|0:125|2026-08-10T10:00:00|"
            "2026-08-10T10:05:00\n"
        ),
    }

    registry.sync(tmp_path, run_command=fake_slurm(responses))

    record = registry.get("201")
    assert record["state"] == "FAILED"
    assert record["scheduler_state"] == "OUT_OF_MEMORY"
    assert record["exit_code"] == 125
    assert record["description"] == "keep me"


def test_sync_maps_cancelled_known_job(registry: Registry, tmp_path: Path):
    registry.start(
        {"identity": "301", "job_id": "301", "state": "PENDING", "source": "hook"}
    )
    responses = {
        "squeue": "",
        "sacct": (
            "301|CANCELLED by 42|0:15|2026-08-10T10:00:00|"
            "2026-08-10T10:02:00\n"
        ),
    }

    registry.sync(tmp_path, run_command=fake_slurm(responses))

    assert registry.get("301")["state"] == "CANCELLED"
