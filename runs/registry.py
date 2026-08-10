#!/usr/bin/env python3
"""Shared registry for CAMEO CG Slurm runs."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent
DEFAULT_DB = RUNS_DIR / "registry.sqlite3"
DEFAULT_MARKDOWN = RUNS_DIR / "REGISTRY.md"
DEFAULT_LOCK = RUNS_DIR / "registry.lock"

FIELDS = (
    "identity",
    "job_id",
    "array_task_id",
    "parent_job_id",
    "run_type",
    "state",
    "scheduler_state",
    "user",
    "job_name",
    "node",
    "partition",
    "submitted_at",
    "started_at",
    "finished_at",
    "exit_code",
    "config_path",
    "runtime_config_path",
    "outputs",
    "command",
    "work_dir",
    "description",
    "tags",
    "source",
    "last_seen_at",
)
JSON_FIELDS = {"outputs", "tags"}
LAUNCHER_FIELDS = {
    "run_type",
    "config_path",
    "runtime_config_path",
    "outputs",
    "description",
    "tags",
    "source",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slurm_identity(
    env: Mapping[str, str],
) -> tuple[str, str, str | None, str | None]:
    """Return canonical identity, allocation job ID, array task, and parent."""
    job_id = env.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise ValueError("SLURM_JOB_ID is not set")
    parent = env.get("SLURM_ARRAY_JOB_ID", "").strip() or None
    task = env.get("SLURM_ARRAY_TASK_ID", "").strip() or None
    identity = f"{parent}_{task}" if parent and task is not None else job_id
    return identity, job_id, task, parent


def read_run_metadata(config_path: Path | None) -> tuple[str | None, list[str]]:
    """Read optional human metadata from a YAML config."""
    if config_path is None or not config_path.is_file():
        return None, []
    data = yaml.safe_load(config_path.read_text()) or {}
    run = data.get("run") or {}
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    description = run.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("run.description must be a string or null")
    tags = run.get("tags") or []
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("run.tags must be a list of strings")
    return description.strip() if description else None, [tag.strip() for tag in tags]


class Registry:
    def __init__(self, db_path: Path, markdown_path: Path, lock_path: Path):
        self.db_path = Path(db_path)
        self.markdown_path = Path(markdown_path)
        self.lock_path = Path(lock_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        columns = ",\n".join(
            ["identity TEXT PRIMARY KEY"]
            + [
                f"{field} {'INTEGER' if field == 'exit_code' else 'TEXT'}"
                for field in FIELDS
                if field != "identity"
            ]
        )
        with self._connect() as connection:
            connection.execute(f"CREATE TABLE IF NOT EXISTS runs ({columns})")

    @staticmethod
    def _encode(field: str, value: object) -> object:
        return json.dumps(value) if field in JSON_FIELDS else value

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for field in JSON_FIELDS:
            record[field] = json.loads(record[field] or "[]")
        return record

    def get(self, identity: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE identity = ?", (identity,)
            ).fetchone()
        return self._decode(row) if row else None

    def all(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs").fetchall()
        return [self._decode(row) for row in rows]

    def start(self, record: dict[str, object]) -> None:
        identity = str(record.get("identity") or "").strip()
        if not identity:
            raise ValueError("identity is required")

        incoming = {field: record.get(field) for field in FIELDS}
        incoming["identity"] = identity
        incoming["outputs"] = list(record.get("outputs") or [])
        incoming["tags"] = list(record.get("tags") or [])
        incoming["state"] = incoming["state"] or "RUNNING"
        incoming["last_seen_at"] = incoming["last_seen_at"] or utc_now()
        if incoming["state"] == "RUNNING":
            incoming["started_at"] = incoming["started_at"] or utc_now()

        existing = self.get(identity)
        if existing:
            merged = dict(existing)
            discovered_over_hook = (
                existing.get("source") == "hook"
                and incoming.get("source") == "discovered"
            )
            for field, value in incoming.items():
                if value in (None, "", []):
                    continue
                if discovered_over_hook and field in LAUNCHER_FIELDS:
                    continue
                merged[field] = value
        else:
            merged = incoming

        values = [self._encode(field, merged.get(field)) for field in FIELDS]
        placeholders = ", ".join("?" for _ in FIELDS)
        updates = ", ".join(
            f"{field}=excluded.{field}" for field in FIELDS if field != "identity"
        )
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO runs ({', '.join(FIELDS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(identity) DO UPDATE SET {updates}",
                values,
            )

    def finish(
        self, identity: str, exit_code: int, finished_at: str | None = None
    ) -> None:
        state = "COMPLETED" if exit_code == 0 else "FAILED"
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET state = ?, scheduler_state = COALESCE(scheduler_state, ?), "
                "exit_code = ?, finished_at = ?, last_seen_at = ? WHERE identity = ?",
                (
                    state,
                    state,
                    exit_code,
                    finished_at or utc_now(),
                    utc_now(),
                    identity,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run identity: {identity}")

    def show(self, identity: str) -> dict[str, Any] | None:
        return self.get(identity)

    def status(self) -> str:
        records = self.all()
        active = sum(record["state"] in {"PENDING", "RUNNING", "UNKNOWN"} for record in records)
        completed = sum(record["state"] == "COMPLETED" for record in records)
        failed = len(records) - active - completed
        return f"{active} active, {completed} completed, {failed} failed/cancelled"

    @staticmethod
    def _markdown_value(value: object) -> str:
        if value in (None, "", []):
            return "—"
        if isinstance(value, list):
            return "<br>".join(Registry._markdown_value(item) for item in value)
        return str(value).replace("|", "\\|").replace("\n", " ")

    def render(self) -> str:
        records = self.all()
        active_states = {"PENDING", "RUNNING", "UNKNOWN"}
        active = sorted(
            (record for record in records if record["state"] in active_states),
            key=lambda record: (record.get("parent_job_id") or record["identity"], record["identity"]),
        )
        recent = sorted(
            (record for record in records if record["state"] not in active_states),
            key=lambda record: record.get("finished_at") or "",
            reverse=True,
        )
        lines = [
            "# Run Registry",
            "",
            f"_Updated: {utc_now()}_",
            "",
            "## Active runs",
            "",
            "| Job | Type | User | Started | Description | Output |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for record in active:
            values = [
                record["identity"], record.get("run_type"), record.get("user"),
                record.get("started_at") or record.get("submitted_at"),
                record.get("description") or record.get("job_name"), record.get("outputs"),
            ]
            lines.append("| " + " | ".join(self._markdown_value(value) for value in values) + " |")
        lines.extend([
            "", "## Recent runs", "",
            "| Job | Status | Started | Finished | Description | Output |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for record in recent:
            values = [
                record["identity"], record.get("state"), record.get("started_at"),
                record.get("finished_at"), record.get("description") or record.get("job_name"),
                record.get("outputs"),
            ]
            lines.append("| " + " | ".join(self._markdown_value(value) for value in values) + " |")
        rendered = "\n".join(lines) + "\n"
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with tempfile.NamedTemporaryFile("w", dir=self.markdown_path.parent, delete=False) as tmp:
                tmp.write(rendered)
                temporary_path = Path(tmp.name)
            temporary_path.replace(self.markdown_path)
        return rendered


def default_registry() -> Registry:
    return Registry(
        Path(os.environ.get("CAMEO_RUN_REGISTRY_DB", DEFAULT_DB)),
        Path(os.environ.get("CAMEO_RUN_REGISTRY_MD", DEFAULT_MARKDOWN)),
        Path(os.environ.get("CAMEO_RUN_REGISTRY_LOCK", DEFAULT_LOCK)),
    )


def _start_record(args: argparse.Namespace) -> tuple[Registry, str]:
    registry = default_registry()
    if args.identity:
        identity = args.identity
        job_id, array_task_id, parent_job_id = args.job_id or args.identity, None, None
    else:
        identity, job_id, array_task_id, parent_job_id = slurm_identity(os.environ)
    config_path = Path(args.config).resolve() if args.config else None
    try:
        description, tags = read_run_metadata(config_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"WARNING: could not read run metadata: {error}", file=os.sys.stderr)
        description, tags = None, []
    description = args.description or description
    if not description and config_path:
        description = config_path.stem
    registry.start(
        {
            "identity": identity,
            "job_id": job_id,
            "array_task_id": array_task_id,
            "parent_job_id": parent_job_id,
            "run_type": args.run_type,
            "state": os.environ.get("SLURM_JOB_STATE", "RUNNING"),
            "user": os.environ.get("SLURM_JOB_USER") or getpass.getuser(),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "config_path": str(config_path) if config_path else None,
            "outputs": [str(Path(path).resolve()) for path in args.output],
            "command": " ".join(os.sys.argv),
            "work_dir": os.environ.get("SLURM_SUBMIT_DIR") or str(Path.cwd()),
            "description": description,
            "tags": tags,
            "source": "hook",
        }
    )
    return registry, identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--identity")
    start.add_argument("--job-id")
    start.add_argument("--run-type", required=True)
    start.add_argument("--config")
    start.add_argument("--output", action="append", default=[])
    start.add_argument("--description")
    finish = subparsers.add_parser("finish")
    finish.add_argument("--identity", required=True)
    finish.add_argument("--exit-code", required=True, type=int)
    subparsers.add_parser("render")
    subparsers.add_parser("status")
    show = subparsers.add_parser("show")
    show.add_argument("identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = default_registry()
    if args.command == "start":
        registry, identity = _start_record(args)
        registry.render()
        print(identity)
    elif args.command == "finish":
        registry.finish(args.identity, args.exit_code)
        registry.render()
    elif args.command == "render":
        registry.render()
    elif args.command == "status":
        print(registry.status())
    elif args.command == "show":
        record = registry.show(args.identity)
        if record is None:
            print(f"Run not found: {args.identity}", file=os.sys.stderr)
            return 1
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
