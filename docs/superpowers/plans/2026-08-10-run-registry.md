# CAMEO CG Run Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared, automatically maintained registry for currently visible and future CAMEO CG Slurm runs, including output locations, metadata, and completion state.

**Architecture:** A standard-library Python CLI stores canonical records in ignored SQLite and atomically renders an ignored Markdown view. Lightweight shell hooks report authoritative launcher metadata and exit status; an idempotent Slurm sync discovers currently visible ad-hoc jobs and repairs terminal states for known jobs.

**Tech Stack:** Python 3 (`argparse`, `sqlite3`, `json`, `subprocess`, `fcntl`, `tempfile`), existing PyYAML, Bash, Slurm CLI, pytest.

## Global Constraints

- Track all current Slurm entry points: training, suites, MD, relative entropy, analysis, profiling, teacher materialization, data preparation, and LAMMPS.
- Import only currently visible and future jobs; do not backfill Slurm history or scan old output directories.
- Keep `runs/registry.sqlite3*`, `runs/registry.lock`, and `runs/REGISTRY.md` untracked.
- Share one registry across checkout users and record the owning user.
- Treat launcher metadata as authoritative over discovered metadata.
- Never let registry failures change a scientific job's exit status.
- Add no new dependency or long-running service.

---

### Task 1: Persistence, identities, metadata, and lifecycle commands

**Files:**
- Create: `runs/registry.py`
- Create: `tests/test_run_registry.py`

**Interfaces:**
- Produces: `Registry(db_path: Path, markdown_path: Path, lock_path: Path)`
- Produces: `slurm_identity(env: Mapping[str, str]) -> tuple[str, str, str | None, str | None]`
- Produces: `read_run_metadata(config_path: Path | None) -> tuple[str | None, list[str]]`
- Produces: `Registry.start(record: dict[str, object]) -> None`
- Produces: `Registry.finish(identity: str, exit_code: int, finished_at: str | None = None) -> None`
- Produces: CLI subcommands `start` and `finish`

- [ ] **Step 1: Write failing identity, metadata, and lifecycle tests**

```python
def test_array_identity_uses_parent_and_task():
    assert slurm_identity({
        "SLURM_JOB_ID": "90123",
        "SLURM_ARRAY_JOB_ID": "90000",
        "SLURM_ARRAY_TASK_ID": "7",
    }) == ("90000_7", "90123", "7", "90000")

def test_metadata_is_optional_and_normalized(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("run:\n  description: compare models\n  tags: [ala2, baseline]\n")
    assert read_run_metadata(config) == ("compare models", ["ala2", "baseline"])

def test_finish_preserves_metadata_and_sets_terminal_state(registry):
    registry.start({
        "identity": "123", "job_id": "123", "state": "RUNNING",
        "run_type": "training", "description": "baseline",
        "outputs": ["/work/run"],
    })
    registry.finish("123", exit_code=2, finished_at="2026-08-10T12:00:00+00:00")
    record = registry.get("123")
    assert record["state"] == "FAILED"
    assert record["exit_code"] == 2
    assert record["description"] == "baseline"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_run_registry.py`

Expected: collection fails because `runs.registry` does not exist.

- [ ] **Step 3: Implement the minimal SQLite model and lifecycle behavior**

Create one `runs` table keyed by `identity`, with scalar scheduler fields and JSON-encoded `outputs` and `tags`. Enable WAL mode and a busy timeout per connection. Implement launcher-authoritative upserts with SQL `COALESCE`/explicit merge logic so empty discovery fields cannot erase known values.

Use this state rule in `finish`:

```python
state = "COMPLETED" if exit_code == 0 else "FAILED"
```

Parse optional YAML metadata with the already-installed `yaml.safe_load`; return `(None, [])` for a missing config or missing `run` section, and raise a concise `ValueError` only for malformed field types.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

Run: `pytest -q tests/test_run_registry.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add runs/registry.py tests/test_run_registry.py
git commit -m "Add run registry lifecycle storage"
```

### Task 2: Atomic Markdown rendering and read commands

**Files:**
- Modify: `runs/registry.py`
- Modify: `tests/test_run_registry.py`

**Interfaces:**
- Consumes: `Registry.get(identity)` and stored run rows from Task 1
- Produces: `Registry.render() -> str`
- Produces: `Registry.status() -> str`
- Produces: `Registry.show(identity: str) -> dict[str, object] | None`
- Produces: CLI subcommands `render`, `status`, and `show`

- [ ] **Step 1: Write failing rendering tests**

```python
def test_render_orders_active_before_recent_and_escapes_tables(registry):
    registry.start({
        "identity": "11", "job_id": "11", "state": "RUNNING",
        "run_type": "md", "description": "A | B", "outputs": ["/work/md"],
    })
    registry.start({
        "identity": "10", "job_id": "10", "state": "RUNNING",
        "run_type": "training", "description": "done", "outputs": [],
    })
    registry.finish("10", 0, "2026-08-10T12:00:00+00:00")
    rendered = registry.render()
    assert rendered.index("## Active runs") < rendered.index("## Recent runs")
    assert "A \\| B" in rendered
    assert "| 10 | COMPLETED |" in rendered
    assert registry.markdown_path.read_text() == rendered
```

- [ ] **Step 2: Run the rendering test and verify RED**

Run: `pytest -q tests/test_run_registry.py::test_render_orders_active_before_recent_and_escapes_tables`

Expected: failure because `Registry.render` is absent.

- [ ] **Step 3: Implement rendering and read-only CLI output**

Render active states (`PENDING`, `RUNNING`, `UNKNOWN`) and terminal states in separate tables. Encode multiple outputs with `<br>`, render missing values as an em dash, group array identities by the stored parent ID, and sort terminal rows newest-first.

Acquire `fcntl.flock(lock_file, LOCK_EX)`, write the complete document to a temporary file in `runs/`, then replace `REGISTRY.md` with `Path.replace`. Make `status` return the same concise summary without querying Slurm. Make `show` return the decoded full row.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_run_registry.py`

Expected: all Task 1–2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add runs/registry.py tests/test_run_registry.py
git commit -m "Render run registry Markdown"
```

### Task 3: Current-job discovery and terminal reconciliation

**Files:**
- Modify: `runs/registry.py`
- Modify: `tests/test_run_registry.py`

**Interfaces:**
- Consumes: `Registry.start`, `Registry.finish`, and `Registry.render`
- Produces: `parse_squeue(text: str) -> list[dict[str, object]]`
- Produces: `parse_sacct(text: str) -> dict[str, dict[str, object]]`
- Produces: `Registry.sync(project_root: Path, run_command: Callable[..., CompletedProcess] = subprocess.run) -> None`

- [ ] **Step 1: Write failing parser and reconciliation tests**

```python
def test_sync_discovers_only_current_jobs_associated_with_checkout(registry, tmp_path):
    root = tmp_path / "cameo_cg"
    root.mkdir()
    responses = {
        "squeue": "101|alice|train|RUNNING|booster|jwb001|2026-08-10T10:00:00|2026-08-10T10:01:00\n"
                  "102|bob|other|RUNNING|booster|jwb002|2026-08-10T10:00:00|2026-08-10T10:01:00\n",
        "scontrol 101": f"JobId=101 WorkDir={root} Command={root}/scripts/run_training.sh\n",
        "scontrol 102": "JobId=102 WorkDir=/elsewhere Command=/elsewhere/job.sh\n",
    }
    registry.sync(root, run_command=fake_slurm(responses))
    assert registry.get("101")["source"] == "discovered"
    assert registry.get("102") is None

def test_sync_uses_sacct_only_for_known_job_that_disappeared(registry, tmp_path):
    registry.start({"identity": "201", "job_id": "201", "state": "RUNNING", "source": "hook"})
    responses = {
        "squeue": "",
        "sacct": "201|OUT_OF_MEMORY|0:125|2026-08-10T10:00:00|2026-08-10T10:05:00\n",
    }
    registry.sync(tmp_path, run_command=fake_slurm(responses))
    assert registry.get("201")["state"] == "FAILED"
```

- [ ] **Step 2: Run sync tests and verify RED**

Run: `pytest -q tests/test_run_registry.py -k sync`

Expected: failure because `Registry.sync` and Slurm parsers are absent.

- [ ] **Step 3: Implement current-only sync**

Run `squeue -h -a -o` with a pipe-delimited format containing job identity, user, name, state, partition, node, submit time, and start time. For each current job, inspect `scontrol show job -o <identity>` and accept it only when `WorkDir` is within `project_root` or `Command` contains the resolved project-root path.

Do not query broad accounting history. Query `sacct -n -X -P -j <known-missing-identities>` only for nonterminal registry rows that were known before this sync but disappeared from `squeue`. Normalize `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, and nonzero completed exit codes to `FAILED`; preserve the original Slurm state in `scheduler_state`; normalize cancellation variants to `CANCELLED`.

Render only after a successful reconciliation transaction. A failed Slurm command raises `RuntimeError` and leaves existing records intact.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_run_registry.py`

Expected: all registry tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add runs/registry.py tests/test_run_registry.py
git commit -m "Reconcile registry with current Slurm jobs"
```

### Task 4: Failure-tolerant Bash hook

**Files:**
- Create: `runs/registry_hook.sh`
- Create: `tests/test_run_registry_hook.py`

**Interfaces:**
- Consumes: `registry.py start` and `registry.py finish`
- Produces: `run_registry_start RUN_TYPE [CONFIG] [OUTPUT ...]`
- Produces: `run_registry_finish EXIT_CODE`

- [ ] **Step 1: Write a failing executable shell-contract test**

```python
def test_registry_failure_does_not_change_workload_exit_code(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\nset -e\n"
        "source runs/registry_hook.sh\n"
        "RUN_REGISTRY_PYTHON=/does/not/exist\n"
        "run_registry_start training '' /tmp/output\n"
        "false\n"
    )
    result = subprocess.run(["bash", str(script)], cwd=PROJECT_ROOT)
    assert result.returncode == 1
```

- [ ] **Step 2: Run the hook test and verify RED**

Run: `pytest -q tests/test_run_registry_hook.py`

Expected: failure because `runs/registry_hook.sh` is absent.

- [ ] **Step 3: Implement the minimal hook**

Resolve `registry.py` relative to `BASH_SOURCE`, choose `${PYTHON_BIN:-python3}`, and invoke CLI commands inside explicit `if ! ...; then echo WARNING >&2; fi` blocks. Store the identity returned by `start` in `RUN_REGISTRY_ID`. Make `run_registry_finish` idempotent and always return zero.

The hook must not install an EXIT trap itself because several launchers already own cleanup traps. Each launcher composes it with its cleanup explicitly.

- [ ] **Step 4: Run hook and registry tests and verify GREEN**

Run: `pytest -q tests/test_run_registry_hook.py tests/test_run_registry.py`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add runs/registry_hook.sh tests/test_run_registry_hook.py
git commit -m "Add failure-tolerant run registry hook"
```

### Task 5: Integrate every current Slurm launcher

**Files:**
- Modify: `scripts/run_training.sh`
- Modify: `scripts/submit_suite.sh`
- Modify: `scripts/submit_md.sh`
- Modify: `scripts/submit_md_parallel.sh`
- Modify: `scripts/submit_md_array.sh`
- Modify: `scripts/run_relative_entropy.sh`
- Modify: `scripts/run_analysis.sh`
- Modify: `scripts/run_profiling.sh`
- Modify: `scripts/submit_teacher_materialization.sh`
- Modify: `data_prep/run_pipeline_gpu.sh`
- Modify: `md_setup/submit_lammps_chemtrain.sh`
- Create: `tests/test_run_registry_launchers.py`

**Interfaces:**
- Consumes: `run_registry_start` and `run_registry_finish` from Task 4
- Produces: start/finish events with the correct run type, config, and resolved output path from each launcher

- [ ] **Step 1: Write failing launcher smoke tests**

Create temporary fake `python3`, `srun`, environment loaders, and workload commands, then execute representative minimal launchers with `SLURM_JOB_ID=123`. Assert against a temporary registry database that training, MD, and a no-config job each record their resolved output and terminal state. Run every modified shell file through `bash -n`.

- [ ] **Step 2: Run launcher tests and verify RED**

Run: `pytest -q tests/test_run_registry_launchers.py`

Expected: registry rows are absent because launchers do not source the hook.

- [ ] **Step 3: Add lifecycle calls to launchers**

After each launcher resolves and creates its output location:

```bash
source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start training "${CONFIG_FILE}" "${RUN_OUTPUT_DIR}"
```

For scripts without existing cleanup, add an EXIT trap that captures `$?`, disables itself, calls `run_registry_finish`, and exits with the original code. For `run_training.sh`, `run_relative_entropy.sh`, and `run_profiling.sh`, compose registry completion into their existing ERR/EXIT cleanup paths without replacing telemetry cleanup or error reporting.

Use these run types: `training`, `training-suite`, `md`, `relative-entropy`, `analysis`, `profiling`, `teacher-materialization`, `data-preparation`, and `lammps`. Pass all real outputs known by the launcher. The submission-only branch of `submit_md_array.sh` does not register a task; each allocated task registers itself. `submit_suite.sh` remains the submitter, while allocated `run_training.sh` tasks hold authoritative task records and parent array IDs.

- [ ] **Step 4: Run launcher and shell syntax tests and verify GREEN**

Run: `pytest -q tests/test_run_registry_launchers.py tests/test_run_registry_hook.py && bash -n scripts/run_training.sh scripts/submit_suite.sh scripts/submit_md.sh scripts/submit_md_parallel.sh scripts/submit_md_array.sh scripts/run_relative_entropy.sh scripts/run_analysis.sh scripts/run_profiling.sh scripts/submit_teacher_materialization.sh data_prep/run_pipeline_gpu.sh md_setup/submit_lammps_chemtrain.sh`

Expected: pytest passes and `bash -n` exits zero.

- [ ] **Step 5: Commit Task 5**

```bash
git add scripts/run_training.sh scripts/submit_suite.sh scripts/submit_md.sh scripts/submit_md_parallel.sh scripts/submit_md_array.sh scripts/run_relative_entropy.sh scripts/run_analysis.sh scripts/run_profiling.sh scripts/submit_teacher_materialization.sh data_prep/run_pipeline_gpu.sh md_setup/submit_lammps_chemtrain.sh tests/test_run_registry_launchers.py
git commit -m "Track Slurm launcher lifecycles"
```

### Task 6: Configuration, ignored runtime files, and operator documentation

**Files:**
- Modify: `.gitignore`
- Modify: `configs/base_config.yaml`
- Create: `runs/README.md`

**Interfaces:**
- Consumes: CLI and metadata behavior from Tasks 1–5
- Produces: documented run metadata, manual commands, and optional five-minute cron setup

- [ ] **Step 1: Add the exact ignored runtime paths**

```gitignore
# --- shared run registry runtime state ---
runs/REGISTRY.md
runs/registry.sqlite3
runs/registry.sqlite3-*
runs/registry.lock
```

- [ ] **Step 2: Document optional metadata in the base config**

```yaml
run:
  description: null  # optional short purpose shown in runs/REGISTRY.md
  tags: []           # optional searchable labels
```

- [ ] **Step 3: Write operator documentation**

Document `status`, `sync`, `show`, and `render`; state that the initial sync imports only currently visible jobs; explain that missing output paths for discovered ad-hoc jobs are expected; and provide this optional cron entry:

```cron
*/5 * * * * cd /e/project1/cameo/schmidt36/cameo_cg && python3 runs/registry.py sync
```

- [ ] **Step 4: Verify tracked/ignored behavior**

Run: `git check-ignore runs/REGISTRY.md runs/registry.sqlite3 runs/registry.sqlite3-wal runs/registry.lock && git check-ignore -v runs/registry.py || true`

Expected: the four runtime paths are ignored; `runs/registry.py` is not reported as ignored. Because the repository has a broad Markdown-ignore rule, force-add only `runs/README.md` while leaving `runs/REGISTRY.md` ignored.

- [ ] **Step 5: Commit Task 6**

```bash
git add .gitignore configs/base_config.yaml
git add -f runs/README.md
git commit -m "Document shared run registry"
```

### Task 7: Final verification

**Files:**
- Verify all files changed in Tasks 1–6

**Interfaces:**
- Consumes: complete feature
- Produces: evidence that tests, shell syntax, CLI smoke behavior, and ignore rules all pass

- [ ] **Step 1: Run focused tests**

Run: `pytest -q tests/test_run_registry.py tests/test_run_registry_hook.py tests/test_run_registry_launchers.py`

Expected: all focused tests pass with no warnings.

- [ ] **Step 2: Run shell syntax verification**

Run: `bash -n scripts/run_training.sh scripts/submit_suite.sh scripts/submit_md.sh scripts/submit_md_parallel.sh scripts/submit_md_array.sh scripts/run_relative_entropy.sh scripts/run_analysis.sh scripts/run_profiling.sh scripts/submit_teacher_materialization.sh data_prep/run_pipeline_gpu.sh md_setup/submit_lammps_chemtrain.sh runs/registry_hook.sh`

Expected: exit zero and no output.

- [ ] **Step 3: Run a temporary-registry CLI smoke test**

Run `start`, `finish`, `status`, `show`, and `render` with `CAMEO_RUN_REGISTRY_DB`, `CAMEO_RUN_REGISTRY_MD`, and `CAMEO_RUN_REGISTRY_LOCK` pointing into a temporary directory. Verify the Markdown contains the fake completed run and no files appear at the default runtime paths.

- [ ] **Step 4: Run the relevant existing test suite**

Run: `pytest -q tests/test_run_registry*.py tests/test_relative_entropy_script.py`

Expected: all selected tests pass.

- [ ] **Step 5: Inspect the final diff and runtime tracking state**

Run: `git diff --check && git status --short && git check-ignore runs/REGISTRY.md runs/registry.sqlite3 runs/registry.sqlite3-wal runs/registry.lock`

Expected: no whitespace errors; only intended feature files plus pre-existing user changes are present; runtime files are ignored.
