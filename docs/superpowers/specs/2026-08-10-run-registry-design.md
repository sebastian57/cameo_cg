# CAMEO CG Run Registry Design

## Goal

Add a shared run registry for every Slurm-backed workflow in `cameo_cg`. The
registry shows jobs that are currently active, where they write output, short
human-provided metadata, and the completion state and time of jobs observed
from this point onward.

Historical jobs that have already left Slurm's current-job view are out of
scope. On first use, the registry imports only jobs that are currently visible
and associated with this checkout.

## Scope

The registry covers:

- Training, including suites and array tasks
- Molecular dynamics in single, parallel, and array modes
- Relative-entropy runs
- Analysis and profiling runs
- Teacher-label materialization
- Data-preparation jobs
- LAMMPS jobs
- Ad-hoc Slurm jobs associated with this checkout

It is shared by all users of this checkout. Every record includes the owning
user.

## Repository Layout

The implementation lives in a new `runs/` directory:

```text
runs/
├── registry.py          # CLI, persistence, Slurm reconciliation, rendering
├── registry_hook.sh     # failure-tolerant shell API for launchers
├── README.md            # commands and optional periodic-sync setup
├── registry.sqlite3     # ignored canonical state
├── REGISTRY.md          # ignored generated human-readable view
└── registry.lock        # ignored cross-process rendering lock
```

The scripts and documentation are tracked. Runtime registry files are ignored
using explicit `.gitignore` entries. SQLite is the canonical data store because
it supports atomic concurrent updates from shared jobs. `REGISTRY.md` is an
automatically regenerated view, not a separately edited source of truth.

The implementation uses the Python standard library and existing Slurm command
line tools. It introduces no service or third-party dependency.

## Run Metadata

All YAML run configurations may contain this optional top-level section:

```yaml
run:
  description: "Compare transition-enriched data against the reference baseline"
  tags: [ala2, transition-data, force-matching]
```

Both fields are optional. When `description` is absent, the registry uses the
config filename or Slurm job name as a concise fallback. The registry reads
this section directly; training and simulation code do not need to consume it.
`configs/base_config.yaml` will document the section.

Jobs without YAML configuration remain valid and use their command or job name
as their description.

## Data Model

Each logical run record contains:

- Slurm cluster, job ID, and optional array task ID as its stable identity
- Parent array job ID when applicable
- Run type
- Slurm state normalized to `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`,
  `CANCELLED`, or `UNKNOWN`
- User, job name, node, and partition when available
- Submission, start, and completion timestamps when available
- Exit code when available
- Original config path and copied runtime/input config paths when available
- Zero or more output paths
- Command and Slurm working directory
- Description and tags
- Whether the record came from a launcher hook or Slurm discovery
- Last reconciliation timestamp

Array tasks have separate records so that their outputs and failures remain
visible. They are linked by parent array job ID and grouped in rendered output.

Launcher-provided config and output information takes precedence over inferred
Slurm information. Reconciliation enriches records without replacing more
specific values with empty or less reliable values.

## Lifecycle

### Instrumented jobs

After a launcher has resolved its configuration and output paths, it calls the
shared hook to register the job. The hook installs or supports an exit/signal
trap that records the process exit code without changing it.

The lifecycle is:

1. Resolve config and output paths.
2. Register the job as `RUNNING`, including metadata and all known outputs.
3. Run the scientific workload.
4. Record a provisional final state and exit code from the shell trap.
5. Let the next reconciliation confirm or correct the final Slurm state.
6. Regenerate `REGISTRY.md` after every successful registry mutation.

Registry failures emit warnings but never fail, cancel, or change the exit code
of a scientific run.

### Slurm reconciliation

`python3 runs/registry.py sync` queries `squeue`, `scontrol`, and `sacct` as
needed. It:

- Discovers currently visible jobs whose working directory is inside this
  checkout or whose command references a path inside it
- Adds currently pending or running ad-hoc jobs
- Refreshes states and scheduler fields for known jobs
- Corrects terminal states such as cancellation, timeout, out-of-memory, and
  node failure when traps cannot report them
- Renders the Markdown view

The first sync deliberately ignores jobs no longer present in the current
Slurm job view. There is no historical backfill command and no scan of old
output directories.

Reconciliation is idempotent. Re-running it does not duplicate records.

### Periodic execution

No long-running watcher is required. `runs/README.md` documents manual sync and
an optional user cron entry that runs every five minutes:

```cron
*/5 * * * * cd /e/project1/cameo/schmidt36/cameo_cg && python3 runs/registry.py sync
```

Cron installation remains an explicit user action because cluster policies and
user environments differ. Lifecycle hooks provide immediate updates even when
periodic sync is not installed; sync supplies discovery and repair.

## Launcher Integration

The shared hook is added to all current Slurm entry points for training,
training suites, MD, relative entropy, analysis, profiling, teacher
materialization, data preparation, and LAMMPS.

Launchers pass resolved information rather than making `registry.py` duplicate
their path-resolution rules. A job may provide multiple output paths. If no
output is known, the registry records no output and renders an em dash rather
than guessing.

Suite and array launchers keep one record per array task. The array parent ID
is stored for grouping. Submission-time registration is optional; a task is
guaranteed to register when it begins execution, and periodic sync can show it
while pending.

Ad-hoc jobs require no launcher modification if reconciliation can associate
their working directory or command with the checkout. Information that Slurm
does not expose, such as a dynamically chosen output directory, remains blank.

## Markdown View

`runs/REGISTRY.md` is replaced atomically after each update. A cross-process
lock prevents concurrent renders from interleaving.

Its primary sections are:

```markdown
# Run Registry

## Active runs
| Job | Type | User | Started | Description | Output |

## Recent runs
| Job | Status | Started | Finished | Description | Output |
```

Active runs appear first. Completed runs are ordered by most recent completion.
Array tasks are grouped under their parent job without hiding per-task status.
Paths are rendered in a readable form and remain unambiguous. Full details are
available through `show` rather than expanding the main tables indefinitely.

## Command Interface

The initial CLI is:

```bash
python3 runs/registry.py status
python3 runs/registry.py sync
python3 runs/registry.py show 1234567
python3 runs/registry.py render
```

Internal `start` and `finish` subcommands support `registry_hook.sh`. They are
documented for maintainers but normal users do not need to call them.

`status` refreshes no external state; it prints the current registry summary.
`sync` performs Slurm reconciliation before rendering. `show` accepts a job ID
or array-task identity and displays the complete record. `render` rebuilds only
the Markdown view from SQLite.

## Concurrency and Failure Handling

- SQLite transactions make record updates atomic.
- A short file lock serializes Markdown rendering and uses atomic replacement.
- Start and sync operations use upserts keyed by stable Slurm identity.
- Missing optional metadata never prevents registration.
- Malformed `run` metadata produces a warning and falls back to inferred text.
- Slurm command failures leave existing records unchanged and return a clear
  nonzero status from manual `sync`.
- The shell hook absorbs registry errors so compute jobs continue unaffected.
- Terminal scheduler states override provisional trap states when they disagree.

## Verification

Focused tests use temporary SQLite databases and mocked Slurm command output.
They cover:

- Idempotent start and sync updates
- Concurrent-safe record creation and rendering
- Completion, failure, cancellation, timeout, out-of-memory, and node-failure
  transitions
- Array-task identity and grouping
- Optional config metadata and fallbacks
- Multiple output paths
- Current ad-hoc job discovery and repository association filtering
- Exclusion of historical jobs that are no longer currently visible
- Preservation of authoritative launcher fields during reconciliation
- Atomic Markdown generation
- Registry failures not altering launcher exit codes

A repository-level smoke test creates a temporary registry, records a fake run,
finishes it, renders Markdown, and verifies the resulting row and timestamps.

## Deliberate Limitations

- The feature does not reconstruct historical jobs from Slurm accounting or
  old output directories.
- Ad-hoc jobs can expose only metadata Slurm makes available.
- Cron setup is documented but not installed automatically.
- The registry is scoped to this shared checkout, not every CAMEO CG checkout
  elsewhere on the filesystem.
