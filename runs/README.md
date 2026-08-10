# Run registry

This directory contains the shared registry tooling for CAMEO CG Slurm jobs.
The scripts are tracked; `REGISTRY.md`, SQLite state, and the lock file are
local runtime files and are intentionally ignored by Git.

Launchers register their config and output locations when a job starts and
record its exit status when it ends. The registry is shared by users of this
checkout. A failed registry update prints a warning but never fails the
scientific job.

## Metadata

Any YAML run config may include:

```yaml
run:
  description: "Compare the transition-enriched model with the baseline"
  tags: [ala2, force-matching]
```

Both fields are optional. The config name or Slurm job name is used when no
description is supplied.

## Commands

Run these from the repository root:

```bash
python3 runs/registry.py status
python3 runs/registry.py sync
python3 runs/registry.py show 1234567
python3 runs/registry.py render
```

- `status` reads the stored summary without contacting Slurm.
- `sync` discovers currently visible jobs associated with this checkout and
  reconciles known jobs with Slurm.
- `show` prints all stored fields for one job or array-task identity.
- `render` rebuilds `runs/REGISTRY.md` from SQLite.

The first sync imports only jobs currently visible through `squeue`. It does
not backfill old accounting records or scan historical output directories.
For discovered ad-hoc jobs, the output is blank when Slurm does not expose it.

## Optional periodic sync

A five-minute user cron job provides discovery and repair for jobs that cannot
run their exit trap:

```cron
*/5 * * * * cd /e/project1/cameo/schmidt36/cameo_cg && python3 runs/registry.py sync
```

Cron installation is intentionally manual because login environments and
cluster policies differ. Ensure the cron environment can find Python, PyYAML,
and the Slurm commands; use the full environment-specific Python path when
needed.
