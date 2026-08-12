"""Campaign launch topology: one replica per job, or many replicas per GPU via `-multidir`.

WHY
    Campaigns currently run ONE GROMACS replica per SLURM array task per GPU. On ala2
    (2,642 atoms) that leaves the node almost idle: 937 ns/day at dt=1 fs is ~92 us/step,
    while a GH200 does that work in single-digit microseconds. The run is kernel-launch
    latency bound, not compute bound, so several replicas share a GPU almost for free.

    `-multidir` runs N replicas as N MPI ranks inside ONE mdrun invocation, which is how
    GROMACS is meant to do this.

THE BLOCKING FACT, AND WHY IT IS ALREADY SATISFIED
    `-multidir` needs a real MPI build. The `gmx` binary the campaigns call today is
    thread-MPI and silently cannot do it. The same module also ships **`gmx_mpi`**
    (ParaStation MPI 5.11.0-1, GPU-aware CUDA), which can. That is the only reason this is a
    launcher change rather than a rebuild.

LAYOUT IS PRESERVED ON PURPOSE
    `collect.py::_discover_cases` finds cases by `replica_*` / `case_*` name prefix and sorts
    lexically, and each case still ends up with `biased.trr`, `unbiased_forces.trr` and
    `biased.tpr` in its own directory. So the collector, the analysis and the seed
    back-mapping in `build_harvest_campaign.py` all work unchanged.

WHAT MULTIDIR CHANGES, AND THE TWO THINGS THAT BITE
  * **Phases must be split.** Per-case `run_case.sh` does grompp -> mdrun -> rerun for one
    replica. Multidir needs ALL the `.tpr`s built first, then one mdrun across the group, then
    one rerun. Hence a parent-level script instead of N independent ones.
  * **Failure granularity gets worse.** A crashed rank aborts the whole invocation, where
    today one array task fails alone (and `collect.py --coords-only` relies on being able to
    skip a dead replica). So replicas are grouped and each GROUP is one array task: a failure
    costs one group, not the campaign. `--replicas-per-job` sets the group size.
  * **The bias server does not multiplex.** `sampling/server.py` is one process per replica,
    `backlog=1`, a strictly serial accept loop, and the wire header carries no replica ID, so
    N replicas against one server would serialise. Under multidir the parent script therefore
    starts N servers and owns their cleanup. Campaigns on the native PLUMED backend
    (`bias_backend: plumed`) need none at all, which is what makes dense packing actually pay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = ["multidir_group_script", "submit_script", "group_ranges",
           "cpus_per_rank", "Topology"]

GMX_MODULES = "Stages/2025 GCC/13.3.0 ParaStationMPI/5.11.0-1 GROMACS/2024.3-PLUMED-2.9.3"

#: Loading the GROMACS stack purges the Stages/2026 stack the venv needs, and vice versa.
#: Every GROMACS call in this repo is wrapped this way (see collect.py, build_harvest_campaign).
_MODULE_BLOCK = f"""module --force purge
module load {GMX_MODULES}"""


def group_ranges(n_replicas: int, per_job: int) -> list[list[int]]:
    """Split replicas into multidir groups. The last group may be short."""
    if per_job < 1:
        raise ValueError(f"replicas_per_job must be >= 1, got {per_job}")
    return [list(range(s, min(s + per_job, n_replicas)))
            for s in range(0, n_replicas, per_job)]


def multidir_group_script(*, case_dirs: Sequence[str], structure_for: Sequence[str],
                          topology: str, ntomp: int, n_gpus: int = 4,
                          repo: str | Path = "", socket_dir: str | None = None,
                          campaign_name: str = "", replicas: Sequence[int] = (),
                          socket_timeout: float = 600.0, use_server: bool = False,
                          mps: bool = False, ranks_per_replica: int = 1) -> str:
    """Render the parent script that runs one multidir GROUP.

    `case_dirs` are relative to the campaign root and become both the grompp working
    directories and the `-multidir` arguments, so every output lands where `collect.py`
    already looks for it.
    """
    n = len(case_dirs)
    if n != len(structure_for):
        raise ValueError("case_dirs and structure_for must be the same length")
    dirs = " ".join(case_dirs)

    grompp = "\n".join(
        f'( cd "{d}" && gmx grompp -f production.mdp -c {s} -p {topology} '
        f'-o biased.tpr -maxwarn 2 )'
        for d, s in zip(case_dirs, structure_for)
    )

    server_start = server_stop = ""
    if use_server:
        if socket_dir is None or not campaign_name or len(replicas) != n:
            raise ValueError("server mode needs socket_dir, campaign_name and one replica id "
                             "per case")
        starts = "\n".join(
            f'SOCK_{i}="{socket_dir}/cgbias_{campaign_name}_r{r}.sock"\n'
            f'rm -f "$SOCK_{i}"\n'
            f'python -m sampling.server --config "$CAMPAIGN/{d}/server.yaml" '
            f'--socket "$SOCK_{i}" --log "$CAMPAIGN/{d}/server_events.log" '
            f'--io-timeout {socket_timeout} > "$CAMPAIGN/{d}/server.log" 2>&1 &\n'
            f'PIDS+=($!)'
            for i, (d, r) in enumerate(zip(case_dirs, replicas))
        )
        waits = "\n".join(
            f'for _ in $(seq 1 120); do [ -S "$SOCK_{i}" ] && break; sleep 1; done\n'
            f'[ -S "$SOCK_{i}" ] || {{ echo "no socket $SOCK_{i}" >&2; exit 1; }}'
            for i in range(n)
        )
        # The servers are started from the PARENT here, not from a per-case script, so the
        # parent must also reap them -- otherwise they outlive the job and hold the sockets.
        server_start = f"""
# ---- bias servers: one per replica (server.py does NOT multiplex) ------------
source {repo}/env_setup/load_modules_2026.sh
source {repo}/../venv_cameocg_jupiter2026/bin/activate
export JAX_PLATFORMS=cpu
PIDS=()
cd {repo}
{starts}
cd "$CAMPAIGN"
trap 'for p in "${{PIDS[@]}}"; do kill $p 2>/dev/null || true; done; rm -f {socket_dir}/cgbias_{campaign_name}_r*.sock' EXIT
{waits}
"""
        server_stop = '\nfor p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done\n'

    mps_block = ""
    if mps:
        # SLURM here exposes GresTypes=gpu only, so MPS is not scheduler-managed; on an
        # exclusive node we can run the daemon ourselves. Without it several ranks still share
        # a GPU by time-slicing, which already helps a latency-bound run -- MPS just overlaps
        # them properly.
        mps_block = """
export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY/log"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || echo "MPS daemon failed to start; continuing time-sliced" >&2
"""

    gpu_ids = "".join(str(i) for i in range(n_gpus))
    total_ranks = n * max(1, int(ranks_per_replica))
    # -gpu_id lists the GPUs available to each node; GROMACS distributes ranks over them.
    # It is only meaningful when several ranks share a node's GPUs, which is the packed
    # regime. With one replica per several GPUs, let GROMACS assign them.
    gpu_flag = f" -gpu_id {gpu_ids}" if ranks_per_replica == 1 else ""
    return f"""#!/bin/bash
# Generated by sampling/launch.py -- do not edit by hand.
# {n} replicas x {ranks_per_replica} rank(s) = {total_ranks} ranks over {n_gpus} GPUs/node.
set -Eeuo pipefail
CAMPAIGN="$(cd "$(dirname "$0")" && pwd)"
cd "$CAMPAIGN"
{server_start}
{_MODULE_BLOCK}
{mps_block}
# ---- 1. grompp every replica FIRST; -multidir needs all .tpr present ---------
{grompp}

# ---- 2. one biased mdrun across the group -----------------------------------
# gmx_mpi, not gmx: the default binary is thread-MPI and cannot do -multidir.
srun -n {total_ranks} gmx_mpi mdrun -multidir {dirs} \\
    -plumed plumed.dat -deffnm biased -ntomp {ntomp}{gpu_flag}

# ---- 3. bias-free force rerun (MANDATORY) -----------------------------------
# No -plumed here, so this phase needs no bias server at all.
srun -n {total_ranks} gmx_mpi mdrun -multidir {dirs} \\
    -s biased.tpr -rerun biased.trr -deffnm unbiased_forces -ntomp {ntomp}
{server_stop}
echo "multidir group complete: {dirs}"
"""


def cpus_per_rank(n_ranks_per_node: int, node_cpus: int = 288, cap: int = 16) -> int:
    """OMP threads per rank, and the value SLURM must be told per task.

    `--cpus-per-task * --ntasks-per-node` must fit the node: asking 6 x 64 on a 288-CPU node
    simply never schedules. Capped as well, because a 2,642-atom system gets nothing from
    dozens of OMP threads and over-subscribing hurts when several ranks share a GPU.
    """
    return max(1, min(cap, node_cpus // max(n_ranks_per_node, 1)))


@dataclass(frozen=True)
class Topology:
    """How a multidir group maps onto nodes, ranks and GPUs.

    TWO REGIMES, ONE KNOB. Packing several replicas onto one GPU pays off only while a
    replica is too small to keep the GPU busy -- the ala2 case, where a step costs ~92 us of
    which almost all is launch latency. A large replica saturates a GPU by itself, and the
    correct move inverts: `ranks_per_replica > 1` and several GPUs per replica, one replica
    at a time. Both are expressed here rather than assumed.
    """

    n_replicas: int                 # replicas in this group
    ranks_per_replica: int = 1      # GROMACS domain-decomposition ranks per replica
    gpus_per_node: int = 4
    node_cpus: int = 288

    @property
    def total_ranks(self) -> int:
        return self.n_replicas * self.ranks_per_replica

    @property
    def replicas_per_node(self) -> int:
        """Replicas that fit on one node, given how many ranks each needs."""
        return max(1, self.gpus_per_node // max(self.ranks_per_replica, 1)) \
            if self.ranks_per_replica > 1 else self.n_replicas

    @property
    def nodes(self) -> int:
        """Nodes required. Single-rank replicas share GPUs and stay on one node; multi-rank
        replicas are sized so a replica never straddles a node boundary."""
        if self.ranks_per_replica <= 1:
            return 1
        per_node = self.replicas_per_node
        return max(1, -(-self.n_replicas // per_node))

    @property
    def ranks_per_node(self) -> int:
        return max(1, -(-self.total_ranks // self.nodes))

    @property
    def cpus_per_task(self) -> int:
        return cpus_per_rank(self.ranks_per_node, self.node_cpus)

    def describe(self) -> str:
        mode = ("replicas packed onto shared GPUs"
                if self.ranks_per_replica == 1 else
                f"{self.ranks_per_replica} ranks/replica (domain-decomposed)")
        return (f"{self.n_replicas} replicas, {self.total_ranks} ranks over {self.nodes} "
                f"node(s) x {self.gpus_per_node} GPUs -- {mode}; "
                f"{self.ranks_per_node} ranks/node x {self.cpus_per_task} cpus")


def submit_script(*, campaign_dir: Path, groups: Sequence[Sequence[int]], job_name: str,
                  hours: float = 2.0, n_gpus: int = 4, node_cpus: int = 288,
                  ranks_per_replica: int = 1,
                  account: str = "cameo", partition: str = "booster") -> str:
    """SLURM array over multidir GROUPS: one array task == one group.

    A group is one node when replicas share GPUs, and several nodes when each replica is
    domain-decomposed over multiple ranks. Grouping is what keeps a single crashed rank from
    taking the whole campaign with it -- `-multidir` aborts the entire invocation on one bad
    rank, unlike the per-case array layout it replaces.
    """
    campaign_dir = Path(campaign_dir).resolve()  # srun runs from an arbitrary cwd
    topo = Topology(n_replicas=max(len(g) for g in groups),
                    ranks_per_replica=ranks_per_replica,
                    gpus_per_node=n_gpus, node_cpus=node_cpus)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name[:14]}
#SBATCH --account={account}
#SBATCH --nodes={topo.nodes}
#SBATCH --ntasks-per-node={topo.ranks_per_node}
#SBATCH --cpus-per-task={topo.cpus_per_task}
#SBATCH --time={int(hours):02d}:{int(round((hours % 1) * 60)):02d}:00
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --array=0-{len(groups) - 1}
#SBATCH --output={campaign_dir}/%A_%a.out
#SBATCH --error={campaign_dir}/%A_%a.err
set -Eeuo pipefail
# {topo.describe()}
srun --ntasks-per-node=1 "{campaign_dir}/run_group_$(printf '%03d' "$SLURM_ARRAY_TASK_ID").sh"
"""
