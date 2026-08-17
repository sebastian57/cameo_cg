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
                          mps: bool = False, ranks_per_replica: int = 1,
                          index: str | None = None, plumed: bool = True) -> str:
    """Render the parent script that runs one multidir GROUP.

    `case_dirs` are relative to the campaign root and become both the grompp working
    directories and the `-multidir` arguments, so every output lands where `collect.py`
    already looks for it.
    """
    n = len(case_dirs)
    if n != len(structure_for):
        raise ValueError("case_dirs and structure_for must be the same length")
    dirs = " ".join(case_dirs)

    # `-n` is required by freeze-group campaigns: `freezegrps` names a custom index group,
    # and grompp cannot resolve it without the index file.
    ndx = f'-n {index} ' if index else ""
    # PARALLEL. grompp is ~1.5 s per case and was run serially, which for short frozen runs
    # cost more than the MD itself (48 cases: 72 s of grompp against 28 s of dynamics).
    # The node has hundreds of cores and each grompp is single-threaded and independent.
    grompp = "\n".join(
        [f'grompp_one() {{ ( cd "$1" && gmx grompp -f production.mdp -c "$2" '
         f'-p {topology} {ndx}-o biased.tpr -maxwarn 2 ) ; }}']
        + [f'grompp_one "{d}" "{s}" &' for d, s in zip(case_dirs, structure_for)]
        + ["wait"]
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
        # WITHOUT MPS, several processes sharing a GPU TIME-SLICE: each gets the whole device
        # in turn, with a context switch between. Measured on 8 replicas over 4 GPUs
        # (job 1342294, 2 processes per GPU): PME GPU mesh went 0.550 s -> 2.504 s (4.5x) and
        # a 1.27 s "Wait GPU NB local" stall appeared, while the CPU-side force time was
        # unchanged at 2.33 s. That is the entire reason packing past one replica per GPU
        # stopped paying. MPS lets the kernels run CONCURRENTLY instead.
        #
        # SLURM here exposes GresTypes=gpu only, so MPS is not scheduler-managed; on an
        # exclusively allocated node we run the daemon ourselves.
        mps_block = """
# ---- CUDA MPS: concurrent GPU sharing instead of time-slicing ----------------
export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY/log"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
if nvidia-cuda-mps-control -d; then
    # Confirm the daemon is actually answering before launching ranks at it: a dead daemon
    # silently degrades to time-slicing, which is exactly the state we are trying to leave,
    # and the run would look like a failed experiment rather than a failed daemon.
    if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "CUDA MPS active ($CUDA_MPS_PIPE_DIRECTORY)"
        trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT
    else
        echo "WARNING: MPS daemon started but not responding; running TIME-SLICED" >&2
    fi
else
    echo "WARNING: MPS daemon failed to start; running TIME-SLICED" >&2
fi
"""

    total_ranks = n * max(1, int(ranks_per_replica))
    # -gpu_id lists the GPUs available on the node. It must be sized so the ranks divide over
    # it: 2 ranks against `-gpu_id 0123` fails with a bare "task assignment failed" (measured,
    # job 1342173). List at most one GPU per rank; 8 ranks over 4 GPUs is fine (2 each).
    gpu_ids = "".join(str(i) for i in range(min(n_gpus, total_ranks)))
    # Only meaningful when several ranks share a node's GPUs (the packed regime). With one
    # replica domain-decomposed over several GPUs, let GROMACS assign them.
    gpu_flag = f" -gpu_id {gpu_ids}" if ranks_per_replica == 1 else ""

    # `-multidir` REQUIRES more than one simulation: with a single directory GROMACS aborts
    # with "The single simulation case is not supported" (measured, job 1342172). A group of
    # one is just an ordinary single-simulation run.
    # A freeze-group campaign holds the CG coordinates with `freezegrps` in the mdp and needs
    # NO bias at all, so PLUMED is dropped from the command line entirely.
    pl = "-plumed plumed.dat " if plumed else ""
    if n == 1:
        biased = (f'( cd "{case_dirs[0]}" && gmx mdrun -deffnm biased {pl}'
                  f'-ntmpi 1 -ntomp {ntomp} )')
    elif not plumed:
        # INDEPENDENT single-rank mdruns instead of one MPI `-multidir` launch. Measured on
        # 64 frozen replicas x 52 ps: 70 s vs 85 s, because `-multidir` pays MPI rank
        # initialisation on top of the CUDA context every replica needs anyway. It also
        # avoids the hard ceiling -- 256 ranks aborts with pscom/PMIx timeouts.
        # Startup is ~0.70 s per replica and does NOT amortise with group size, so it, not
        # the dynamics, sets the cost of a many-state campaign.
        ind_ntomp = max(1, min(int(ntomp), 256 // max(n, 1)))
        biased = "\n".join([
            "pids=()", "i=0",
            f'for d in {dirs}; do',
            f'    ( cd "$d" && CUDA_VISIBLE_DEVICES=$(( i % {n_gpus} )) gmx mdrun '
            f'-s biased.tpr -deffnm biased -ntmpi 1 -ntomp {ind_ntomp} -pin off ) &',
            "    pids+=($!); i=$(( i + 1 ))",
            "done",
            'fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done',
            '[ "$fail" -eq 0 ] || { echo "an mdrun failed" >&2; exit 1; }',
        ])
    else:
        biased = (f"srun -n {total_ranks} gmx_mpi mdrun -multidir {dirs} \\\n"
                  f"    {pl}-deffnm biased -ntomp {ntomp}{gpu_flag}")

    # PLUMED suffixes every output file with the replica index under -multidir: a
    # `PRINT ... FILE=colvar.dat` lands as `colvar.0.dat`, `colvar.1.dat`, ... (measured, job
    # 1342295). Downstream readers use the plain name -- `build_harvest_campaign.py:148` opens
    # `<case>/colvar.dat` literally and otherwise reports "no colvar.dat found", i.e. the
    # window check that justifies running a pilot at all silently reports nothing. Strip the
    # index here so the on-disk layout matches the per-case path exactly.
    # No-op when n == 1, where PLUMED does not suffix at all.
    normalise = "\n".join([
        "# ---- 2b. strip PLUMED's per-replica index from output filenames ---------------",
        "i=0",
        f"for d in {dirs}; do",
        '    for f in "$d"/*.$i.*; do',
        '        case "$(basename "$f")" in bck.*) continue ;; esac',
        '        [ -e "$f" ] && mv -f "$f" "${f%.$i.*}.${f##*.}"',
        "    done",
        '    [ -e "$d/HILLS.$i" ] && mv -f "$d/HILLS.$i" "$d/HILLS"',
        "    i=$(( i + 1 ))",
        "done",
    ]) if (n > 1 and plumed) else "# (no PLUMED, or single simulation: no per-replica filename suffixes)"

    # `-rerun` does NOT support multi-simulation either: "Multiple simulations not supported
    # by rerun" (GROMACS 2024.3 src/gromacs/mdrun/rerun.cpp:258; measured, jobs 1342184/5/6 --
    # the biased mdrun completed and reported Performance, then this phase aborted the job).
    # So the rerun is N independent single-simulation mdruns. They are run CONCURRENTLY,
    # round-robined over the node's GPUs, rather than serially: a rerun evaluates forces on
    # stored frames with no dynamics, so it is cheap, but N of them in series is not.
    rerun_ntomp = max(1, min(int(ntomp), 288 // max(n, 1)))
    if not plumed:
        # NO BIAS WAS APPLIED, so `biased.trr` already holds the physical forces and the
        # bias-free rerun is redundant. Verified on state_0000 of the freeze probe: forces
        # from the frozen run and from a fresh no-freeze rerun are IDENTICAL (585.037 both).
        # `freezegrps` suppresses the integration of those atoms, not the force computation.
        # Skipping it halves stage-3 storage and removes a whole mdrun phase.
        rerun = "\n".join([
            "# ---- 3. no rerun: no bias was applied, so biased.trr already has the forces ---",
            f'for d in {dirs}; do ln -sf biased.trr "$d/unbiased_forces.trr"; done',
        ])
    else:
      rerun = "\n".join([
          "pids=()", "i=0",
          f'for d in {dirs}; do',
          f'    ( cd "$d" && CUDA_VISIBLE_DEVICES=$(( i % {n_gpus} )) gmx mdrun -s biased.tpr '
          f'-rerun biased.trr -deffnm unbiased_forces -ntmpi 1 -ntomp {rerun_ntomp} ) &',
          "    pids+=($!); i=$(( i + 1 ))",
          "done",
          'fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done',
          '[ "$fail" -eq 0 ] || { echo "a bias-free rerun failed" >&2; exit 1; }',
      ])
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

# ---- 2. biased production ----------------------------------------------------
# gmx_mpi, not gmx: the default binary is thread-MPI and cannot do -multidir.
{biased}

{normalise}

# ---- 3. bias-free force rerun (MANDATORY, and NOT multidir) -------------------
# No -plumed here, so this phase needs no bias server at all.
{rerun}
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
                  ranks_per_replica: int = 1, groups_per_task: int = 1,
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
#SBATCH --array=0-{(len(groups) + groups_per_task - 1) // groups_per_task - 1}
#SBATCH --output={campaign_dir}/%A_%a.out
#SBATCH --error={campaign_dir}/%A_%a.err
set -Eeuo pipefail
# {topo.describe()}
# NOT `srun ... run_group_NNN.sh`. The group script runs its own
# `srun -n N gmx_mpi mdrun -multidir` internally, and an outer srun step holds the
# allocation, so the inner one can never be created:
#   srun: Job step creation temporarily disabled, retrying (Requested nodes are busy)
# repeated until walltime. That killed jobs 1324907-1324910 and 1324913 on 2026-08-12
# with zero output. The batch script already runs on the first allocated node, so the
# group script is invoked directly and its inner srun sees the whole allocation.
# Several GROUPS per array task: each task pays the SLURM prolog and module load once and
# then runs its groups back to back. With one group per task a 656-group campaign would
# spend hours in prolog alone.
NG={len(groups)}
PER={groups_per_task}
start=$(( SLURM_ARRAY_TASK_ID * PER ))
for k in $(seq "$start" $(( start + PER - 1 ))); do
    [ "$k" -ge "$NG" ] && break
    echo "=== group $k ==="
    bash "{campaign_dir}/run_group_$(printf '%03d' "$k").sh"
done
"""
