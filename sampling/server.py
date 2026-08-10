#!/usr/bin/env python3
"""Unix-socket bias server for PLUMED-driven CG enhanced sampling.

Loads whatever bias terms a run config declares, sums their forces, and answers
requests from the CGBias PLUMED action. One process per replica.

    python -m sampling.server --config run.yaml --socket /tmp/rep0.sock

Run config:

    mapping: ala2_backbone_cb_6        # optional; validates bead count
    report_every: 500                  # diagnostics cadence in MD steps
    biases:
      - type: mlcg_teacher
        training_config_path: .../exports/..._config.yaml
        params_path: .../exports/..._params.pkl
        dataset_path: .../ala2_cg_backbone_cb_6bead_200k.npz
        alpha: 1.0
        equilibrate_steps: 5000
        ramp_steps: 5000
      - type: tica_regional
        bias_npz: .../reference_bias.npz
        scale: 1.0

An empty or omitted `biases` list is legal and yields zero force -- useful as an
unbiased control that still exercises the whole plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sampling import protocol  # noqa: E402
from sampling.biases import build_biases, evaluate_all  # noqa: E402
from sampling.mapping import MAPPINGS  # noqa: E402


def read_exact(conn: socket.socket, size: int) -> bytes | None:
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--backlog", type=int, default=1)
    p.add_argument("--connect-timeout", type=float, default=600.0,
                   help="seconds to wait for GROMACS to connect (grompp + startup); "
                        "exceeded means the MD side never came up")
    p.add_argument("--io-timeout", type=float, default=300.0,
                   help="seconds to wait for the next request once running; exceeded "
                        "means GROMACS died or hung, so holding the allocation is waste")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text()) or {}

    n_beads = None
    mapping_name = cfg.get("mapping")
    if mapping_name:
        if mapping_name not in MAPPINGS:
            raise KeyError(f"unknown mapping {mapping_name!r}; have {sorted(MAPPINGS)}")
        n_beads = MAPPINGS[mapping_name].n_beads

    terms = build_biases(cfg.get("biases", []), n_beads=n_beads)
    report_every = int(cfg.get("report_every", 0))

    log_path = args.log or args.socket.with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", buffering=1)

    def emit(**kw):
        kw["t"] = round(time.time(), 3)
        log.write(json.dumps(kw) + "\n")

    emit(event="startup", socket=str(args.socket), config=str(args.config),
         protocol_version=protocol.PROTOCOL_VERSION, mapping=mapping_name,
         n_beads=n_beads, biases=[t.describe() for t in terms])
    if not terms:
        emit(event="warning", message="no bias terms declared; serving zero force")

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(args.backlog)
    server.settimeout(args.connect_timeout)
    emit(event="listening", connect_timeout_s=args.connect_timeout,
         io_timeout_s=args.io_timeout)

    n_calls = 0
    last_step = -1
    try:
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                emit(event="fatal", message="no client connected before timeout",
                     socket=str(args.socket), timeout_s=args.connect_timeout,
                     calls_served=n_calls)
                raise SystemExit(
                    f"bias server: GROMACS never connected to {args.socket} within "
                    f"{args.connect_timeout}s -- refusing to hold the allocation"
                )
            conn.settimeout(args.io_timeout)
            with conn:
                while True:
                    try:
                        head = read_exact(conn, protocol.header_size())
                    except socket.timeout:
                        emit(event="fatal", message="client stalled mid-run",
                             socket=str(args.socket), timeout_s=args.io_timeout,
                             last_step=last_step, calls_served=n_calls)
                        raise SystemExit(
                            f"bias server: no request on {args.socket} for "
                            f"{args.io_timeout}s after step {last_step} "
                            f"({n_calls} calls served) -- GROMACS appears dead"
                        )
                    if head is None:
                        break
                    _, _, _, n_atoms = protocol.peek_header(head)
                    rest = read_exact(
                        conn, protocol.request_struct(n_atoms).size - protocol.header_size()
                    )
                    if rest is None:
                        break
                    step, positions_nm = protocol.unpack_request(head + rest)
                    last_step = step

                    if n_beads is not None and n_atoms != n_beads:
                        emit(event="fatal", message="bead count mismatch",
                             plugin_sent=n_atoms, mapping_expects=n_beads)
                        raise SystemExit(
                            f"PLUMED sent {n_atoms} atoms, mapping expects {n_beads}"
                        )

                    positions_A = positions_nm / protocol.NM_PER_A
                    energy_kcal, forces_kcal, per_term = evaluate_all(terms, positions_A, step)

                    if not np.all(np.isfinite(forces_kcal)):
                        emit(event="fatal", step=step, message="non-finite bias force")
                        raise SystemExit("non-finite bias force; refusing to continue")

                    energy_kj = energy_kcal * protocol.KJ_PER_KCAL
                    forces_kj_nm = forces_kcal * protocol.KJ_PER_KCAL / protocol.NM_PER_A
                    conn.sendall(protocol.pack_response(step, energy_kj, forces_kj_nm))

                    n_calls += 1
                    if report_every and step % report_every == 0:
                        rec = {"event": "report", "step": step, "calls": n_calls,
                               "energy_kcal": round(energy_kcal, 6),
                               "max_force_kcal_A": round(float(np.abs(forces_kcal).max()), 6),
                               "per_term_kcal": {k: round(v, 6) for k, v in per_term.items()}}
                        for t in terms:
                            rec.update(t.diagnostics())
                        emit(**rec)
    except KeyboardInterrupt:
        emit(event="interrupted", calls=n_calls)
    finally:
        emit(event="shutdown", calls=n_calls)
        server.close()
        args.socket.unlink(missing_ok=True)
        log.close()


if __name__ == "__main__":
    main()
