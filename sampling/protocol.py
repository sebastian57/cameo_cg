"""Wire protocol between the PLUMED plugin and the bias server.

Version 2 (2026-08-01). Clean break from v1, which hardcoded five atoms into
fixed-size 15-double payloads on both sides. v2 carries `n_atoms` in the header and
sizes the payload from it, so the same plugin and server work for any CG mapping.

Layout (native byte order, as both ends run on the same node):

    request   : magic(Q) version(Q) step(q) n_atoms(q)  positions_nm[3*n_atoms](d)
    response  : magic(Q) version(Q) step(q) n_atoms(q)  energy_kj(d) forces_kj_nm[3*n_atoms](d)

Units on the wire are PLUMED's: nm and kJ/mol. Bias implementations work in A and
kcal/mol; conversion happens here, in one place.
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np

__all__ = [
    "PROTOCOL_VERSION",
    "REQUEST_MAGIC",
    "RESPONSE_MAGIC",
    "NM_PER_A",
    "KJ_PER_KCAL",
    "request_struct",
    "response_struct",
    "pack_request",
    "unpack_request",
    "pack_response",
    "unpack_response",
]

PROTOCOL_VERSION = 2
REQUEST_MAGIC = 0x4347425245513200      # "CGBREQ2\0"
RESPONSE_MAGIC = 0x4347425245533200     # "CGBRES2\0"

NM_PER_A = 0.1
KJ_PER_KCAL = 4.184

_HEADER = "@QQqq"
_HEADER_SIZE = struct.calcsize(_HEADER)


def request_struct(n_atoms: int) -> struct.Struct:
    return struct.Struct(f"@QQqq{3 * int(n_atoms)}d")


def response_struct(n_atoms: int) -> struct.Struct:
    return struct.Struct(f"@QQqqd{3 * int(n_atoms)}d")


def header_size() -> int:
    return _HEADER_SIZE


def peek_header(payload: bytes) -> Tuple[int, int, int, int]:
    """Read (magic, version, step, n_atoms) so the reader can size the rest."""
    if len(payload) < _HEADER_SIZE:
        raise ValueError(f"short header: {len(payload)} < {_HEADER_SIZE}")
    return struct.unpack_from(_HEADER, payload, 0)


def pack_request(step: int, positions_nm: np.ndarray) -> bytes:
    p = np.asarray(positions_nm, dtype=np.float64).reshape(-1, 3)
    s = request_struct(len(p))
    return s.pack(REQUEST_MAGIC, PROTOCOL_VERSION, int(step), len(p), *p.ravel())


def unpack_request(payload: bytes) -> Tuple[int, np.ndarray]:
    magic, version, step, n_atoms = peek_header(payload)
    if magic != REQUEST_MAGIC:
        raise ValueError(f"bad request magic 0x{magic:016x}")
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"protocol version mismatch: plugin sent v{version}, server speaks "
            f"v{PROTOCOL_VERSION}. Rebuild the plugin."
        )
    values = request_struct(n_atoms).unpack(payload)
    positions_nm = np.asarray(values[4:], dtype=np.float64).reshape(n_atoms, 3)
    return int(step), positions_nm


def pack_response(step: int, energy_kj: float, forces_kj_nm: np.ndarray) -> bytes:
    f = np.asarray(forces_kj_nm, dtype=np.float64).reshape(-1, 3)
    s = response_struct(len(f))
    return s.pack(
        RESPONSE_MAGIC, PROTOCOL_VERSION, int(step), len(f), float(energy_kj), *f.ravel()
    )


def unpack_response(payload: bytes) -> Tuple[int, float, np.ndarray]:
    magic, version, step, n_atoms = peek_header(payload)
    if magic != RESPONSE_MAGIC:
        raise ValueError(f"bad response magic 0x{magic:016x}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"protocol version mismatch: server v{version}")
    values = response_struct(n_atoms).unpack(payload)
    energy_kj = float(values[4])
    forces = np.asarray(values[5:], dtype=np.float64).reshape(n_atoms, 3)
    return int(step), energy_kj, forces
