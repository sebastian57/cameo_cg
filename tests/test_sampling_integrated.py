"""Regression tests for the integrated enhanced-sampling package (CR-15).

Before this file the `sampling/` package had no repository tests: the TICA
projection, protocol v2, bias registry, and server were validated only by one-off
scripts outside the repo, so nothing caught a regression. The earlier "10/10 TICA
validation" referred to the external pre-integration implementation.

Covers, per the code review's minimum list:
  1. TICA force vs finite differences
  2. symmetry behaviour + zero net force/torque
  3. protocol round trip and corrupt magic/version/count rejection
  4. bias composition and disabled-term behaviour
  5. artifact schema, non-default lambda, omitted-feature-bead
  6. a real Unix-socket client/server integration test
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling import protocol                                    # noqa: E402
from sampling.biases import BIAS_REGISTRY, build_biases, evaluate_all, register_bias  # noqa: E402
from sampling.biases.base import BiasTerm                        # noqa: E402
from sampling.biases.local_inversion import LocalInversionUmbrella  # noqa: E402
from sampling.biases.tica_regional import TICAProjection         # noqa: E402
from sampling.mapping import (                                   # noqa: E402
    dihedral_deg, get_mapping, normalized_signed_volume, wrap_deg,
)

MAPPING = "ala2_backbone_cb_6"


@pytest.fixture(scope="module")
def structure() -> np.ndarray:
    """A physically-shaped ala2 bb6 frame (Angstrom)."""
    rng = np.random.default_rng(20260804)
    base = np.array([
        [-2.35, 1.05, 0.10], [-1.10, 0.55, -0.20], [0.05, 1.45, 0.05],
        [0.30, 2.35, -1.10], [1.35, 0.75, 0.55], [2.45, 1.30, 0.15],
    ])
    return base + 0.02 * rng.standard_normal(base.shape)


# ---------------------------------------------------------------- 1 + 2
def _fd_gradient(fn, R, h=1e-6):
    g = np.zeros_like(R)
    for i in range(R.shape[0]):
        for d in range(3):
            Rp, Rm = R.copy(), R.copy()
            Rp[i, d] += h
            Rm[i, d] -= h
            g[i, d] = (fn(Rp) - fn(Rm)) / (2 * h)
    return g


def test_inversion_force_matches_finite_differences(structure):
    bias = LocalInversionUmbrella(mapping=MAPPING, chi_target=-0.2, force_constant=750.0)
    energy, forces = bias.evaluate(structure, step=10**9)
    fd = -_fd_gradient(lambda R: bias.evaluate(R, step=10**9)[0], structure)
    assert np.abs(forces - fd).max() < 1e-5
    assert np.isfinite(energy)


def test_inversion_force_has_zero_net_force_and_torque(structure):
    bias = LocalInversionUmbrella(mapping=MAPPING, chi_target=0.0, force_constant=500.0)
    _, F = bias.evaluate(structure, step=10**9)
    assert np.abs(F.sum(axis=0)).max() < 1e-8
    torque = np.cross(structure - structure.mean(axis=0), F).sum(axis=0)
    assert np.abs(torque).max() < 1e-7


def test_chi_is_parity_odd_and_rotation_invariant(structure):
    """chi must flip sign under reflection and be invariant under rotation.

    This is the whole reason the bias exists: pair-distance TICA is reflection
    invariant, so it cannot supply a direction that distinguishes enantiomers.
    """
    c, nb = 2, (1, 3, 4)
    chi = normalized_signed_volume(structure[None], c, nb)[0]

    mirrored = structure * np.array([1.0, 1.0, -1.0])
    assert normalized_signed_volume(mirrored[None], c, nb)[0] == pytest.approx(-chi, abs=1e-12)

    theta = 0.7
    Q = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    rotated = structure @ Q.T + np.array([3.0, -1.0, 2.0])
    assert normalized_signed_volume(rotated[None], c, nb)[0] == pytest.approx(chi, abs=1e-12)


def test_chi_is_scale_invariant_in_each_branch(structure):
    """No radial component: stretching a branch must not change chi.

    The unnormalised signed volume fails this, and a Cartesian bias built on one
    stretched ala2 CA-bonds 3-5% instead of driving the transition (2026-08-03).
    """
    c, nb = 2, (1, 3, 4)
    chi = normalized_signed_volume(structure[None], c, nb)[0]
    stretched = structure.copy()
    stretched[nb[0]] = structure[c] + 1.9 * (structure[nb[0]] - structure[c])
    assert normalized_signed_volume(stretched[None], c, nb)[0] == pytest.approx(chi, abs=1e-12)


def test_tica_force_matches_finite_differences():
    rng = np.random.default_rng(7)
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [3, 5]])
    proj = TICAProjection(pairs, rng.normal(size=len(pairs)),
                          rng.normal(size=(len(pairs), 2)), declared_n_beads=6)
    R = rng.normal(scale=2.0, size=(6, 3))
    z, jac = proj.value_and_jacobian(R)
    for k in range(len(z)):
        fd = _fd_gradient(lambda X, k=k: proj.transform(X)[k], R)
        assert np.abs(jac[k] - fd).max() < 1e-6


# ---------------------------------------------------------------- 3
def test_protocol_round_trip():
    rng = np.random.default_rng(3)
    for n in (1, 5, 6, 23):
        pos = rng.normal(size=(n, 3))
        blob = protocol.pack_request(step=7, positions_nm=pos)
        step, back = protocol.unpack_request(blob)
        assert step == 7
        assert np.array_equal(back, pos)

        f = rng.normal(size=(n, 3))
        rblob = protocol.pack_response(step=7, energy_kj=1.25, forces_kj_nm=f)
        rstep, energy, rf = protocol.unpack_response(rblob)
        assert rstep == 7
        assert energy == pytest.approx(1.25)
        assert np.array_equal(rf, f)


def test_protocol_rejects_corrupt_header():
    pos = np.zeros((6, 3))
    good = bytearray(protocol.pack_request(step=1, positions_nm=pos))

    bad_magic = bytearray(good)
    bad_magic[0:8] = (0xDEADBEEF).to_bytes(8, sys.byteorder)
    with pytest.raises(Exception):
        protocol.unpack_request(bytes(bad_magic))

    bad_version = bytearray(good)
    bad_version[8:16] = (protocol.PROTOCOL_VERSION + 99).to_bytes(8, sys.byteorder)
    with pytest.raises(Exception):
        protocol.unpack_request(bytes(bad_version))

    with pytest.raises(Exception):
        protocol.unpack_request(bytes(good[:-8]))      # truncated payload


def test_peek_header_reports_atom_count():
    pos = np.zeros((6, 3))
    blob = protocol.pack_request(step=11, positions_nm=pos)
    *_, n_atoms = protocol.peek_header(blob[:protocol.header_size()])
    assert n_atoms == 6


# ---------------------------------------------------------------- 4
@register_bias("_test_constant")
class _ConstantBias(BiasTerm):
    def __init__(self, value=1.0, name="_test_constant", enabled=True):
        super().__init__(name=name, enabled=enabled)
        self.value = float(value)

    def evaluate(self, positions_A, step):
        f = np.zeros_like(positions_A)
        f[0, 0] = self.value
        return self.value, f


def test_bias_composition_is_additive(structure):
    terms = build_biases([{"type": "_test_constant", "value": 2.0},
                          {"type": "_test_constant", "value": 5.0, "name": "second"}])
    energy, forces, per_term = evaluate_all(terms, structure, step=0)
    assert energy == pytest.approx(7.0)
    assert forces[0, 0] == pytest.approx(7.0)
    assert len(per_term) == 2


def test_disabled_term_contributes_nothing(structure):
    terms = build_biases([{"type": "_test_constant", "value": 2.0},
                          {"type": "_test_constant", "value": 5.0, "name": "off",
                           "enabled": False}])
    energy, forces, _ = evaluate_all(terms, structure, step=0)
    assert energy == pytest.approx(2.0)
    assert forces[0, 0] == pytest.approx(2.0)


def test_empty_bias_list_yields_zero_force(structure):
    """A legal unbiased control that still exercises the whole plumbing."""
    energy, forces, per_term = evaluate_all(build_biases([]), structure, step=0)
    assert energy == pytest.approx(0.0)
    assert np.abs(forces).max() == 0.0
    assert per_term == {} or len(per_term) == 0


def test_registry_contains_expected_terms():
    for name in ("mlcg_teacher", "tica_regional", "local_inversion_umbrella"):
        assert name in BIAS_REGISTRY


# ---------------------------------------------------------------- 5
def test_tica_projection_validates_schema():
    pairs = np.array([[0, 1], [1, 2]])
    mean = np.zeros(2)
    coef = np.zeros((2, 2))

    TICAProjection(pairs, mean, coef, declared_n_beads=3).validate()

    with pytest.raises(ValueError):      # declared count too small for the pairs
        TICAProjection(pairs, mean, coef, declared_n_beads=2).validate()
    with pytest.raises(ValueError):      # mean length mismatch
        TICAProjection(pairs, np.zeros(3), coef, declared_n_beads=3).validate()
    with pytest.raises(ValueError):      # coefficient row mismatch
        TICAProjection(pairs, mean, np.zeros((3, 2)), declared_n_beads=3).validate()


def test_declared_n_beads_beats_inference_when_last_bead_has_no_feature():
    """CR-12: max(pairs)+1 undercounts if the final bead carries no pair."""
    pairs = np.array([[0, 1], [1, 2]])          # bead 3 appears in no pair
    proj = TICAProjection(pairs, np.zeros(2), np.zeros((2, 2)), declared_n_beads=4)
    assert proj.n_beads == 4
    inferred = TICAProjection(pairs, np.zeros(2), np.zeros((2, 2)))
    assert inferred.n_beads == 3                # the wrong answer, retained as fallback


# ---------------------------------------------------------------- mapping
def test_inversion_centers_are_degree_three_only():
    m = get_mapping(MAPPING)
    assert m.inversion_centers() == {2: (1, 3, 4)}


def test_mapping_bond_graph_matches_topology():
    m = get_mapping(MAPPING)
    assert set(m.bonds) == {(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)}
    assert m.neighbors(2) == (1, 3, 4)


def test_screen_identifies_rigid_stereocentre(structure):
    m = get_mapping(MAPPING)
    R = np.repeat(structure[None], 50, axis=0)
    info = m.screen_inversion_centers(R)[2]
    assert info["is_candidate"] and info["sign_purity"] == pytest.approx(1.0)


def test_inversion_rejects_invalid_configurations():
    bad = [
        dict(center=2, neighbors=[1, 1, 3]),        # duplicates
        dict(center=2, neighbors=[1, 3, 99]),       # out of range
        dict(center=2, neighbors=[1, 3, 5]),        # bead 5 not bonded to CA
        dict(force_constant=-1.0),
        dict(chi_target=2.0),
        dict(ramp_steps=-1),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            LocalInversionUmbrella(mapping=MAPPING, **{"chi_target": -0.3, **kw})


def test_inversion_ramp_reaches_target_and_then_holds(structure):
    b = LocalInversionUmbrella(mapping=MAPPING, chi_target=0.0, force_constant=100.0,
                               equilibrate_steps=100, ramp_steps=100)
    b.evaluate(structure, step=0)
    start = b.last_target
    b.evaluate(structure, step=50)
    assert b.last_target == pytest.approx(start)          # held during equilibration
    b.evaluate(structure, step=150)
    assert start > b.last_target > 0.0 or start < b.last_target < 0.0   # mid-ramp
    b.evaluate(structure, step=100000)
    assert b.last_target == pytest.approx(0.0)            # fixed window afterwards


# ---------------------------------------------------------------- 6
def test_server_socket_integration(tmp_path):
    """Start the real server over a real Unix socket and exchange one frame."""
    cfg = tmp_path / "server.yaml"
    cfg.write_text(
        "mapping: %s\nreport_every: 0\nbiases:\n"
        "  - type: local_inversion_umbrella\n"
        "    mapping: %s\n    chi_target: -0.3\n    force_constant: 500.0\n"
        % (MAPPING, MAPPING)
    )
    sock_path = tmp_path / "t.sock"
    proc = subprocess.Popen(
        [sys.executable, "-m", "sampling.server", "--config", str(cfg),
         "--socket", str(sock_path), "--log", str(tmp_path / "s.log"),
         "--connect-timeout", "60", "--io-timeout", "30"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(600):
            if sock_path.exists():
                break
            if proc.poll() is not None:
                pytest.fail(f"server exited early: {proc.stderr.read().decode()[-2000:]}")
            time.sleep(0.1)
        assert sock_path.exists(), "server never created its socket"

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(30)
        client.connect(str(sock_path))

        pos_A = np.array([
            [-2.35, 1.05, 0.10], [-1.10, 0.55, -0.20], [0.05, 1.45, 0.05],
            [0.30, 2.35, -1.10], [1.35, 0.75, 0.55], [2.45, 1.30, 0.15],
        ])
        client.sendall(protocol.pack_request(step=3, positions_nm=pos_A * protocol.NM_PER_A))

        size = protocol.response_struct(6).size
        buf = b""
        while len(buf) < size:
            chunk = client.recv(size - len(buf))
            assert chunk, "server closed the connection"
            buf += chunk
        step, energy_kj, forces_kj_nm = protocol.unpack_response(buf)

        assert step == 3
        assert np.isfinite(energy_kj) and energy_kj > 0.0
        forces_kcal_A = forces_kj_nm / protocol.KJ_PER_KCAL * protocol.NM_PER_A
        assert np.abs(forces_kcal_A.sum(axis=0)).max() < 1e-6   # still Newton's third law

        expected, _ = LocalInversionUmbrella(
            mapping=MAPPING, chi_target=-0.3, force_constant=500.0
        ).evaluate(pos_A, step=3)
        assert energy_kj / protocol.KJ_PER_KCAL == pytest.approx(expected, rel=1e-9)

        client.close()
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    events = [json.loads(l) for l in (tmp_path / "s.log").read_text().splitlines() if l.strip()]
    startup = [e for e in events if e.get("event") == "startup"]
    assert startup and startup[0]["n_beads"] == 6
    listening = [e for e in events if e.get("event") == "listening"]
    assert listening and listening[0]["io_timeout_s"] == 30      # CR-14 wired through


def test_server_connect_timeout_is_fatal(tmp_path):
    """CR-14: no client => exit, rather than holding the allocation to walltime."""
    cfg = tmp_path / "server.yaml"
    cfg.write_text("mapping: %s\nreport_every: 0\nbiases: []\n" % MAPPING)
    proc = subprocess.Popen(
        [sys.executable, "-m", "sampling.server", "--config", str(cfg),
         "--socket", str(tmp_path / "n.sock"), "--log", str(tmp_path / "n.log"),
         "--connect-timeout", "2", "--io-timeout", "2"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        rc = proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("server hung despite --connect-timeout")
    assert rc != 0
    events = [json.loads(l) for l in (tmp_path / "n.log").read_text().splitlines() if l.strip()]
    assert any(e.get("event") == "fatal" for e in events)


# ---------------------------------------------------------------- tica_metad
TICA_ARTIFACT = ("/e/project1/cameo/schmidt36/SAMPLING/tica_regional_weighting/results/"
                 "ala2_bb6_reference/smooth_reference_bias_lambda0p25.npz")
_have_tica = Path(TICA_ARTIFACT).exists()
metad_only = pytest.mark.skipif(not _have_tica, reason="TICA artifact not available")


def _metad(**kw):
    from sampling.biases.tica_metad import TICAWellTemperedMetaD
    base = dict(bias_npz=TICA_ARTIFACT, height=0.15, sigma=(0.15, 0.09), pace=500,
                bias_factor=8.0, temperature=298.0)
    base.update(kw)
    return TICAWellTemperedMetaD(**base)


@metad_only
def test_metad_force_matches_finite_differences(structure):
    b = _metad()
    rng = np.random.default_rng(5)
    for s in (0, 500, 1000, 1500):
        b.evaluate(structure + 0.01 * rng.standard_normal(structure.shape), s)
    _, F = b.evaluate(structure, 10**7)
    fd = -_fd_gradient(lambda R: b.evaluate(R, 10**7)[0], structure)
    assert np.abs(F - fd).max() < 1e-6


@metad_only
def test_metad_zero_net_force_and_torque(structure):
    b = _metad()
    for s in (0, 500, 1000):
        b.evaluate(structure, s)
    _, F = b.evaluate(structure, 10**7)
    assert np.abs(F.sum(axis=0)).max() < 1e-9
    assert np.abs(np.cross(structure - structure.mean(axis=0), F).sum(axis=0)).max() < 1e-9


@metad_only
def test_metad_is_well_tempered(structure):
    """Hill heights must DECAY where bias accumulates, else the bias grows unbounded."""
    b = _metad()
    for s in range(0, 5000, 500):
        b.evaluate(structure, s)
    h = b._heights
    assert len(h) >= 8
    assert h[0] == pytest.approx(0.15)
    assert np.all(np.diff(h) < 0), "heights must decrease when revisiting the same z"


@metad_only
def test_metad_deposits_on_pace_despite_recompute_stride(structure):
    """`step` advances by RECOMPUTE_STRIDE; deposition must not be skipped.

    Testing `step % pace == 0` would deposit nothing whenever pace is not a multiple
    of the stride -- a silently inert bias.
    """
    b = _metad(pace=300)
    for s in range(0, 3000, 7):        # stride 7 never hits a multiple of 300
        b.evaluate(structure, s)
    assert len(b._heights) == pytest.approx(3000 / 300, abs=1)


@metad_only
def test_metad_equilibration_delays_first_hill(structure):
    b = _metad(equilibrate_steps=1000)
    for s in range(0, 900, 100):
        b.evaluate(structure, s)
    assert len(b._heights) == 0
    b.evaluate(structure, 1000)
    assert len(b._heights) == 1


@metad_only
def test_metad_hills_round_trip(tmp_path, structure):
    p = tmp_path / "hills.npz"
    b = _metad(hills_path=str(p))
    for s in range(0, 2000, 500):
        b.evaluate(structure, s)
    b.save_hills()
    b2 = _metad(hills_path=str(p))
    assert np.allclose(b2._centers, b._centers)
    assert np.allclose(b2._heights, b._heights)
    assert b2.evaluate(structure, 10**7)[0] == pytest.approx(b.evaluate(structure, 10**7)[0])


@metad_only
def test_metad_rejects_invalid_config():
    for kw in (dict(height=0.0), dict(pace=0), dict(bias_factor=1.0),
               dict(sigma=(0.1,)), dict(sigma=(0.1, -0.1)), dict(equilibrate_steps=-1)):
        with pytest.raises(ValueError):
            _metad(**kw)
