"""CG mapping registry: bead identity, collective variables, and AA atom indices.

SCOPE -- this is an ATOM-SELECTION mapping, not a general CG projection
----------------------------------------------------------------------
Each bead is exactly one retained AA atom (`aa_atom_indices_1based`). The PLUMED
plugin sends those atoms' positions and applies the returned bias force directly
back to the same atoms, which is correct *only* because the coordinate map is the
identity on a selected subset.

COM / weighted / many-to-one maps are NOT supported. Those need an explicit
`z = M x` with the bias force back-projected as `F_x = M^T F_z` (or the Jacobian
transpose for a nonlinear map); nothing here does that, and applying a bead force
to a single representative atom of a COM bead would be silently wrong.

Note this is a separate question from the FORCE map used to build training labels:
reference CG *forces* come from an aggforce-fitted weighted map over all AA atoms
(see KB DESIGN/CG_FORCE_MAPPING.md). Identity coordinate selection and weighted
force mapping coexist deliberately -- do not assume one implies the other.

Single source of truth for "which bead is which atom" and "how is phi/psi defined".
Every analysis, bias and PLUMED input generator should read from here rather than
hardcoding index tuples, because getting this wrong is silent: a dihedral over the
wrong quadruple still produces a plausible-looking angle.

Cautionary example: the 5-bead ala2 mapping retains only the ALA heavy atoms
(N, CA, CB, C, O). Its bead quadruple (0,1,2,3) is therefore N-CA-CB-C, the Calpha
chirality improper -- NOT the Ramachandran phi. Analyses in this project labelled it
"phi" for months. See KB DESIGN/ALLEGRO_PARITY_INVARIANCE.md.

Angle convention
----------------
`dihedral_deg` returns the IUPAC-style torsion of a bead quadruple. CV definitions
below carry `shift_deg`; applying it reproduces the convention used throughout this
project (and gives the standard Ramachandran for the 6-bead mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "CollectiveVariable",
    "CGMapping",
    "MAPPINGS",
    "get_mapping",
    "dihedral_deg",
    "wrap_deg",
]


def wrap_deg(x):
    """Wrap angles to (-180, 180]."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def dihedral_deg(R: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Torsion angle in degrees for a bead quadruple.

    R: (..., n_beads, 3). indices: four bead indices.
    """
    i0, i1, i2, i3 = (int(i) for i in indices)
    p0, p1, p2, p3 = R[..., i0, :], R[..., i1, :], R[..., i2, :], R[..., i3, :]
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    u = b1 / np.linalg.norm(b1, axis=-1)[..., None]
    v = b0 - np.sum(b0 * u, axis=-1)[..., None] * u
    w = b2 - np.sum(b2 * u, axis=-1)[..., None] * u
    return np.degrees(
        np.arctan2(np.sum(np.cross(u, v) * w, axis=-1), np.sum(v * w, axis=-1))
    )


def signed_volume(R: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Parity-ODD scalar for a bead quadruple; sign flips under reflection."""
    i0, i1, i2, i3 = (int(i) for i in indices)
    a = R[..., i1, :] - R[..., i0, :]
    b = R[..., i2, :] - R[..., i1, :]
    c = R[..., i3, :] - R[..., i2, :]
    return np.sum(np.cross(a, b) * c, axis=-1)


@dataclass(frozen=True)
class CollectiveVariable:
    """A dihedral CV defined on bead indices.

    `atom_indices_1based` is the equivalent AA selection for PLUMED TORSION lines.
    """

    name: str
    bead_indices: Tuple[int, int, int, int]
    shift_deg: float = 180.0
    description: str = ""

    def evaluate(self, R: np.ndarray) -> np.ndarray:
        return wrap_deg(dihedral_deg(R, self.bead_indices) + self.shift_deg)

    def atom_indices_1based(self, mapping: "CGMapping") -> Tuple[int, ...]:
        return tuple(mapping.aa_atom_indices_1based[i] for i in self.bead_indices)


def normalized_signed_volume(R: np.ndarray, center: int,
                             neighbors: Sequence[int]) -> np.ndarray:
    """Parity-odd local inversion coordinate chi in [-1, 1].

    chi = [(z_i - z_c) x (z_j - z_c)] . (z_k - z_c) / (|..| |..| |..|)

    Normalising by the three branch lengths makes chi **scale-invariant in each
    branch**, so d(chi)/d|z_n - z_c| = 0 and a bias built on it exerts purely
    angular force. That matters: the unnormalised signed volume has a radial
    component, and a Cartesian bias with one stretched ala2 CA-bonds 3-5% rather
    than driving the transition it was aimed at.

    chi ~ 0 is the locally planar inversion region; the sign distinguishes the two
    handednesses.
    """
    i, j, k = (int(n) for n in neighbors)
    a = R[..., i, :] - R[..., center, :]
    b = R[..., j, :] - R[..., center, :]
    c = R[..., k, :] - R[..., center, :]
    num = np.sum(np.cross(a, b) * c, axis=-1)
    den = (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
           * np.linalg.norm(c, axis=-1))
    return num / den


@dataclass(frozen=True)
class CGMapping:
    """A coarse-grained mapping and everything derived from it."""

    name: str
    bead_labels: Tuple[str, ...]
    aa_atom_indices_1based: Tuple[int, ...]   # GROMACS numbering
    masses_amu: Tuple[float, ...]
    cvs: Dict[str, CollectiveVariable] = field(default_factory=dict)
    chirality_cv: str | None = None
    # CG bond graph as bead-index pairs. Projected from the AA topology (two beads
    # are bonded when an AA covalent bond crosses between their atom groups) --
    # never guessed from distances. Needed to enumerate inversion centers.
    bonds: Tuple[Tuple[int, int], ...] = ()
    notes: str = ""

    def neighbors(self, bead: int) -> Tuple[int, ...]:
        out = [b for a, b in self.bonds if a == bead] + [a for a, b in self.bonds if b == bead]
        return tuple(sorted(out))

    def inversion_centers(self) -> Dict[int, Tuple[int, ...]]:
        """Beads with EXACTLY three bonded branches -- unambiguous chi candidates.

        Degree-4+ beads are deliberately excluded rather than truncated to their
        first three neighbours: which triplet you pick changes chi's meaning, so the
        choice would silently depend on bead numbering. Declare `neighbors` explicitly
        for those.

        **The graph alone does not identify a stereocentre.** A degree-3 bead may be a
        planar sp2 junction, where chi ~ 0 and biasing it is meaningless. This returns
        *topological candidates only*; screen them against reference data with
        `screen_inversion_centers()` before trusting one.
        """
        return {b: self.neighbors(b) for b in range(self.n_beads)
                if len(self.neighbors(b)) == 3}

    def screen_inversion_centers(self, R_ref: np.ndarray,
                                 min_abs_chi: float = 0.3,
                                 min_sign_purity: float = 0.99) -> Dict[int, Dict]:
        """Rank topological candidates by their behaviour in reference data.

        Topology says where a chi *can* be defined; only the data says whether it is a
        genuine stereocentre. A real one keeps |chi| well away from zero and never
        changes sign; a planar junction sits near zero, and a labile centre flips.

        Returns {center: {neighbors, mean_abs_chi, sign_purity, is_candidate, reason}}.
        A centre that flips sign is *labile* and must NOT be given a blocking bias --
        including both rigid and labile examples is what makes training transferable.
        """
        out: Dict[int, Dict] = {}
        for c, nb in self.inversion_centers().items():
            chi = normalized_signed_volume(R_ref, c, nb)
            mean_abs = float(np.abs(chi).mean())
            purity = float(max((chi > 0).mean(), (chi < 0).mean()))
            ok = mean_abs >= min_abs_chi and purity >= min_sign_purity
            if mean_abs < min_abs_chi:
                reason = f"near-planar (mean |chi| {mean_abs:.3f} < {min_abs_chi})"
            elif purity < min_sign_purity:
                reason = f"labile: changes sign ({purity:.3f} < {min_sign_purity})"
            else:
                reason = "rigid stereocentre"
            out[c] = {"neighbors": nb, "mean_abs_chi": mean_abs,
                      "sign_purity": purity, "is_candidate": ok, "reason": reason}
        return out

    @property
    def n_beads(self) -> int:
        return len(self.bead_labels)

    def validate(self) -> None:
        n = self.n_beads
        if not (len(self.aa_atom_indices_1based) == len(self.masses_amu) == n):
            raise ValueError(f"{self.name}: bead/atom/mass length mismatch")
        for cv in self.cvs.values():
            if max(cv.bead_indices) >= n:
                raise ValueError(
                    f"{self.name}: CV {cv.name!r} references bead "
                    f"{max(cv.bead_indices)} but mapping has {n} beads"
                )

    def plumed_atom_selection(self) -> str:
        """Comma-separated 1-based AA atom list for PLUMED ATOMS=."""
        return ",".join(str(i) for i in self.aa_atom_indices_1based)

    def describe(self) -> str:
        rows = [
            "  bead %d: %-6s  AA atom %3d  %.3f amu"
            % (i, lab, idx, m)
            for i, (lab, idx, m) in enumerate(
                zip(self.bead_labels, self.aa_atom_indices_1based, self.masses_amu)
            )
        ]
        cvs = [
            "  %-10s beads %s (+%.0f deg)  %s"
            % (c.name, c.bead_indices, c.shift_deg, c.description)
            for c in self.cvs.values()
        ]
        return "\n".join([f"{self.name} ({self.n_beads} beads)"] + rows + ["CVs:"] + cvs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ala2 = ACE-ALA-NME, GROMACS topology
# /p/project1/cameo/edelkoetter2/ala2/constrained/topol.top
#   5 C(ACE)  7 N(ALA)  9 CA(ALA)  11 CB(ALA)  15 C(ALA)  16 O(ALA)  17 N(NME)

_ALA2_BB6 = CGMapping(
    name="ala2_backbone_cb_6",
    bead_labels=("C_ace", "N", "CA", "CB", "C", "N_nme"),
    aa_atom_indices_1based=(5, 7, 9, 11, 15, 17),
    masses_amu=(12.011, 14.007, 12.011, 12.011, 12.011, 14.007),
    cvs={
        "phi": CollectiveVariable(
            "phi", (0, 1, 2, 4), 180.0, "C(ACE)-N-CA-C : standard Ramachandran phi"
        ),
        "psi": CollectiveVariable(
            "psi", (1, 2, 4, 5), 180.0, "N-CA-C-N(NME) : standard Ramachandran psi"
        ),
        "chirality": CollectiveVariable(
            # shift 180 like phi/psi so evaluate() matches PLUMED TORSION output and all
            # three CVs share one convention. With shift 0 the reference sat at -54.5 deg
            # and biased runs at +123, which looks like a chirality flip and is not.
            "chirality", (1, 2, 3, 4), 180.0, "N-CA-CB-C improper : Calpha stereocenter"
        ),
    },
    chirality_cv="chirality",
    # Projected from amber99sb-ildn topol.top [bonds]: AA bonds 5-7, 7-9, 9-11, 9-15,
    # 15-17 are the only ones whose endpoints are both beads. CA (bead 2) is the sole
    # degree-3 bead -> the unique inversion center.
    bonds=((0, 1), (1, 2), (2, 3), (2, 4), (4, 5)),
    notes=(
        "Verified against topol.top and frame-0 bond geometry: C_ace-N 1.367, N-CA 1.497, "
        "CA-CB 1.481, CA-C 1.550, C-N_nme 1.368, CB..C 2.509 A. With the +180 convention "
        "the reference gives 96.8% phi<0 and basins at (-68,+158) C7eq and (-142,+158) "
        "beta/PPII, matching Chen et al. JCTC 2026 (phi = C-N-CA-C, psi = N-CA-C-N). "
        "Chirality improper median +125.5 deg in this convention (range 102.8-151.3), "
        "never approaching the -55 deg mirror value."
    ),
)

_ALA2_5 = CGMapping(
    name="ala2_ala_heavy_5",
    bead_labels=("N", "CA", "CB", "C", "O"),
    aa_atom_indices_1based=(7, 9, 11, 15, 16),
    masses_amu=(14.007, 12.011, 12.011, 12.011, 15.999),
    cvs={
        "chirality": CollectiveVariable(
            "chirality", (0, 1, 2, 3), 180.0,
            "N-CA-CB-C improper. HISTORICALLY MISLABELLED 'phi' in this project.",
        ),
        "psi_proxy": CollectiveVariable(
            "psi_proxy", (1, 2, 3, 4), 180.0,
            "CA-CB-C-O. Tracks backbone psi up to an offset; NOT standard psi.",
        ),
    },
    chirality_cv="chirality",
    # Projected from topol.top [bonds]: AA 7-9 (N-CA), 9-11 (CA-CB), 9-15 (CA-C),
    # 15-16 (C-O). CA (bead 1) is the sole degree-3 bead, i.e. the stereocentre.
    bonds=((0, 1), (1, 2), (1, 3), (3, 4)),
    notes=(
        "Legacy mapping for all ala2 work up to 2026-07-31. Retains only ALA heavy atoms, "
        "so it contains NO true Ramachandran phi (needs C of ACE). Analyses that called "
        "(0,1,2,3) 'phi' were plotting the chirality improper: its +180-shifted median of "
        "+125 deg is the SAME coordinate as the 6-bead chirality CV. Kept so old campaigns remain "
        "reproducible; do not use for new work."
    ),
)

MAPPINGS: Dict[str, CGMapping] = {m.name: m for m in (_ALA2_BB6, _ALA2_5)}
for _m in MAPPINGS.values():
    _m.validate()


def get_mapping(name: str) -> CGMapping:
    if name not in MAPPINGS:
        raise KeyError(f"unknown mapping {name!r}; available: {sorted(MAPPINGS)}")
    return MAPPINGS[name]


if __name__ == "__main__":
    for m in MAPPINGS.values():
        print(m.describe())
        print("  PLUMED ATOMS=%s" % m.plumed_atom_selection())
        print()
