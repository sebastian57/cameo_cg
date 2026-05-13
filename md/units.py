"""Unit conversion for MD config parameters.

By default the md: config block uses physical units:
  dt      — fs   (femtoseconds)
  kT      — K    (Kelvin; converted via kB)
  gamma   — ps   (friction timescale τ = 1/γ; Langevin)
  mass    — amu  (already AKMA-native; no conversion)

JAX-MD internally uses AKMA units:
  length  — Å
  energy  — kcal/mol
  mass    — amu
  time    — 1 AKMA ≈ 48.888 fs   (derived from F = ma in the above)

Set  units: akma  in the config to pass values straight through
(backward-compatible with configs that pre-compute AKMA values).

Adding a new convertible field
------------------------------
Register it in _CONVERTERS with a (forward_fn, reverse_fn, description) tuple:
  forward  — physical → AKMA
  reverse  — AKMA → physical  (used for human-readable logging)
  description — shown in log output
"""

from typing import Any, Dict, Tuple, Callable, Optional
from utils.logging import md_logger

# ── Physical constants ────────────────────────────────────────────────────────
FS_PER_AKMA    = 48.888          # 1 AKMA ≈ 48.888 fs
KB_KCAL_PER_K  = 1.9872041e-3   # Boltzmann constant  [kcal/mol/K]

# ── Converter registry ────────────────────────────────────────────────────────
# Each entry: field_name -> (to_akma, from_akma, log_unit_physical, log_unit_akma)
#   to_akma(v)   : physical → AKMA
#   from_akma(v) : AKMA     → physical  (for human-readable log output)

_ConvEntry = Tuple[Callable, Callable, str, str]

_CONVERTERS: Dict[str, _ConvEntry] = {
    # dt: femtoseconds → AKMA time units
    "dt": (
        lambda v: v / FS_PER_AKMA,
        lambda v: v * FS_PER_AKMA,
        "fs", "AKMA",
    ),
    # kT: Kelvin → kcal/mol   (kT = kB * T)
    "kT": (
        lambda v: v * KB_KCAL_PER_K,
        lambda v: v / KB_KCAL_PER_K,
        "K", "kcal/mol",
    ),
    # gamma: friction timescale τ in ps → friction rate γ in AKMA⁻¹
    #   γ [AKMA⁻¹] = 1/τ [AKMA] = FS_PER_AKMA / (τ_ps × 1000 fs/ps)
    "gamma": (
        lambda v: FS_PER_AKMA / (v * 1000.0),
        lambda v: FS_PER_AKMA / (v * 1000.0),   # symmetric: ps ↔ AKMA⁻¹ via same formula
        "ps (τ)", "AKMA⁻¹ (γ=1/τ)",
    ),
    # mass: amu is already the AKMA mass unit — identity conversion
    "mass": (
        lambda v: v,
        lambda v: v,
        "amu", "amu",
    ),
}


def register_converter(
    field: str,
    to_akma: Callable[[float], float],
    from_akma: Callable[[float], float],
    unit_physical: str,
    unit_akma: str,
) -> None:
    """Register a new physical→AKMA converter for a config field.

    Args:
        field:         Key in the md: config dict (e.g. "pressure").
        to_akma:       Converts physical-unit value → AKMA value.
        from_akma:     Converts AKMA value → physical-unit value (for logging).
        unit_physical: Human label shown in logs (e.g. "bar").
        unit_akma:     AKMA label shown in logs (e.g. "kcal/mol/Å³").
    """
    _CONVERTERS[field] = (to_akma, from_akma, unit_physical, unit_akma)


def to_akma(md_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an md: config dict from physical units to AKMA units.

    Reads  md_cfg["units"]  to decide:
      "physical" (default) — apply registered converters
      "akma"               — pass all values through unchanged

    Returns a new dict; the input is not modified.
    """
    unit_system = str(md_cfg.get("units", "physical")).strip().lower()
    if unit_system not in ("physical", "akma"):
        raise ValueError(
            f"md.units={unit_system!r} is not recognised. "
            "Expected 'physical' (default) or 'akma'."
        )

    converted = dict(md_cfg)   # shallow copy

    if unit_system == "akma":
        md_logger.info("[Units] unit_system=akma — passing values through unchanged.")
        return converted

    # unit_system == "physical"
    conversions_applied = []
    for field, (to_fn, _, unit_in, unit_out) in _CONVERTERS.items():
        if field not in md_cfg:
            continue
        raw   = float(md_cfg[field])
        akma  = to_fn(raw)
        converted[field] = akma
        conversions_applied.append(
            f"  {field}: {raw:.6g} {unit_in}  →  {akma:.6g} {unit_out}"
        )

    if conversions_applied:
        md_logger.info("[Units] Converted physical → AKMA:")
        for line in conversions_applied:
            md_logger.info(line)

    return converted


def describe_akma(md_cfg_akma: Dict[str, Any]) -> str:
    """Return a human-readable summary of an AKMA config dict."""
    lines = []
    for field, (_, from_fn, unit_in, _) in _CONVERTERS.items():
        if field not in md_cfg_akma:
            continue
        phys = from_fn(float(md_cfg_akma[field]))
        lines.append(f"  {field} = {phys:.6g} {unit_in}")
    return "\n".join(lines)
