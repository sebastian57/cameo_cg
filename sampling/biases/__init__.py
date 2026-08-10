"""Composable bias terms for CG-guided atomistic enhanced sampling.

Declare any combination in a run config; the server sums them:

    biases:
      - type: mlcg_teacher
        params_path: .../params.pkl
        alpha: 1.0
      - type: tica_regional
        bias_npz: .../reference_bias.npz

Teacher only, TICA only, both, or a new term = editing that list. A new bias type is
one class decorated with @register_bias; nothing else changes.
"""

from .base import (  # noqa: F401
    BIAS_REGISTRY,
    BiasTerm,
    build_bias,
    build_biases,
    evaluate_all,
    register_bias,
)
from . import teacher as _teacher   # noqa: F401  (registers mlcg_teacher)
from . import tica_regional as _tica  # noqa: F401  (registers tica_regional)
from . import local_inversion as _inv  # noqa: F401  (registers local_inversion_umbrella)
from . import tica_metad as _metad  # noqa: F401  (registers tica_metad)

__all__ = [
    "BIAS_REGISTRY", "BiasTerm", "build_bias", "build_biases",
    "evaluate_all", "register_bias",
]
