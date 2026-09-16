"""Source-level contract tests for the lean Allegro cuEq backend.

Runtime numerical tests live in the GPU benchmark because importing cuEq on a
login node requires the CUDA runtime.  These checks remain runnable anywhere
and make the intended scope of the clean backend explicit.
"""

from __future__ import annotations

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CLEAN_MODEL = MODEL_DIR / "allegro_cueq_fast_clean.py"


def _defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_clean_backend_exists_and_keeps_only_standard_tp_path() -> None:
    source = CLEAN_MODEL.read_text()
    tree = ast.parse(source, filename=str(CLEAN_MODEL))
    names = _defined_names(tree)

    assert {"Allegro", "AllegroLayer", "allegro_neighborlist_pp"} <= names
    assert "_tensor_product_mixed" in names
    assert "_tensor_product_blockwise" not in names
    assert "_tensor_product_per_irrep" not in names
    assert "_tensor_product_fused_sp" not in names

    forbidden_tokens = (
        "SmoothingEnvelope",
        "AllegroFastForceHead",
        "AllegroCentralForceHead",
        "scatter_central_pair_forces",
        "chirality",
        "return_al_features",
        "direct_forces",
        "fast_forces",
        "block_uniform_1d",
        "uniform_1d",
        "fused_sp",
    )
    for token in forbidden_tokens:
        assert token not in source


def test_clean_backend_guards_edge_distance_calculation() -> None:
    source = CLEAN_MODEL.read_text()
    tree = ast.parse(source, filename=str(CLEAN_MODEL))

    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "allegro_neighborlist_pp"
    )
    wrapper_norms = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "norm"
    ]
    assert len(wrapper_norms) == 1, (
        "The wrapper should have at most one optional edge-distance norm; "
        "radial-feature norms belong inside the model core."
    )

    gated_distance_blocks = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.If) and "edge_distance_gate" in ast.unparse(node.test)
    ]
    assert gated_distance_blocks, "Edge-distance gate handling must be explicit."
    assert any(
        wrapper_norms[0] in ast.walk(block) for block in gated_distance_blocks
    ), "The wrapper edge-distance norm must be inside the gate guard."


def test_original_backend_is_still_separate() -> None:
    assert (MODEL_DIR / "allegro_cueq_fast_1103.py").exists()
    assert CLEAN_MODEL != MODEL_DIR / "allegro_cueq_fast_1103.py"
