"""cuEquivariance Allegro experimental model with optional fused-SP TP backend."""

import os
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import cuequivariance as cue
import cuequivariance_jax as cuex
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from jax_md import space, partition, util as md_util

# Resolve helper module paths used by the fast backend implementation.
# The helper files live in the sibling repository directory:
#   /p/project1/cameo/schmidt36/cueq_allegro/{layers.py,layers_cueq.py,utils.py}
_CUEQ_HELPER_DIR = Path(__file__).resolve().parents[2] / "cueq_allegro"


def _load_helper_module(module_stem: str):
    """Load cueq_allegro helper modules by explicit file path."""
    helper_file = _CUEQ_HELPER_DIR / f"{module_stem}.py"
    if not helper_file.is_file():
        raise ImportError(f"Missing helper module file: {helper_file}")
    module_name = f"_cueq_fast_helper_{module_stem}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, helper_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for helper module: {helper_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_layers_mod = _load_helper_module("layers")
_layers_cueq_mod = _load_helper_module("layers_cueq")
_utils_mod = _load_helper_module("utils")

CueLinear = _layers_mod.CueLinear
RadialBesselLayer = _layers_mod.RadialBesselLayer
SmoothingEnvelope = _layers_mod.SmoothingEnvelope
AtomicEnergyLayer = _layers_cueq_mod.AtomicEnergyLayer
polynomial_envelope = _utils_mod.polynomial_envelope
segment_sum_map_back = _utils_mod.segment_sum_map_back


def _irrep_to_str(ir: Any) -> str:
    return f"{ir.l}{'e' if ir.p == 1 else 'o'}"


def _sorted_unique_irreps(irreps: cue.Irreps) -> List[Any]:
    """Return unique irreps sorted by (l, p) for deterministic descriptor builds."""
    uniq: Dict[Tuple[int, int], Any] = {}
    for _mul, ir in irreps:
        uniq[(int(ir.l), int(ir.p))] = ir
    return [uniq[k] for k in sorted(uniq.keys())]


def _irrep_block_slices(irreps: cue.Irreps) -> List[Tuple[int, Any, int, int]]:
    """Build contiguous [start, end) slices for each irrep block in ir_mul layout."""
    blocks: List[Tuple[int, Any, int, int]] = []
    offset = 0
    for mul, ir in irreps:
        width = int(mul) * int(ir.dim)
        blocks.append((int(mul), ir, offset, offset + width))
        offset += width
    return blocks


def _normalize_tp_mode(tp_mode: str) -> str:
    """Normalize TP mode aliases to internal names."""
    aliases = {
        "block_linear_1d": "block_uniform_1d",
    }
    return aliases.get(tp_mode, tp_mode)


def _normalize_tp_method(tp_method: str) -> str:
    """Normalize TP method aliases to cuEquivariance names."""
    aliases = {
        "linear_1d": "uniform_1d",
    }
    return aliases.get(tp_method, tp_method)


def _normalize_tp_backend(tp_backend: str) -> str:
    """Normalize TP backend aliases to internal names."""
    aliases = {
        "baseline": "baseline_mixed",
        "mixed": "baseline_mixed",
        "fused": "fused_sp",
    }
    return aliases.get(tp_backend, tp_backend)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _parse_mode_csv(modes: Optional[Union[str, Iterable[str]]]) -> Optional[Tuple[str, ...]]:
    if modes is None:
        return None
    if isinstance(modes, str):
        parsed = tuple(m.strip() for m in modes.split(",") if m.strip())
    else:
        parsed = tuple(str(m).strip() for m in modes if str(m).strip())
    return parsed or None


def _mesh_safe_softplus(x):
    """Softplus variant that avoids Manual-vs-Auto sharding mismatches."""
    x = jnp.asarray(x)
    zero = jnp.zeros_like(x)
    return jnp.maximum(x, zero) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def _is_indexed_linear_candidate(stp: cue.SegmentedTensorProduct) -> bool:
    """Heuristic: check if STP looks like i,j,+ij (linear-like indexed contraction)."""
    if stp.num_operands != 3:
        return False
    subs = stp.subscripts
    operands = list(subs.operands)
    if len(operands) != 3:
        return False
    return operands[2] == "" and subs.coefficients == operands[0] + operands[1]


def summarize_sp_operations(
    tp_desc: cue.EquivariantPolynomial,
) -> Tuple[cue.SegmentedPolynomial, List[Dict[str, Any]]]:
    """Return consolidated SP plus per-operation canonicalization summary."""
    polynomial = tp_desc.polynomial.consolidate()
    summary = _summarize_segmented_polynomial(polynomial)
    return polynomial, summary


def _summarize_segmented_polynomial(
    polynomial: cue.SegmentedPolynomial,
) -> List[Dict[str, Any]]:
    """Summarize operations of a consolidated segmented polynomial."""
    summary: List[Dict[str, Any]] = []
    for op_idx, (_ope, stp) in enumerate(polynomial.operations):
        modes = list(stp.subscripts.modes())
        summary.append(
            {
                "op_idx": int(op_idx),
                "subscripts": str(stp.subscripts),
                "num_modes": int(len(modes)),
                "modes": modes,
                "num_segments": [int(operand.num_segments) for operand in stp.operands],
                "sizes": [int(operand.size) for operand in stp.operands],
                "num_paths": int(stp.num_paths),
                "uniform_1d_eligible": bool(len(modes) == 1),
                "indexed_linear_candidate": bool(_is_indexed_linear_candidate(stp)),
            }
        )
    return summary


def _extract_irrep_subset(
    rep: cuex.RepArray,
    target_l: int,
    target_p: int,
) -> cuex.RepArray:
    """Return only channels with irrep (l=target_l, p=target_p) in ir_mul layout."""
    irreps = rep.irreps if isinstance(rep.irreps, cue.Irreps) else rep.irreps.irreps
    offset = 0
    parts: List[jnp.ndarray] = []
    terms: List[str] = []

    for mul, ir in irreps:
        width = int(mul) * int(ir.dim)
        chunk = rep.array[:, offset:offset + width]
        if int(ir.l) == int(target_l) and int(ir.p) == int(target_p):
            parts.append(chunk)
            terms.append(f"{int(mul)}x{int(ir.l)}{'e' if int(ir.p) == 1 else 'o'}")
        offset += width

    if parts:
        out_array = jnp.concatenate(parts, axis=-1)
        out_irreps = cue.Irreps("O3", " + ".join(terms))
    else:
        out_array = jnp.zeros((rep.array.shape[0], 0), dtype=rep.array.dtype)
        out_irreps = cue.Irreps("O3", "0x0e")

    return cuex.RepArray(cue.IrrepsAndLayout(out_irreps, cue.ir_mul), out_array)


def build_tp_left_input(
    Y: jnp.ndarray,
    a: jnp.ndarray,
    senders: jnp.ndarray,
    mode: Literal["node_agg", "edge_local"],
    eps: float,
    n_nodes: int,
    norm: Optional[Literal["deg_sqrt", "deg"]] = None,
    norm_delta: float = 1e-8,
) -> jnp.ndarray:
    """Build TP-left input with either sender-node aggregation or pure edge-local form.

    Args:
        Y: Spherical harmonics array [E, K].
        a: Scalar steering weights [E, g].
        senders: Sender node ids [E].
        mode: "node_agg" (baseline) or "edge_local" (approximation).
        eps: Layer normalization scalar.
        n_nodes: Number of nodes.
        norm: Optional degree normalization, only for edge_local mode.
        norm_delta: Numerical stabilizer for degree normalization.

    Returns:
        Array [E, g, K] used as left TP input.
    """
    valid_sender = jnp.logical_and(senders >= 0, senders < n_nodes)
    senders_safe = jnp.where(valid_sender, senders, 0)

    left = a[:, :, None] * Y[:, None, :]
    left = jnp.where(valid_sender[:, None, None], left, 0.0)

    if mode == "node_agg":
        if norm is not None:
            raise ValueError("tp_left_norm is only supported with tp_left_mode='edge_local'.")
        return segment_sum_map_back(left, senders_safe, n_nodes) * eps

    if mode != "edge_local":
        raise ValueError(f"Unknown tp_left_mode {mode!r}. Expected 'node_agg' or 'edge_local'.")

    if norm is not None:
        deg_sender = jax.ops.segment_sum(
            jnp.where(valid_sender, 1.0, 0.0).astype(left.dtype),
            senders_safe,
            num_segments=n_nodes,
        )
        deg_on_edge = deg_sender[senders_safe]
        if norm == "deg_sqrt":
            scale = jax.lax.rsqrt(deg_on_edge + norm_delta)
        elif norm == "deg":
            scale = 1.0 / (deg_on_edge + norm_delta)
        else:
            raise ValueError(f"Unknown tp_left_norm {norm!r}. Expected None, 'deg_sqrt', or 'deg'.")
        scale = jnp.where(valid_sender, scale, 0.0).astype(left.dtype)
        left = left * scale[:, None, None]

    return left * eps


class AllegroEmbedding(hk.Module):
    """Initial feature embedding for Allegro.

    Converts edge vectors and species information into scalar and equivariant features.
    """

    def __init__(
        self,
        num_species: int,
        embed_n_hidden: Iterable[int],
        n_radial_basis: int,
        envelope_p: int,
        mlp_n_hidden: int,
        irreps: Union[cue.Irreps, str],
        mlp_activation: Callable = jax.nn.silu,
        species_embed: Optional[int] = None,
        name: Optional[str] = None
    ):
        """Initialize AllegroEmbedding.

        Args:
            num_species: Number of atomic species
            embed_n_hidden: Hidden layer sizes for embedding MLP
            n_radial_basis: Number of radial basis functions
            envelope_p: Polynomial order for envelope
            mlp_n_hidden: Output size for scalar features
            irreps: Output irreps for equivariant features
            mlp_activation: Activation function for embedding MLP
            species_embed: Species embedding dimension (default: embed_n_hidden[0] // 2)
            name: Module name
        """
        super().__init__(name=name)

        if isinstance(irreps, str):
            irreps = cue.Irreps("O3", irreps)

        self.irreps = irreps
        self.envelope_p = envelope_p
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_activation = mlp_activation

        embed_n_hidden = list(embed_n_hidden)
        if species_embed is None:
            species_embed = embed_n_hidden[0] // 2

        self.species_embed_dim = species_embed

        self.radial_basis = RadialBesselLayer(
            cutoff=1.0,
            num_radial=n_radial_basis,
            envelope_p=envelope_p
        )

        self.species_embedding = hk.Embed(num_species, species_embed)

        self.embed_layers = embed_n_hidden + [mlp_n_hidden]

    def __call__(
        self,
        vectors: cuex.RepArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray
    ) -> Tuple[jnp.ndarray, cuex.RepArray]:
        """Compute initial embeddings.

        Args:
            vectors: Edge vectors, RepArray shape [n_edges, 3], irreps "1o"
            senders: Sender node indices, shape [n_edges]
            receivers: Receiver node indices, shape [n_edges]
            species: Species for each node, shape [n_nodes]

        Returns:
            x: Scalar features, shape [n_edges, mlp_n_hidden]
            V: Equivariant features, RepArray shape [n_edges, irreps_dim]
        """
        n_edges = vectors.array.shape[0]

        distances = jnp.linalg.norm(vectors.array, axis=-1)

        radial_features = self.radial_basis(distances)

        species_sender = self.species_embedding(species[senders])
        species_receiver = self.species_embedding(species[receivers])

        x = jnp.concatenate([radial_features, species_sender, species_receiver], axis=-1)

        x = e3nn.haiku.MultiLayerPerceptron(
            self.embed_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)

        envelope = polynomial_envelope(distances, p=self.envelope_p, cutoff=1.0)
        x = envelope[:, None] * x

        irreps_Y = self._filter_irreps_by_parity(vectors)

        with cue.assume(cue.O3, cue.ir_mul):
            ls = sorted(set(ir.l for mul, ir in irreps_Y))
            Y_single = cuex.spherical_harmonics(ls, vectors, normalize=True)

            Y_parts = []
            for mul, ir in irreps_Y:
                l_idx = ls.index(ir.l)
                l_dim = 2 * ir.l + 1
                l_start = sum(2 * ls[j] + 1 for j in range(l_idx))
                l_end = l_start + l_dim
                Y_l = Y_single.array[:, l_start:l_end]  # (E, l_dim), single copy in ir_mul
                Y_l_expanded = jnp.repeat(Y_l, mul, axis=-1)  # (E, l_dim * mul) in ir_mul
                Y_parts.append(Y_l_expanded)

            Y_array = jnp.concatenate(Y_parts, axis=-1)
            Y_irreps = cue.Irreps("O3", " + ".join([f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"
                                                    for mul, ir in irreps_Y]))
            Y = cuex.RepArray(cue.IrrepsAndLayout(Y_irreps, cue.ir_mul), Y_array)

            species_irreps = cue.Irreps("O3", f"{self.species_embed_dim}x0e")
            species_sender_rep = cuex.RepArray(
                cue.IrrepsAndLayout(species_irreps, cue.ir_mul),
                species_sender
            )
            species_receiver_rep = cuex.RepArray(
                cue.IrrepsAndLayout(species_irreps, cue.ir_mul),
                species_receiver
            )
            V = cuex.concatenate([Y, species_sender_rep, species_receiver_rep])

            V_irreps = V.irreps if isinstance(V.irreps, cue.Irreps) else V.irreps.irreps
            num_irreps = sum(mul for mul, ir in V_irreps)

            w = e3nn.haiku.MultiLayerPerceptron(
                (num_irreps,), None, output_activation=False
            )(x)

            E = x.shape[0]
            V_parts = []
            v_offset = 0
            w_offset = 0
            for mul, ir in V_irreps:
                ir_dim = ir.dim
                block = V.array[:, v_offset:v_offset + mul * ir_dim]   # (E, ir_dim * mul)
                block = block.reshape(E, ir_dim, mul)                  # (E, ir_dim, mul)
                w_k = w[:, w_offset:w_offset + mul]                    # (E, mul)
                block = (block * w_k[:, None, :]) / num_irreps         # (E, ir_dim, mul)
                V_parts.append(block.reshape(E, ir_dim * mul))
                v_offset += mul * ir_dim
                w_offset += mul
            V = cuex.RepArray(V.irreps, jnp.concatenate(V_parts, axis=-1))

        return x, V

    def _filter_irreps_by_parity(self, vectors: cuex.RepArray) -> cue.Irreps:
        """Filter irreps to match vector parity.

        Only keep irreps where parity matches: (-1)^l == vector_parity
        """
        vector_parity = 1 if "1o" in str(vectors.irreps) else -1  # 1o is odd, 1e is even

        filtered_irreps = []
        for mul, ir in self.irreps:
            ir_parity = (-1) ** ir.l
            if ir_parity * vector_parity == ir.p:
                filtered_irreps.append((mul, ir))

        irreps_str = " + ".join([f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"
                                  for mul, ir in filtered_irreps])

        return cue.Irreps("O3", irreps_str) if irreps_str else cue.Irreps("O3", "0e")


class AllegroLayer(hk.Module):
    """Tensor product layer matching e3nn's AllegroLayer.

    This layer implements the exact operations from e3nn:
    1. Weight generation from scalar features (using V.irreps.mul_gcd)
    2. Spherical harmonics computation
    3. Scatter sum (neighbor aggregation with map_back)
    4. Tensor product
    5. Scalar extraction and concatenation
    6. MLP on scalars
    7. Envelope weighting
    8. Linear projection on equivariant features
    """

    def __init__(
        self,
        epsilon: float,
        max_ell: int,
        output_irreps: Union[cue.Irreps, str],
        mlp_n_hidden: int,
        mlp_n_layers: int,
        p: int = 6,
        mlp_activation = jax.nn.silu,
        tp_backend: str = "baseline_mixed",
        tp_fused_flatten_modes: Optional[Union[str, Iterable[str]]] = None,
        tp_mode: str = "mixed_naive",
        tp_method: str = "naive",
        tp_batch_strategy: str = "nested_vmap",
        tp_left_mode: Literal["node_agg", "edge_local"] = "node_agg",
        tp_left_norm: Optional[Literal["deg_sqrt", "deg"]] = None,
        name: Optional[str] = None
    ):
        """Initialize AllegroLayer.

        Args:
            epsilon: Normalization constant (1/sqrt(1 + softplus(avg_neighbors)))
            max_ell: Maximum l for spherical harmonics
            output_irreps: Output irreps for equivariant features
            mlp_n_hidden: Hidden size for MLPs
            mlp_n_layers: Number of MLP layers
            p: Polynomial order for envelope
            mlp_activation: Activation function
            tp_backend: TP backend:
                - "baseline_mixed": existing cuex.equivariant_polynomial path
                - "fused_sp": one consolidated cuex.segmented_polynomial call
            tp_fused_flatten_modes: Optional mode list (e.g. "i,j") applied to the
                fused SP descriptor before execution. Use "auto" to flatten all
                but the last canonical mode. Intended for Option B1 probes.
            tp_mode: TP implementation mode:
                - "mixed_naive": single mixed-irrep TP descriptor
                - "block_naive": per-irrep blockwise TP with method="naive"
                - "block_uniform_1d": per-irrep blockwise TP with method="uniform_1d"
                - alias: "block_linear_1d" -> "block_uniform_1d"
            tp_method: Method used by mixed TP path.
                - alias: "linear_1d" -> "uniform_1d"
            tp_batch_strategy: How to batch mixed TP ("flatten_Eg" or "nested_vmap")
            tp_left_mode: TP-left construction mode:
                - "node_agg": sender-node segment-sum then map-back (baseline)
                - "edge_local": no cross-edge aggregation (edge-local approximation)
            tp_left_norm: Optional degree normalization for edge_local TP-left.
            name: Module name
        """
        super().__init__(name=name)

        if isinstance(output_irreps, str):
            output_irreps = cue.Irreps("O3", output_irreps)

        self.epsilon = epsilon
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = p
        self.mlp_activation = mlp_activation
        self.tp_backend = _normalize_tp_backend(tp_backend)
        self.tp_fused_flatten_modes = _parse_mode_csv(tp_fused_flatten_modes)
        self.tp_mode = _normalize_tp_mode(tp_mode)
        self.tp_method = _normalize_tp_method(tp_method)
        self.tp_batch_strategy = tp_batch_strategy
        self.tp_left_mode = tp_left_mode
        self.tp_left_norm = tp_left_norm
        self.print_sp_summary = _env_flag("ALLEGRO_PRINT_SP_SUMMARY", default=False)
        self.tp_method_fallback = os.environ.get("ALLEGRO_TP_METHOD_FALLBACK", "error").strip().lower()
        if self.tp_backend not in ("baseline_mixed", "fused_sp"):
            raise ValueError(
                f"Unknown tp_backend {self.tp_backend!r}. "
                "Expected 'baseline_mixed' or 'fused_sp'."
            )
        if self.tp_method_fallback not in ("error", "naive"):
            raise ValueError(
                f"Unknown ALLEGRO_TP_METHOD_FALLBACK={self.tp_method_fallback!r}. "
                "Expected 'error' or 'naive'."
            )

    def _tensor_product_mixed(
        self,
        wY_axis: jnp.ndarray,
        V_axis: jnp.ndarray,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
        tp_desc: cue.EquivariantPolynomial,
        n_edges: int,
        mul_gcd: int,
    ) -> Tuple[jnp.ndarray, cue.Irreps]:
        """Apply the original mixed-irrep TP path."""
        tp_output_irreps = tp_desc.outputs[0].irreps if hasattr(tp_desc.outputs[0], "irreps") else tp_desc.outputs[0]
        if self.tp_batch_strategy == "nested_vmap":
            def tp_single_slice(Y_slice, V_slice):
                Y_rep = cuex.RepArray(cue.IrrepsAndLayout(Y_irreps, cue.ir_mul), Y_slice)
                V_rep = cuex.RepArray(cue.IrrepsAndLayout(V_red_irreps, cue.ir_mul), V_slice)
                result = cuex.equivariant_polynomial(tp_desc, [Y_rep, V_rep], method="naive")
                if isinstance(result, list):
                    result = result[0]
                return result.array

            tp_vmapped = jax.vmap(jax.vmap(tp_single_slice, in_axes=0), in_axes=0)
            return tp_vmapped(wY_axis, V_axis), tp_output_irreps

        Eg = n_edges * mul_gcd
        wY_flat = wY_axis.reshape(Eg, -1)
        V_flat = V_axis.reshape(Eg, -1)
        wY_rep = cuex.RepArray(cue.IrrepsAndLayout(Y_irreps, cue.ir_mul), wY_flat)
        V_rep = cuex.RepArray(cue.IrrepsAndLayout(V_red_irreps, cue.ir_mul), V_flat)
        out_flat = cuex.equivariant_polynomial(tp_desc, [wY_rep, V_rep], method=self.tp_method)
        if isinstance(out_flat, list):
            out_flat = out_flat[0]
        return out_flat.array.reshape(n_edges, mul_gcd, -1), tp_output_irreps

    def _tensor_product_blockwise(
        self,
        wY_axis: jnp.ndarray,
        V_axis: jnp.ndarray,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
        tp_desc: cue.EquivariantPolynomial,
        n_edges: int,
        mul_gcd: int,
        method: str,
    ) -> Tuple[jnp.ndarray, cue.Irreps]:
        """Apply per-irrep TP calls and concatenate block outputs."""
        Eg = n_edges * mul_gcd
        wY_flat = wY_axis.reshape(Eg, -1)
        V_flat = V_axis.reshape(Eg, -1)
        dtype = wY_flat.dtype

        y_blocks = _irrep_block_slices(Y_irreps)
        v_blocks = _irrep_block_slices(V_red_irreps)

        tp_output_irreps_mixed = tp_desc.outputs[0].irreps if hasattr(tp_desc.outputs[0], "irreps") else tp_desc.outputs[0]
        out_irreps_unique = _sorted_unique_irreps(tp_output_irreps_mixed)

        out_parts: List[jnp.ndarray] = []
        out_terms: List[str] = []
        desc_cache: Dict[Tuple[int, ...], Optional[cue.EquivariantPolynomial]] = {}

        for y_mul, y_ir, y_start, y_end in y_blocks:
            y_irreps_single = cue.Irreps("O3", f"{y_mul}x{_irrep_to_str(y_ir)}")
            y_rep = cuex.RepArray(
                cue.IrrepsAndLayout(y_irreps_single, cue.ir_mul),
                wY_flat[:, y_start:y_end],
            )

            for v_mul, v_ir, v_start, v_end in v_blocks:
                v_irreps_single = cue.Irreps("O3", f"{v_mul}x{_irrep_to_str(v_ir)}")
                v_rep = cuex.RepArray(
                    cue.IrrepsAndLayout(v_irreps_single, cue.ir_mul),
                    V_flat[:, v_start:v_end],
                )

                for out_ir in out_irreps_unique:
                    cache_key = (
                        int(y_ir.l), int(y_ir.p), int(y_mul),
                        int(v_ir.l), int(v_ir.p), int(v_mul),
                        int(out_ir.l), int(out_ir.p),
                    )
                    desc = desc_cache.get(cache_key)
                    if desc is None and cache_key not in desc_cache:
                        out_filter = [out_ir]
                        desc = cue.descriptors.full_tensor_product(
                            y_irreps_single,
                            v_irreps_single,
                            irreps3_filter=out_filter,
                        )
                        out_dim = desc.outputs[0].dim if hasattr(desc.outputs[0], "dim") else 0
                        desc_cache[cache_key] = desc if out_dim > 0 else None
                        desc = desc_cache[cache_key]
                    if desc is None:
                        continue

                    out_part = cuex.equivariant_polynomial(desc, [y_rep, v_rep], method=method)
                    if isinstance(out_part, list):
                        out_part = out_part[0]
                    out_parts.append(out_part.array)

                    out_irreps = out_part.irreps.irreps if hasattr(out_part.irreps, "irreps") else out_part.irreps
                    for mul, ir in out_irreps:
                        out_terms.append(f"{mul}x{_irrep_to_str(ir)}")

        if out_parts:
            out_flat = jnp.concatenate(out_parts, axis=-1)
            out_irreps = cue.Irreps("O3", " + ".join(out_terms))
        else:
            out_flat = jnp.zeros((Eg, 0), dtype=dtype)
            out_irreps = cue.Irreps("O3", "0x0e")
        return out_flat.reshape(n_edges, mul_gcd, -1), out_irreps

    def _build_tp_descriptor(
        self,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
    ) -> cue.EquivariantPolynomial:
        """Build TP descriptor on reduced irreps with deterministic filtering."""
        filter_irreps = _sorted_unique_irreps(self.output_irreps)
        has_scalar = any(int(ir.l) == 0 and int(ir.p) == 1 for ir in filter_irreps)
        if not has_scalar:
            filter_irreps = filter_irreps + [ir for _, ir in cue.Irreps("O3", "1x0e")]
        return cue.descriptors.full_tensor_product(
            Y_irreps,
            V_red_irreps,
            irreps3_filter=filter_irreps,
        )

    def _build_per_irrep_tp_descriptors(
        self,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
    ) -> Dict[Tuple[int, int], cue.EquivariantPolynomial]:
        """Build separate TP descriptors for each (l, p) irrep block.
        
        This enables using uniform_1d method which requires single-mode descriptors.
        """
        irrep_descriptors = {}
        
        for mul, ir in self.output_irreps:
            l, p = ir.l, ir.p
            
            if (l, p) in irrep_descriptors:
                continue
            
            Y_filtered = cue.Irreps("O3", f"1x{l}{'e' if p == 1 else 'o'}")

            def keep_fn(mul_ir):
                mul_, ir_ = mul_ir
                return int(ir_.l) == l and int(ir_.p) == p
            V_filtered = V_red_irreps.filter(keep=keep_fn)
            
            if V_filtered.dim == 0:
                continue
            
            try:
                tp_desc = cue.descriptors.full_tensor_product(
                    Y_filtered,
                    V_filtered,
                    irreps3_filter=cue.Irreps("O3", f"{mul}x{l}{'e' if p == 1 else 'o'}"),
                )
                irrep_descriptors[(l, p)] = tp_desc
            except Exception:
                continue
        
        return irrep_descriptors

    def _tensor_product_per_irrep(
        self,
        wY_axis: jnp.ndarray,
        V_axis: jnp.ndarray,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
        n_edges: int,
        mul_gcd: int,
    ) -> Tuple[jnp.ndarray, cue.Irreps]:
        """Apply TP using per-irrep descriptors with uniform_1d method.
        
        This splits the TP into separate calls per (l, p) irrep block,
        each of which is eligible for the fast uniform_1d kernel.
        """
        irrep_descriptors = self._build_per_irrep_tp_descriptors(Y_irreps, V_red_irreps)
        
        if not irrep_descriptors:
            return jnp.zeros((n_edges, mul_gcd, 0)), cue.Irreps("O3", "0x0e")
        
        results = []
        output_irreps_list = []
        
        Eg = n_edges * mul_gcd
        
        for (l, p), tp_desc in sorted(irrep_descriptors.items()):
            try:
                wY_dim = tp_desc.inputs[0].dim
                V_dim = tp_desc.inputs[1].dim
                
                wY_flat = wY_axis.reshape(Eg, -1)
                V_flat = V_axis.reshape(Eg, -1)
                
                wY_slice = wY_flat[:, :wY_dim]
                V_slice = V_flat[:, :V_dim]
                
                wY_rep = cuex.RepArray(tp_desc.inputs[0], wY_slice)
                V_rep = cuex.RepArray(tp_desc.inputs[1], V_slice)
                
                out = cuex.equivariant_polynomial(tp_desc, [wY_rep, V_rep], method="uniform_1d")
                
                if isinstance(out, list):
                    out = out[0]
                
                out_arr = out.array.reshape(n_edges, mul_gcd, -1)
                results.append(out_arr)
                output_irreps_list.append(tp_desc.outputs[0].irreps)
                
            except Exception as e:
                if hk.running_init():
                    print(f"[{self.name}] Failed TP for ({l},{p}): {e}", flush=True)
                continue
        
        if not results:
            return jnp.zeros((n_edges, mul_gcd, 0)), cue.Irreps("O3", "0x0e")
        
        out_axis = jnp.concatenate(results, axis=-1)
        
        output_irreps = cue.Irreps("O3", "")
        for irreps in output_irreps_list:
            for mul, ir in irreps:
                output_irreps = output_irreps + cue.Irreps("O3", f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}")
        
        return out_axis, output_irreps

    def _tensor_product_fused_sp(
        self,
        wY_axis: jnp.ndarray,
        V_axis: jnp.ndarray,
        tp_desc: cue.EquivariantPolynomial,
        n_edges: int,
        mul_gcd: int,
    ) -> Tuple[jnp.ndarray, cue.Irreps]:
        """Apply one consolidated segmented polynomial call."""
        Eg = n_edges * mul_gcd
        wY_flat = wY_axis.reshape(Eg, -1)
        V_flat = V_axis.reshape(Eg, -1)

        poly, op_summary = summarize_sp_operations(tp_desc)
        if len(poly.operations) != 1:
            raise ValueError(
                f"{self.name}: fused_sp backend expects consolidated polynomial with one operation, "
                f"got {len(poly.operations)}."
            )

        applied_flatten_modes: Optional[List[str]] = None
        if self.tp_fused_flatten_modes is not None:
            requested_modes = list(self.tp_fused_flatten_modes)
            if len(requested_modes) == 1 and requested_modes[0].lower() == "auto":
                mode_list = list(op_summary[0]["modes"]) if op_summary else []
                requested_modes = mode_list[:-1]  # keep the final mode (typically output segment mode)

            if requested_modes:
                try:
                    poly = poly.flatten_modes(requested_modes)
                except ValueError as exc:
                    if self.tp_method_fallback == "naive":
                        if hk.running_init():
                            print(
                                f"[{self.name}] failed to flatten_modes={requested_modes!r} "
                                f"({exc}); continuing without flattening.",
                                flush=True,
                            )
                        requested_modes = []
                    else:
                        raise ValueError(
                            f"{self.name}: failed to flatten fused_sp modes {requested_modes!r}: {exc}"
                        ) from exc

            if requested_modes:
                if len(poly.operations) != 1:
                    raise ValueError(
                        f"{self.name}: flattened fused_sp polynomial must keep one operation, "
                        f"got {len(poly.operations)}."
                    )
                op_summary = _summarize_segmented_polynomial(poly)
                applied_flatten_modes = requested_modes

        if hk.running_init() and self.print_sp_summary:
            print(
                f"[{self.name}] fused_sp canonicalization: "
                f"num_operations={len(poly.operations)} method={self.tp_method!r}",
                flush=True,
            )
            if applied_flatten_modes is not None:
                print(
                    f"[{self.name}] fused_sp flatten_modes={applied_flatten_modes!r}",
                    flush=True,
                )
            for entry in op_summary:
                print(
                    f"[{self.name}] op={entry['op_idx']} subscripts={entry['subscripts']} "
                    f"num_modes={entry['num_modes']} modes={entry['modes']} "
                    f"num_segments={entry['num_segments']} sizes={entry['sizes']} "
                    f"num_paths={entry['num_paths']} "
                    f"uniform_1d_eligible={entry['uniform_1d_eligible']} "
                    f"indexed_linear_candidate={entry['indexed_linear_candidate']}",
                    flush=True,
                )

        requested_method = self.tp_method
        effective_method = requested_method

        if requested_method == "indexed_linear":
            if self.tp_method_fallback == "naive":
                effective_method = "naive"
                if hk.running_init():
                    print(
                        f"[{self.name}] method={requested_method!r} is unsupported in this fused_sp "
                        "path (indices=None); using fallback method='naive'.",
                        flush=True,
                    )
            else:
                raise ValueError(
                    f"{self.name}: method={requested_method!r} is unsupported for this fused_sp path "
                    "(indices=None). Set ALLEGRO_TP_METHOD_FALLBACK=naive or use tp_method='naive'."
                )

        if requested_method == "uniform_1d":
            uniform_ok = all(bool(entry["uniform_1d_eligible"]) for entry in op_summary)
            if not uniform_ok:
                if self.tp_method_fallback == "naive":
                    effective_method = "naive"
                    if hk.running_init():
                        print(
                            f"[{self.name}] method='uniform_1d' requires one-mode canonicalization; "
                            "descriptor is not eligible, falling back to method='naive'.",
                            flush=True,
                        )
                else:
                    raise ValueError(
                        f"{self.name}: method='uniform_1d' requested but descriptor is not "
                        "uniform_1d-eligible. Set ALLEGRO_TP_METHOD_FALLBACK=naive or use tp_method='naive'."
                    )

        out_shape = [jax.ShapeDtypeStruct((Eg, -1), wY_flat.dtype)]
        try:
            [out_flat] = cuex.segmented_polynomial(
                poly,
                [wY_flat, V_flat],
                out_shape,
                method=effective_method,
            )
        except ValueError as exc:
            if self.tp_method_fallback == "naive" and effective_method != "naive":
                if hk.running_init():
                    print(
                        f"[{self.name}] tp_method={effective_method!r} unsupported for fused_sp "
                        "descriptor on this platform; falling back to method='naive' "
                        "(ALLEGRO_TP_METHOD_FALLBACK=naive).",
                        flush=True,
                    )
                [out_flat] = cuex.segmented_polynomial(
                    poly,
                    [wY_flat, V_flat],
                    out_shape,
                    method="naive",
                )
            else:
                raise ValueError(
                    f"{self.name}: failed to execute segmented_polynomial with method={effective_method!r}. "
                    "This TP descriptor/method combination is unsupported. "
                    "Set ALLEGRO_TP_METHOD_FALLBACK=naive to continue with a runtime fallback."
                ) from exc

        tp_output_irreps = (
            tp_desc.outputs[0].irreps
            if hasattr(tp_desc.outputs[0], "irreps")
            else tp_desc.outputs[0]
        )
        return out_flat.reshape(n_edges, mul_gcd, -1), tp_output_irreps

    def __call__(
        self,
        vectors: cuex.RepArray,
        x: jnp.ndarray,
        V: cuex.RepArray,
        senders: jnp.ndarray,
        species: jnp.ndarray,
        num_nodes: int
    ) -> Tuple[jnp.ndarray, cuex.RepArray]:
        """Apply tensor product layer using axis-based approach.

        This implementation follows the gcd-axis factorization strategy to avoid
        the combinatorial explosion when encoding gcd as multiplicities.

        Key changes from previous version:
        1. Keep wY as (E, gcd, Y_dim) - don't flatten into multiplicities
        2. Factor V using unflatten_mul_to_axis
        3. Build TP descriptor on reduced irreps
        4. Vmap TP over edges and gcd axis
        5. Apply small projection per-slice
        6. Fold axis back with flatten_axis_to_mul

        Args:
            vectors: Edge vectors [n_edges, 3]
            x: Scalar features [n_edges, x_dim]
            V: Equivariant features RepArray [n_edges, V_dim]
            senders: Sender node indices [n_edges]
            species: Species per node [n_nodes]
            num_nodes: Number of nodes

        Returns:
            y: Scalar feature update [n_edges, mlp_n_hidden]
            V_out: Equivariant feature update RepArray [n_edges, output_dim]
        """
        compute_mul_gcd = _utils_mod.compute_mul_gcd
        extract_and_filter_scalars = _utils_mod.extract_and_filter_scalars
        flatten_axis_to_mul = _utils_mod.flatten_axis_to_mul
        unflatten_mul_to_axis = _utils_mod.unflatten_mul_to_axis

        n_edges = x.shape[0]

        mul_gcd = compute_mul_gcd(V.irreps)
        w = e3nn.haiku.MultiLayerPerceptron((mul_gcd,), None)(x)


        with cue.assume(cue.O3, cue.ir_mul):
            Y = cuex.spherical_harmonics(
                list(range(self.max_ell + 1)),
                vectors,
                normalize=True
            )

            Y_irreps = Y.irreps.irreps if hasattr(Y.irreps, 'irreps') else Y.irreps

            wY_axis = build_tp_left_input(
                Y=Y.array,
                a=w,
                senders=senders,
                mode=self.tp_left_mode,
                eps=self.epsilon,
                n_nodes=num_nodes,
                norm=self.tp_left_norm,
            )


            V_axis, V_red_irreps = unflatten_mul_to_axis(V.array, V.irreps, mul_gcd)


            tp_desc = self._build_tp_descriptor(Y_irreps, V_red_irreps)

            if hk.running_init():
                print(
                    f"[{self.name}] tp_backend={self.tp_backend!r} "
                    f"tp_mode={self.tp_mode!r} "
                    f"tp_method={self.tp_method!r} "
                    f"tp_fused_flatten_modes={self.tp_fused_flatten_modes!r} "
                    f"tp_batch_strategy={self.tp_batch_strategy!r} "
                    f"tp_left_mode={self.tp_left_mode!r} "
                    f"tp_left_norm={self.tp_left_norm!r}",
                    flush=True,
                )

            if self.tp_method == "uniform_1d":
                out_axis, tp_output_irreps = self._tensor_product_per_irrep(
                    wY_axis, V_axis, Y_irreps, V_red_irreps, n_edges, mul_gcd
                )
            elif self.tp_backend == "fused_sp":
                if self.tp_mode != "mixed_naive":
                    raise ValueError(
                        f"tp_backend='fused_sp' currently supports only tp_mode='mixed_naive', "
                        f"got {self.tp_mode!r}."
                    )
                out_axis, tp_output_irreps = self._tensor_product_fused_sp(
                    wY_axis, V_axis, tp_desc, n_edges, mul_gcd
                )
            else:
                if self.tp_mode == "mixed_naive":
                    out_axis, tp_output_irreps = self._tensor_product_mixed(
                        wY_axis, V_axis, Y_irreps, V_red_irreps, tp_desc, n_edges, mul_gcd
                    )
                elif self.tp_mode == "block_naive":
                    out_axis, tp_output_irreps = self._tensor_product_blockwise(
                        wY_axis, V_axis, Y_irreps, V_red_irreps, tp_desc, n_edges, mul_gcd, method="naive"
                    )
                elif self.tp_mode == "block_uniform_1d":
                    out_axis, tp_output_irreps = self._tensor_product_blockwise(
                        wY_axis, V_axis, Y_irreps, V_red_irreps, tp_desc, n_edges, mul_gcd, method="uniform_1d"
                    )
                else:
                    raise ValueError(
                        f"Unknown tp_mode {self.tp_mode!r}. "
                        "Expected one of: mixed_naive, block_naive, block_uniform_1d."
                    )

            V_new_array, V_new_irreps = flatten_axis_to_mul(
                out_axis,
                tp_output_irreps,
                mul_gcd
            )

            V_new = cuex.RepArray(
                cue.IrrepsAndLayout(V_new_irreps, cue.ir_mul),
                V_new_array
            )


            x_new, V_filtered = extract_and_filter_scalars(x, V_new)


            V_out = CueLinear(self.output_irreps)(V_filtered)


        y = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False
        )(x_new)

        lengths = jnp.linalg.norm(vectors.array, axis=-1)
        envelope = polynomial_envelope(lengths, p=self.envelope_p, cutoff=1.0)
        y = envelope[:, None] * y

        return y, V_out


class AllegroFastForceHead(hk.Module):
    """Optional fast force head built from edge-level l=1 channels."""

    def __init__(
        self,
        aggregate: Literal["receiver", "sender"] = "receiver",
        degree_norm: Optional[Literal["deg_sqrt", "deg"]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.aggregate = aggregate
        self.degree_norm = degree_norm

    def __call__(
        self,
        V_edge: cuex.RepArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        num_nodes: int,
    ) -> jnp.ndarray:
        """Predict per-node fast forces [N, 3] with equivariant operations only."""
        if self.aggregate not in ("receiver", "sender"):
            raise ValueError(
                f"Unknown fast-force aggregate mode {self.aggregate!r}. "
                "Expected 'receiver' or 'sender'."
            )

        dst = receivers if self.aggregate == "receiver" else senders
        with cue.assume(cue.O3, cue.ir_mul):
            V_l1 = _extract_irrep_subset(V_edge, target_l=1, target_p=-1)
            if V_l1.array.shape[-1] == 0:
                return jnp.zeros((num_nodes, 3), dtype=V_edge.array.dtype)

            edge_vec_rep = CueLinear("1x1o")(V_l1)
            edge_vec = edge_vec_rep.array.reshape(edge_vec_rep.array.shape[0], 3)
            node_vec = jax.ops.segment_sum(edge_vec, dst, num_segments=num_nodes)

            if self.degree_norm is not None:
                deg = jax.ops.segment_sum(
                    jnp.ones((dst.shape[0],), dtype=edge_vec.dtype),
                    dst,
                    num_segments=num_nodes,
                )
                if self.degree_norm == "deg_sqrt":
                    node_vec = node_vec * jax.lax.rsqrt(deg[:, None] + 1e-8)
                elif self.degree_norm == "deg":
                    node_vec = node_vec / (deg[:, None] + 1e-8)
                else:
                    raise ValueError(
                        f"Unknown fast-force degree_norm {self.degree_norm!r}. "
                        "Expected None, 'deg_sqrt', or 'deg'."
                    )

        return node_vec


class AllegroReadout(hk.Module):
    """Readout layer to produce per-edge energies.

    Uses baseline approach: concatenate x (scalar) with V (equivariant),
    then apply CueLinear to output 0e.
    """

    def __init__(
        self,
        output_n_hidden: int,
        output_n_layers: int,
        envelope_p: int = 6,
        output_activation = jax.nn.silu,
        name: Optional[str] = None
    ):
        """Initialize AllegroReadout.

        Args:
            output_n_hidden: Hidden size for output MLP
            output_n_layers: Number of output MLP layers
            envelope_p: Polynomial order for envelope
            output_activation: Activation function
            name: Module name
        """
        super().__init__(name=name)
        self.output_n_hidden = output_n_hidden
        self.output_n_layers = output_n_layers
        self.envelope_p = envelope_p
        self.output_activation = output_activation

    @staticmethod
    def _e3nn_style_linear_no_bias(
        x: jnp.ndarray,
        output_size: int,
        name: str,
    ) -> jnp.ndarray:
        """Apply a bias-free linear map with e3nn-style fan-in normalization."""
        input_size = x.shape[-1]
        alpha = 1.0 / float(max(input_size, 1))
        w = hk.get_parameter(
            f"{name}_w",
            shape=(input_size, output_size),
            init=hk.initializers.RandomNormal(stddev=1.0),
        )
        return jnp.sqrt(alpha) * jnp.matmul(x, w)

    @staticmethod
    def _extract_scalar_channels(rep: cuex.RepArray) -> jnp.ndarray:
        """Extract 0e channels from a RepArray in ir_mul layout."""
        irreps = rep.irreps if isinstance(rep.irreps, cue.Irreps) else rep.irreps.irreps
        offset = 0
        scalar_parts = []

        for mul, ir in irreps:
            width = mul * ir.dim
            chunk = rep.array[:, offset:offset + width]
            if ir.l == 0 and ir.p == 1:
                scalar_parts.append(chunk)
            offset += width

        if scalar_parts:
            return jnp.concatenate(scalar_parts, axis=-1)
        return jnp.zeros((rep.array.shape[0], 0), dtype=rep.array.dtype)

    def __call__(
        self,
        vectors: cuex.RepArray,
        x: jnp.ndarray,
        V: cuex.RepArray,
        return_intermediates: bool = False,
    ) -> cuex.RepArray:
        """Compute per-edge energies by concatenating x and V, then applying linear.

        Matches baseline: MLP on x, then linear(x + V) → 0e
        """
        n_edges = x.shape[0]

        for i in range(self.output_n_layers):
            x = self._e3nn_style_linear_no_bias(
                x,
                self.output_n_hidden,
                name=f"output_mlp_{i}",
            )
            if i < self.output_n_layers - 1:
                x = self.output_activation(x)

        with cue.assume(cue.O3, cue.ir_mul):
            x_irreps = cue.Irreps("O3", f"{x.shape[-1]}x0e")
            x_rep = cuex.RepArray(
                cue.IrrepsAndLayout(x_irreps, cue.ir_mul),
                x
            )

            V_irreps = V.irreps if isinstance(V.irreps, cue.Irreps) else V.irreps.irreps
            if hasattr(V.irreps, 'layout') and V.irreps.layout != cue.ir_mul:
                V = V.change_layout(cue.ir_mul)

            xV = cuex.concatenate([x_rep, V])
            h_in_array = xV.array

            scalar_features = self._extract_scalar_channels(xV)
            fan_in = scalar_features.shape[-1]

            if fan_in == 0:
                h_lin_array = jnp.zeros((n_edges, 1), dtype=xV.array.dtype)
                mu_dot_w = jnp.array(0.0, dtype=xV.array.dtype)
                scalar_mu_l2 = jnp.array(0.0, dtype=xV.array.dtype)
                scalar_mu_max_abs = jnp.array(0.0, dtype=xV.array.dtype)
            else:
                W = hk.get_parameter(
                    "final_linear_weights",
                    shape=(fan_in, 1),
                    init=hk.initializers.VarianceScaling(
                        1.0, "fan_in", "truncated_normal"
                    ),
                )
                h_lin_array = jnp.matmul(scalar_features, W)
                scalar_mu = jnp.mean(scalar_features, axis=0)
                scalar_mu_l2 = jnp.linalg.norm(scalar_mu)
                scalar_mu_max_abs = jnp.max(jnp.abs(scalar_mu))
                mu_dot_w = jnp.sum(scalar_mu[:, None] * W)

            output_irreps = cue.Irreps("O3", "0e")
            energies = cuex.RepArray(
                cue.IrrepsAndLayout(output_irreps, cue.ir_mul),
                h_lin_array,
            )

            distances = jnp.linalg.norm(vectors.array, axis=-1)
            envelope = polynomial_envelope(distances, p=self.envelope_p, cutoff=1.0)

            energies_array = energies.array * envelope[:, None]
            energies = cuex.RepArray(energies.irreps, energies_array)

        if return_intermediates:
            return energies, {
                "h_in": h_in_array,
                "h_lin": h_lin_array,
                "h_out": energies.array,
                "scalar_fan_in": int(fan_in),
                "scalar_mu_l2": float(scalar_mu_l2),
                "scalar_mu_max_abs": float(scalar_mu_max_abs),
                "mu_dot_w": float(mu_dot_w),
            }

        return energies


def compute_tensor_product_irreps(irreps1: cue.Irreps, irreps2: cue.Irreps) -> cue.Irreps:
    """Compute output irreps from tensor product of two irreps.

    Uses Clebsch-Gordan rules:
    - l1 ⊗ l2 → |l1 - l2|, |l1 - l2| + 1, ..., l1 + l2
    - parity: p_out = p1 * p2

    Args:
        irreps1: First set of irreps
        irreps2: Second set of irreps

    Returns:
        Output irreps (unique, sorted)
    """
    output_set = set()

    for mul1, ir1 in irreps1:
        for mul2, ir2 in irreps2:
            l_min = abs(ir1.l - ir2.l)
            l_max = ir1.l + ir2.l

            p_out = ir1.p * ir2.p

            for l_out in range(l_min, l_max + 1):
                output_set.add((l_out, p_out))

    output_list = sorted(output_set)
    irreps_str = " + ".join([f"{l}{'e' if p == 1 else 'o'}" for l, p in output_list])
    return cue.Irreps("O3", irreps_str)


def filter_layers(layer_irreps: List[cue.Irreps], max_ell: int) -> List[cue.Irreps]:
    """Filter irreps backward from output to input.

    Matches e3nn's filter_layers exactly:
    - Start from output
    - For each layer going backward: filter irreps by what can appear in
      tensor_product(next_layer_filtered, SH(lmax))
    - NO dropping of 0e here - that's done in forward pass via filter_ir_out
    
    Args:
        layer_irreps: List of irreps for each stage
        max_ell: Maximum l for spherical harmonics

    Returns:
        Filtered list of irreps
    """
    layer_irreps = list(layer_irreps)

    filtered = [layer_irreps[-1]]

    sh_irreps = cue.Irreps("O3", " + ".join([f"{l}{'e' if l % 2 == 0 else 'o'}"
                                               for l in range(max_ell + 1)]))

    for i, irreps in enumerate(reversed(layer_irreps[:-1])):
        next_layer_filtered = filtered[0]
        
        tp_irreps = compute_tensor_product_irreps(next_layer_filtered, sh_irreps)
        
        tp_irreps_regrouped = tp_irreps.regroup()
        
        def keep_fn(mul_ir):
            mul, ir = mul_ir
            for _, tp_ir in tp_irreps_regrouped:
                if int(ir.l) == int(tp_ir.l) and int(ir.p) == int(tp_ir.p):
                    return True
            return False
        
        filtered_irreps = irreps.filter(keep=keep_fn)
        
        if filtered_irreps.num_irreps == 0:
            filtered_irreps = cue.Irreps("O3", "0x0e")

        filtered.insert(0, filtered_irreps)

    return filtered


class Allegro(hk.Module):
    """Allegro model for molecular property prediction.

    Pure cuEquivariance implementation for maximum CUDA performance.
    """

    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        hidden_irreps: Union[cue.Irreps, str] = "128x0o + 128x1o + 128x1e + 128x2e + 128x2o + 128x3o + 128x3e",
        output_irreps: Union[cue.Irreps, str] = "0e",
        mlp_activation = jax.nn.silu,
        mlp_output_activation = None,
        mlp_n_hidden: int = 1024,
        mlp_n_layers: int = 3,
        embed_n_hidden: Iterable[int] = (64, 128, 256),
        species_embed: Optional[int] = None,
        num_species: int = 100,
        envelope_p: int = 6,
        n_radial_basis: int = 8,
        num_layers: int = 1,
        tp_backend: str = "baseline_mixed",
        tp_fused_option_b1_layer0: bool = False,
        tp_fused_option_b1_modes: Union[str, Iterable[str]] = "auto",
        tp_mode: str = "mixed_naive",
        tp_method: str = "naive",
        tp_method_by_layer: Optional[Iterable[str]] = None,
        tp_batch_strategy: str = "nested_vmap",
        tp_left_mode: Literal["node_agg", "edge_local"] = "node_agg",
        tp_left_norm: Optional[Literal["deg_sqrt", "deg"]] = None,
        remat_layers: bool = False,
        enable_fast_force_head: bool = False,
        fast_force_source: str = "layer0",
        fast_force_aggregate: Literal["receiver", "sender"] = "receiver",
        fast_force_degree_norm: Optional[Literal["deg_sqrt", "deg"]] = None,
        name: str = 'Allegro'
    ):
        """Initialize Allegro model.

        Args:
            avg_num_neighbors: Average number of neighbors (for normalization)
            max_ell: Maximum l for spherical harmonics
            hidden_irreps: Hidden layer irreps
            output_irreps: Output irreps (should be "0e" for energy)
            mlp_activation: Activation function
            mlp_output_activation: Output/readout activation function.
                If None, falls back to mlp_activation for backward compatibility.
            mlp_n_hidden: MLP hidden size
            mlp_n_layers: Number of MLP layers
            embed_n_hidden: Embedding MLP hidden sizes
            species_embed: Species embedding dimension
            num_species: Number of atomic species
            envelope_p: Polynomial order for envelope
            n_radial_basis: Number of radial basis functions
            num_layers: Number of AllegroLayers
            tp_backend: TP backend for AllegroLayer.
            tp_fused_option_b1_layer0: If True, enable Option B1 flattened-mode
                fused SP transform for layer 0.
            tp_fused_option_b1_modes: Coefficient modes to flatten for Option B1.
                Use "auto" to flatten all but the final canonical mode.
            tp_mode: Tensor-product mode for AllegroLayer.
            tp_method: TP method used by mixed mode.
            tp_method_by_layer: Optional per-layer TP method overrides.
                If provided, must have length == num_layers.
                Example: ["naive", "indexed_linear"].
            tp_batch_strategy: TP batching strategy in mixed mode.
            tp_left_mode: TP-left construction mode ("node_agg" or "edge_local").
            tp_left_norm: Optional degree normalization for edge_local TP-left.
            remat_layers: If True, rematerialize each AllegroLayer call during
                backward pass to reduce activation memory at the cost of extra
                recomputation.
            enable_fast_force_head: If True, instantiate optional fast force head.
            fast_force_source: Source for force head features: "embedding", "layer0", ...
            fast_force_aggregate: Node aggregation mode for fast force head.
            fast_force_degree_norm: Optional degree normalization in fast force head.
            name: Model name
        """
        super().__init__(name=name)

        if isinstance(hidden_irreps, str):
            hidden_irreps = cue.Irreps("O3", hidden_irreps)
        if isinstance(output_irreps, str):
            output_irreps = cue.Irreps("O3", output_irreps)

        self.output_irreps = output_irreps
        self.mlp_n_hidden = mlp_n_hidden
        self.enable_fast_force_head = bool(enable_fast_force_head)
        self.fast_force_source = fast_force_source
        self.remat_layers = bool(remat_layers)
        self.tp_fused_option_b1_layer0 = bool(tp_fused_option_b1_layer0)
        self.tp_fused_option_b1_modes = _parse_mode_csv(tp_fused_option_b1_modes)
        if self.tp_fused_option_b1_layer0 and self.tp_fused_option_b1_modes is None:
            raise ValueError(
                "tp_fused_option_b1_layer0=True requires non-empty tp_fused_option_b1_modes."
            )

        epsilon_init = jnp.sqrt(avg_num_neighbors)
        epsilon = hk.get_parameter(
            "varepsilon",
            shape=(),
            init=hk.initializers.Constant(epsilon_init)
        )
        self.epsilon = 1.0 / jnp.sqrt(1.0 + _mesh_safe_softplus(epsilon))

        self.alpha = hk.get_parameter(
            "residual_alpha",
            shape=(),
            init=hk.initializers.Constant(0.0)
        )

        layer_irreps = [hidden_irreps] * num_layers + [output_irreps]
        filtered_irreps = filter_layers(layer_irreps, max_ell)

        self.embedding_layer = AllegroEmbedding(
            num_species=num_species,
            embed_n_hidden=embed_n_hidden,
            species_embed=species_embed,
            n_radial_basis=n_radial_basis,
            envelope_p=envelope_p,
            mlp_n_hidden=mlp_n_hidden,
            irreps=filtered_irreps[0],  # Use filtered irreps for embedding
            mlp_activation=mlp_activation,
        )

        if tp_method_by_layer is None:
            layer_methods = [tp_method] * num_layers
        else:
            layer_methods = list(tp_method_by_layer)
            if len(layer_methods) != num_layers:
                raise ValueError(
                    f"tp_method_by_layer must have length {num_layers}, got {len(layer_methods)}."
                )

        self.layers = []
        for i in range(num_layers):
            layer_flatten_modes = (
                self.tp_fused_option_b1_modes
                if (self.tp_fused_option_b1_layer0 and i == 0 and tp_backend == "fused_sp")
                else None
            )
            self.layers.append(
                AllegroLayer(
                    epsilon=self.epsilon,
                    max_ell=max_ell,
                    output_irreps=filtered_irreps[i + 1],  # Use filtered irreps for this layer
                    mlp_n_hidden=mlp_n_hidden,
                    mlp_n_layers=mlp_n_layers,
                    p=envelope_p,
                    mlp_activation=mlp_activation,
                    tp_backend=tp_backend,
                    tp_fused_flatten_modes=layer_flatten_modes,
                    tp_mode=tp_mode,
                    tp_method=layer_methods[i],
                    tp_batch_strategy=tp_batch_strategy,
                    tp_left_mode=tp_left_mode,
                    tp_left_norm=tp_left_norm,
                    name=f"layer_{i}"
                )
            )

        self.readout_layer = AllegroReadout(
            output_n_hidden=mlp_n_hidden,
            output_n_layers=1,
            output_activation=(
                mlp_activation if mlp_output_activation is None else mlp_output_activation
            ),
            envelope_p=envelope_p
        )

        self.fast_force_head = (
            AllegroFastForceHead(
                aggregate=fast_force_aggregate,
                degree_norm=fast_force_degree_norm,
                name="fast_force_head",
            )
            if self.enable_fast_force_head
            else None
        )

    def __call__(
        self,
        vectors: cuex.RepArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray,
        num_nodes: int,
        return_fast_forces: bool = False,
        compute_energy: bool = True,
    ) -> Union[cuex.RepArray, jnp.ndarray, Tuple[cuex.RepArray, jnp.ndarray]]:
        """Predict per-edge energies and optionally fast per-node forces.

        Args:
            vectors: Edge vectors, RepArray [n_edges, 3], irreps "1o"
            senders: Sender node indices, [n_edges]
            receivers: Receiver node indices, [n_edges]
            species: Species per node, [n_nodes]
            num_nodes: Number of nodes (concrete value, not traced)
            return_fast_forces: If True, also compute fast per-node force head.
            compute_energy: If False, skip readout and only compute fast forces.

        Returns:
            Per-edge energies, or per-node fast forces, or both.
        """
        if return_fast_forces and self.fast_force_head is None:
            raise ValueError(
                "return_fast_forces=True requires enable_fast_force_head=True in Allegro init."
            )

        x, V = self.embedding_layer(vectors, senders, receivers, species)
        source_V = V if self.fast_force_source == "embedding" else None

        for i, layer in enumerate(self.layers):
            if self.remat_layers:
                y, V_new = hk.remat(
                    lambda vectors_, x_, V_, senders_, species_, layer=layer: layer(
                        vectors_, x_, V_, senders_, species_, num_nodes
                    )
                )(vectors, x, V, senders, species)
            else:
                y, V_new = layer(vectors, x, V, senders, species, num_nodes)

            alpha = _mesh_safe_softplus(self.alpha)
            x = (x + alpha * y) / (1.0 + alpha)
            V = V_new
            if self.fast_force_source == f"layer{i}":
                source_V = V

        fast_forces = None
        if return_fast_forces:
            if source_V is None:
                source_V = V
            fast_forces = self.fast_force_head(source_V, senders, receivers, num_nodes)

        if compute_energy:
            energies = self.readout_layer(vectors, x, V)
            if return_fast_forces:
                return energies, fast_forces
            return energies

        if return_fast_forces:
            return fast_forces

        raise ValueError("At least one of compute_energy or return_fast_forces must be True.")


def allegro_neighborlist_pp(
    displacement: space.DisplacementFn,
    r_cutoff: float,
    n_species: int = 100,
    positions_test: jnp.ndarray = None,
    neighbor_test: partition.NeighborList = None,
    max_edge_multiplier: float = 1.1,
    max_edges=None,
    avg_num_neighbors: float = None,
    mode: str = "energy",
    per_particle: bool = False,
    positive_species: bool = False,
    logging: bool = True,
    **allegro_kwargs
):
    """Allegro model wrapper for neighbor list-based energy prediction.

    Args:
        displacement: JAX-MD displacement function
        r_cutoff: Cutoff radius for neighbor list
        n_species: Number of atomic species
        positions_test: Unused compatibility placeholder (matches cuEq wrapper API).
        neighbor_test: Unused compatibility placeholder (matches cuEq wrapper API).
        max_edge_multiplier: Unused compatibility placeholder (matches cuEq wrapper API).
        max_edges: Unused compatibility placeholder (matches cuEq wrapper API).
        avg_num_neighbors: Average neighbors (required)
        mode: Prediction mode:
            - "energy": total energy only
            - "energy_and_fast_forces": total energy + fast per-node forces
            - "fast_forces": fast per-node forces only
        per_particle: If True in energy mode, return per-particle energies.
        positive_species: If True, species IDs start from 1 (subtract 1)
        logging: Unused compatibility placeholder (matches cuEq wrapper API).
        **allegro_kwargs: Additional arguments for Allegro model

    Returns:
        init_fn: Parameter initialization function
        apply_fn: Energy evaluation function
    """
    r_cutoff = jnp.array(r_cutoff, dtype=jnp.float32)

    assert avg_num_neighbors is not None, "avg_num_neighbors is required"
    if mode not in ("energy", "energy_and_fast_forces", "fast_forces"):
        raise NotImplementedError(f"Mode {mode} not implemented")

    # Keep wrapper API compatible with allegro_cueq_v2 and ignore wrapper-only args.
    _ = positions_test, neighbor_test, max_edge_multiplier, max_edges, logging
    allegro_kwargs = dict(allegro_kwargs)
    allegro_kwargs.pop("positions_test", None)
    allegro_kwargs.pop("neighbor_test", None)
    allegro_kwargs.pop("max_edge_multiplier", None)
    allegro_kwargs.pop("max_edges", None)
    allegro_kwargs.pop("logging", None)
    allegro_kwargs.pop("mlp_dtype", None)
    allegro_kwargs.pop("num_types", None)
    if "mlp_activation" not in allegro_kwargs and "mlp_hidden_activation" in allegro_kwargs:
        allegro_kwargs["mlp_activation"] = allegro_kwargs["mlp_hidden_activation"]
    allegro_kwargs.pop("mlp_hidden_activation", None)
    if mode in ("energy_and_fast_forces", "fast_forces"):
        allegro_kwargs.setdefault("enable_fast_force_head", True)
    # Honor config-provided activations while keeping historical defaults.
    allegro_kwargs.setdefault("mlp_activation", jax.nn.mish)
    allegro_kwargs.setdefault("mlp_output_activation", None)
    # Drop unknown keys so this strict backend remains compatible with
    # config dictionaries used by the more permissive cuEq wrapper.
    allowed_init_keys = {
        key
        for key in inspect.signature(Allegro.__init__).parameters
        if key not in {"self", "avg_num_neighbors", "num_species"}
    }
    allegro_kwargs = {
        key: value for key, value in allegro_kwargs.items() if key in allowed_init_keys
    }

    @hk.without_apply_rng
    @hk.transform
    def model(
        position: jnp.ndarray,
        neighbor: partition.NeighborList,
        species: jnp.ndarray = None,
        mask: jnp.ndarray = None,
        **dynamic_kwargs
    ):
        """Model function compatible with JAX-MD."""
        n_nodes = position.shape[0]

        if species is None:
            species = jnp.zeros(n_nodes, dtype=jnp.int32)
        elif positive_species:
            species = species - 1

        if mask is None:
            mask = jnp.ones(n_nodes, dtype=jnp.bool_)

        dyn_displacement = lambda Ra, Rb: displacement(Ra, Rb, **dynamic_kwargs)

        if neighbor.format == partition.Sparse:
            receivers, senders = neighbor.idx
            receivers = jnp.asarray(receivers, dtype=jnp.int32)
            senders = jnp.asarray(senders, dtype=jnp.int32)
        elif neighbor.format == partition.Dense:
            dense_idx = jnp.asarray(neighbor.idx, dtype=jnp.int32)
            if dense_idx.ndim != 2:
                raise ValueError(
                    f"Dense neighbor idx must be rank-2, got shape={dense_idx.shape}."
                )
            n_centers, n_slots = dense_idx.shape
            senders = jnp.repeat(jnp.arange(n_centers, dtype=jnp.int32), n_slots)
            receivers = dense_idx.reshape(-1)
        else:
            raise NotImplementedError(
                f"Unsupported neighbor list format: {neighbor.format!r}. "
                "Expected partition.Sparse or partition.Dense."
            )

        valid_edges = jnp.logical_and(
            jnp.logical_and(senders >= 0, senders < n_nodes),
            jnp.logical_and(receivers >= 0, receivers < n_nodes),
        )
        senders_safe = jnp.where(valid_edges, senders, 0)
        receivers_safe = jnp.where(valid_edges, receivers, 0)
        vectors = jax.vmap(dyn_displacement)(
            position[senders_safe],
            position[receivers_safe],
        )
        fallback_vec = jnp.array([r_cutoff, 0.0, 0.0], dtype=vectors.dtype)
        vectors = jnp.where(valid_edges[:, None], vectors, fallback_vec)

        vectors = vectors / r_cutoff

        vector_irreps = cue.IrrepsAndLayout(cue.Irreps("O3", "1o"), cue.ir_mul)
        vectors_rep = cuex.RepArray(vector_irreps, vectors)

        net = Allegro(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            **allegro_kwargs
        )

        if mode == "energy":
            per_edge_energies = net(
                vectors_rep,
                senders,
                receivers,
                species,
                n_nodes,
                return_fast_forces=False,
                compute_energy=True,
            )
            per_node_energies = jax.ops.segment_sum(
                per_edge_energies.array.squeeze(-1),
                senders,
                num_segments=n_nodes
            )

            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)

            per_atom_energies = per_atom_energies * mask

            if per_particle:
                return per_atom_energies
            return md_util.high_precision_sum(per_atom_energies)

        if mode == "energy_and_fast_forces":
            per_edge_energies, fast_forces = net(
                vectors_rep,
                senders,
                receivers,
                species,
                n_nodes,
                return_fast_forces=True,
                compute_energy=True,
            )
            per_node_energies = jax.ops.segment_sum(
                per_edge_energies.array.squeeze(-1),
                senders,
                num_segments=n_nodes
            )
            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies = per_atom_energies * mask
            total_energy = md_util.high_precision_sum(per_atom_energies)
            return total_energy, fast_forces

        fast_forces = net(
            vectors_rep,
            senders,
            receivers,
            species,
            n_nodes,
            return_fast_forces=True,
            compute_energy=False,
        )
        return fast_forces

    return jax.jit(model.init), jax.jit(model.apply)
