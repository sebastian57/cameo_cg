"""Lean cuEquivariance Allegro backend for the normal energy-training path.

This module intentionally contains only the representation, tensor-product,
readout, and neighbor-list code used by the standard energy model.  The older
``allegro_cueq_fast_1103`` module remains the compatibility backend for
experimental force and feature modes.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

import cuequivariance as cue
import cuequivariance_jax as cuex
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from jax_md import partition, space
from jax_md import util as md_util

from training.edge_distance_gate import compute_edge_distance_gate


_CUEQ_HELPER_DIR = Path(__file__).resolve().parents[2] / "cueq_allegro"


def _load_helper_module(module_stem: str):
    helper_file = _CUEQ_HELPER_DIR / f"{module_stem}.py"
    if not helper_file.is_file():
        raise ImportError(f"Missing helper module file: {helper_file}")
    module_name = f"_cueq_fast_clean_helper_{module_stem}"
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
AtomicEnergyLayer = _layers_cueq_mod.AtomicEnergyLayer
polynomial_envelope = _utils_mod.polynomial_envelope
segment_sum_map_back = _utils_mod.segment_sum_map_back


def _sorted_unique_irreps(irreps: cue.Irreps) -> List[Any]:
    unique = {(int(ir.l), int(ir.p)): ir for _mul, ir in irreps}
    return [unique[key] for key in sorted(unique)]


def _mesh_safe_softplus(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.maximum(x, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def _expand_spherical_harmonics(
    base_Y: cuex.RepArray,
    target_irreps: cue.Irreps,
) -> cuex.RepArray:
    """Expand one-copy spherical harmonics into the embedding representation."""
    base_irreps = (
        base_Y.irreps
        if isinstance(base_Y.irreps, cue.Irreps)
        else base_Y.irreps.irreps
    )
    offsets: dict[int, int] = {}
    offset = 0
    for _mul, ir in base_irreps:
        offsets[int(ir.l)] = offset
        offset += int(ir.dim)

    parts = []
    terms = []
    for mul, ir in target_irreps:
        start = offsets[int(ir.l)]
        width = int(ir.dim)
        part = base_Y.array[:, start : start + width]
        parts.append(jnp.repeat(part, int(mul), axis=-1))
        parity = "e" if int(ir.p) == 1 else "o"
        terms.append(f"{int(mul)}x{int(ir.l)}{parity}")

    if parts:
        array = jnp.concatenate(parts, axis=-1)
    else:
        array = jnp.zeros((base_Y.array.shape[0], 0), dtype=base_Y.array.dtype)
    irreps = cue.Irreps("O3", " + ".join(terms) if terms else "0e")
    return cuex.RepArray(cue.IrrepsAndLayout(irreps, cue.ir_mul), array)


def build_tp_left_input(
    Y: jnp.ndarray,
    a: jnp.ndarray,
    senders: jnp.ndarray,
    eps: jax.Array,
    n_nodes: int,
) -> jnp.ndarray:
    """Build the standard sender-node aggregated TP-left input."""
    valid_sender = jnp.logical_and(senders >= 0, senders < n_nodes)
    senders_safe = jnp.where(valid_sender, senders, 0)
    left = a[:, :, None] * Y[:, None, :]
    left = jnp.where(valid_sender[:, None, None], left, 0.0)
    return segment_sum_map_back(left, senders_safe, n_nodes) * eps


class AllegroEmbedding(hk.Module):
    """Initial radial, species, and spherical-harmonic feature embedding."""

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
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if isinstance(irreps, str):
            irreps = cue.Irreps("O3", irreps)

        hidden = list(embed_n_hidden)
        if species_embed is None:
            species_embed = hidden[0] // 2

        self.irreps = irreps
        self.envelope_p = envelope_p
        self.mlp_activation = mlp_activation
        self.species_embed_dim = species_embed
        self.radial_basis = RadialBesselLayer(
            cutoff=1.0,
            num_radial=n_radial_basis,
            envelope_p=envelope_p,
        )
        self.species_embedding = hk.Embed(num_species, species_embed)
        self.embed_layers = hidden + [mlp_n_hidden]

    def __call__(
        self,
        vectors: cuex.RepArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray,
        distances: jnp.ndarray,
        base_Y: cuex.RepArray,
    ) -> Tuple[jnp.ndarray, cuex.RepArray]:
        radial_features = self.radial_basis(distances)
        species_sender = self.species_embedding(species[senders])
        species_receiver = self.species_embedding(species[receivers])

        x = jnp.concatenate(
            [radial_features, species_sender, species_receiver],
            axis=-1,
        )
        x = e3nn.haiku.MultiLayerPerceptron(
            self.embed_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)
        x = polynomial_envelope(
            distances,
            p=self.envelope_p,
            cutoff=1.0,
        )[:, None] * x

        irreps_Y = self._filter_irreps_by_parity(vectors)
        with cue.assume(cue.O3, cue.ir_mul):
            Y = _expand_spherical_harmonics(base_Y, irreps_Y)
            species_irreps = cue.Irreps("O3", f"{self.species_embed_dim}x0e")
            species_sender_rep = cuex.RepArray(
                cue.IrrepsAndLayout(species_irreps, cue.ir_mul),
                species_sender,
            )
            species_receiver_rep = cuex.RepArray(
                cue.IrrepsAndLayout(species_irreps, cue.ir_mul),
                species_receiver,
            )
            V = cuex.concatenate([Y, species_sender_rep, species_receiver_rep])

            V_irreps = V.irreps if isinstance(V.irreps, cue.Irreps) else V.irreps.irreps
            num_irreps = sum(int(mul) for mul, _ir in V_irreps)
            w = e3nn.haiku.MultiLayerPerceptron(
                (num_irreps,),
                None,
                output_activation=False,
            )(x)

            n_edges = x.shape[0]
            parts = []
            v_offset = 0
            w_offset = 0
            for mul, ir in V_irreps:
                ir_dim = int(ir.dim)
                width = int(mul) * ir_dim
                block = V.array[:, v_offset : v_offset + width]
                block = block.reshape(n_edges, ir_dim, int(mul))
                weights = w[:, w_offset : w_offset + int(mul)]
                block = (block * weights[:, None, :]) / num_irreps
                parts.append(block.reshape(n_edges, width))
                v_offset += width
                w_offset += int(mul)
            V = cuex.RepArray(V.irreps, jnp.concatenate(parts, axis=-1))

        return x, V

    def _filter_irreps_by_parity(self, vectors: cuex.RepArray) -> cue.Irreps:
        vector_parity = 1 if "1o" in str(vectors.irreps) else -1
        filtered = []
        for mul, ir in self.irreps:
            if ((-1) ** int(ir.l)) * vector_parity == int(ir.p):
                filtered.append((mul, ir))

        terms = [
            f"{int(mul)}x{int(ir.l)}{'e' if int(ir.p) == 1 else 'o'}"
            for mul, ir in filtered
        ]
        return cue.Irreps("O3", " + ".join(terms) if terms else "0e")


class AllegroLayer(hk.Module):
    """One standard mixed-irrep Allegro tensor-product layer."""

    def __init__(
        self,
        epsilon: jax.Array,
        max_ell: int,
        output_irreps: Union[cue.Irreps, str],
        mlp_n_hidden: int,
        mlp_n_layers: int,
        p: int = 6,
        mlp_activation: Callable = jax.nn.silu,
        tp_backend: str = "baseline_mixed",
        tp_mode: str = "mixed_naive",
        tp_method: str = "naive",
        tp_batch_strategy: str = "nested_vmap",
        tp_left_mode: Literal["node_agg"] = "node_agg",
        tp_left_norm: None = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if isinstance(output_irreps, str):
            output_irreps = cue.Irreps("O3", output_irreps)
        requested = {
            "tp_backend": tp_backend,
            "tp_mode": tp_mode,
            "tp_method": tp_method,
            "tp_batch_strategy": tp_batch_strategy,
            "tp_left_mode": tp_left_mode,
        }
        expected = {
            "tp_backend": "baseline_mixed",
            "tp_mode": "mixed_naive",
            "tp_method": "naive",
            "tp_batch_strategy": "nested_vmap",
            "tp_left_mode": "node_agg",
        }
        for key, value in requested.items():
            if value != expected[key]:
                raise ValueError(
                    f"Clean Allegro backend requires {key}={expected[key]!r}; got {value!r}."
                )
        if tp_left_norm is not None:
            raise ValueError("Clean Allegro backend requires tp_left_norm=None.")

        self.epsilon = epsilon
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = p
        self.mlp_activation = mlp_activation

    def _build_tp_descriptor(
        self,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
    ) -> cue.EquivariantPolynomial:
        filter_irreps = _sorted_unique_irreps(self.output_irreps)
        if not any(int(ir.l) == 0 and int(ir.p) == 1 for ir in filter_irreps):
            filter_irreps.append(cue.Irreps("O3", "0e")[0][1])
        return cue.descriptors.full_tensor_product(
            Y_irreps,
            V_red_irreps,
            irreps3_filter=filter_irreps,
        )

    def _tensor_product_mixed(
        self,
        wY_axis: jnp.ndarray,
        V_axis: jnp.ndarray,
        Y_irreps: cue.Irreps,
        V_red_irreps: cue.Irreps,
        tp_desc: cue.EquivariantPolynomial,
    ) -> Tuple[jnp.ndarray, cue.Irreps]:
        output_irreps = (
            tp_desc.outputs[0].irreps
            if hasattr(tp_desc.outputs[0], "irreps")
            else tp_desc.outputs[0]
        )

        def apply_one(Y_slice: jnp.ndarray, V_slice: jnp.ndarray) -> jnp.ndarray:
            Y_rep = cuex.RepArray(
                cue.IrrepsAndLayout(Y_irreps, cue.ir_mul),
                Y_slice,
            )
            V_rep = cuex.RepArray(
                cue.IrrepsAndLayout(V_red_irreps, cue.ir_mul),
                V_slice,
            )
            result = cuex.equivariant_polynomial(
                tp_desc,
                [Y_rep, V_rep],
                method="naive",
            )
            if isinstance(result, list):
                result = result[0]
            return result.array

        apply_axis = jax.vmap(jax.vmap(apply_one, in_axes=0), in_axes=0)
        return apply_axis(wY_axis, V_axis), output_irreps

    def __call__(
        self,
        base_Y: cuex.RepArray,
        distances: jnp.ndarray,
        x: jnp.ndarray,
        V: cuex.RepArray,
        senders: jnp.ndarray,
        num_nodes: int,
    ) -> Tuple[jnp.ndarray, cuex.RepArray]:
        compute_mul_gcd = _utils_mod.compute_mul_gcd
        extract_and_filter_scalars = _utils_mod.extract_and_filter_scalars
        flatten_axis_to_mul = _utils_mod.flatten_axis_to_mul
        unflatten_mul_to_axis = _utils_mod.unflatten_mul_to_axis

        mul_gcd = compute_mul_gcd(V.irreps)
        weights = e3nn.haiku.MultiLayerPerceptron(
            (mul_gcd,),
            None,
            output_activation=False,
        )(x)

        with cue.assume(cue.O3, cue.ir_mul):
            Y = base_Y
            Y_irreps = Y.irreps if isinstance(Y.irreps, cue.Irreps) else Y.irreps.irreps
            wY_axis = build_tp_left_input(
                Y=Y.array,
                a=weights,
                senders=senders,
                eps=self.epsilon,
                n_nodes=num_nodes,
            )
            V_axis, V_red_irreps = unflatten_mul_to_axis(
                V.array,
                V.irreps,
                mul_gcd,
            )
            tp_desc = self._build_tp_descriptor(Y_irreps, V_red_irreps)
            out_axis, output_irreps = self._tensor_product_mixed(
                wY_axis,
                V_axis,
                Y_irreps,
                V_red_irreps,
                tp_desc,
            )
            V_new_array, V_new_irreps = flatten_axis_to_mul(
                out_axis,
                output_irreps,
                mul_gcd,
            )
            V_new = cuex.RepArray(
                cue.IrrepsAndLayout(V_new_irreps, cue.ir_mul),
                V_new_array,
            )
            x_new, V_filtered = extract_and_filter_scalars(x, V_new)
            V_out = CueLinear(self.output_irreps)(V_filtered)

        y = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(x_new)
        envelope = polynomial_envelope(
            distances,
            p=self.envelope_p,
            cutoff=1.0,
        )
        return envelope[:, None] * y, V_out


class AllegroReadout(hk.Module):
    """Single linear scalar readout used by the normal energy model."""

    def __init__(
        self,
        output_n_hidden: int,
        output_n_layers: int = 1,
        envelope_p: int = 6,
        output_activation: Callable = jax.nn.silu,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if output_n_layers != 1:
            raise ValueError("Clean Allegro readout requires output_n_layers=1.")
        self.output_n_hidden = output_n_hidden
        self.envelope_p = envelope_p

    @staticmethod
    def _extract_scalar_channels(rep: cuex.RepArray) -> jnp.ndarray:
        irreps = rep.irreps if isinstance(rep.irreps, cue.Irreps) else rep.irreps.irreps
        offset = 0
        parts = []
        for mul, ir in irreps:
            width = int(mul) * int(ir.dim)
            chunk = rep.array[:, offset : offset + width]
            if int(ir.l) == 0 and int(ir.p) == 1:
                parts.append(chunk)
            offset += width
        if parts:
            return jnp.concatenate(parts, axis=-1)
        return jnp.zeros((rep.array.shape[0], 0), dtype=rep.array.dtype)

    def __call__(
        self,
        distances: jnp.ndarray,
        x: jnp.ndarray,
        V: cuex.RepArray,
    ) -> cuex.RepArray:
        input_size = x.shape[-1]
        alpha = 1.0 / float(max(input_size, 1))
        mlp_weights = hk.get_parameter(
            "output_mlp_0_w",
            shape=(input_size, self.output_n_hidden),
            init=hk.initializers.RandomNormal(stddev=1.0),
        )
        x = jnp.sqrt(alpha) * jnp.matmul(x, mlp_weights)

        with cue.assume(cue.O3, cue.ir_mul):
            x_irreps = cue.Irreps("O3", f"{x.shape[-1]}x0e")
            x_rep = cuex.RepArray(
                cue.IrrepsAndLayout(x_irreps, cue.ir_mul),
                x,
            )
            if hasattr(V.irreps, "layout") and V.irreps.layout != cue.ir_mul:
                V = V.change_layout(cue.ir_mul)
            xV = cuex.concatenate([x_rep, V])
            scalar_features = self._extract_scalar_channels(xV)
            fan_in = scalar_features.shape[-1]
            if fan_in == 0:
                h_linear = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
            else:
                final_weights = hk.get_parameter(
                    "final_linear_weights",
                    shape=(fan_in, 1),
                    init=hk.initializers.VarianceScaling(
                        1.0,
                        "fan_in",
                        "truncated_normal",
                    ),
                )
                h_linear = jnp.matmul(scalar_features, final_weights)

            output_irreps = cue.Irreps("O3", "0e")
            energies = cuex.RepArray(
                cue.IrrepsAndLayout(output_irreps, cue.ir_mul),
                h_linear,
            )
            envelope = polynomial_envelope(
                distances,
                p=self.envelope_p,
                cutoff=1.0,
            )
            return cuex.RepArray(
                energies.irreps,
                energies.array * envelope[:, None],
            )


def compute_tensor_product_irreps(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
) -> cue.Irreps:
    output_set = set()
    for _mul1, ir1 in irreps1:
        for _mul2, ir2 in irreps2:
            for l_out in range(abs(int(ir1.l) - int(ir2.l)), int(ir1.l) + int(ir2.l) + 1):
                output_set.add((l_out, int(ir1.p) * int(ir2.p)))
    terms = [
        f"{l}{'e' if parity == 1 else 'o'}"
        for l, parity in sorted(output_set)
    ]
    return cue.Irreps("O3", " + ".join(terms))


def filter_layers(layer_irreps: List[cue.Irreps], max_ell: int) -> List[cue.Irreps]:
    filtered = [layer_irreps[-1]]
    sh_irreps = cue.Irreps(
        "O3",
        " + ".join(
            f"{l}{'e' if l % 2 == 0 else 'o'}"
            for l in range(max_ell + 1)
        ),
    )
    for irreps in reversed(layer_irreps[:-1]):
        possible = compute_tensor_product_irreps(filtered[0], sh_irreps).regroup()

        def keep_fn(mul_ir):
            _mul, ir = mul_ir
            return any(
                int(ir.l) == int(possible_ir.l)
                and int(ir.p) == int(possible_ir.p)
                for _possible_mul, possible_ir in possible
            )

        kept = irreps.filter(keep=keep_fn)
        filtered.insert(0, kept if kept.num_irreps else cue.Irreps("O3", "0x0e"))
    return filtered


class Allegro(hk.Module):
    """Lean Allegro model with the production mixed-naive TP path."""

    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        hidden_irreps: Union[cue.Irreps, str] = "128x0o + 128x1o + 128x1e + 128x2e + 128x2o + 128x3o + 128x3e",
        output_irreps: Union[cue.Irreps, str] = "0e",
        mlp_activation: Callable = jax.nn.silu,
        mlp_output_activation: Optional[Callable] = None,
        mlp_n_hidden: int = 1024,
        mlp_n_layers: int = 3,
        embed_n_hidden: Iterable[int] = (64, 128, 256),
        species_embed: Optional[int] = None,
        num_species: int = 100,
        envelope_p: int = 6,
        n_radial_basis: int = 8,
        num_layers: int = 1,
        tp_backend: str = "baseline_mixed",
        tp_mode: str = "mixed_naive",
        tp_method: str = "naive",
        tp_method_by_layer: Optional[Iterable[str]] = None,
        tp_batch_strategy: str = "nested_vmap",
        tp_left_mode: Literal["node_agg"] = "node_agg",
        tp_left_norm: None = None,
        remat_layers: bool = False,
        name: str = "Allegro",
    ):
        super().__init__(name=name)
        if isinstance(hidden_irreps, str):
            hidden_irreps = cue.Irreps("O3", hidden_irreps)
        if isinstance(output_irreps, str):
            output_irreps = cue.Irreps("O3", output_irreps)

        if tp_method_by_layer is None:
            layer_methods = [tp_method] * num_layers
        else:
            layer_methods = list(tp_method_by_layer)
            if len(layer_methods) != num_layers:
                raise ValueError(
                    f"tp_method_by_layer must have length {num_layers}, got {len(layer_methods)}."
                )

        epsilon_init = jnp.sqrt(avg_num_neighbors)
        epsilon = hk.get_parameter(
            "varepsilon",
            shape=(),
            init=hk.initializers.Constant(epsilon_init),
        )
        epsilon = 1.0 / jnp.sqrt(1.0 + _mesh_safe_softplus(epsilon))
        self.alpha = hk.get_parameter(
            "residual_alpha",
            shape=(),
            init=hk.initializers.Constant(0.0),
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
            irreps=filtered_irreps[0],
            mlp_activation=mlp_activation,
        )
        self.layers = [
            AllegroLayer(
                epsilon=epsilon,
                max_ell=max_ell,
                output_irreps=filtered_irreps[i + 1],
                mlp_n_hidden=mlp_n_hidden,
                mlp_n_layers=mlp_n_layers,
                p=envelope_p,
                mlp_activation=mlp_activation,
                tp_backend=tp_backend,
                tp_mode=tp_mode,
                tp_method=layer_methods[i],
                tp_batch_strategy=tp_batch_strategy,
                tp_left_mode=tp_left_mode,
                tp_left_norm=tp_left_norm,
                name=f"layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.readout_layer = AllegroReadout(
            output_n_hidden=mlp_n_hidden,
            output_n_layers=1,
            output_activation=(
                mlp_activation if mlp_output_activation is None else mlp_output_activation
            ),
            envelope_p=envelope_p,
        )
        self.remat_layers = bool(remat_layers)
        self.max_ell = max_ell

    def __call__(
        self,
        vectors: cuex.RepArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray,
        num_nodes: int,
    ) -> cuex.RepArray:
        distances = jnp.linalg.norm(vectors.array, axis=-1)
        with cue.assume(cue.O3, cue.ir_mul):
            base_Y = cuex.spherical_harmonics(
                list(range(self.max_ell + 1)),
                vectors,
                normalize=True,
            )
            x, V = self.embedding_layer(
                vectors,
                senders,
                receivers,
                species,
                distances,
                base_Y,
            )

        for layer in self.layers:
            if self.remat_layers:
                x_update, V_new = hk.remat(
                    lambda vectors_, distances_, x_, V_, layer_=layer: layer_(
                        base_Y,
                        distances_,
                        x_,
                        V_,
                        senders,
                        num_nodes,
                    )
                )(vectors, distances, x, V)
            else:
                x_update, V_new = layer(
                    base_Y,
                    distances,
                    x,
                    V,
                    senders,
                    num_nodes,
                )
            alpha = _mesh_safe_softplus(self.alpha)
            x = (x + alpha * x_update) / (1.0 + alpha)
            V = V_new

        return self.readout_layer(distances, x, V)


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
    edge_distance_gate=None,
    **allegro_kwargs,
):
    """Build the normal energy/per-particle neighbor-list functions."""
    if mode != "energy":
        raise NotImplementedError(
            "The clean Allegro backend implements mode='energy' only."
        )
    if avg_num_neighbors is None:
        raise AssertionError("avg_num_neighbors is required")

    _ = positions_test, neighbor_test, max_edge_multiplier, max_edges, logging
    allegro_kwargs = dict(allegro_kwargs)
    for key in (
        "positions_test",
        "neighbor_test",
        "max_edge_multiplier",
        "max_edges",
        "logging",
        "mlp_dtype",
        "num_types",
    ):
        allegro_kwargs.pop(key, None)
    if "mlp_activation" not in allegro_kwargs and "mlp_hidden_activation" in allegro_kwargs:
        allegro_kwargs["mlp_activation"] = allegro_kwargs["mlp_hidden_activation"]
    allegro_kwargs.pop("mlp_hidden_activation", None)
    allegro_kwargs.setdefault("mlp_activation", jax.nn.mish)
    allegro_kwargs.setdefault("mlp_output_activation", None)

    allowed_init_keys = {
        key
        for key in inspect.signature(Allegro.__init__).parameters
        if key not in {"self", "avg_num_neighbors", "num_species"}
    }
    allegro_kwargs = {
        key: value
        for key, value in allegro_kwargs.items()
        if key in allowed_init_keys
    }

    @hk.without_apply_rng
    @hk.transform
    def model(
        position: jnp.ndarray,
        neighbor: partition.NeighborList,
        species: jnp.ndarray = None,
        mask: jnp.ndarray = None,
        **dynamic_kwargs,
    ):
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
                f"Unsupported neighbor list format: {neighbor.format!r}."
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

        vector_irreps = cue.IrrepsAndLayout(
            cue.Irreps("O3", "1o"),
            cue.ir_mul,
        )
        vectors_rep = cuex.RepArray(vector_irreps, vectors)
        net = Allegro(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            **allegro_kwargs,
        )
        per_edge_energies = net(
            vectors_rep,
            senders,
            receivers,
            species,
            n_nodes,
        )
        per_edge_values = per_edge_energies.array.squeeze(-1)

        if edge_distance_gate is not None:
            if getattr(edge_distance_gate, "has_ala2_combined_gate", False):
                raise NotImplementedError(
                    "The clean Allegro backend supports distance-only gates, "
                    "not latent feature gates."
                )
            distances = jnp.linalg.norm(vectors, axis=-1)
            edge_alpha = compute_edge_distance_gate(
                distances=distances,
                senders=senders,
                receivers=receivers,
                species=species,
                valid_edges=valid_edges,
                bank=edge_distance_gate,
                positions=position,
            )
            per_edge_values = per_edge_values * edge_alpha

        per_node_energies = jax.ops.segment_sum(
            per_edge_values,
            senders,
            num_segments=n_nodes,
        )
        per_atom_energies = AtomicEnergyLayer(n_species)(
            per_node_energies,
            species,
        )
        per_atom_energies = per_atom_energies * mask
        if per_particle:
            return per_atom_energies
        return md_util.high_precision_sum(per_atom_energies)

    return jax.jit(model.init), jax.jit(model.apply)
