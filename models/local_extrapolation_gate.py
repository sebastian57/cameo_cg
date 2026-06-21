from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def smoothstep_gate(scores, *, onset: float, offset: float):
    width = jnp.maximum(jnp.asarray(float(offset) - float(onset), dtype=jnp.float32), 1.0e-6)
    t = jnp.clip((jnp.asarray(scores, dtype=jnp.float32) - float(onset)) / width, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - smooth).astype(jnp.float32)


def build_jax_geometric_descriptor(R, mask, *, cutoff: float = 10.0, num_rbf: int = 8):
    """Build the 20D JAX geometric local-env descriptor for every bead center."""
    R = jnp.asarray(R, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    n = R.shape[0]
    cutoff_f = float(cutoff)
    eps = jnp.asarray(1.0e-6, dtype=jnp.float32)

    rel = R[None, :, :] - R[:, None, :]
    dist = jnp.sqrt(jnp.sum(rel * rel, axis=-1) + eps)
    valid_pair = (mask[:, None] > 0.5) & (mask[None, :] > 0.5) & (~jnp.eye(n, dtype=bool)) & (dist <= cutoff_f)
    neigh_mask = valid_pair.astype(jnp.float32)
    mask3 = neigh_mask[..., None]

    count = jnp.sum(neigh_mask, axis=1, keepdims=True)
    safe_count = jnp.maximum(count, 1.0)
    dist_masked = dist * neigh_mask

    radial_mean = jnp.sum(dist_masked, axis=1, keepdims=True) / safe_count
    radial_var = jnp.sum(((dist - radial_mean) * neigh_mask) ** 2, axis=1, keepdims=True) / safe_count
    radial_std = jnp.sqrt(radial_var + eps)
    radial_min = jnp.min(jnp.where(neigh_mask > 0.5, dist, cutoff_f), axis=1, keepdims=True)
    radial_max = jnp.max(jnp.where(neigh_mask > 0.5, dist, 0.0), axis=1, keepdims=True)
    radial_stats = jnp.concatenate([count, radial_mean, radial_std, radial_min, radial_max], axis=1)

    centers = jnp.linspace(0.0, cutoff_f, int(num_rbf), dtype=jnp.float32)
    sigma = jnp.asarray(cutoff_f / max(int(num_rbf) - 1, 1), dtype=jnp.float32)
    rbf = jnp.exp(-0.5 * ((dist[..., None] - centers[None, None, :]) / jnp.maximum(sigma, eps)) ** 2) * neigh_mask[..., None]
    rbf_density = jnp.sum(rbf, axis=1) / safe_count

    safe_dist = jnp.maximum(dist, 0.2)
    inv2 = jnp.sum(jnp.where(neigh_mask > 0.5, (1.0 / safe_dist) ** 2, 0.0), axis=1, keepdims=True)
    inv6 = jnp.sum(jnp.where(neigh_mask > 0.5, (1.0 / safe_dist) ** 6, 0.0), axis=1, keepdims=True)
    soft_clash = jnp.sum(jnp.where(neigh_mask > 0.5, jnp.exp(-dist / 0.35), 0.0), axis=1, keepdims=True)
    clash = jnp.concatenate([inv2, inv6, soft_clash], axis=1)

    weighted_R = rel * mask3
    second = jnp.einsum("nki,nkj->nij", weighted_R, weighted_R) / safe_count[:, :, None]
    trace = (second[:, 0, 0] + second[:, 1, 1] + second[:, 2, 2])[:, None]
    second_sq_trace = jnp.sum(second * jnp.swapaxes(second, 1, 2), axis=(1, 2))[:, None]
    isotropic_sq_trace = (trace ** 2) / 3.0
    anisotropy = jnp.maximum(second_sq_trace - isotropic_sq_trace, 0.0) / jnp.maximum(trace ** 2, eps)
    det = (
        second[:, 0, 0] * (second[:, 1, 1] * second[:, 2, 2] - second[:, 1, 2] * second[:, 2, 1])
        - second[:, 0, 1] * (second[:, 1, 0] * second[:, 2, 2] - second[:, 1, 2] * second[:, 2, 0])
        + second[:, 0, 2] * (second[:, 1, 0] * second[:, 2, 1] - second[:, 1, 1] * second[:, 2, 0])
    )[:, None]
    shape = jnp.concatenate([trace, second_sq_trace, anisotropy, det], axis=1)

    desc = jnp.concatenate([radial_stats, rbf_density, clash, shape], axis=1).astype(jnp.float32)
    return jnp.where(mask[:, None] > 0.5, desc, 0.0)


def build_sequence_bio_descriptor_from_frame(R, mask):
    """Build 19D sequence-local biological plausibility descriptors per bead."""
    R = jnp.asarray(R, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    n = int(R.shape[0])
    idx = jnp.arange(n, dtype=jnp.int32)

    def neighbor_delta(offset):
        j = idx + int(offset)
        present = (j >= 0) & (j < n) & (mask > 0.5)
        j_clip = jnp.clip(j, 0, n - 1)
        present = present & (mask[j_clip] > 0.5)
        vec = R[j_clip] - R
        dist = jnp.sqrt(jnp.sum(vec * vec, axis=1, keepdims=True) + 1.0e-6)
        present_f = present.astype(jnp.float32)[:, None]
        return present_f, dist * present_f, vec * present_f

    prev1_p, prev1_d, prev_vec = neighbor_delta(-1)
    next1_p, next1_d, next_vec = neighbor_delta(1)
    prev2_p, prev2_d, _ = neighbor_delta(-2)
    next2_p, next2_d, _ = neighbor_delta(2)

    seq1_count = prev1_p + next1_p
    seq1_sum = prev1_d + next1_d
    seq1_mean = seq1_sum / jnp.maximum(seq1_count, 1.0)
    seq1_var = (prev1_p * (prev1_d - seq1_mean) ** 2 + next1_p * (next1_d - seq1_mean) ** 2) / jnp.maximum(seq1_count, 1.0)
    seq1_std = jnp.sqrt(seq1_var + 1.0e-6)
    seq1_min = jnp.where(seq1_count > 0.5, jnp.minimum(jnp.where(prev1_p > 0.5, prev1_d, 1.0e6), jnp.where(next1_p > 0.5, next1_d, 1.0e6)), 0.0)
    seq1_max = jnp.maximum(prev1_d, next1_d)

    seq2_count = prev2_p + next2_p
    seq2_sum = prev2_d + next2_d
    seq2_mean = seq2_sum / jnp.maximum(seq2_count, 1.0)
    seq2_var = (prev2_p * (prev2_d - seq2_mean) ** 2 + next2_p * (next2_d - seq2_mean) ** 2) / jnp.maximum(seq2_count, 1.0)
    seq2_std = jnp.sqrt(seq2_var + 1.0e-6)

    have_angle = (prev1_p > 0.5) & (next1_p > 0.5)
    denom = jnp.maximum(jnp.sqrt(jnp.sum(prev_vec * prev_vec, axis=1, keepdims=True) * jnp.sum(next_vec * next_vec, axis=1, keepdims=True)), 1.0e-6)
    cos_prev_next = jnp.sum(prev_vec * next_vec, axis=1, keepdims=True) / denom
    cos_prev_next = jnp.where(have_angle, cos_prev_next, 0.0)

    stretch = prev1_p * jnp.maximum(prev1_d - 4.5, 0.0) ** 2 + next1_p * jnp.maximum(next1_d - 4.5, 0.0) ** 2
    compression = prev1_p * jnp.maximum(3.0 - prev1_d, 0.0) ** 2 + next1_p * jnp.maximum(3.0 - next1_d, 0.0) ** 2

    desc = jnp.concatenate(
        [
            prev1_p, next1_p, prev2_p, next2_p,
            prev1_d, next1_d, prev2_d, next2_d,
            seq1_count, seq1_mean, seq1_std, seq1_min, seq1_max,
            seq2_count, seq2_mean, seq2_std,
            cos_prev_next, stretch, compression,
        ],
        axis=1,
    )
    return jnp.where(mask[:, None] > 0.5, desc, 0.0).astype(jnp.float32)


def build_jax_geometric_bio_descriptor(R, mask, *, cutoff: float = 10.0, num_rbf: int = 8):
    geom = build_jax_geometric_descriptor(R, mask, cutoff=cutoff, num_rbf=num_rbf)
    bio = build_sequence_bio_descriptor_from_frame(R, mask)
    return jnp.concatenate([geom, bio], axis=1).astype(jnp.float32)


@dataclass(frozen=True)
class LocalExtrapolationGate:
    artifact: Mapping[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "LocalExtrapolationGate":
        with Path(path).open("rb") as handle:
            artifact = pickle.load(handle)
        return cls(artifact)

    @property
    def cutoff(self) -> float:
        return float(self.artifact.get("cutoff", 10.0))

    @property
    def onset(self) -> float:
        return float(self.artifact["onset"])

    @property
    def offset(self) -> float:
        return float(self.artifact["offset"])

    def compute_scores(self, R, mask):
        descriptor = str(self.artifact.get("descriptor", "jax_geometric"))
        if descriptor == "jax_geometric_bio":
            desc = build_jax_geometric_bio_descriptor(R, mask, cutoff=self.cutoff)
        else:
            desc = build_jax_geometric_descriptor(R, mask, cutoff=self.cutoff)
        mean = jnp.asarray(self.artifact["scale_mean"], dtype=jnp.float32)
        std = jnp.asarray(self.artifact["scale_std"], dtype=jnp.float32)
        z = (desc - mean) / jnp.maximum(std, 1.0e-6)
        mode = str(self.artifact.get("mode", "center_distance"))
        if mode == "mahalanobis":
            center = jnp.asarray(
                self.artifact.get("mahalanobis_mean", self.artifact.get("center_z", jnp.zeros((z.shape[1],), dtype=jnp.float32))),
                dtype=jnp.float32,
            )
            inv_cov = jnp.asarray(self.artifact["mahalanobis_inv_cov"], dtype=jnp.float32)
            delta = z - center[None, :]
            q = jnp.einsum("nd,dd,nd->n", delta, inv_cov, delta)
            scores = jnp.sqrt(jnp.maximum(q, 0.0))
        elif mode == "flax_teacher":
            try:
                import flax.linen as nn
            except Exception as exc:
                raise RuntimeError("flax is required for local_extrapolation_gate mode='flax_teacher'") from exc

            class DescriptorTeacher(nn.Module):
                out_dim: int
                width: int
                depth: int

                @nn.compact
                def __call__(self, x):
                    for _ in range(self.depth):
                        x = nn.Dense(self.width)(x)
                        x = nn.tanh(x)
                    return nn.Dense(self.out_dim)(x)

            model = DescriptorTeacher(
                out_dim=int(self.artifact.get("out_dim", z.shape[1])),
                width=int(self.artifact.get("hidden_width", 64)),
                depth=int(self.artifact.get("hidden_depth", 2)),
            )
            pred = model.apply({"params": self.artifact["params"]}, z)
            scores = jnp.sqrt(jnp.mean((pred - z) ** 2, axis=1))
        else:
            center = jnp.asarray(self.artifact.get("center_z", jnp.zeros((z.shape[1],), dtype=jnp.float32)), dtype=jnp.float32)
            scores = jnp.sqrt(jnp.mean((z - center[None, :]) ** 2, axis=1))
        return jnp.where(jnp.asarray(mask) > 0.5, scores.astype(jnp.float32), 0.0)

    def compute_gates(self, R, mask):
        scores = self.compute_scores(R, mask)
        gates = smoothstep_gate(scores, onset=self.onset, offset=self.offset)
        return jnp.where(jnp.asarray(mask) > 0.5, gates, 1.0).astype(jnp.float32)
