"""Dense online attention admitted by both fused flash and generic MMA-DAG."""

from __future__ import annotations

import math

import torch

import helion.language as hl


def overlapping_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_major: hl.constexpr,
) -> torch.Tensor:
    batch, seq, dim = q.size(0), q.size(1), q.size(2)
    out = torch.empty([batch, seq, dim], dtype=q.dtype, device=q.device)
    scale = math.log2(math.e) / math.sqrt(dim)
    for tb, tm in hl.tile([batch, seq], block_size=[1, None]):
        maximum = hl.full([tm], float("-inf"), dtype=torch.float32)
        denominator = hl.zeros([tm], dtype=torch.float32)
        accumulator = hl.zeros([tm, dim], dtype=torch.float32)
        query = q[tb.begin, tm, :]
        for tn in hl.tile(seq):
            score = hl.dot(query, k[tb.begin, tn, :].T) * scale
            next_max = torch.maximum(maximum, torch.amax(score, dim=-1))
            probability = torch.exp2(score - next_max[:, None])
            alpha = torch.exp2(maximum - next_max)
            denominator = denominator * alpha + probability.sum(-1)
            value = v[tb.begin, :, tn].T if k_major else v[tb.begin, tn, :]
            accumulator = hl.dot(
                probability.to(v.dtype), value, acc=accumulator * alpha[:, None]
            )
            maximum = next_max
        out[tb.begin, tm, :] = (accumulator / denominator[:, None]).to(q.dtype)
    return out
