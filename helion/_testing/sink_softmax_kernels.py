"""gpt-oss's sink-softmax normalize, for the whole-row tile seed's tests.

Shared by ``test/test_cute_pointwise_resident_seeds.py`` (the seed and its
render, on CPU) and ``test/test_cute_pointwise_vec.py`` (its numerics on
the GPU).
"""

from __future__ import annotations

import torch

import helion.language as hl


def sink_normalize(
    x: torch.Tensor, mask: torch.Tensor, total: torch.Tensor
) -> torch.Tensor:
    """``[h, q, k]`` scores, a ``[q, k]`` mask broadcast over the heads and a
    divisor per row."""
    out = torch.empty_like(x)
    for tile_h, tile_q, tile_k in hl.tile(x.size()):
        e = torch.exp(
            x[tile_h, tile_q, tile_k].float() + mask[tile_q, tile_k].float()[None]
        )
        out[tile_h, tile_q, tile_k] = (e / total[tile_h, tile_q][:, :, None]).to(
            out.dtype
        )
    return out
