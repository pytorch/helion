"""Kernels over a flattened 4-D tile shared by the row-packet tests.

``test/test_cute_flat_row_packets.py`` renders them on CPU and
``test/test_cute_pointwise_vec.py`` runs them on a GPU.  They are plain
functions: each module wraps them with the kernel settings it needs.
"""

from __future__ import annotations

import torch

import helion.language as hl


def cast_copy(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    out = torch.empty([b, h, s, d], dtype=torch.float32, device=x.device)
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        out[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d].to(
            torch.float32
        )
    return out


def cast_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The tile walks ``out``'s shape; ``x`` may be larger."""
    b, h, s, d = out.shape
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        out[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d].to(
            torch.float32
        )
    return out


def scale_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        out[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d] * 2
    return out


def scale_in_place(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        x[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d] * 2 + 1
    return x


def scale_between_views(x: torch.Tensor) -> torch.Tensor:
    """Loads through one host view of ``x``'s storage and stores through
    another: two tensors to the kernel, one to the barrier pass."""
    src = x.permute(0, 2, 1, 3)
    dst = x.permute(0, 2, 1, 3)
    b, h, s, d = src.shape
    for tile_b, tile_h, tile_s, tile_d in hl.tile([b, h, s, d]):
        dst[tile_b, tile_h, tile_s, tile_d] = (
            src[tile_b, tile_h, tile_s, tile_d] * 2 + 1
        )
    return dst
