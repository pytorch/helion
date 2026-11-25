# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.dist-moe file in this directory.

"""Dist-MoE's two expert projections for EP2, NVFP4 inference, and G8 SwiGLU.

The input and weight tensors are packed E2M1, their block scales are E4M3,
and the destination buffers are BF16. See nvfp4_projections.md for the call
signatures, the historical GB300 configs, and how to render W2.
"""

from __future__ import annotations

from typing import Annotated

import torch

import helion
import helion.language as hl

# Original baseline for W13, which requires the development compiler.
_W13_BASE = helion.Config(
    block_sizes=[16, 16, 8],
    cute_scaled_mma=True,
    cute_collective_mma=False,
    cute_scaled_tile_n=128,
    cute_scaled_tile_k=128,
    cute_scaled_stages=2,
    cute_scaled_cluster_m=1,
    cute_scaled_persistent=True,
    cute_scaled_prepare="fused",
)


def _atom_offset(row: torch.Tensor, group: torch.Tensor, groups: int) -> torch.Tensor:
    """Return the byte offset within one expert's padded 128-by-4 scale atoms."""
    tile = (row // 128) * ((groups + 3) // 4) + group // 4
    return tile * 512 + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


@helion.kernel(
    backend="cute",
    static_shapes=True,
    fast_math=False,
    cute_exact_division=True,
    config=_W13_BASE,
)
def _nvfp4_g8_batched_gate(
    a: torch.Tensor,
    w: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    out: torch.Tensor,
    row_scale: torch.Tensor,
    post: torch.Tensor,
    edges: torch.Tensor,
    alpha: Annotated[float | None, hl.constexpr] = None,
    limit: Annotated[float | None, hl.constexpr] = None,
) -> torch.Tensor:
    """Contract each expert on its own device-bounded rows, rounding once to BF16."""
    groups = a.shape[2]
    for batch, m, n in hl.tile(
        (a.shape[0], a.shape[1], w.shape[3]), block_size=(1, None, None)
    ):
        e = batch.begin
        acc = hl.zeros([m, n], dtype=torch.float32)
        for g in hl.tile(groups):
            left = sa[e, _atom_offset(m.index[:, None], g.index[None, :], groups)].to(
                torch.float32
            )
            right = sw[e, _atom_offset(n.index[None, :], g.index[:, None], groups)].to(
                torch.float32
            )
            for byte in hl.static_range(8):
                a0, a1 = hl.float4_e2m1fn_x2_to_float32(a[e, m, g, byte])
                b0, b1 = hl.float4_e2m1fn_x2_to_float32(w[e, g, byte, n])
                acc = acc + (
                    (
                        (
                            a0.unsqueeze(2) * b0.unsqueeze(0)
                            + a1.unsqueeze(2) * b1.unsqueeze(0)
                        )
                        * left.unsqueeze(2)
                    )
                    * right.unsqueeze(0)
                ).sum(1)
        value = ((acc * row_scale[e, m].unsqueeze(1)) * post[e]).to(torch.bfloat16)
        # W13 preparation uses the stock Dist-MoE eight-channel gate/up layout.
        panels = value.reshape(m.block_size, n.block_size // 16, 2, 8).permute(
            0, 1, 3, 2
        )
        x, y = hl.split(panels)
        x = x.reshape(m.block_size, n.block_size // 2).float()
        y = y.reshape(m.block_size, n.block_size // 2).float()
        if limit is None:
            sig = 1.0 / (1.0 + torch.exp(-x))
        else:
            x = torch.where(x < limit, x, limit)
            lo = torch.where(y > -limit, y, -limit)
            y = torch.where(lo < limit, lo, limit) + 1.0
            # Clamped SwiGLU supplies both constexpr coefficients.
            sig = 1.0 / (1.0 + torch.exp(-(alpha * x)))  # pyrefly: ignore [unsupported-operation]
        result = ((x * sig) * y).to(torch.bfloat16)
        live = edges[e + 1] - edges[e]
        cols = n.begin // 2 + hl.arange(n.block_size // 2)
        hl.store(out, [e, m, cols], result, extra_mask=(m.index < live)[:, None])
    return out


@helion.kernel(
    backend="cute",
    static_shapes=True,
    config=helion.Config(block_sizes=[16, 16, 8], cute_scaled_mma=False),
)
def _nvfp4_batched_peer_rows(
    a: torch.Tensor,
    w: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    row_scale: torch.Tensor,
    post: torch.Tensor,
    edges: torch.Tensor,
    routes: torch.Tensor,
    owners: torch.Tensor,
    dest0: torch.Tensor,
    dest1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = a.shape[2]
    for batch, m, n in hl.tile(
        (a.shape[0], a.shape[1], w.shape[3]), block_size=(1, None, None)
    ):
        e = batch.begin
        acc = hl.zeros([m, n], dtype=torch.float32)
        for g in hl.tile(groups):
            left = sa[e, _atom_offset(m.index[:, None], g.index[None, :], groups)].to(
                torch.float32
            )
            right = sw[e, _atom_offset(n.index[None, :], g.index[:, None], groups)].to(
                torch.float32
            )
            for byte in hl.static_range(8):
                a0, a1 = hl.float4_e2m1fn_x2_to_float32(a[e, m, g, byte])
                b0, b1 = hl.float4_e2m1fn_x2_to_float32(w[e, g, byte, n])
                acc = acc + (
                    (
                        (
                            a0.unsqueeze(2) * b0.unsqueeze(0)
                            + a1.unsqueeze(2) * b1.unsqueeze(0)
                        )
                        * left.unsqueeze(2)
                    )
                    * right.unsqueeze(0)
                ).sum(1)
        value = ((acc * row_scale[e, m].unsqueeze(1)) * post[e]).to(torch.bfloat16)
        live = (m.index < edges[e + 1] - edges[e])[:, None]
        code = routes[e, m]
        slots = dest0.shape[0]
        offset = code % slots
        tag = e * a.shape[1] + m.index
        for peer in hl.static_range(2):
            dest = dest0 if peer == 0 else dest1
            belongs = (code // slots == peer) & (owners[peer, offset] == tag)
            hl.store(dest, [offset, n], value, extra_mask=live & belongs[:, None])
    return dest0, dest1


def projection_inputs(
    role: str, device: str | torch.device
) -> tuple[torch.Tensor | float, ...]:
    """Build one rank's synthetic T1024 operands, including two local W2 destinations."""
    if role not in ("w13", "w2"):
        raise ValueError(f"Unknown projection: {role}")
    experts, rows, live, slots = 4, 2048, 512, 2048
    groups, cols = (224, 6144) if role == "w13" else (192, 3584)

    def packed(shape: tuple[int, ...]) -> torch.Tensor:
        # E2M1 0x2 encodes 1.0 in each half byte.
        return torch.full(shape, 0x22, dtype=torch.uint8, device=device).view(
            torch.float4_e2m1fn_x2
        )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        # E4M3 0x38 encodes a block scale of 1.0.
        return torch.full(shape, 0x38, dtype=torch.uint8, device=device).view(
            torch.float8_e4m3fn
        )

    a = packed((experts, rows, groups, 8))
    w = packed((experts, cols, groups, 8)).permute(0, 2, 3, 1)
    sa = scales((experts, rows * groups))
    sw = scales((experts, cols * groups))
    row_scale = torch.ones((experts, rows), dtype=torch.float32, device=device)
    post = torch.ones(experts, dtype=torch.float32, device=device)
    edges = torch.arange(experts + 1, dtype=torch.int64, device=device) * live
    if role == "w13":
        out = torch.zeros(
            (experts, rows, cols // 2), dtype=torch.bfloat16, device=device
        )
        return a, w, sa, sw, out, row_scale, post, edges, 1.702, 7.0

    row = torch.arange(rows, dtype=torch.int32, device=device)
    expert = torch.arange(experts, dtype=torch.int32, device=device)[:, None]
    # Alternate live rows between peers. Each destination slot has one owner.
    slot = expert * (live // 2) + row // 2
    routes = torch.where(row < live, (row % 2) * slots + slot, 0)
    owners = torch.full((2, slots), -1, dtype=torch.int32, device=device)
    owners[(row[:live] % 2).long(), slot[:, :live].long()] = expert * rows + row[:live]
    dest0 = torch.zeros((slots, cols), dtype=torch.bfloat16, device=device)
    dest1 = torch.zeros_like(dest0)
    return a, w, sa, sw, row_scale, post, edges, routes, owners, dest0, dest1
