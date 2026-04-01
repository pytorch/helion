"""Subprocess worker: full transposed_matmul_bwd with d_sliced/d_remainder.

Both transpose_C branches are included (if/else).

Usage:
    python ima_worker_d_sliced_bwd.py <config_json> <skip|normalize>

Exit codes:
    0 = results are finite
    1 = NaN/Inf detected
    crash = CUDA IMA or other fatal error
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import torch
from unittest.mock import patch

import helion
import helion._compat
import helion.language as hl
from helion.autotuner.config_spec import ConfigSpec


@helion.kernel(static_shapes=False)
def transposed_matmul_bwd(
    d_out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    transpose_A: hl.constexpr = False,
    transpose_C: hl.constexpr = False,
    d_sliced: Optional[torch.Tensor] = None,
    d_remainder: Optional[torch.Tensor] = None,
    slice_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    Batch = A.size(0)
    N = hl.specialize(B.size(0))
    K = hl.specialize(B.size(1))
    if transpose_A:
        M = hl.specialize(A.size(2))
        d_A = torch.empty(
            (Batch, K, M), device=A.device, dtype=torch.bfloat16
        )
    else:
        M = hl.specialize(A.size(1))
        d_A = torch.empty(
            (Batch, M, K), device=A.device, dtype=torch.bfloat16
        )
    d_B = torch.empty((Batch, N, K), device=B.device, dtype=torch.bfloat16)
    if bias is not None:
        d_bias = torch.empty(
            (Batch, N), device=B.device, dtype=torch.bfloat16
        )
    for tile_b in hl.tile(Batch, block_size=1):
        for tile_n, tile_k in hl.tile([N, K]):
            d_B_acc = hl.zeros([tile_n, tile_k], dtype=torch.float32)
            if bias is not None:
                d_bias_acc = hl.zeros([tile_n], dtype=torch.float32)
            for tile_m in hl.tile(M):
                if transpose_C:
                    d_out_tile = d_out[tile_b.begin, tile_n, tile_m].to(
                        torch.bfloat16
                    )
                else:
                    d_out_tile = d_out[tile_b.begin, tile_m, tile_n].to(
                        torch.bfloat16
                    )
                    d_out_tile = d_out_tile.t()
                if d_sliced is not None and slice_size:
                    if transpose_C:
                        d_sliced_tile = hl.load(
                            d_sliced,
                            [tile_b.begin, tile_n, tile_m],
                            extra_mask=(tile_n.index < slice_size)[:, None],
                        )
                    else:
                        d_sliced_tile = hl.load(
                            d_sliced,
                            [tile_b.begin, tile_m, tile_n],
                            extra_mask=(tile_n.index < slice_size)[None, :],
                        )
                        d_sliced_tile = d_sliced_tile.t()
                    d_out_tile = d_out_tile + d_sliced_tile.to(torch.bfloat16)
                if d_remainder is not None and slice_size:
                    remainder_idx = torch.clamp(
                        tile_n.index - slice_size, min=0
                    )
                    if transpose_C:
                        d_remainder_tile = hl.load(
                            d_remainder,
                            [tile_b.begin, remainder_idx, tile_m],
                            extra_mask=(tile_n.index >= slice_size)[:, None],
                        )
                    else:
                        d_remainder_tile = hl.load(
                            d_remainder,
                            [tile_b.begin, tile_m, remainder_idx],
                            extra_mask=(tile_n.index >= slice_size)[None, :],
                        )
                        d_remainder_tile = d_remainder_tile.t()
                    d_out_tile = d_out_tile + d_remainder_tile.to(
                        torch.bfloat16
                    )
                if transpose_A:
                    a_tile = A[tile_b.begin, tile_k, tile_m].to(
                        torch.bfloat16
                    )
                    a_tile = a_tile.t()
                else:
                    a_tile = A[tile_b.begin, tile_m, tile_k].to(
                        torch.bfloat16
                    )
                d_B_acc = torch.addmm(d_B_acc, d_out_tile, a_tile)
                if bias is not None:
                    d_bias_acc = d_bias_acc + torch.sum(d_out_tile, dim=1)
            d_B[tile_b.begin, tile_n, tile_k] = d_B_acc.to(d_B.dtype)
            if bias is not None:
                d_bias[tile_b.begin, tile_n] = d_bias_acc.to(d_bias.dtype)
        for tile_m3, tile_k3 in hl.tile([M, K]):
            d_A_acc = hl.zeros([tile_m3, tile_k3], dtype=torch.float32)
            for tile_n3 in hl.tile(N):
                if transpose_C:
                    d_out_tile = d_out[tile_b.begin, tile_n3, tile_m3].to(
                        torch.bfloat16
                    )
                    if d_sliced is not None and slice_size:
                        d_sliced_tile = hl.load(
                            d_sliced,
                            [tile_b.begin, tile_n3, tile_m3],
                            extra_mask=(tile_n3.index < slice_size)[:, None],
                        )
                        d_out_tile = d_out_tile + d_sliced_tile.to(
                            torch.bfloat16
                        )
                    if d_remainder is not None and slice_size:
                        remainder_idx = torch.clamp(
                            tile_n3.index - slice_size, min=0
                        )
                        d_remainder_tile = hl.load(
                            d_remainder,
                            [tile_b.begin, remainder_idx, tile_m3],
                            extra_mask=(tile_n3.index >= slice_size)[:, None],
                        )
                        d_out_tile = d_out_tile + d_remainder_tile.to(
                            torch.bfloat16
                        )
                    d_out_tile = d_out_tile.t()
                else:
                    d_out_tile = d_out[tile_b.begin, tile_m3, tile_n3].to(
                        torch.bfloat16
                    )
                    if d_sliced is not None and slice_size:
                        d_sliced_tile = hl.load(
                            d_sliced,
                            [tile_b.begin, tile_m3, tile_n3],
                            extra_mask=(tile_n3.index < slice_size)[None, :],
                        )
                        d_out_tile = d_out_tile + d_sliced_tile.to(
                            torch.bfloat16
                        )
                    if d_remainder is not None and slice_size:
                        remainder_idx = torch.clamp(
                            tile_n3.index - slice_size, min=0
                        )
                        d_remainder_tile = hl.load(
                            d_remainder,
                            [tile_b.begin, tile_m3, remainder_idx],
                            extra_mask=(tile_n3.index >= slice_size)[None, :],
                        )
                        d_out_tile = d_out_tile + d_remainder_tile.to(
                            torch.bfloat16
                        )
                b_tile = B[tile_n3, tile_k3].to(torch.bfloat16)
                d_A_acc = torch.addmm(d_A_acc, d_out_tile, b_tile)
            if transpose_A:
                d_A[tile_b.begin, tile_k3, tile_m3] = d_A_acc.to(
                    d_A.dtype
                ).t()
            else:
                d_A[tile_b.begin, tile_m3, tile_k3] = d_A_acc.to(d_A.dtype)
    d_B = d_B.sum(dim=0)
    if bias is not None:
        d_bias = d_bias.sum(dim=0)
    else:
        d_bias = None
    return d_A, d_B, d_bias


if __name__ == "__main__":
    config = helion.Config(**json.loads(sys.argv[1]))
    skip_normalize = sys.argv[2] == "skip"

    B_sz, K_sz, M_sz, N_sz, SLICE_SIZE = 8, 48, 128, 192, 96
    torch.manual_seed(42)
    A = torch.randn(
        B_sz, K_sz, M_sz, device="cuda", dtype=torch.float32
    )
    W = torch.randn(N_sz, K_sz, device="cuda", dtype=torch.float32)
    d_out = torch.randn(
        B_sz, N_sz, M_sz, device="cuda", dtype=torch.bfloat16
    )
    d_sliced = torch.randn(
        B_sz, N_sz, M_sz, device="cuda", dtype=torch.bfloat16
    )
    d_remainder = torch.randn(
        B_sz, N_sz - SLICE_SIZE, M_sz, device="cuda", dtype=torch.bfloat16
    )

    args = (d_out, A, W, None, True, True, d_sliced, d_remainder, SLICE_SIZE)
    with patch.object(
        helion._compat, "_supports_tensor_descriptor", lambda: False
    ):
        bound = transposed_matmul_bwd.bind(args)
        if skip_normalize:
            with patch.object(
                ConfigSpec, "normalize", lambda self, cfg, **kw: None
            ):
                compiled = bound.compile_config(config)
        else:
            compiled = bound.compile_config(config)
        d_A, d_B, d_bias = compiled(*args)
        torch.cuda.synchronize()
        if torch.isfinite(d_A).all() and torch.isfinite(d_B).all():
            sys.exit(0)
        else:
            print("NaN/Inf detected", file=sys.stderr)
            sys.exit(1)
