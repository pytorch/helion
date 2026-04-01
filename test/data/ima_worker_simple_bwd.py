"""Subprocess worker: simple transposed_matmul_bwd kernel (no d_sliced).

Usage:
    python ima_worker_simple_bwd.py <config_json> <skip|normalize>

Exit codes:
    0 = results are finite
    1 = NaN/Inf detected
    crash = CUDA IMA or other fatal error
"""
from __future__ import annotations

import json
import sys

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
) -> tuple[torch.Tensor, torch.Tensor]:
    Batch = A.size(0)
    N = hl.specialize(B.size(0))
    K = hl.specialize(B.size(1))
    M = hl.specialize(A.size(2))
    d_A = torch.empty((Batch, K, M), device=A.device, dtype=torch.bfloat16)
    d_B = torch.zeros((N, K), device=B.device, dtype=torch.float32)
    for tile_b in hl.tile(Batch, block_size=1):
        for tile_n, tile_k in hl.tile([N, K]):
            d_B_acc = hl.zeros([tile_n, tile_k], dtype=torch.float32)
            for tile_m in hl.tile(M):
                d_out_tile = d_out[tile_b.begin, tile_n, tile_m].to(
                    torch.bfloat16
                )
                a_tile = A[tile_b.begin, tile_k, tile_m].to(torch.bfloat16)
                a_tile = a_tile.t()
                d_B_acc = torch.addmm(d_B_acc, d_out_tile, a_tile)
            hl.atomic_add(d_B, [tile_n, tile_k], d_B_acc)
        for tile_m3, tile_k3 in hl.tile([M, K]):
            d_A_acc = hl.zeros([tile_m3, tile_k3], dtype=torch.float32)
            for tile_n3 in hl.tile(N):
                d_out_tile = d_out[tile_b.begin, tile_n3, tile_m3].to(
                    torch.bfloat16
                )
                d_out_tile = d_out_tile.t()
                b_tile = B[tile_n3, tile_k3].to(torch.bfloat16)
                d_A_acc = torch.addmm(d_A_acc, d_out_tile, b_tile)
            d_A[tile_b.begin, tile_k3, tile_m3] = d_A_acc.to(d_A.dtype).t()
    d_B = d_B.to(torch.bfloat16)
    return d_A, d_B


if __name__ == "__main__":
    config = helion.Config(**json.loads(sys.argv[1]))
    skip_normalize = sys.argv[2] == "skip"

    B_sz, K_sz, M_sz, N_sz = 8, 48, 64, 96
    torch.manual_seed(42)
    A = torch.randn(
        B_sz, K_sz, M_sz, device="cuda", dtype=torch.float32
    ) * 0.01
    W = torch.randn(N_sz, K_sz, device="cuda", dtype=torch.float32) * 0.01
    d_out = torch.randn(
        B_sz, N_sz, M_sz, device="cuda", dtype=torch.float32
    ) * 0.01

    args = (d_out, A, W)
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
        d_A, d_B = compiled(*args)
        torch.cuda.synchronize()
        if torch.isfinite(d_A).all() and torch.isfinite(d_B).all():
            sys.exit(0)
        else:
            print("NaN/Inf detected", file=sys.stderr)
            sys.exit(1)
