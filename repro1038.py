#!/usr/bin/env python3
import torch
import helion
import helion.language as hl

@helion.kernel(
    config=helion.Config(
        block_sizes=[32],
        indexing='block_ptr',
        load_eviction_policies=['last', 'last'],
        num_stages=8,
        num_warps=1,
        pid_type='flat',
        range_flattens=[None],
        range_multi_buffers=[None],
        range_num_stages=[0],
        range_unroll_factors=[0],
        range_warp_specializes=[None],
    ),
    static_shapes=True,
)

def se_block_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        sigmoid_result = torch.sigmoid(x_tile @ w[:, :])
        acc = hl.full([1], 2.0, dtype=x.dtype) * x_tile * sigmoid_result
        out[tile_m, :] = acc.to(x.dtype)

    return out


def main():
    M, N = 4096, 128
    dtype = torch.bfloat16

    x = torch.randn(M, N, dtype=dtype, device="cuda")
    w = torch.randn(N, N, dtype=dtype, device="cuda")

    se_block_fwd(x, w)


if __name__ == "__main__":
    main()
