from __future__ import annotations
from unittest import result

import torch

import helion
import helion.language as hl

def baseline_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1)



# Looped Reduction Long Sum
# Example Config:
# @helion.kernel(config=helion.Config(block_sizes=[[32768], [1]], num_warps=16, num_stages=5, indexing='pointer'))
@helion.kernel()
def long_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    # Call register_block_size to know block_size_n outside of the reduction loop.
    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_size_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_size_n): # The reduction loop for n that doesn't fit in a tile.
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


# Long Sum using Helion's reduction feature
# Config: reduction_loop allows Helion to generate a looped reduction (same as the naive impl above)
# Example Config:
# @helion.kernel(config=helion.Config(block_sizes=[[1]], reduction_loops=[None], num_warps=32, num_stages=4, indexing='block_ptr'))
@helion.kernel()
def long_sum_reduction(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def check(m: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, n], device="cuda", dtype=torch.float32)

    helion_out = long_sum(x)
    torch.testing.assert_close(helion_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ Naive Helion")

    helion_red_out = long_sum_reduction(x)
    torch.testing.assert_close(helion_red_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ Reduction Helion")

    sec = do_bench(lambda: long_sum(x))
    red_sec = do_bench(lambda: long_sum_reduction(x))
    baseline_sec = do_bench(lambda: baseline_sum(x))
    print(
        f"Helion time: {sec:.4f}s, Helion Reduction Time: {red_sec:.4f},  torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x {baseline_sec / red_sec:.2f}x"
    )


def main() -> None:
    check(4, 131072) # seq_len = 128k


if __name__ == "__main__":
    main()
