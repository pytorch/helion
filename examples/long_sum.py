from __future__ import annotations
from unittest import result

import torch

import helion
import helion.language as hl

def baseline_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1)


@helion.kernel(
    config=helion.Config(
        block_sizes=[[1]],
        reduction_loops=[32768], # [None] for non-looped reduction, [tile_size] for looped reduction
        num_warps=16,
        num_stages=5,
        indexing="pointer",
    )
)
def long_sum_reduction(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


# This generates the same code as above, but manually implements looped reduction.
@helion.kernel(config=helion.Config(block_sizes=[[32768], [1]], num_warps=16, num_stages=5, indexing='pointer'))
def long_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    # Call register_block_size to know block_size_n outside of the reduction loop.
    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_size_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_size_n): # Reduction loop
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


def check(m: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, n], device="cuda", dtype=torch.float32)

    helion_out = long_sum(x)
    torch.testing.assert_close(helion_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ Naive Looped Reduction")

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
    check(4, 130000) # seq_len = 128k


if __name__ == "__main__":
    main()
