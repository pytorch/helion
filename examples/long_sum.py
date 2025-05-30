from __future__ import annotations
from unittest import result

import torch

import helion
import helion.language as hl

def baseline_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1)

# Naive Reduction: Load the entire reduction dim at once, and reduce in reg.
@helion.kernel(config=helion.Config(block_sizes=[[1]], reduction_loops=[None], num_warps=32, num_stages=4, indexing='block_ptr'))
def longsum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


# Looped reduction
@helion.kernel(
    config=helion.Config(
        block_sizes=[[1]],
        reduction_loops=[32768], # [None] for naive reduction, [tile_size] for looped reduction
        num_warps=16,
        num_stages=5,
        indexing="pointer",
    )
)
def longsum_w_red_loop(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty(
        [m], dtype=x.dtype, device=x.device
    )

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


# This generates the same code as above, but manually implements looped reduction.
@helion.kernel(config=helion.Config(block_sizes=[[32768], [1]], num_warps=16, num_stages=5, indexing='pointer'))
def longsum_manual(x: torch.Tensor) -> torch.Tensor:
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

    helion_out = longsum(x)
    torch.testing.assert_close(helion_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ naive reduction")

    helion_red_loop_out = longsum_w_red_loop(x)
    torch.testing.assert_close(helion_red_loop_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ Reduction Loop")

    helion_manual_out = longsum_manual(x)
    torch.testing.assert_close(helion_manual_out, baseline_sum(x), rtol=1e-2, atol=1e-1)
    print("✅ Results Match ✅ Manual Reduction Loop")

    sec = do_bench(lambda: longsum(x))
    loop_sec = do_bench(lambda: longsum_w_red_loop(x))
    manual_loop_sec = do_bench(lambda: longsum_manual(x))
    baseline_sec = do_bench(lambda: baseline_sum(x))
    print(
        f"Helion Naive time: {sec:.4f}s, Helion Looped Time: {loop_sec:.4f},  Helion Manual Loop Time: {manual_loop_sec:.4f} torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x {baseline_sec / loop_sec:.2f}x {baseline_sec / manual_loop_sec:.2f}x"
    )


def main() -> None:
    check(4, 130000) # seq_len = 128k


if __name__ == "__main__":
    main()
