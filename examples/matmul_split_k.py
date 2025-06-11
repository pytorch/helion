from __future__ import annotations

import torch

import helion
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.zeros(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


def check(m: int, k: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    result = matmul_split_k(x, y)
    torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1)
    sec = do_bench(lambda: matmul_split_k(x, y))
    baseline_sec = do_bench(lambda: torch.matmul(x, y))
    print(
        f"Helion time: {sec:.4f}ms, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


def main() -> None:
    check(64, 32768, 64)


if __name__ == "__main__":
    main()
