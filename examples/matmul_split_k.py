from __future__ import annotations

import torch

import helion
from helion._testing import run_example
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
        for inner_k in hl.tile(hl.tile_begin(outer_k), hl.tile_end(outer_k)):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    run_example(matmul_split_k, torch.matmul, (x, y), atol=1)


def main() -> None:
    check(64, 32768, 64)


if __name__ == "__main__":
    main()
