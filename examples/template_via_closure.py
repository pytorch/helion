from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(
    # This was tuned on a 3090 and likely isn't optimal for other GPUs
    config=helion.Config(
        block_sizes=[[64, 64], [16]],
        loop_orders=[[0, 1]],
        num_warps=2,
        num_stages=3,
        indexing="block_ptr",
        l2_grouping=32,
    ),
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def matmul_with_epilogue(
    x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]
) -> Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, [tile_m, tile_n])
    return out


def autotune(n: int, k: int, m: int) -> None:
    x = torch.randn([n, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, m], device="cuda", dtype=torch.float16)
    bias = torch.randn([1, m], device="cuda", dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile]))
    matmul_with_epilogue.configs.clear()
    best_config = matmul_with_epilogue.autotune(args)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


def check(n: int, k: int, m: int) -> None:
    x = torch.randn([n, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, m], device="cuda", dtype=torch.float16)
    bias = torch.randn([1, m], device="cuda", dtype=torch.float16)
    # The epilogue can use the captured bias tensor that is implicitly lifted to a kernel arg
    result = matmul_with_epilogue(x, y, lambda acc, tile: torch.relu(acc + bias[tile]))
    torch.testing.assert_close(
        result,
        torch.relu(x @ y + bias),
        rtol=1e-2,
        atol=1e-1,
    )
    print("ok")


if __name__ == "__main__":
    # autotune(1024, 1024, 1024)
    check(1024, 1024, 1024)
