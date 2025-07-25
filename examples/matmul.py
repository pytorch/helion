from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    epilogue: Callable[[torch.Tensor, list[torch.Tensor]], torch.Tensor] = lambda acc,
    tile: acc,
) -> torch.Tensor:
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


def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)

    # Test without bias
    kernel_no_bias = lambda x, y: matmul(x, y)  # noqa: E731
    expected_no_bias = lambda x, y: torch.matmul(x, y)  # noqa: E731
    run_example(kernel_no_bias, expected_no_bias, (x, y))

    # Test with bias
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    kernel_with_bias = lambda x, y: matmul(x, y, lambda acc, tile: acc + bias[tile[1]])  # noqa: E731
    expected_with_bias = lambda x, y: torch.matmul(x, y) + bias  # noqa: E731
    run_example(kernel_with_bias, expected_with_bias, (x, y))


def main() -> None:
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
