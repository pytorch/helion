from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def matmul(
    x: Tensor,
    y: Tensor,
    epilogue: Callable[[Tensor, list[Tensor]], Tensor] = lambda acc, tile: acc,
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


def autotune(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
    best_config = matmul.autotune(args, force=True)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)

    # Test without bias
    run_example(matmul, torch.matmul, (x, y))

    # Test with bias
    def helion_linear(x: Tensor, y: Tensor, bias: Tensor) -> Tensor:
        return matmul(x, y, lambda acc, tile: acc + bias[tile[1]])

    def baseline_linear(x: Tensor, y: Tensor, bias: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, y.T, bias)

    run_example(helion_linear, baseline_linear, (x, y, bias))

    # Test more complex epilogue
    def epilogue(acc: Tensor, tile: list[Tensor]) -> Tensor:
        # The epilogue can use the captured bias tensor that is implicitly lifted to a kernel arg
        return torch.relu(acc + bias[tile[1]])

    def kernel_wrapper(x: Tensor, y: Tensor) -> Tensor:
        return matmul(x, y, epilogue)

    def baseline_wrapper(x: Tensor, y: Tensor) -> Tensor:
        return torch.relu(x @ y + bias)

    run_example(
        kernel_wrapper,
        baseline_wrapper,
        (x, y),
    )


def matmul_tritonbench(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None
) -> Callable:
    """Wrapper for tritonbench that matches its interface."""
    if bias is not None:
        return lambda: matmul(a, b, lambda acc, tile: acc + bias[tile[1]])
    return lambda: matmul(a, b)


def main() -> None:
    # autotune(1024, 1024, 1024)
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
