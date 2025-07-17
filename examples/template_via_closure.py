from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl

import torch
from helion._testing import run_example
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(
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
    """
    Autotunes the matmul_with_epilogue kernel and saves the best configuration.

    Creates random tensors and runs the autotuning process to find the optimal
    configuration for the kernel with the given dimensions.

    Args:
        n: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        m: Second dimension of the second matrix
    """
    x = torch.randn([n, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, m], device="cuda", dtype=torch.float16)
    bias = torch.randn([1, m], device="cuda", dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile]))
    best_config = matmul_with_epilogue.autotune(args, force=True)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


def check(n: int, k: int, m: int) -> None:
    """
    Verify the matmul_with_epilogue kernel implementation against a PyTorch baseline.

    Tests matrix multiplication with a ReLU + bias epilogue function.

    Args:
        n: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        m: Second dimension of the second matrix
    """
    x = torch.randn([n, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, m], device="cuda", dtype=torch.float16)
    bias: torch.Tensor = torch.randn([1, m], device="cuda", dtype=torch.float16)

    def epilogue(acc: torch.Tensor, tile: list[torch.Tensor]) -> torch.Tensor:
        # The epilogue can use the captured bias tensor that is implicitly lifted to a kernel arg
        return torch.relu(acc + bias[tile])

    def kernel_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return matmul_with_epilogue(x, y, epilogue)

    def baseline_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ y + bias)

    run_example(
        kernel_wrapper,
        baseline_wrapper,
        (x, y),
    )


def main() -> None:
    """
    Main entry point that runs the matmul_with_epilogue kernel verification.

    Tests with 1024x1024 matrices and a ReLU + bias epilogue function.
    Uncomment the autotune line to run autotuning instead.
    """
    # autotune(1024, 1024, 1024)
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
