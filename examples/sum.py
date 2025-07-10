from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Sum 2D tensor along the last dimension."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)

    return out


def check(m: int, n: int) -> None:
    x = torch.randn([m, n], device="cuda", dtype=torch.float32)
    kernels = {"helion": sum_kernel}
    run_example(kernels, lambda x: x.sum(-1), (x,))


def main() -> None:
    check(512, 256)
    check(1024, 1024)


if __name__ == "__main__":
    main()
