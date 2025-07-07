from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    for tile in hl.tile(output.size()):
        output[tile] = x[tile] + y[tile]

    return output


def check(m: int, n: int) -> None:
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    y = torch.randn([m, n], device="cuda", dtype=torch.float16)
    run_example(vector_add, torch.add, (x, y))


def main() -> None:
    check(1024, 1024)


if __name__ == "__main__":
    main()
