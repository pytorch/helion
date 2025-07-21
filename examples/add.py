from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # match pytorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        # match type promotion of torch.add
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # tile will be a tuple of blocks
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def check(m: int, n: int) -> None:
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    y = torch.randn([m, n], device="cuda", dtype=torch.float16)
    run_example(add, torch.add, (x, y))


def main() -> None:
    check(1024, 1024)


if __name__ == "__main__":
    main()
