from __future__ import annotations

import helion
import helion.language as hl

import torch
from helion._testing import run_example


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise with broadcasting support.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        A new tensor containing the element-wise sum of x and y
    """
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
    """
    Verify the add kernel implementation against PyTorch's native add function.

    Args:
        m: First dimension of the test tensors
        n: Second dimension of the test tensors
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    y = torch.randn([m, n], device="cuda", dtype=torch.float16)
    run_example(add, torch.add, (x, y))


def main() -> None:
    """
    Main entry point that runs the add kernel verification with 1024x1024 tensors.
    """
    check(1024, 1024)


if __name__ == "__main__":
    main()
