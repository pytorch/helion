from __future__ import annotations

import helion
import helion.language as hl

import torch
from helion._testing import run_example


@helion.kernel()
def exp(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential of all elements in the input tensor.

    Args:
        x: Input tensor

    Returns:
        Output tensor with the exponential of each element in the input
    """
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out


def exp_tritonbench(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Wrapper for tritonbench that returns output in expected format."""
    return {"output": exp(x)}


def check(n: int) -> None:
    """
    Verify the exp kernel implementation against PyTorch's native exp function.

    Args:
        n: Size of the test tensor
    """
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    run_example(exp, torch.exp, (x,))


def main() -> None:
    """
    Main entry point that runs the exp kernel verification with a tensor of size 1M elements.
    """
    check(1024 * 1024)


if __name__ == "__main__":
    main()
