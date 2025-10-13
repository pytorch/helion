"""
Tensor Concatenation Example
============================

This example demonstrates how to implement a tensor concatenation operation using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Concatenation Kernel
# --------------------


# %%
@helion.kernel()
def concat2d_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two 2D tensors along dimension 1 (columns).

    Args:
        x: First input tensor of shape [M, N1]
        y: Second input tensor of shape [M, N2] with same first dimension as x

    Returns:
        Output tensor of shape [M, N1+N2] containing the concatenation of x and y along dimension 1
    """
    assert x.size(0) == y.size(0)
    out = torch.empty(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    for tile0, tile1 in hl.tile(out.size()):
        # Most masking is automatic in helion, but tile1 spans both x and y we need to do some manual masking
        x_part = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index < x.size(1))[None, :]
        )
        y_part = hl.load(
            y,
            [tile0, tile1.index - x.size(1)],
            extra_mask=(tile1.index >= x.size(1))[None, :],
        )
        out[tile0, tile1] = torch.where(
            (tile1.index < x.size(1))[None, :], x_part, y_part
        )
    return out


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the concatenation kernel verification.
    Tests with two tensors of shapes [1500, 400] and [1500, 600].
    """
    x = torch.randn([1500, 400], device=DEVICE)
    y = torch.randn([1500, 600], device=DEVICE)
    run_example(concat2d_dim1, lambda x, y: torch.cat([x, y], dim=1), (x, y))


if __name__ == "__main__":
    main()
