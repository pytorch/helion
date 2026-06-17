"""
Tensor Concatenation Examples
=============================

This example demonstrates two approaches to implementing tensor concatenation in Helion:
a simple version that tiles rows and uses slices for each source tensor, and a masked
version that tiles both dimensions using ``hl.load`` with ``extra_mask``.
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
# Simple Concatenation Kernel
# ---------------------------
# Tiles only the row dimension and uses slices to copy each source tensor
# into the corresponding region of the output. This produces two separate
# load/store pairs with no manual masking.


# %%
@helion.kernel()
def concat2d_dim1_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two 2D tensors along dimension 1 using separate stores.

    Tiles the row dimension and writes each source tensor into its
    corresponding slice of the output, avoiding manual masking.

    Args:
        x: First input tensor of shape [M, N1]
        y: Second input tensor of shape [M, N2] with same first dimension as x

    Returns:
        Output tensor of shape [M, N1+N2]
    """
    assert x.size(0) == y.size(0)
    out = torch.empty(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    n1 = x.size(1)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :n1] = x[tile_m, :]
        out[tile_m, n1:] = y[tile_m, :]
    return out


# %%
# Masked Concatenation Kernel
# ----------------------------
# Tiles both dimensions of the output. Because a single tile along the
# column dimension can span both source tensors, ``hl.load`` with
# ``extra_mask`` is used to selectively load from each source, and
# ``torch.where`` merges the results.


# %%
@helion.kernel()
def concat2d_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two 2D tensors along dimension 1 using masked loads.

    Tiles both dimensions and uses ``hl.load`` with ``extra_mask`` to
    handle tiles that straddle the boundary between the two source tensors.

    Args:
        x: First input tensor of shape [M, N1]
        y: Second input tensor of shape [M, N2] with same first dimension as x

    Returns:
        Output tensor of shape [M, N1+N2]
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
    kernels = {
        "simple": concat2d_dim1_simple,
        "masked": concat2d_dim1,
    }
    run_example(kernels, lambda x, y: torch.cat([x, y], dim=1), (x, y))


if __name__ == "__main__":
    main()
