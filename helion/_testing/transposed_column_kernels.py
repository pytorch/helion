"""The kernel the ``cute_transposed_column_reduction`` seed test modules share:
the copy of a permuted tensor and its sums over the permuted dim, the first
kernel of a split column sum that helion.inductor emits for the clone of
``x.permute(...)`` and the sum that follows it."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def copy_and_column_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The contiguous copy of the ``[splits, rows, columns]`` input and the
    fp32 sums of each split's rows (dim 1); the tests pass an input whose
    stride-1 dim is the rows (``torch.randn(s, c, r).permute(0, 2, 1)``)."""
    splits, rows, columns = x.size()
    y = torch.empty([splits, rows, columns], dtype=x.dtype, device=x.device)
    total = torch.empty([splits, columns], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, columns]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            v = x[tile_s, tile_r, tile_c].to(torch.float32)
            y[tile_s, tile_r, tile_c] = v.to(x.dtype)
            acc = acc + torch.sum(v, dim=1)
        total[tile_s, tile_c] = acc
    return y, total


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_sums(x: torch.Tensor) -> torch.Tensor:
    """The fp32 sums of each ``[splits, columns]`` row of a contiguous
    ``[splits, columns, rows]`` input over its last dim, walked by a device
    loop: the loop's tile is the input's stride-1 dim, but it is the summed
    value's innermost dim too (a row sum, not a column sum)."""
    splits, columns, rows = x.size()
    total = torch.empty([splits, columns], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, columns]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            acc = acc + torch.sum(x[tile_s, tile_c, tile_r].to(torch.float32), dim=-1)
        total[tile_s, tile_c] = acc
    return total


def permuted(
    splits: int, rows: int, columns: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """A ``[splits, rows, columns]`` tensor whose stride-1 dim is the rows."""
    return torch.randn(splits, columns, rows, device=device, dtype=dtype).permute(
        0, 2, 1
    )
