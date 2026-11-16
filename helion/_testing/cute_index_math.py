"""The kernels shared by the CuTe index-math tests: the per-channel gather of a
flat NCHW tile, ``scale[(i // (H * W)) % C]`` (one quotient per vector, proven
unsigned and unguarded), a row index scaled past Int32 under int32 and int64
indexing, and the split moments of an NCHW input read through a
de-linearized ``x[r // (H * W), r % (H * W), c]`` (its masks compared once
per vector under ``cute_uniform_lane_compares``)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from .cute_codegen import cpu_bind
import helion
import helion.language as hl

if TYPE_CHECKING:
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def flat_channel_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = x.shape
    flat = x.view(-1)
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        channel = (tile.index // (height * width)) % channels
        out[tile] = (flat[tile].to(torch.float32) * scale[channel]).to(out.dtype)
    return out.view_as(x)


def flat_channel_config(
    bound: BoundKernel[Any], *, proven_bounds: bool = False
) -> helion.Config:
    """1024-element tiles over 128 threads in 8-wide vectors."""
    return bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[1024],
            num_threads=[128],
            cute_vector_widths=[8],
            cute_proven_bounds=proven_bounds,
        )
    )


def flat_channel_code(side: int, *, proven_bounds: bool = False) -> str:
    """The source ``flat_channel_scale`` renders on CPU for a
    [4, 24, side, side] bf16 input under ``flat_channel_config``."""
    bound = cpu_bind(
        cast("Kernel[object]", flat_channel_scale),
        (
            torch.empty(4, 24, side, side, dtype=torch.bfloat16),
            torch.empty(24, dtype=torch.float32),
        ),
    )
    return bound.to_code(flat_channel_config(bound, proven_bounds=proven_bounds))


def _row_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.int64)
    for tile in hl.tile(x.size(0)):
        out[tile] = (tile.index // 8) * 1048576
    return out


# ``(i // 8) * 2**20`` passes Int32 from i = 2**14 on; int64 indexing keeps it
# Int64.
row_scale = {
    index_dtype: helion.kernel(
        _row_scale,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        index_dtype=index_dtype,
    )
    for index_dtype in (torch.int32, torch.int64)
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def split_row_moments(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [N, H*W, C], a permuted view of an NCHW tensor (H*W has stride 1).
    # Split s of 4 sums the flat (n, h, w) elements s * rows onwards in a
    # device loop, as BatchNorm's split Welford partials do: per channel,
    # the count, the sum and the sum of squares of each split's elements.
    batch, hw, channels = x.shape
    total = batch * hw
    rows = (total + 3) // 4
    count = torch.empty([4, channels], dtype=torch.float32, device=x.device)
    sums = torch.empty_like(count)
    squares = torch.empty_like(count)
    for tile_s, tile_c in hl.tile([4, channels]):
        n = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        s1 = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        s2 = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            r = tile_s.index[:, None] * rows + tile_r.index[None, :]
            valid = (r < total) & (tile_r.index[None, :] < rows)
            values = hl.load(
                x, [r // hw, r % hw, tile_c], extra_mask=valid[:, :, None]
            ).to(torch.float32)
            n = n + torch.sum(valid.to(torch.float32), dim=1)[:, None]
            s1 = s1 + torch.sum(values, dim=1)
            s2 = s2 + torch.sum(values * values, dim=1)
        count[tile_s, tile_c] = n
        sums[tile_s, tile_c] = s1
        squares[tile_s, tile_c] = s2
    return count, sums, squares


# The same kernel under dynamic shapes: the split length and the row bounds
# are kernel arguments.
split_row_moments_dynamic = helion.kernel(
    split_row_moments.fn, backend="cute", static_shapes=False, autotune_effort="none"
)


def split_row_config(
    bound: BoundKernel[Any], *, uniform_compares: bool | None = False
) -> helion.Config:
    """128-row chunks of the device loop over 16 threads in 8-wide vectors;
    ``cute_uniform_lane_compares`` left out when ``uniform_compares`` is
    None."""
    return bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[1, 1, 128],
            num_threads=[1, 0, 16],
            cute_vector_widths=[1, 1, 8],
            cute_uniform_lane_compares=uniform_compares,
        )
    )


def split_row_input(
    batch: int, hw: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """A [batch, 16, hw] bf16 NCHW input read as [batch, hw, 16]."""
    planes = torch.randn(batch, 16, hw, device=device).to(torch.bfloat16)
    return planes.permute(0, 2, 1)


def split_row_code(*, uniform_compares: bool | None = False) -> str:
    """The source ``split_row_moments`` renders on CPU for a [6, 16, 192]
    input (splits of 288 rows: two whole chunks and a partial one) under
    ``split_row_config``."""
    bound = cpu_bind(
        cast("Kernel[object]", split_row_moments), (split_row_input(6, 192),)
    )
    return bound.to_code(split_row_config(bound, uniform_compares=uniform_compares))
