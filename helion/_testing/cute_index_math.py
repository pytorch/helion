"""The kernels shared by the CuTe index-math tests: the per-channel gather of a
flat NCHW tile, ``scale[(i // (H * W)) % C]`` (one quotient per vector, proven
unsigned and unguarded), and a row index scaled past Int32 under int32 and
int64 indexing."""

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
