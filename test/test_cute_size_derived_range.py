"""CuTe ranges whose length is computed from a reduction dim's size (CPU codegen).

CuTe runs an ``hl.arange`` over the lanes of the dim registered for its
shape, which keeps the length traced over the reduction dim's block size:
``hl.arange(row.size(1) // 2)`` for ``row = x[tile, :]`` at 129 columns, in
64-column chunks, ran 32 lanes where the row's extent gives 64.  CuTe refuses
such a length unless the block size gives it the value the extent gives, any
tile size in it taken from the config on both sides.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def half_row_iota(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        n = row.size(1) // 2
        out[tile] = row.sum(-1) + hl.arange(n).to(torch.float32).sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_per_tile_row_iota(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        n = row.size(1) // row.size(0)
        out[tile] = row.sum(-1) + hl.arange(n).to(torch.float32).sum(-1)
    return out


@pytest.mark.parametrize(
    "kernel",
    [
        # Half of 128 columns, in one 128-column chunk: 64.
        pytest.param(half_row_iota, id="half"),
        # 128 columns over the 16-row tile: 8, the tile's size from the config
        # on the extent's side too.
        pytest.param(row_per_tile_row_iota, id="per_tile_row"),
    ],
)
def test_a_length_the_block_gives_as_the_extent_does_renders(
    kernel: helion.Kernel,
) -> None:
    bound = cpu_bind(kernel, (torch.empty(64, 128),))
    bound.to_code(helion.Config(block_sizes=[16], reduction_loops=[128]))


@pytest.mark.parametrize(
    "kernel,columns",
    [
        # Half a 64-column chunk is 32; half of 129 columns is 64.
        pytest.param(half_row_iota, 129, id="half"),
        # A 64-column chunk over the 16-row tile is 4; 128 columns give 8.
        pytest.param(row_per_tile_row_iota, 128, id="per_tile_row"),
    ],
)
def test_a_length_the_block_gives_otherwise_is_refused(
    kernel: helion.Kernel, columns: int
) -> None:
    bound = cpu_bind(kernel, (torch.empty(64, columns),))
    with pytest.raises(
        helion.exc.BackendUnsupported, match="computed from a reduction dim's size"
    ):
        bound.to_code(helion.Config(block_sizes=[16], reduction_loops=[64]))
