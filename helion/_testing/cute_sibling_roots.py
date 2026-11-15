"""Two sibling root loops whose launch one of them widens.

Shared by the CuTe launch-sharing codegen test
(``test/test_cute_multigrid_reduction_threads.py``) and its GPU twin
(``test/test_cute_tile_dim_reduction_threads.py``): the first root reduces
over a 128-thread column tile, the second tiles 512 rows on the same thread
axis, so the column tile's reduction runs widened to the launch the roots
share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from ..runtime.kernel import BoundKernel


def row_sums_beside_a_wider_tile(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two sibling root loops on one thread axis: the first reduces over a
    128-thread column tile, the second tiles 512 rows."""
    m, n = x.size()
    partial = torch.empty([m], dtype=torch.float32, device=x.device)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        partial[tile_m] = x[tile_m, tile_n].sum(dim=1)
    for tile_m in hl.tile(m):
        out[tile_m] = y[tile_m, :].sum(-1)
    return partial, out


def sibling_config(bound: BoundKernel, pid_type: str) -> helion.Config:
    """The column tile on 128 threads of axis 1 beside a 512-row tile there."""
    config = dict(bound.config_spec.default_config().config)
    config.update(block_sizes=[1, 128, 512], reduction_loops=[128], pid_type=pid_type)
    return helion.Config.from_dict(config)
