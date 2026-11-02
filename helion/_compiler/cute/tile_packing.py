"""Several grid tiles per CTA as thread sub-groups (``cute_tiles_per_block``).

A kernel whose grid axis is fixed at one element per tile (``hl.tile(M,
block_size=1)``, ``hl.register_block_size(1, 1)``) launches one CTA per row.
At thousands of tiny tiles the kernel is bound by CTA dispatch, not by the
bytes it moves: the gemv of ``examples/nvfp4_gemv.py`` reads 2 KB per CTA
and Triton packs 4-16 rows per program.  With ``cute_tiles_per_block = T``
the flat program id of that axis becomes ``block_idx * T + thread_idx()[a]``
on a thread axis ``a`` of extent ``T``: every thread sub-group owns its own
tile, the launch grid shrinks by ``T``, and the tile's begin/index values are
per thread rather than per CTA (``block_uniform`` stops treating them as
uniform; cross-thread reductions below the axis see ``T`` interleaved
siblings).  ``T`` must divide the static tile count, so no tail masks are
needed; the pid type must be flat, no CTA cluster split may be active and no
L2 grouping (``L2GroupingProgramIDs`` decodes the pid without the pack term).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...runtime.config import Config
    from ..device_ir import DeviceIR

TILES_PER_BLOCK_CHOICES: tuple[int, ...] = (1, 2, 4, 8)


def tile_pack_counts(env: CompileEnvironment, device_ir: DeviceIR) -> dict[int, int]:
    """Static tile counts of the root-grid axes fixed at one element per tile."""
    if len(device_ir.grid_block_ids) != 1:
        return {}
    counts: dict[int, int] = {}
    pinned_to_one = {
        spec.block_id
        for spec in env.config_spec.block_sizes
        if spec.min_size == 1 and spec.max_size == 1
    }
    for block_id in device_ir.grid_block_ids[0]:
        info = env.block_sizes[block_id]
        source = info.block_size_source
        fixed_one = isinstance(source, FixedBlockSizeSource) and source.value == 1
        if not fixed_one and block_id not in pinned_to_one:
            continue
        if not isinstance(info.size, (int, torch.SymInt)):
            # A data-dependent bound (``hl.grid(tensor)``) has no static count.
            continue
        numel = env.specialize_expr(sympy.sympify(info.numel))
        if isinstance(numel, sympy.Integer) and int(numel) > 1:
            counts[block_id] = int(numel)
    return counts


def l2_grouped(config: Mapping[str, object]) -> bool:
    """Whether any grid axis of ``config`` uses an L2 grouping above one."""
    groupings = config.get("l2_groupings", ())
    return isinstance(groupings, (list, tuple)) and any(
        isinstance(group, int) and group > 1 for group in groupings
    )


def packed_grid_block(
    env: CompileEnvironment, config: Config
) -> tuple[int, int] | None:
    """``(block_id, tiles_per_block)`` of the packed grid axis under ``config``."""
    pack = config.get("cute_tiles_per_block", 1)
    if type(pack) is not int or pack <= 1:
        return None
    if config.get("pid_type", "flat") != "flat":
        return None
    if config.get("cute_cluster_n", 1) != 1:
        return None
    if l2_grouped(config):
        return None
    for block_id, count in sorted(env.config_spec.cute_tile_pack_counts.items()):
        if count % pack == 0:
            return block_id, pack
    return None
