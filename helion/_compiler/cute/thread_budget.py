"""Thread budget validation for CuTe layout planning.

Centralizes the 1024-thread-per-block limit enforcement that was
previously scattered across backend.py and tile_strategy.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo

MAX_THREADS_PER_BLOCK = 1024


def tile_loop_thread_count(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    graphs: Sequence[GraphInfo],
    config: Config,
    *,
    inactive_block_ids: set[int] | frozenset[int] = frozenset(),
) -> int:
    """Estimate the launch budget for the configured non-reduction tiles.

    Sibling loops reuse a hardware axis; nested loops multiply the budget.
    The launch must accommodate the maximum extent of *each* axis across
    paths, which can exceed the largest individual path's thread count.
    """
    from .loop_nesting import tile_loop_paths

    launch_extents: list[int] = []
    for path in tile_loop_paths(device_ir, graphs):
        extents: list[int] = []
        seen: set[int] = set()
        for block_ids in path:
            order = env.config_spec.loop_orders.config_get(
                config.loop_orders, block_ids[0]
            ) or range(len(block_ids))
            for position in order:
                block_id = block_ids[position]
                if block_id in seen or block_id in inactive_block_ids:
                    continue
                seen.add(block_id)
                info = env.block_sizes[block_id]
                if info.reduction:
                    continue
                size = info.from_config(config)
                if not isinstance(size, int):
                    continue
                threads = int(
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                )
                extent = threads if threads > 0 else size
                if extent > 1:
                    extents.append(extent)
        for axis, extent in enumerate(extents):
            if axis == len(launch_extents):
                launch_extents.append(extent)
            else:
                launch_extents[axis] = max(launch_extents[axis], extent)
    return math.prod(launch_extents)


def check_thread_limit(
    num_threads: int,
    *,
    context: str = "",
) -> None:
    """Raise ``BackendUnsupported`` if *num_threads* exceeds 1024.

    This is the single source of truth for the CuTe thread-per-block limit.
    Both the scattered checks in ``backend.py`` and the layout planner call
    this function.

    Args:
        num_threads: Concrete thread count to validate.
        context: Human-readable description for the error message
                 (e.g. block sizes or node name).
    """
    if num_threads > MAX_THREADS_PER_BLOCK:
        from ..compile_environment import CompileEnvironment

        backend_name = CompileEnvironment.current().backend.name
        msg = f"thread block too large for {backend_name} kernel: {context or num_threads}"
        raise exc.BackendUnsupported(backend_name, msg)
