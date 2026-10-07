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
# CUDA's per-axis limits on a thread block: 1024 along x and y, 64 along z.
MAX_THREAD_BLOCK_DIMS = (1024, 1024, 64)
# Largest per-thread register tile (reduction lanes x vector elements) that a
# scalar synthetic reduction lane unrolls at trace time outside the one-vector
# tile wrappers (``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).
# Every unrolled element of every loaded tensor stays live across the register
# tile, so wider products keep the rolled lane loop.
CUTE_REGISTER_TILE_MAX_ELEMENTS = 64
# Largest number of values one register tile keeps live across its unrolled
# element loops: the intermediates stashed for the consume pass (one entry per
# lane per element for every stashed name) plus the per-element accumulators of
# every reduction, one register each.  Three full tiles already approach the
# 255-register file; wider tiles reject the config instead of spilling.
CUTE_REGISTER_TILE_MAX_LIVE_VALUES = 3 * CUTE_REGISTER_TILE_MAX_ELEMENTS


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


def check_thread_block_dims(dims: Sequence[int], *, context: str = "") -> None:
    """Raise ``BackendUnsupported`` for a launch shape CUDA cannot start.

    A thread block is limited along each axis (1024, 1024, 64) as well as in
    total (:func:`check_thread_limit`); a config whose thread counts land a
    wide tile axis on z (``num_threads=[2, 128]`` launching ``block=(1, 2,
    128)``) passed the total and failed at launch with
    ``cudaErrorInvalidValue``.  Every ``block=(x, y, z)`` the backend emits
    goes through here, except the symbolic shape a kernel argument sizes,
    which the host wrapper checks at launch through
    :func:`checked_thread_block_dims`.
    """
    from ..compile_environment import CompileEnvironment

    for axis, (size, limit) in enumerate(zip(dims, MAX_THREAD_BLOCK_DIMS, strict=True)):
        if size > limit:
            backend_name = CompileEnvironment.current().backend.name
            raise exc.BackendUnsupported(
                backend_name,
                f"thread block axis {axis} of {context or tuple(dims)} exceeds "
                f"the launch limit of {limit}",
            )
    check_thread_limit(math.prod(dims), context=context or str(tuple(dims)))


def checked_thread_block_dims(dims: Sequence[int]) -> tuple[int, int, int]:
    """Validate a launch shape only known at launch time and hand it back.

    A thread extent that is a kernel argument (``hl.tile(n, block_size=bsz)``
    with ``bsz`` an int argument) renders the launch shape symbolically, so
    :func:`check_thread_block_dims` cannot see it at codegen; the host wrapper
    wraps that ``block=`` tuple in this call instead.  It runs outside any
    compile environment, so the backend is spelled out.
    """
    x, y, z = (int(dim) for dim in dims)
    for axis, (size, limit) in enumerate(
        zip((x, y, z), MAX_THREAD_BLOCK_DIMS, strict=True)
    ):
        if size > limit:
            raise exc.BackendUnsupported(
                "cute",
                f"thread block axis {axis} of {(x, y, z)} exceeds the launch "
                f"limit of {limit}",
            )
    if x * y * z > MAX_THREADS_PER_BLOCK:
        raise exc.BackendUnsupported(
            "cute", f"thread block too large for cute kernel: {(x, y, z)}"
        )
    return x, y, z


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
