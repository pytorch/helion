"""The ``cute_cluster_n`` split of a lane-looped tile axis across a thread-block cluster.

Role: planner (the ``cute_cluster_n`` decision of a per-thread strategy).  ``PerThreadNDTileStrategy`` asks the backend once per
strategy, before its first device loop, whether the knob applies
(``CuteBackend.apply_lane_cluster_split``); when it does, the loop's single
lane-looped axis divides its elements across ``cluster_n`` CTAs in addition
to the CTA's own threads (``LaneLoopContext.cluster_n``) and the device
function's ``cute_state.simt_cluster_n`` is set so the host-side call emits
the extra grid dim and the cluster launch shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..host_function import HostFunction

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import PerThreadNDTileStrategy


def apply_cluster_split(
    strategy: PerThreadNDTileStrategy, env: CompileEnvironment, state: CodegenState
) -> None:
    """Record the cluster split of ``strategy``'s lane-looped axis when the
    ``cute_cluster_n`` knob applies to it: a single lane-looped axis whose
    one tile covers the whole extent, no atomics in the kernel, no sibling
    axis on a thread dim, a warp-multiple thread count and a per-CTA slice
    that keeps a whole vec chunk per thread.
    """
    cl = strategy.fn.config.config.get("cute_cluster_n", 1)
    if not isinstance(cl, int) or cl <= 1:
        return
    if strategy.fn.cute_state.lanes.simt_cluster_n > 1:
        # Another device loop's strategy already claimed the cluster
        # rank; splitting a second axis on the same rank would leave
        # only the diagonal rank x rank blocks covered.
        return
    if len(strategy._lane_var_by_block) != 1:
        return
    # Statements outside the cluster-split loop execute once per
    # cluster CTA — benign for plain (idempotent) stores, but a
    # read-modify-write would repeat ``cluster_n`` times.
    if HostFunction.current().device_ir.has_atomic_ops():
        return
    # The cluster reduce combines across the WHOLE CTA, so the grid
    # strategy must not put sibling axes on thread dims (e.g. a
    # multi-row grid tile with block_m > 1) — those rows would be
    # folded together.
    grid_axis_sizes = getattr(
        state.codegen.current_grid_state, "thread_axis_sizes", None
    )
    if isinstance(grid_axis_sizes, dict) and any(
        size > 1 for size in grid_axis_sizes.values()
    ):
        return
    [block_id] = strategy._lane_var_by_block.keys()
    vec = strategy.lane_context(block_id).vec_width
    idx = strategy.block_ids.index(block_id)
    nt = strategy._shared_thread_extents.get(block_id, strategy.num_threads[idx])
    static_bs = strategy._configured_block_size_int(strategy.block_size[idx])
    if static_bs is None or nt <= 0 or nt % 32 != 0:
        return
    numel = env.block_sizes[block_id].numel
    try:
        numel_int = int(numel)
    except (TypeError, ValueError):
        return
    # Single trip (the whole extent in one tile) so each cluster reduce
    # call site executes exactly once (fixed mbarrier phase), and the
    # per-CTA slice must keep a whole vec chunk per thread.
    if static_bs < numel_int or static_bs % (nt * cl * vec) != 0:
        return
    # All sibling axes must not use a thread axis (the cluster reduce
    # combines across the whole CTA).
    for other_id, other_bs in zip(strategy.block_ids, strategy.block_size, strict=True):
        if other_id == block_id:
            continue
        extent = strategy._static_thread_extent_for_block(other_id, other_bs)
        if extent is not None and extent != 1:
            return
    strategy.lane_context(block_id).cluster_n = cl
    strategy.fn.cute_state.lanes.simt_cluster_n = cl
