"""Thread budget validation for CuTe layout planning.

Re-exports the per-block thread limit of the shared per-thread planner
(``per_thread_planning.MAX_THREADS_PER_BLOCK`` / ``check_thread_limit``) for
the CuTe layout passes, and hosts the two register-tile admission inputs the
reduction strategies look up at call time: ``shared_memory_budget_bytes`` (the device's per-block shared memory,
opt-in when larger) and ``register_tile_shape`` (whether a scalar synthetic
reduction lane unrolls as a per-thread register tile).  Both are
``mock.patch`` seams -- the SMEM-probe tests and
``helion._testing.patch_target_facts`` patch the budget, the register-tile
tests patch the gate -- so patch them on this module, not on their callers.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import torch

from ... import exc
from ..per_thread_planning import MAX_THREADS_PER_BLOCK
from ..per_thread_planning import check_thread_limit
from .simplify_proven_bounds import _axis_index
from .state import cute_codegen_state
from .state import cute_env_state

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState

__all__ = [
    "CUTE_REGISTER_TILE_MAX_ELEMENTS",
    "CUTE_REGISTER_TILE_MAX_LIVE_VALUES",
    "MAX_THREADS_PER_BLOCK",
    "MAX_THREAD_BLOCK_DIMS",
    "check_thread_block_dims",
    "check_thread_limit",
    "checked_thread_block_dims",
    "record_root_read_axes",
    "register_tile_shape",
    "shared_memory_budget_bytes",
    "thread_idx_axis",
    "tile_loop_thread_count",
]

# CUDA's per-axis limits on a thread block: 1024 along x and y, 64 along z.
MAX_THREAD_BLOCK_DIMS = (1024, 1024, 64)
# Largest per-thread register tile (reduction lanes x vector elements) that a
# scalar synthetic reduction lane unrolls at trace time outside the one-vector
# tile wrappers (``DeviceGridState.nest_reduction_lane_outside_vector_tiles``).
# Every unrolled element of every loaded tensor stays live across the register
# tile, so wider products keep the rolled lane loop.
CUTE_REGISTER_TILE_MAX_ELEMENTS = 64
# Largest number of values one register tile keeps live across its unrolled
# element loops: the intermediates stashed for the later nests (one register
# per distinct value of every stashed name, i.e. the product of the extents of
# the loops the value depends on; a per-row value is held once per lane, a
# per-column value once per element, and the compiler merges the unrolled
# duplicates) plus the per-element accumulators of every reduction.  Three
# full tiles already approach the 255-register file; wider tiles reject the
# config instead of spilling.
CUTE_REGISTER_TILE_MAX_LIVE_VALUES = 3 * CUTE_REGISTER_TILE_MAX_ELEMENTS


def thread_idx_axis(node: ast.AST) -> int | None:
    """The axis of a ``cute.arch.thread_idx()[k]`` subscript, else ``None``."""
    return _axis_index(node, "thread_idx")


def referenced_thread_axes(statements: Iterable[ast.AST]) -> list[int]:
    """The axis of every ``cute.arch.thread_idx()[axis]`` in ``statements``.

    A structural walk with ``thread_idx_axis`` (a subscript of the bare call
    by an integer literal in ``range(3)``). An axis may be reported more than
    once; the callers take maxima or sets.
    """
    return [
        axis
        for statement in statements
        for node in ast.walk(statement)
        if (axis := thread_idx_axis(node)) is not None
    ]


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
            order = (
                env.config_spec.loop_orders.config_get(config.loop_orders, block_ids[0])
                if len(block_ids) > 1
                else None
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
        # A launch has three thread axes; tiles beyond them run as single
        # thread lane loops (``PerThreadNDTileStrategy`` demotes them) and
        # add no threads.
        del extents[3:]
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


def record_root_read_axes(
    codegen: GenerateAST, grid_state: DeviceGridState, body: list[ast.AST]
) -> None:
    """Record the launch axes the root loop ``grid_state`` reads with their
    thread extents (``CuteCodegenState.root_read_axis_sizes``): its index
    setup in the
    statement list that receives the body, the body, the lane setup and
    the prefix and suffix around the lane loops."""
    statements: list[ast.AST] = [
        *(grid_state.hoist_parent_statements or ()),
        *grid_state.outer_prefix,
        *grid_state.lane_setup_statements,
        *body,
        *grid_state.outer_suffix,
    ]
    read_axes = set(referenced_thread_axes(statements))
    cute_codegen_state(codegen).root_read_axis_sizes.append(
        {
            axis: size
            for axis, size in grid_state.root_axis_sizes.items()
            if axis in read_axes
        }
    )


def shared_memory_budget_bytes() -> int:
    """The current CUDA device's per-block shared memory budget (opt-in when
    larger).  Read through this module at call time: tests patch it here."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    default_shared = int(props.shared_memory_per_block)
    optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(default_shared, optin_shared)


def register_tile_shape(fn: DeviceFunction, block_index: int, lane_extent: int) -> bool:
    """Whether a scalar synthetic reduction lane of ``lane_extent`` over
    reduction block ``block_index`` can nest outside the tile lane loops of
    ``fn`` as a per-thread register tile.

    Mirrors ``DeviceGridState.nest_reduction_lane_outside_vector_tiles`` from
    the tile strategies' construction-time state: every tile block that has a
    lane loop must be vectorized with one V-wide fragment per thread (elements
    per thread == V), at least one such block must exist, no matmul lowering
    may replace the lane bodies, the unrolled per-thread element count stays
    within ``CUTE_REGISTER_TILE_MAX_ELEMENTS``, the device IR admits the
    reduction block (``ConfigSpec.cute_register_tile_reduction_blocks``) and no
    earlier pass over this kernel rejected the register tile
    (``CuteEnvState.register_tile_disabled``).
    """
    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    env = CompileEnvironment.current()
    if (
        cute_env_state(env).register_tile_disabled
        or env.config_spec.matmul_facts
        or block_index not in env.config_spec.cute_register_tile_reduction_blocks
    ):
        return False
    unrolled = lane_extent
    found = False
    for strategy in fn.tile_strategy.strategies:
        if isinstance(strategy, PerThreadFlattenedTileStrategy):
            if strategy._lane_var is None:
                continue
            vec_width = strategy.lane_context(strategy.block_ids[-1]).vec_width
            if vec_width <= 1 or strategy._elements_per_thread != vec_width:
                return False
            unrolled *= vec_width
            found = True
            continue
        if not isinstance(strategy, PerThreadNDTileStrategy):
            continue
        for block_id in strategy._lane_var_by_block:
            vec_width = strategy.lane_context(block_id).vec_width
            if (
                vec_width <= 1
                or strategy._elements_per_thread_for_block(block_id) != vec_width
            ):
                return False
            unrolled *= vec_width
            found = True
    return found and unrolled <= CUTE_REGISTER_TILE_MAX_ELEMENTS
