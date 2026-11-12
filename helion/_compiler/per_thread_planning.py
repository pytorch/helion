"""The per-thread tile planner shared by the CuTe and Metal backends.

Both backends run one element per thread with cooperative hardware primitives
for matmul, so they build their tile loops with the same
``PerThreadNDTileStrategy`` / ``PerThreadFlattenedTileStrategy`` planning:
:func:`plan_per_thread_loop_strategy` resolves the thread count of every axis
of one tile loop under the 1024-thread budget (``MAX_THREADS_PER_BLOCK``),
drops the axes no statement observes (:func:`tile_loop_inactive_block_ids`)
and reserves the threads a matmul's contraction axis needs on the scalar
fallback (:func:`matmul_contraction_thread_reserve`).

The backend supplies the policy: the grid axes that carry no threads
(``Backend.boundary_only_grid_block_ids``), the live thread total
(``tile_loop_thread_count``), the tile loops that share thread axes
(``tile_loop_paths``), the threads each launch axis takes
(``thread_axis_extent_limits``) and whether a reduction lays the tiles out
itself (``per_thread_reduction_lays_out_tiles``), the packed grid axis
(``packed_grid_block``), the reduction axis
reservation (``reduction_axis_first``, ``max_reduction_threads``) for the
reductions that get a strategy (``reduction_strategy_block_ids``, through
``constexpr_live_graph_ids``) and the tensor-core detectors
(``per_thread_mma_candidate_loop``, ``per_thread_matmul_loop_has_mma_path``,
``per_thread_mma_mode``, ``per_thread_root_thread_counts``), whose core
defaults describe a backend without a tensor-core path.  The matmul operand
facts the planner reads are :mod:`helion._compiler.matmul_utils`'s
(``matmul_contraction_block_ids``, ``kernel_has_half_precision_matmul``).
"""

from __future__ import annotations

import functools
from itertools import starmap
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from .. import exc
from .backend import _active_loop_block_ids
from .backend import _largest_divisor_at_most
from .backend import _loop_contains_matmul

# ``matmul_utils`` (the matmul operand facts) is imported at call time: this
# module loads while ``compile_environment`` is still initializing.
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .backend import Backend
    from .device_function import DeviceFunction
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .tile_strategy import TileStrategy

# The CUDA / Metal hard cap on threads per block (threadgroup).
MAX_THREADS_PER_BLOCK = 1024


def check_thread_limit(
    num_threads: int,
    *,
    context: str = "",
) -> None:
    """Raise ``BackendUnsupported`` if *num_threads* exceeds 1024.

    The single source of truth for the per-block thread limit of the
    per-thread backends: the planner below and the CuTe layout passes call it.

    Args:
        num_threads: Concrete thread count to validate.
        context: Human-readable description for the error message
                 (e.g. block sizes or node name).
    """
    if num_threads > MAX_THREADS_PER_BLOCK:
        from .compile_environment import CompileEnvironment

        backend_name = CompileEnvironment.current().backend.name
        msg = f"thread block too large for {backend_name} kernel: {context or num_threads}"
        raise exc.BackendUnsupported(backend_name, msg)


# -- matmul-contraction thread reserve ----------------------------------------
# A reduction block that is the contraction (K) axis of a matmul lowered
# through the scalar fallback must keep its full extent on real hardware
# threads (OPTION B in ``plan_per_thread_loop_strategy``): the cross-thread
# reduction sums the thread lanes only, never a synthetic lane loop's
# partials.  Only ``backend.max_reduction_threads()`` is backend-dependent.


def matmul_contraction_reduction_block_ids() -> set[int]:
    """Reduction block ids that are also a matmul-contraction (K) axis.

    These are the blocks that must keep real threads for the whole K extent
    instead of being split into ``threads x synthetic-lane`` (see OPTION B in
    :func:`plan_per_thread_loop_strategy` and
    ``matmul_contraction_block_ids``).
    """
    from .compile_environment import CompileEnvironment
    from .matmul_utils import matmul_contraction_block_ids

    env = CompileEnvironment.current()
    contraction = matmul_contraction_block_ids()
    if not contraction:
        return set()
    result: set[int] = set()
    for info in env.block_sizes:
        if not info.reduction:
            continue
        block_id = env.canonical_block_id(info.block_id)
        # Reduction lowering may materialize an output-range block whose
        # extent aliases an already-active *tile* block.  Such an alias
        # reuses the tile strategy and must not reserve a second copy of
        # the contraction threads.  Canonical reduction aliases, on the
        # other hand, still need one (deduplicated) reserve.
        if block_id in contraction and env.block_sizes[block_id].reduction:
            result.add(block_id)
    return result


def extent_size_hint(block_id: int) -> int:
    """The size hint of block ``block_id``'s extent."""
    from .._compat import shape_env_size_hint
    from .compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    numel = env.block_sizes[block_id].numel
    if isinstance(numel, (int, sympy.Integer)):
        return int(numel)
    if isinstance(numel, sympy.Expr):
        return shape_env_size_hint(env.shape_env, numel)
    return env.size_hint(numel)


def matmul_contraction_extents(tile_block_ids: list[int]) -> list[int]:
    """Size hints of the matmul-contraction reduction axes (above one)
    that are *not* one of ``tile_block_ids``, i.e. handled by a separate
    reduction strategy, not this tile strategy."""
    tile_ids = set(tile_block_ids)
    extents: list[int] = []
    for block_id in matmul_contraction_reduction_block_ids():
        if block_id in tile_ids:
            continue
        size_hint = extent_size_hint(block_id)
        if size_hint > 1:
            extents.append(size_hint)
    return extents


def matmul_contraction_thread_reserve(
    backend: Backend, tile_block_ids: list[int]
) -> int:
    """Threads to reserve for matmul-contraction reduction axes.

    Returns the product of the per-axis full thread extents (power-of-two,
    capped at ``backend.max_reduction_threads()``) of every reduction block
    that is a matmul-contraction axis and is *not* one of ``tile_block_ids``
    (i.e. it is handled by a separate reduction strategy, not this tile
    strategy).
    """
    from torch._inductor.runtime.runtime_utils import next_power_of_2

    max_reduction_threads = backend.max_reduction_threads()
    if max_reduction_threads is None:
        return 1
    reserve = 1
    for size_hint in matmul_contraction_extents(tile_block_ids):
        reserve *= next_power_of_2(min(size_hint, max_reduction_threads))
    return reserve


def free_auto_thread_axis_count(fn: DeviceFunction, config: Config) -> int:
    """Count the kernel's free (non-reduction) tile axes that auto-thread.

    These are the axes that compete for the thread budget left over after a
    matmul-contraction reduction has reserved its slice (see OPTION B in
    :func:`plan_per_thread_loop_strategy`).  The reserve's ``thread_limit``
    shrink is applied per ``create_loop_strategy`` call, but a kernel may
    build the M and N tile axes in *separate* calls (e.g. M is the grid, N is
    a device loop).  Each call only sees its own axes, so dividing the
    per-call ``thread_limit`` by the reserve once is not enough: the product
    of every free axis' threads must stay within ``1024 // reserve``.
    Counting all the free auto-threaded axes lets each call take only its
    fair share.
    """
    from .compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    count = 0
    for block_id in _active_loop_block_ids(fn):
        info = env.block_sizes[block_id]
        if info.reduction:
            continue
        block_size = info.from_config(config)
        if not isinstance(block_size, int) or block_size <= 1:
            continue
        threads = int(
            env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
        )
        # Only auto-threaded (``num_threads == 0``) axes participate in the
        # budget split; explicitly-threaded axes keep their configured count.
        if threads == 0:
            count += 1
    return max(count, 1)


def fit_reduction_threads(threads: int, other_threads: int) -> int:
    """``threads`` of a reduction, halved until they fit in one CTA beside
    the ``other_threads`` of the tiles and reductions it runs with."""
    while other_threads * threads > MAX_THREADS_PER_BLOCK and threads > 1:
        threads //= 2
    return threads


def reductions_keep_threads(
    reductions: Iterable[tuple[int, int]], tile_threads: int
) -> bool:
    """Whether one of ``reductions``, each an ``(extent, num_threads)``
    pair, keeps several live threads beside ``tile_threads`` threads of
    tiles, so it claims a launch thread axis
    (``ReductionStrategy._claims_thread_axis``).

    The reductions are planned after the tiles.  Each asks for its
    ``num_threads`` (a power of two when that is 0, which halves through 2),
    which the backend halves until the reduction fits in the CTA beside the
    tiles (``fit_reduction_threads``, as
    ``Backend.adjust_reduction_thread_count`` does): it is left one thread
    when the config asks for one, when its extent is 1, or when the tiles
    take more than half the CTA.  The planner asks it of the configured
    tiles, and so does the CuTe autotuner's elements-per-thread shrink where
    it follows the planner.
    """
    return any(
        extent > 1 and fit_reduction_threads(requested or 2, tile_threads) > 1
        for extent, requested in reductions
    )


def _reduction_surely_claims_thread_axis(
    fn: DeviceFunction,
    backend: Backend,
    config: Config,
    tile_threads: int,
    reduction_block_ids: Iterable[int],
    tile_paths: Sequence[Sequence[tuple[int, ...]]],
) -> bool:
    """Whether one of ``reduction_block_ids``, the reductions that get a
    strategy, surely keeps several live threads under ``config``
    (:func:`reductions_keep_threads`), so it claims the leading thread axis.
    A conservative test: where it cannot tell, it says no.

    ``tile_threads``, the threads of the configured tiles
    (``Backend.tile_loop_thread_count``), bounds what the tiles take only on
    a single path of nested loops (``tile_paths``, the
    ``Backend.tile_loop_paths``): sibling loops and roots share axes, and
    the launch widens each to its widest sibling before the reductions take
    their share (``TileStrategyDispatch._normalize_shared_tile_thread_extents``),
    so beside them no reduction is sure to keep its threads.
    """
    from .compile_environment import CompileEnvironment

    # One path also stands for several loops over the same blocks, whose
    # shared strategy the reductions' fit counts once per loop
    # (``_loops_share_a_strategy``).
    if len(tile_paths) > 1 or _loops_share_a_strategy(fn):
        return False
    if (packed := backend.packed_grid_block(config)) is not None:
        # The packed grid axis takes ``tiles_per_block`` threads at block
        # size 1.
        tile_threads *= packed[1]
    env = CompileEnvironment.current()
    return reductions_keep_threads(
        (
            (
                extent_size_hint(block_id),
                int(
                    env.config_spec.num_threads.config_get(
                        config.num_threads, block_id, 0
                    )
                ),
            )
            for block_id in reduction_block_ids
        ),
        tile_threads,
    )


def _loops_share_a_strategy(fn: DeviceFunction) -> bool:
    """Whether two loops of ``fn``, roots or device loops, run over the same
    blocks, as loops sharing an ``hl.register_block_size`` do.  They run on
    one strategy, which ``TileStrategyDispatch.strategies`` lists once per
    loop, and ``Backend.adjust_reduction_thread_count`` counts every entry
    of that list, so the reductions get a smaller share of the CTA than one
    path of the loops suggests."""
    from .device_ir import ForLoopGraphInfo
    from .device_ir import ReductionLoopGraphInfo
    from .host_function import HostFunction

    blocks = [
        tuple(block_ids)
        for block_ids in HostFunction.current().device_ir.grid_block_ids
    ]
    blocks += [
        tuple(info.block_ids)
        for info in fn.codegen.codegen_graphs
        if isinstance(info, ForLoopGraphInfo)
        and not isinstance(info, ReductionLoopGraphInfo)
        and info.block_ids
    ]
    return len(set(blocks)) < len(blocks)


# -- the tensor-core contraction candidate ----------------------------------------

_TENSOR_CORE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def matmul_loop_contracts_root_grid(
    fn: DeviceFunction,
    block_ids: list[int],
    *,
    block_sizes: Sequence[int | torch.SymInt],
    num_threads_config: Sequence[int],
    grid_ids: set[int],
) -> bool:
    """Whether the device loop over ``block_ids`` is a tensor-core contraction
    candidate (``Backend.per_thread_mma_candidate_loop``'s default).

    True when ``block_ids`` is a device loop, the root grid has two or three
    axes none of which the loop serializes partially (a configured thread
    count between one and the block size), and the loop graph over exactly
    these block ids contains an accumulating matmul of fp32 / fp16 / bf16
    operands (``addmm`` / ``baddbmm`` / ``hl.dot``, or an ``mm`` / ``bmm``
    whose result an add consumes) whose output axes are the root grid's, in
    order, and whose contraction axis is one of ``block_ids``.  The planner
    then keeps the loop's axes active, its configured thread counts and the
    full thread budget: the loop is the kernel's K loop and the per-thread
    layout must stay the matmul's.  The CuTe backend answers the same question
    with its own operand analysis (``CuteBackend.per_thread_mma_candidate_loop``).
    """
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction
    from .matmul_utils import matmul_operand_indices

    if not any(block_id not in grid_ids for block_id in block_ids):
        return False
    device_ir = HostFunction.current().device_ir
    if len(device_ir.grid_block_ids) != 1:
        return False
    root_grid_block_ids = tuple(device_ir.grid_block_ids[0])
    if len(root_grid_block_ids) not in (2, 3):
        return False
    root_grid_ids = set(root_grid_block_ids)
    resolved_threads = [
        threads
        if threads > 0
        else int(block_size)
        if isinstance(block_size, int)
        else 0
        for block_size, threads in zip(block_sizes, num_threads_config, strict=True)
    ]
    # A root-grid axis the loop serializes partially would reuse one
    # tensor-core fragment for several logical lanes.
    if any(
        block_id in root_grid_ids and 0 < threads < block_size
        for block_id, block_size, threads in zip(
            block_ids, block_sizes, resolved_threads, strict=False
        )
    ):
        return False
    env = CompileEnvironment.current()
    operand_indices = matmul_operand_indices()
    accumulating_targets = {
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
    }
    add_targets = {torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor}
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        if graph_info.block_ids != block_ids:
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            indices = operand_indices.get(node.target)
            if indices is None:
                continue
            lhs_index, rhs_index = indices
            if len(node.args) <= max(lhs_index, rhs_index):
                continue
            lhs_arg, rhs_arg = node.args[lhs_index], node.args[rhs_index]
            if not isinstance(lhs_arg, torch.fx.Node) or not isinstance(
                rhs_arg, torch.fx.Node
            ):
                continue
            lhs, rhs = lhs_arg.meta.get("val"), rhs_arg.meta.get("val")
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                continue
            if lhs.ndim < 2 or rhs.ndim < 2:
                continue
            if (
                lhs.dtype not in _TENSOR_CORE_DTYPES
                or rhs.dtype not in _TENSOR_CORE_DTYPES
            ):
                continue
            k_block_id = env.resolve_block_id(lhs.shape[-1])
            if k_block_id is None or k_block_id not in block_ids:
                continue
            output_block_ids = tuple(
                env.resolve_block_id(size)
                for size in (*lhs.shape[:-2], lhs.shape[-2], rhs.shape[-1])
            )
            if output_block_ids != root_grid_block_ids:
                continue
            accumulates = (
                node.target in accumulating_targets
                or node.target
                not in (
                    torch.ops.aten.mm.default,
                    torch.ops.aten.bmm.default,
                    torch.ops.aten.bmm.dtype,
                )
                or any(user.target in add_targets for user in node.users)
            )
            if accumulates:
                return True
    return False


def kernel_has_contraction_candidate_loop(
    backend: Backend, fn: DeviceFunction, config: Config
) -> bool:
    """Whether any device loop of the kernel is a tensor-core contraction
    candidate (``Backend.per_thread_mma_candidate_loop``) under ``config``:
    the default answer of ``Backend.per_thread_matmul_loop_has_mma_path``.

    Walks every ``ForLoopGraphInfo`` whose block ids are not all root-grid
    axes and whose block sizes ``config`` resolves, once per distinct block
    id tuple, the way the CuTe planner's specialized-path probe did for every
    backend before the planner moved to core.
    """
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo
    from .host_function import HostFunction

    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    grid_ids = {bid for ids in device_ir.grid_block_ids for bid in ids}
    seen: set[tuple[int, ...]] = set()
    for graph_info in fn.codegen.codegen_graphs:
        if not isinstance(graph_info, ForLoopGraphInfo):
            continue
        block_ids = tuple(graph_info.block_ids)
        if block_ids in seen or not any(bid not in grid_ids for bid in block_ids):
            continue
        seen.add(block_ids)
        block_sizes: list[int | torch.SymInt] = []
        for bid in block_ids:
            size = env.block_sizes[bid].from_config(config)
            if not isinstance(size, (int, torch.SymInt)):
                break
            block_sizes.append(size)
        else:
            num_threads_config = [
                int(env.config_spec.num_threads.config_get(config.num_threads, bid, 0))
                for bid in block_ids
            ]
            if backend.per_thread_mma_candidate_loop(
                fn,
                list(block_ids),
                block_sizes=block_sizes,
                num_threads_config=num_threads_config,
                grid_ids=grid_ids,
            ):
                return True
    return False


# -- inactive tile axes ---------------------------------------------------------


def tile_loop_inactive_block_ids(
    fn: DeviceFunction,
    block_ids: list[int],
    inactive_grid_block_ids: set[int],
) -> set[int]:
    """Axes of a multi-dimensional tile loop that carry no threads or lanes.

    The boundary-only grid axes, plus every axis of the loop whose element
    coordinate no graph observes (sizes known equal to an observed axis stay
    active, they alias its coordinate).  The root strategy and the
    specialized-MMA detection use the same set so they agree on the output
    tile a matmul owns.
    """
    from .compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    inactive_block_ids = inactive_grid_block_ids.intersection(block_ids)
    if len(block_ids) <= 1:
        return inactive_block_ids
    used_block_ids = _graph_used_block_ids(fn, block_ids)
    if not used_block_ids:
        used_block_ids = set(block_ids)
    for block_id in tuple(used_block_ids):
        block_size = env.block_sizes[block_id].size
        if block_size is None or not isinstance(block_size, (int, torch.SymInt)):
            continue
        for other_block_id in block_ids:
            if other_block_id == block_id:
                continue
            other_size = env.block_sizes[other_block_id].size
            if other_size is None or not isinstance(other_size, (int, torch.SymInt)):
                continue
            if env.known_equal(block_size, other_size):
                used_block_ids.add(other_block_id)
    inactive_block_ids.update(set(block_ids) - used_block_ids)
    return inactive_block_ids


def _tile_loop_path(
    paths: Sequence[Sequence[tuple[int, ...]]], block_ids: list[int]
) -> Sequence[tuple[int, ...]] | None:
    """The tiles above the loop over ``block_ids`` on its path of ``paths``
    (``Backend.tile_loop_paths``), outermost first: its root's grid tile, then
    the device loops enclosing it.  ``None`` when no path holds the loop."""
    for path in paths:
        if tuple(block_ids) in path:
            return path[: path.index(tuple(block_ids))]
    return None


def _graph_used_block_ids(
    fn: DeviceFunction,
    block_ids: list[int],
) -> set[int]:
    from ..language import memory_ops
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo
    from .device_ir import RootGraphInfo
    from .device_ir import control_flow_child_graph_ids
    from .host_function import HostFunction
    from .variable_origin import BlockSizeOrigin

    env = CompileEnvironment.current()
    host_function = HostFunction.current()
    device_ir = host_function.device_ir
    candidate_block_ids = set(block_ids)
    used: set[int] = set()

    def visit_value(value: object) -> None:
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                visit_value(dim)
            return
        if isinstance(value, torch.SymInt):
            block_id = env.get_block_id(value)
            if block_id is not None and block_id in candidate_block_ids:
                used.add(block_id)
            raw_expr = value.node._expr
            if isinstance(raw_expr, sympy.Expr):
                visit_value(raw_expr)
            return
        if isinstance(value, sympy.Expr):
            for symbol in value.free_symbols:
                block_id = env.get_block_id(symbol)
                if block_id is not None and block_id in candidate_block_ids:
                    used.add(block_id)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit_value(key)
                visit_value(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit_value(item)

    def is_tensor_like_value(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(
                is_tensor_like_value(key) or is_tensor_like_value(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(is_tensor_like_value(item) for item in value)
        return False

    for block_id in candidate_block_ids:
        block_info = env.block_sizes[block_id]
        if block_info.reduction:
            used.add(block_id)

    def graph_matches_loop(graph_info: GraphInfo) -> bool:
        if isinstance(graph_info, ForLoopGraphInfo):
            return graph_info.block_ids == block_ids
        if not isinstance(graph_info, RootGraphInfo):
            return False
        phase_index = graph_info.phase_index
        return (
            0 <= phase_index < len(device_ir.grid_block_ids)
            and device_ir.grid_block_ids[phase_index] == block_ids
        )

    graph_by_id = {
        graph_info.graph_id: graph_info for graph_info in fn.codegen.codegen_graphs
    }
    pending = [
        graph_info
        for graph_info in fn.codegen.codegen_graphs
        if graph_matches_loop(graph_info)
    ]
    visited: set[int] = set()
    while pending:
        graph_info = pending.pop()
        if graph_info.graph_id in visited:
            continue
        visited.add(graph_info.graph_id)
        for node in graph_info.graph.nodes:
            # A root coordinate can first be read inside a captured device
            # scope. Its thread distribution must remain live there as well.
            pending.extend(
                graph_by_id[graph_id] for graph_id in control_flow_child_graph_ids(node)
            )
            if node.target is memory_ops.store:
                # A store has no value; its index names every tile axis it
                # writes, including axes a broadcast value does not carry.
                # Like ``SubscriptIndexing.compute_shape``, only a tile itself
                # spans its block; ``tile.id`` / ``tile.begin`` drop the dim.
                for index in node.args[1]:
                    if not isinstance(index, torch.fx.Node):
                        continue
                    index_value = index.meta.get("val")
                    if not isinstance(index_value, torch.SymInt):
                        continue
                    origin = host_function.expr_to_origin.get(index_value._sympy_())
                    if origin is not None and isinstance(
                        origin.origin, BlockSizeOrigin
                    ):
                        visit_value(index_value)
            value = node.meta.get("val")
            if is_tensor_like_value(value):
                visit_value(value)
                for arg in node.args:
                    if is_tensor_like_value(arg):
                        visit_value(arg)
                for arg in node.kwargs.values():
                    if is_tensor_like_value(arg):
                        visit_value(arg)
    return used


# -- the planner ---------------------------------------------------------------
# The rules below lay out an N-D tile loop's axes; the planner applies them
# to the loop at codegen, and the CuTe autotuner's elements-per-thread shrink
# applies them to a config it bounds (``CuteConfigGenerationExtension``).


def tile_loop_keeps_nd_layout(
    device_ir: DeviceIR,
    graphs: Sequence[GraphInfo],
    block_ids: Sequence[int],
    inactive_grid_block_ids: set[int],
) -> bool:
    """Whether :func:`plan_per_thread_loop_strategy` lays out the tile loop
    over ``block_ids`` as an N-D tile even when its ``flatten_loops`` entry
    asks for a flattened one: in a kernel with a device loop (one of
    ``graphs``) or several root loops, when a block's size is not known (a
    data-dependent bound, a registered block size two loops of different
    sizes share) or when the loop holds a boundary-only grid block
    (``inactive_grid_block_ids``)."""
    from .compile_environment import CompileEnvironment
    from .device_ir import ForLoopGraphInfo
    from .device_ir import ReductionLoopGraphInfo

    env = CompileEnvironment.current()
    return (
        any(
            isinstance(graph, ForLoopGraphInfo)
            and not isinstance(graph, ReductionLoopGraphInfo)
            for graph in graphs
        )
        or any(env.block_sizes[i].size is None for i in block_ids)
        or len(device_ir.grid_block_ids) != 1
        or bool(inactive_grid_block_ids.intersection(block_ids))
    )


def demote_auto_thread_axes(
    num_threads: list[int],
    available_axes: int,
    takes_axis: Callable[[int, int], bool],
    demotable: Callable[[int], bool],
) -> None:
    """While more of a tile's axes take a launch thread axis
    (``takes_axis``, given each axis's current count) than the
    ``available_axes`` the launch leaves the tile, run the automatic axis
    (``num_threads`` entry 0) latest in block order that ``demotable``
    allows on one thread."""
    while sum(starmap(takes_axis, enumerate(num_threads))) > available_axes:
        candidates = [
            i
            for i, threads in enumerate(num_threads)
            if threads == 0 and takes_axis(i, threads) and demotable(i)
        ]
        if not candidates:
            return
        num_threads[candidates[-1]] = 1


def shrink_auto_thread_counts(
    block_sizes: Sequence[object], num_threads: list[int], thread_limit: int
) -> int:
    """Shrink a tile's automatic thread counts (``num_threads`` entries 0, a
    thread per element of the block size), the widest first, each to the
    next divisor of its block size below, until the axes of static block
    size take at most ``thread_limit`` threads together; returns the threads
    they take."""
    int_positions: list[int] = []
    int_block_sizes: dict[int, int] = {}
    for i, block_size in enumerate(block_sizes):
        if isinstance(block_size, int):
            int_positions.append(i)
            int_block_sizes[i] = block_size
    resolved_threads = [
        num_threads[i] if num_threads[i] > 0 else int_block_sizes[i]
        for i in int_positions
    ]
    auto_positions = {
        pos
        for pos, block_idx in enumerate(int_positions)
        if num_threads[block_idx] == 0
    }
    static_threads = functools.reduce(operator.mul, resolved_threads, 1)
    while static_threads > thread_limit and auto_positions:
        shrink_idx = max(
            (pos for pos in auto_positions if resolved_threads[pos] > 1),
            key=lambda pos: resolved_threads[pos],
            default=None,
        )
        if shrink_idx is None:
            break
        block_idx = int_positions[shrink_idx]
        block_size = int_block_sizes[block_idx]
        next_threads = _largest_divisor_at_most(
            block_size, resolved_threads[shrink_idx] - 1
        )
        if next_threads == resolved_threads[shrink_idx]:
            break
        resolved_threads[shrink_idx] = next_threads
        num_threads[block_idx] = next_threads
        static_threads = functools.reduce(operator.mul, resolved_threads, 1)
    return static_threads


def cap_auto_thread_axes(
    num_threads: list[int],
    block_sizes: Sequence[object],
    loop_order: Iterable[int],
    first_axis: int,
    axis_extent_limits: Sequence[int],
    takes_axis: Callable[[int, int], bool],
    capped: Callable[[int], bool],
) -> None:
    """Cap a tile's automatic thread counts at what their launch axes take.

    In ``loop_order`` the axes that take a thread axis (``takes_axis``) take
    the launch's from ``first_axis`` on; one that ``capped`` allows, whose
    count (its block size when 0) passes its axis's limit
    (``axis_extent_limits``, 64 threads on CUDA's z), keeps the largest
    divisor of its block size within it and walks the rest of the tile in a
    lane loop.  The axes past the last limit keep their counts.
    """
    axis = first_axis
    for i in loop_order:
        if not takes_axis(i, num_threads[i]):
            continue
        if axis >= len(axis_extent_limits):
            break
        limit = axis_extent_limits[axis]
        axis += 1
        block_size = block_sizes[i]
        if (
            capped(i)
            and isinstance(block_size, int)
            and (num_threads[i] or block_size) > limit
        ):
            num_threads[i] = _largest_divisor_at_most(block_size, limit)


def plan_per_thread_loop_strategy(
    backend: Backend, fn: DeviceFunction, block_ids: list[int], config: Config
) -> TileStrategy:
    """Plan the per-thread tile strategy of one tile loop
    (``PerThreadNDTileStrategy`` / ``PerThreadFlattenedTileStrategy``).

    Shared by CuTe and Metal (one element per thread, cooperative hardware
    primitives for matmul).  ``backend`` supplies the policy through
    ``boundary_only_grid_block_ids``, ``tile_loop_thread_count``,
    ``thread_axis_extent_limits``, ``packed_grid_block`` and the
    ``per_thread_*`` tensor-core hooks (see the module docstring).
    """
    from .compile_environment import CompileEnvironment
    from .host_function import HostFunction
    from .matmul_utils import kernel_has_half_precision_matmul
    from .matmul_utils import matmul_contraction_block_ids
    from .tile_dispatch import reduction_strategy_block_ids
    from .tile_strategy import PerThreadFlattenedTileStrategy
    from .tile_strategy import PerThreadNDTileStrategy

    if len(set(block_ids)) != len(block_ids):
        # ``hl.tile([m, n], block_size=[bs, bs])`` with one registered
        # block size hands both dimensions the same block: they would
        # share one index and one mask, walking the diagonal.
        raise backend.unsupported(
            "a tile whose dimensions share one block size symbol "
            f"(block ids {block_ids}); register a block size per dimension"
        )
    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    block_size_infos = [env.block_sizes[i] for i in block_ids]
    flattened = block_size_infos[0].is_flattened(config)
    loop_order = env.config_spec.loop_orders.config_get_for_tile(
        config.loop_orders, block_ids
    ) or [*range(len(block_ids))]
    l2_grouping = env.config_spec.l2_groupings.config_get(
        config.l2_groupings, block_ids[0], 1
    )
    grid_ids = {bid for ids in device_ir.grid_block_ids for bid in ids}
    num_threads_config = [
        int(env.config_spec.num_threads.config_get(config.num_threads, block_id, 0))
        for block_id in block_ids
    ]
    inactive_grid_block_ids = backend.boundary_only_grid_block_ids(fn)

    total_threads = backend.tile_loop_thread_count(fn, config, inactive_grid_block_ids)
    if total_threads > MAX_THREADS_PER_BLOCK:
        for i, block_id in enumerate(block_ids):
            if num_threads_config[i] == 0 and block_id not in grid_ids:
                num_threads_config[i] = 1
    if (len(block_ids) > 1 and not flattened) or tile_loop_keeps_nd_layout(
        device_ir, fn.codegen.codegen_graphs, block_ids, inactive_grid_block_ids
    ):
        nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
        original_num_threads_config = list(num_threads_config)
        mma_candidate = backend.per_thread_mma_candidate_loop(
            fn,
            block_ids,
            block_sizes=nd_block_size,
            num_threads_config=original_num_threads_config,
            grid_ids=grid_ids,
        )
        inactive_block_ids = tile_loop_inactive_block_ids(
            fn, block_ids, inactive_grid_block_ids
        )
        reduction_block_ids = reduction_strategy_block_ids(
            fn, backend.constexpr_live_graph_ids(fn)
        )
        for i, block_id in enumerate(block_ids):
            if block_id in inactive_block_ids:
                num_threads_config[i] = 1
        is_device_loop = any(bid not in grid_ids for bid in block_ids)
        # Only a reduction that gets a strategy claims an axis: not one on
        # the untaken side of a branch that is constant for this config.
        reduction_axis_reserve = (
            1 if backend.reduction_axis_first() and reduction_block_ids else 0
        )

        # ``cute_tiles_per_block``: the packed grid axis is a thread axis
        # of extent ``pack`` even when the body never distributes over it
        # (``PerThreadNDTileStrategy`` reactivates it); budget it here.
        packed = backend.packed_grid_block(config)
        packed_block_id = None if packed is None else packed[0]

        def uses_thread_axis_for(
            block_id: int, block_size: object, num_threads: int
        ) -> bool:
            if block_id == packed_block_id:
                return True
            if block_id in inactive_block_ids or block_id in inactive_grid_block_ids:
                return False
            if num_threads > 0:
                return num_threads > 1
            return not (isinstance(block_size, int) and block_size == 1)

        def grid_thread_axes(grid_block_ids: Iterable[int]) -> int:
            """The thread axes the configured tiles of ``grid_block_ids``
            take: a root's grid tiles, or an enclosing device loop's."""
            return sum(
                uses_thread_axis_for(
                    grid_block_id,
                    env.block_sizes[grid_block_id].from_config(config),
                    int(
                        env.config_spec.num_threads.config_get(
                            config.num_threads, grid_block_id, 0
                        )
                    ),
                )
                for grid_block_id in grid_block_ids
            )

        consumed_grid_axes = grid_thread_axes(grid_ids) if is_device_loop else 0
        available_axes = max(0, 3 - reduction_axis_reserve - consumed_grid_axes)

        def takes_axis(i: int, threads: int) -> bool:
            return uses_thread_axis_for(block_ids[i], nd_block_size[i], threads)

        demote_auto_thread_axes(
            num_threads_config,
            available_axes,
            takes_axis,
            lambda i: not is_device_loop or block_ids[i] not in grid_ids,
        )
        thread_limit = MAX_THREADS_PER_BLOCK
        if len(block_ids) > 1 and _loop_contains_matmul(fn, block_ids):
            if mma_candidate or backend.per_thread_matmul_loop_has_mma_path(
                fn, block_ids, config
            ):
                thread_limit = MAX_THREADS_PER_BLOCK
            else:
                # Matmul-heavy kernels with no viable MMA path, and
                # especially atomic-accumulating split-K loops, can be
                # register/smem limited well before the 1024-thread hard
                # cap. Keep those auto-threaded ND tiles within 256
                # threads and let lane loops cover the rest.
                thread_limit = min(thread_limit, 256)
        # Each tile the packed grid axis holds runs on threads of its own,
        # so the auto counts fit the tile into what the pack leaves of a block.
        if packed is not None:
            thread_limit //= packed[1]
        if len(block_ids) > 1 and mma_candidate:
            inactive_block_ids.clear()
            num_threads_config = [
                env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
                for block_id in block_ids
            ]
        # OPTION B (matmul-contraction synthetic-lane fix): when a reduction
        # block is the *contraction* (K) axis of a matmul lowered through the
        # scalar fallback, it must keep enough real hardware threads to cover
        # its full extent so the cross-warp shared-memory reduction sums the
        # whole K.  Otherwise the budget below would hand the free tile axes
        # the threads and leave K split into ``threads x synthetic-lane`` -
        # the reduction then sums only the thread lanes, never the synthetic
        # lanes, so each contracted dot product covers only a fraction of K.
        # Reserve the K block's full thread extent up front by shrinking the
        # thread limit available to the free tile axes; the contraction axis
        # then claims that budget when its reduction strategy is created and
        # the synthetic lane is pushed onto the free tile axes instead.
        reserved_contraction_threads = backend.matmul_contraction_thread_reserve(
            fn, block_ids
        )
        if reserved_contraction_threads > 1:
            # Budget left for the free tile axes after the contraction axis
            # has claimed its full thread extent.
            free_budget = max(1, thread_limit // reserved_contraction_threads)
            # The free axes are built across separate ``create_loop_strategy``
            # calls but their thread counts multiply, so split the budget so
            # the product over every free axis stays within ``free_budget``.
            free_axes = backend.free_auto_thread_axis_count(fn, config)
            per_axis_limit = free_budget
            while per_axis_limit > 1 and per_axis_limit**free_axes > free_budget:
                per_axis_limit //= 2
            thread_limit = max(1, per_axis_limit)
        auto_positions = {
            i for i, threads in enumerate(num_threads_config) if threads == 0
        }
        static_threads = shrink_auto_thread_counts(
            nd_block_size, num_threads_config, thread_limit
        )

        # The tensor-core detectors are the backend's (``mma_mode``: the
        # dedicated MMA, flash and warp-MMA bodies suppress the synthetic lane
        # loops of a device loop; a tensor-core root plan sets the thread
        # counts of the root tile).  The core defaults describe a backend
        # without such a path, so Metal never runs them.
        mma_mode = False
        root_threads: list[int] | None = None
        if is_device_loop:
            mma_mode = backend.per_thread_mma_mode(
                fn, block_ids, config, block_sizes=nd_block_size
            )
        elif (
            len(device_ir.grid_block_ids) == 1
            and block_ids == device_ir.grid_block_ids[0]
        ):
            root_threads = backend.per_thread_root_thread_counts(
                fn,
                block_ids,
                config,
                block_sizes=nd_block_size,
                num_threads_config=num_threads_config,
                original_num_threads_config=original_num_threads_config,
            )
            if root_threads is not None:
                num_threads_config = root_threads
                static_threads = functools.reduce(
                    operator.mul,
                    (
                        threads
                        if threads > 0
                        else int(block_size)
                        if isinstance(block_size, int)
                        else 0
                        for block_size, threads in zip(
                            nd_block_size,
                            num_threads_config,
                            strict=True,
                        )
                    ),
                    1,
                )

        axis_extent_limits = backend.thread_axis_extent_limits()
        if (
            axis_extent_limits is not None
            and any(
                (num_threads_config[i] or nd_block_size[i]) > min(axis_extent_limits)
                for i in auto_positions
                if isinstance(nd_block_size[i], int)
            )
            and not (
                mma_mode
                or mma_candidate
                or root_threads is not None
                or backend.per_thread_reduction_lays_out_tiles(
                    env.config_spec, config, total_threads
                )
            )
        ):
            # An automatic thread count above what its launch axis takes (64
            # threads on CUDA's z) keeps that many threads and walks the rest
            # of the tile in a lane loop.  The axes below the loop are counted
            # low, so a wrong count misses a cap instead of adding one to a
            # tile that launched before: the reduction's only when a reduction
            # surely keeps several threads (one left a single thread claims
            # none) and, below a device loop, the grid axes of its root (the
            # first tile of its path; the root with the fewest when no path
            # holds the loop) and those of the loops enclosing it (each at
            # most the axes the grid leaves it), without the tiles no graph
            # observes.  A sibling root's reductions only move the tiles up.
            # A reduction that lays the tiles out itself, such as a register
            # tile putting them below its own axis, leaves the counts uncapped.
            # Codegen knows each tile's axis exactly where it demotes the
            # blocks past the third one
            # (``PerThreadNDTileStrategy._demote_blocks_beyond_thread_axes``),
            # but the cap belongs here: what is decided before codegen reads
            # the planned counts (sibling tiles widened to the widest on their
            # axis, the reductions' fit beside the tiles in
            # ``adjust_reduction_thread_count``, a shared launch's per-axis
            # budget from ``thread_block_dims``), so they must be capped.
            tile_paths = backend.tile_loop_paths(fn)
            axis = 0
            if reduction_axis_reserve and _reduction_surely_claims_thread_axis(
                fn, backend, config, total_threads, reduction_block_ids, tile_paths
            ):
                axis = 1
            if is_device_loop:
                path = _tile_loop_path(tile_paths, block_ids)
                for level in path[1:] if path else ():
                    axis += min(
                        available_axes,
                        grid_thread_axes(
                            set(level)
                            - tile_loop_inactive_block_ids(
                                fn, list(level), inactive_grid_block_ids
                            )
                        ),
                    )
                axis += min(
                    grid_thread_axes(
                        set(root_block_ids)
                        - tile_loop_inactive_block_ids(
                            fn, root_block_ids, inactive_grid_block_ids
                        )
                    )
                    for root_block_ids in (
                        [list(path[0])] if path else device_ir.grid_block_ids
                    )
                )
            cap_auto_thread_axes(
                num_threads_config,
                nd_block_size,
                loop_order,
                axis,
                axis_extent_limits,
                takes_axis,
                lambda i: i in auto_positions and block_ids[i] != packed_block_id,
            )
            static_threads = functools.reduce(
                operator.mul,
                (
                    threads or block_size
                    for block_size, threads in zip(
                        nd_block_size, num_threads_config, strict=True
                    )
                    if isinstance(block_size, int)
                ),
                1,
            )

        check_thread_limit(static_threads, context=str(tuple(nd_block_size)))
        return PerThreadNDTileStrategy(
            fn,
            block_ids,
            block_size=nd_block_size,
            loop_order=loop_order,
            l2_grouping=l2_grouping,
            num_threads=num_threads_config,
            mma_mode=mma_mode,
            inactive_block_ids=inactive_block_ids,
        )
    nd_block_size = [bs.from_config_assert(config) for bs in block_size_infos]
    block_size = functools.reduce(operator.mul, nd_block_size)  # pyrefly: ignore[incompatible-overload-residual]
    # Resolve per-axis thread counts then flatten to a single total
    all_auto = all(nt <= 0 for nt in num_threads_config)
    flat_num_threads = functools.reduce(
        operator.mul,
        (
            nt if nt > 0 else (int(bs) if isinstance(bs, int) else 0)
            for nt, bs in zip(num_threads_config, nd_block_size, strict=True)
        ),
        1,
    )
    if (
        isinstance(block_size, int)
        and flat_num_threads > MAX_THREADS_PER_BLOCK
        and all_auto
    ):
        # Auto thread budget exceeds the 1024-per-CTA cap: fall back to a
        # lane loop (each thread owns block_size // 1024 elements).
        flat_num_threads = MAX_THREADS_PER_BLOCK
    if (
        isinstance(block_size, int)
        and flat_num_threads > 1
        and all_auto
        # A tile that is itself a contraction axis (``x[tile, :].T @
        # g[tile, :]``) keeps its threads: the fallback reduces it across
        # threads only, not across a flattened tile's lane loop.
        and not matmul_contraction_block_ids().intersection(
            env.canonical_block_id(block_id) for block_id in block_ids
        )
        # A 16-bit matmul may take the direct grouped-N warp-MMA plan,
        # which needs one tile row per thread.
        and not kernel_has_half_precision_matmul()
    ):
        # A flattened tile reserves a matmul's contraction axis its full
        # thread extent (see the N-D branch above), so ``x[tile, :] @
        # w[:, :]`` keeps K on real threads and the tile takes the lane
        # loop, rather than splitting K into synthetic lanes.  A
        # contraction past the thread cap is folded serially whatever its
        # threads (``PersistentReductionStrategy`` gives it one): the tile
        # keeps its threads then.
        contraction_extents = matmul_contraction_extents(block_ids)
        reserved_contraction_threads = (
            backend.matmul_contraction_thread_reserve(fn, block_ids)
            if max(contraction_extents, default=0) <= MAX_THREADS_PER_BLOCK
            else 1
        )
        if reserved_contraction_threads > 1:
            free_budget = max(1, MAX_THREADS_PER_BLOCK // reserved_contraction_threads)
            free_axes = backend.free_auto_thread_axis_count(fn, config)
            per_axis_limit = free_budget
            while per_axis_limit > 1 and per_axis_limit**free_axes > free_budget:
                per_axis_limit //= 2
            flat_num_threads = _largest_divisor_at_most(
                block_size, min(flat_num_threads, per_axis_limit)
            )
    if isinstance(block_size, int) and flat_num_threads > 0:
        check_thread_limit(flat_num_threads, context=str(block_size))
    return PerThreadFlattenedTileStrategy(
        fn,
        block_ids,
        block_size=block_size,
        loop_order=loop_order,
        num_threads=flat_num_threads,
    )
