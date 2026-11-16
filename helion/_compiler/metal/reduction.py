"""Metal reduction expression emitters.

Turns Helion's backend-neutral reduction requests into MSL text built on
Inductor's ``c10::metal::`` reduction helpers (see :mod:`.msl_reduction` for
what comes from where).

Reduction group layout
----------------------
``MetalBackend.reduction_axis_first()`` is ``True``, so a reduction owns a whole
thread axis and tiles occupy the axes above it.  ``tid`` is laid out with
``tid[0]`` fastest-varying, and Metal assigns SIMD groups in that linear order,
so a reduction on axis 0 covers a contiguous run of lanes.  Three cases follow,
in increasing cost:

======================  ==========================================  =============
span (reduction lanes)  emission                                    cost
======================  ==========================================  =============
``< 32``                ``helion_red::seg_*`` butterfly             shuffles only
``== 32``               ``helion_red::simd_*``                      shuffles only
``% 32 == 0``           ``helion_red::tg_*`` + threadgroup scratch  2 barriers
======================  ==========================================  =============

Spans are always powers of two (``static_rdim_size`` rounds the reduction
extent up), so a span above the SIMD width is automatically a whole number of
SIMD groups.  Spans wider than one threadgroup never reach here: the config
spec rolls those into a ``for`` loop over the reduction dimension, and only the
per-thread accumulator is combined across threads.

Launched-axis collectives
-------------------------
A sibling loop, another root, or an MPP graph can launch a tile's axis wider
than its input span. The collective covers the launched width; extra positions
contribute the identity and receive their row's result. Final launch extents
are known only after codegen, so metal_jit chooses each site's collective and
scratch size from the actual launch.

Contiguous power-of-two axes use the seg/simd/tg helpers at the launched width.
Strided rows mask all but one row at a time and reduce the entire contiguous
group with those helpers. Each thread retains its own row's result. Groups
that are not aligned whole SIMD groups or power-of-two segments use the same
row masks with a bounded shared-memory tree.

A reduction over a user tile is owned by the tile's strategy, so its axis
follows loop nesting rather than ``reduction_axis_first``;
:func:`plan_reduced_tile_axes` ranks such tiles onto the lowest axes, where
they take the contiguous paths.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from ... import exc
from ..ast_extension import statement_from_string
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .msl_expr import cast_source
from .msl_reduction import REDUCTION_NAMESPACE as NS
from .msl_reduction import SUPPORTED_ARGREDUCTIONS
from .msl_reduction import SUPPORTED_REDUCTIONS

if TYPE_CHECKING:
    import torch

    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

#: Prefix of the generated module-level globals that carry threadgroup scratch
#: declarations from codegen to ``metal_jit._generate_msl``.  Mirrors how
#: ``_BLOCK_SIZE_*`` globals reach the same place.
TG_BUFFER_GLOBAL_PREFIX = "_METAL_TG_BUF_"

#: Prefix of the generated module-level globals that describe a group
#: reduction site, ``(site, axis)``, for ``metal_jit``: it picks
#: the site's ``group_path`` from the launch and declares the site name as that
#: path's index.
RED_SITE_GLOBAL_PREFIX = "_METAL_RED_SITE_"

#: MSL constants ``metal_jit`` declares with the launch's threadgroup size.
LAUNCH_DIM_VARS = ("_TG_SIZE_X", "_TG_SIZE_Y", "_TG_SIZE_Z")


def _metal_acc_dtype(dtype: torch.dtype) -> str:
    """MSL *accumulator* type for a reduction over values of ``dtype``.

    Not the storage type: ``c10::metal::threadgroup_sum`` takes its scratch as
    ``threadgroup opmath_t<T>*``, and ``c10/metal/utils.h`` promotes
    ``char``/``short``/``uchar`` to ``int`` -- so passing a ``threadgroup char*``
    has no viable overload and the shader fails to compile.  ``simd_shuffle_xor``
    likewise has no ``bool`` overload.  ``Backend.acc_type`` already encodes the
    right promotion for every dtype Metal supports.
    """
    from ..compile_environment import CompileEnvironment

    return CompileEnvironment.current().backend.acc_type(dtype)


def alloc_threadgroup_buffer(
    metal_dtype: str, slots: int | str, prefix: str = "_red_scratch"
) -> str:
    """Reserve a ``threadgroup`` scratch array and return its MSL name.

    Every reduction site gets its own buffer.  Sharing one buffer between two
    reductions in the same kernel (LayerNorm's mean and variance, say) races on
    Apple GPUs even with barriers in between.  *slots* is a count, or a group
    reduction's site name, which ``metal_jit`` sizes with
    :func:`group_scratch_slots` once the launch is known.
    """
    from ..device_function import DeviceFunction

    device_fn = DeviceFunction.current()
    name = device_fn.new_var(prefix)
    device_fn.codegen.module_statements.append(
        statement_from_string(
            f"{TG_BUFFER_GLOBAL_PREFIX}{name} = ({name!r}, {metal_dtype!r}, {slots!r})"
        )
    )
    return name


@dataclasses.dataclass(frozen=True)
class ReductionGroup:
    """Where a reduction's threads live in the Metal threadgroup.

    ``axis`` is the ``tid`` component the reduction spans and ``span`` how
    many threads it spans.  How those threads lie on SIMD lanes depends on the
    launch, so ``metal_jit`` resolves it (see :func:`group_path`).
    """

    axis: int
    span: int


def resolve_group(
    device_fn: DeviceFunction, block_size_var: str | None
) -> ReductionGroup | None:
    """Locate the reduction strategy owning *block_size_var* and describe it."""
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import BlockSizeTileStrategy

    tile_strategy = device_fn.tile_strategy
    for strategy in tile_strategy.strategies:
        if not isinstance(strategy, ReductionStrategy):
            continue
        if strategy.block_size_var(strategy.block_index) != block_size_var:
            continue
        span = strategy._reduction_thread_count()
        return ReductionGroup(strategy._get_thread_axis(), span) if span > 0 else None

    # ``BlockReductionStrategy`` reduces over a user tile, so the block size var
    # belongs to the tile strategy that owns the axis instead.
    for strategy in tile_strategy.strategies:
        if not isinstance(strategy, BlockSizeTileStrategy):
            continue
        for block_id in strategy.block_ids:
            if strategy.block_size_var(block_id) != block_size_var:
                continue
            axis = _tile_thread_axis(device_fn, block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 0:
                return None
            return ReductionGroup(axis, extent)
    return None


def plan_reduced_tile_axes(
    graphs: list[GraphInfo], tile_strategy: TileStrategyDispatch
) -> None:
    """Rank the strategies owning a reduced tile block onto the lowest axes.

    Thread axes follow loop nesting, so in ``for tile_m in hl.tile(m): for
    tile_n in hl.tile(n): torch.amax(x[tile_m, tile_n], dim=1)`` the outer
    grid takes ``tid[0]`` and the reduced ``tile_n`` lands ``block_m`` lanes
    apart: every reduction pays the strided path, and loads along the reduced
    dimension (usually the contiguous one) do not coalesce.  This is the tile
    counterpart of ``reduction_axis_first``, which already puts reduction
    *strategies* first.  Emitted as a grid execution plan (the mechanism the
    CuTe layout planner uses), which both the body's axis offsets and the
    launch block dims follow.

    A reduced tile on ``tid[0]`` takes a contiguous group path, also when
    another loop launches the axis wider.  The plan is skipped:

    - while a reduction strategy reserves the leading axes (those keep them);
    - when two threaded tile loops are siblings (``for tile_n ...`` next to
      ``for tile_k ...`` with different block ids): the body lets siblings
      share their nesting level's axes, but a plan ranks every block-id set
      onto axes of its own, so the sibling would claim a fresh axis and every
      thread on it would repeat the body.

    Blocks within one multi-block strategy keep their order (``loop_orders``
    does not move Metal thread axes), so a reduction over the inner block of
    ``hl.tile([m, n])`` stays strided and takes the ``strided`` collective
    (``masked`` when the stride is not a power of two).
    """
    from ..cute.execution_plans import CuTeGridExecutionPlan
    from ..cute.loop_nesting import tile_loop_paths
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction
    from ..inductor_lowering import ReductionLowering
    from ..reduction_strategy import ReductionStrategy

    device_ir = HostFunction.current().device_ir
    if tile_strategy.reserved_reduction_axes() or _has_threaded_sibling_loops(
        tile_loop_paths(device_ir, graphs), tile_strategy
    ):
        return
    tile_blocks = [
        block_id
        for strategy in tile_strategy.strategies
        if not isinstance(strategy, ReductionStrategy)
        for block_id in strategy.block_ids
    ]
    reduced = {
        lowering.block_index
        for graph in graphs
        for node in graph.graph.nodes
        if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
    }
    if not any(
        (tile_strategy.thread_extent_for_block_id(block_id) or 0) > 1
        for block_id in reduced.intersection(tile_blocks)
    ):
        return
    plan = CuTeGridExecutionPlan(
        scoped_block_ids=frozenset(tile_blocks),
        block_axis_priority={
            block_id: 0 if block_id in reduced else 1 for block_id in tile_blocks
        },
        disable_reduction_axis_reservation_for=frozenset(),
    )
    for graph in graphs:
        if isinstance(graph, RootGraphInfo):
            graph.cute_grid_execution_plans = (*graph.cute_grid_execution_plans, plan)


def _has_threaded_sibling_loops(
    paths: tuple[tuple[tuple[int, ...], ...], ...],
    tile_strategy: TileStrategyDispatch,
) -> bool:
    """Whether two mutually exclusive tile loops of one root both use threads.

    *paths* are the maximal stacks of simultaneously active tile loops, root
    grid first.  Dropping the loops that occupy no thread axis, the threaded
    loops of each root must form a single stack.
    """

    def threaded(block_ids: tuple[int, ...]) -> bool:
        return any(
            (tile_strategy.thread_extent_for_block_id(block_id) or 0) > 1
            for block_id in block_ids
        )

    stacks: dict[tuple[int, ...], set[tuple[tuple[int, ...], ...]]] = {}
    for root, *loops in paths:
        stacks.setdefault(root, set()).add(
            tuple(block_ids for block_ids in loops if threaded(block_ids))
        )
    for root_stacks in stacks.values():
        longest = max(root_stacks, key=len)
        if any(stack != longest[: len(stack)] for stack in root_stacks):
            return True
    return False


def _tile_thread_axis(device_fn: DeviceFunction, block_id: int) -> int | None:
    """Thread axis a tile block occupies in the active loop nest or, once its
    loop has exited, the one it used there (:func:`exited_block_thread_axis`)."""
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState

    codegen = device_fn.codegen
    current_grid = codegen.current_grid_state
    if isinstance(current_grid, DeviceGridState):
        axis = current_grid.block_thread_axes.get(block_id)
        if axis is not None:
            return axis
    for loops in codegen.active_device_loops.values():
        for loop_state in loops:
            if not isinstance(loop_state, (DeviceGridState, DeviceLoopState)):
                continue
            axis = loop_state.block_thread_axes.get(block_id)
            if axis is not None:
                return axis
    if not codegen.active_device_loops.get(block_id):
        return exited_block_thread_axis(device_fn, block_id)
    return None


def exited_block_thread_axis(device_fn: DeviceFunction, block_id: int) -> int | None:
    """Thread axis a tile block used in its loop, after that loop has exited.

    Only the dispatcher's plan is left once the loop state is gone.  Under a
    grid execution plan that gives the block an axis priority, the body took
    its axis offset from ``thread_axis_for_strategy``
    (``BlockSizeTileStrategy._compute_thread_axis_offset``), which
    ``thread_axis_for_block_id`` is built on, so the two agree.  Without such a
    plan the body counts the active loops' axes instead, and the two diverge:
    threaded sibling loops share their nesting level's axis in the body but
    get separate axes from the dispatcher.  Then there is no axis to trust and
    this returns ``None``.
    """
    tile_strategy = device_fn.tile_strategy
    plan = tile_strategy.current_cute_grid_execution_plan(block_ids=(block_id,))
    if plan is None or plan.priority_for_block(block_id) is None:
        return None
    return tile_strategy.thread_axis_for_block_id(block_id)


def group_path(axis: int, dims: tuple[int, int, int]) -> str:
    """Choose a launched-axis collective from the physical thread layout."""
    extent = dims[axis]
    below = math.prod(dims[:axis])
    if below == 1 and extent & (extent - 1) == 0:
        return "contiguous"
    size = extent * below
    if size % SIMD_WIDTH == 0 or (size < SIMD_WIDTH and size & (size - 1) == 0):
        return "strided" if below & (below - 1) == 0 else "masked"
    return "shared_tree"


def group_scratch_slots(path: str, dims: tuple[int, int, int], axis: int) -> int:
    """Scratch for one scalar pass of a launched-axis collective."""
    threads = math.prod(dims)
    below = math.prod(dims[:axis])
    size = dims[axis] * below
    if path in ("contiguous", "masked"):
        return threads // SIMD_WIDTH if size > SIMD_WIDTH else 0
    if path == "strided":
        # One partial per SIMD group and run.
        return (
            threads // SIMD_WIDTH * min(below, SIMD_WIDTH) if size > SIMD_WIDTH else 0
        )
    assert path == "shared_tree"
    return threads


def _group_site(group: ReductionGroup) -> str:
    """Register a group reduction site for ``metal_jit`` and return its name."""
    from ..device_function import DeviceFunction

    device_fn = DeviceFunction.current()
    site = device_fn.new_var("_red_site")
    device_fn.codegen.module_statements.append(
        statement_from_string(
            f"{RED_SITE_GLOBAL_PREFIX}{site} = ({site!r}, {group.axis})"
        )
    )
    return site


def _group_args(group: ReductionGroup, span: int) -> str:
    """The axis, input span, and final launch dimensions of a collective."""
    dims = ", ".join(LAUNCH_DIM_VARS)
    return f"tid, {group.axis}, {span}, {dims}"


def _unsupported_span(reduction_type: str, span: int) -> exc.BackendUnsupported:
    return exc.BackendUnsupported(
        "metal",
        f"{reduction_type} reduction over {span} threads (the reduction must "
        f"span a power-of-two group of at most {MAX_THREADS_PER_THREADGROUP} "
        f"threads, and beyond {SIMD_WIDTH} a whole number of SIMD groups)",
    )


def _cast(expr: str, dtype: torch.dtype | None) -> tuple[str, str]:
    """Cast *expr* to the accumulation dtype and return ``(expr, metal_type)``.

    Casting up front pins the template argument of the ``c10::metal::`` helper,
    which is what makes the scratch buffer's element type predictable: a masked
    fp16 load can still carry its storage dtype into an fp32 accumulation.
    """
    if dtype is None:
        raise exc.BackendUnsupported("metal", "reduction without an accumulator dtype")
    metal_type = _metal_acc_dtype(dtype)
    return cast_source(metal_type, expr), metal_type


def reduction_expr(
    input_name: str,
    reduction_type: str,
    *,
    block_size_var: str | None,
    dtype: torch.dtype | None,
) -> str:
    from ..device_function import DeviceFunction

    if reduction_type not in SUPPORTED_REDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")

    group = resolve_group(DeviceFunction.current(), block_size_var)
    value, metal_type = _cast(input_name, dtype)

    if group is None:
        raise exc.BackendUnsupported(
            "metal", f"unresolved thread group for reduction {reduction_type!r}"
        )
    span = group.span
    if span <= 1:
        return value
    if span & (span - 1) or span > MAX_THREADS_PER_THREADGROUP:
        raise _unsupported_span(reduction_type, span)

    site = _group_site(group)
    partials = alloc_threadgroup_buffer(metal_type, site)
    return (
        f"{NS}.group_{reduction_type}({site}, {partials}, {value}, "
        f"{_group_args(group, span)})"
    )


def combine_expr(reduction_type: str, lhs: str, rhs: str) -> str:
    """*lhs* and *rhs* combined by *reduction_type*, one of
    ``SUPPORTED_REDUCTIONS``.  Needs no compile environment, so it also
    serves rewrites that run at launch."""
    assert reduction_type in SUPPORTED_REDUCTIONS, reduction_type
    if reduction_type == "sum":
        return f"({lhs} + {rhs})"
    if reduction_type == "prod":
        return f"({lhs} * {rhs})"
    # c10::metal::max/min propagate NaN, matching torch semantics; ::metal:: does not.
    return f"c10.metal.{reduction_type}({lhs}, {rhs})"


def reduction_combine_expr(
    reduction_type: str, acc: str, val: str, dtype: torch.dtype
) -> str:
    """Per-iteration combine for ``LoopedReductionStrategy``'s accumulator."""
    value, _ = _cast(val, dtype)
    if reduction_type not in SUPPORTED_REDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction combine {reduction_type!r}")
    return combine_expr(reduction_type, acc, value)


def argreduce_result_expr(
    input_name: str,
    index_value: str,
    reduction_type: str,
    output_dtype: torch.dtype,
    *,
    block_size_var: str | None,
    index_dtype: torch.dtype | None,
    dtype: torch.dtype | None,
) -> str:
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction

    if reduction_type not in SUPPORTED_ARGREDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")
    if index_dtype is None:
        raise exc.BackendUnsupported("metal", "missing index_dtype for argreduce")

    backend = CompileEnvironment.current().backend
    group = resolve_group(DeviceFunction.current(), block_size_var)
    value, value_type = _cast(input_name, dtype)
    index_type = backend.index_type_str(index_dtype)
    index = cast_source(index_type, index_value)
    out_type = backend.dtype_str(output_dtype)

    if group is None:
        raise exc.BackendUnsupported(
            "metal", f"unresolved thread group for reduction {reduction_type!r}"
        )
    span = group.span
    if span <= 1:
        return cast_source(out_type, index)
    if span & (span - 1) or span > MAX_THREADS_PER_THREADGROUP:
        raise _unsupported_span(reduction_type, span)

    site = _group_site(group)
    value_partials = alloc_threadgroup_buffer(value_type, site)
    index_partials = alloc_threadgroup_buffer(index_type, site)
    result = (
        f"{NS}.group_{reduction_type}({site}, {value_partials}, "
        f"{index_partials}, {value}, {index}, "
        f"{_group_args(group, span)})"
    )
    return cast_source(out_type, result)


def argreduce_loop_update_statements(
    *,
    reduction_type: str,
    acc: str,
    acc_index: str,
    value: str,
    index: str,
    dtype: torch.dtype | None,
) -> list[str]:
    """Branch-free accumulator update for a rolled argmin/argmax.

    Lowest index wins a tie, matching Inductor's MPS semantics (``mps.py``) and
    torch.  The tie is explicit rather than implied by a strict comparison: the
    accumulator index starts at INT_MAX (``backend.full_expr`` over
    ``torch.iinfo(index_dtype).max`` in ``reduction_strategy``), so a row that
    never strictly beats the identity -- an all-``-inf`` row under argmax, say
    -- would otherwise never write ``acc_index`` and return the sentinel as if
    it were a position.

    NaN counts as a new extremum for floats, but only over a non-NaN
    accumulator.  Letting it win against a NaN too would keep moving the index
    forward and report the *last* NaN a thread saw; the equality tie-break
    cannot cover this, since ``NaN != NaN``.
    """
    from ..device_function import DeviceFunction

    if reduction_type not in SUPPORTED_ARGREDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")
    op = ">" if reduction_type == "argmax" else "<"
    better = (
        f"(({value}) {op} ({acc}))"
        f" or ((({value}) == ({acc})) and (({index}) < ({acc_index})))"
    )
    if dtype is not None and dtype.is_floating_point:
        better = f"({better} or (metal.isnan({value}) and not metal.isnan({acc})))"
    flag = DeviceFunction.current().new_var("_argreduce_better")
    return [
        f"{flag} = {better}",
        f"{acc_index} = ({index} if {flag} else {acc_index})",
        f"{acc} = ({value} if {flag} else {acc})",
    ]
