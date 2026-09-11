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
``% 32 == 0``           ``helion_red::tg_*`` + threadgroup scratch  3 barriers
======================  ==========================================  =============

Spans are always powers of two (``static_rdim_size`` rounds the reduction
extent up), so a span above the SIMD width is automatically a whole number of
SIMD groups.  Spans wider than one threadgroup never reach here: the config
spec rolls those into a ``for`` loop over the reduction dimension, and only the
per-thread accumulator is combined across threads.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ... import exc
from ..ast_extension import statement_from_string
from ._constants import MAX_SIMD_GROUPS
from ._constants import MAX_THREADS_PER_THREADGROUP
from ._constants import SIMD_WIDTH
from .msl_reduction import REDUCTION_NAMESPACE as NS
from .msl_reduction import SUPPORTED_ARGREDUCTIONS
from .msl_reduction import SUPPORTED_REDUCTIONS

if TYPE_CHECKING:
    import torch

    from ..device_function import DeviceFunction

#: Prefix of the generated module-level globals that carry threadgroup scratch
#: declarations from codegen to ``metal_jit._generate_msl``.  Mirrors how
#: ``_BLOCK_SIZE_*`` globals reach the same place.
TG_BUFFER_GLOBAL_PREFIX = "_METAL_TG_BUF_"

#: Name of the MSL kernel parameter bound to
#: ``[[simdgroup_index_in_threadgroup]]``, used to pick a reduction group's
#: slice of the shared scratch buffer.
SIMD_GROUP_VAR = "_simd_group"


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


def alloc_threadgroup_buffer(metal_dtype: str) -> str:
    """Reserve a ``threadgroup`` scratch array and return its MSL name.

    Every reduction site gets its own buffer.  Sharing one buffer between two
    reductions in the same kernel (LayerNorm's mean and variance, say) races on
    Apple GPUs even with barriers in between, and ``slots`` is only 32 entries,
    so the isolation is cheap.
    """
    from ..device_function import DeviceFunction

    device_fn = DeviceFunction.current()
    name = device_fn.new_var("_red_scratch")
    device_fn.codegen.module_statements.append(
        statement_from_string(
            f"{TG_BUFFER_GLOBAL_PREFIX}{name} = "
            f"({name!r}, {metal_dtype!r}, {MAX_SIMD_GROUPS})"
        )
    )
    return name


@dataclasses.dataclass(frozen=True)
class ReductionGroup:
    """Where a reduction's threads live in the Metal threadgroup.

    ``axis`` is the ``tid`` component the reduction spans, ``span`` how many
    threads it spans, and ``stride`` the distance between consecutive members
    on the linear thread index (``tid[0] + tid[1] * dim0 + ...``).  Only
    ``stride == 1`` groups map onto SIMD lanes, which every Metal reduction
    primitive requires.
    """

    axis: int
    span: int
    stride: int

    @property
    def index_expr(self) -> str:
        return f"tid[{self.axis}]"


def resolve_group(
    device_fn: DeviceFunction, block_size_var: str | None
) -> ReductionGroup | None:
    """Locate the reduction strategy owning *block_size_var* and describe it."""
    from ..reduction_strategy import ReductionStrategy
    from ..tile_strategy import BlockSizeTileStrategy

    tile_strategy = device_fn.tile_strategy
    dims = tile_strategy.thread_block_dims()

    def make(axis: int, span: int) -> ReductionGroup:
        stride = 1
        for lower in range(min(axis, len(dims))):
            stride *= max(1, dims[lower])
        return ReductionGroup(axis, span, stride)

    for strategy in tile_strategy.strategies:
        if not isinstance(strategy, ReductionStrategy):
            continue
        if strategy.block_size_var(strategy.block_index) != block_size_var:
            continue
        span = strategy._reduction_thread_count()
        return make(strategy._get_thread_axis(), span) if span > 0 else None

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
            return make(axis, extent)
    return None


def _tile_thread_axis(device_fn: DeviceFunction, block_id: int) -> int | None:
    """Thread axis a tile block occupies in the active loop nest, if any."""
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
    return None


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
    return f"(static_cast<{metal_type}>({expr}))", metal_type


def _reject_strided_group(reduction_type: str, group: ReductionGroup | None) -> None:
    """Metal's SIMD reductions fold adjacent lanes, so a group whose members are
    not lane-contiguous would combine unrelated rows."""
    if group is not None and group.stride != 1:
        raise exc.BackendUnsupported(
            "metal",
            f"{reduction_type} over a strided thread axis (axis {group.axis}, "
            f"lane stride {group.stride}); Metal needs the reduced dimension to "
            "be the fastest-varying thread axis",
        )


def reduction_expr(
    input_name: str,
    reduction_type: str,
    *,
    block_size_var: str | None,
    threads_in_group: int | None,
    dtype: torch.dtype | None,
) -> str:
    from ..device_function import DeviceFunction

    if reduction_type not in SUPPORTED_REDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")

    device_fn = DeviceFunction.current()
    group = resolve_group(device_fn, block_size_var)
    span = (
        threads_in_group
        if threads_in_group is not None
        else (group.span if group is not None else None)
    )
    value, metal_type = _cast(input_name, dtype)

    if span is None:
        raise exc.BackendUnsupported(
            "metal", f"unresolved thread group for reduction {reduction_type!r}"
        )
    if span <= 1:
        return value
    if span & (span - 1):
        raise _unsupported_span(reduction_type, span)
    _reject_strided_group(reduction_type, group)

    if span < SIMD_WIDTH:
        return f"{NS}.seg_{reduction_type}({value}, {span})"
    if span == SIMD_WIDTH:
        return f"{NS}.simd_{reduction_type}({value})"
    if span % SIMD_WIDTH or span > MAX_THREADS_PER_THREADGROUP:
        raise _unsupported_span(reduction_type, span)

    if group is None:
        # ``c10::metal::threadgroup_*`` needs each thread's index *within* the
        # reduction group to pick its scratch slot; guessing ``tid[0]`` would
        # make every thread claim slot 0 and race.  The sub-SIMD-group and
        # single-SIMD-group tiers above need no such index, so only the shared
        # path has to bail.
        raise exc.BackendUnsupported(
            "metal",
            f"{reduction_type} reduction over {span} threads whose thread axis "
            "could not be resolved",
        )
    buffer = alloc_threadgroup_buffer(metal_type)
    return (
        f"{NS}.tg_{reduction_type}("
        f"{buffer} + {_group_base_expr(span)}, {value}, {group.index_expr}, {span})"
    )


def _group_base_expr(span: int) -> str:
    """Offset of this reduction group's slice of the shared scratch buffer.

    ``[[simdgroup_index_in_threadgroup]]`` counts SIMD groups over the whole
    threadgroup, and a span that is a multiple of the SIMD width means each
    reduction group owns ``span / 32`` consecutive SIMD groups.
    """
    simd_groups = span // SIMD_WIDTH
    return f"({SIMD_GROUP_VAR} / {simd_groups}) * {simd_groups}"


def reduction_combine_expr(
    reduction_type: str, acc: str, val: str, dtype: torch.dtype
) -> str:
    """Per-iteration combine for ``LoopedReductionStrategy``'s accumulator."""
    value, _ = _cast(val, dtype)
    if reduction_type == "sum":
        return f"({acc} + {value})"
    if reduction_type == "prod":
        return f"({acc} * {value})"
    # c10::metal::max/min propagate NaN, matching torch semantics; ::metal:: does not.
    if reduction_type in ("max", "min"):
        return f"c10.metal.{reduction_type}({acc}, {value})"
    raise exc.BackendUnsupported("metal", f"reduction combine {reduction_type!r}")


def argreduce_result_expr(
    input_name: str,
    index_value: str,
    reduction_type: str,
    output_dtype: torch.dtype,
    *,
    block_size_var: str | None,
    index_dtype: torch.dtype | None,
    threads_in_group: int | None,
    dtype: torch.dtype | None,
) -> str:
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction

    if reduction_type not in SUPPORTED_ARGREDUCTIONS:
        raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")
    if index_dtype is None:
        raise exc.BackendUnsupported("metal", "missing index_dtype for argreduce")

    backend = CompileEnvironment.current().backend
    device_fn = DeviceFunction.current()
    group = resolve_group(device_fn, block_size_var)
    span = (
        threads_in_group
        if threads_in_group is not None
        else (group.span if group is not None else None)
    )
    value, value_type = _cast(input_name, dtype)
    index_type = backend.index_type_str(index_dtype)
    index = f"(static_cast<{index_type}>({index_value}))"
    out_type = backend.dtype_str(output_dtype)

    if span is None:
        raise exc.BackendUnsupported(
            "metal", f"unresolved thread group for reduction {reduction_type!r}"
        )
    if span <= 1:
        return f"(static_cast<{out_type}>({index}))"
    if span & (span - 1) or span > MAX_THREADS_PER_THREADGROUP:
        raise _unsupported_span(reduction_type, span)
    _reject_strided_group(reduction_type, group)
    if span < SIMD_WIDTH:
        result = f"{NS}.seg_{reduction_type}({value}, {index}, {span})"
        return f"(static_cast<{out_type}>({result}))"

    if span == SIMD_WIDTH:
        result = f"{NS}.simd_{reduction_type}({value}, {index})"
        return f"(static_cast<{out_type}>({result}))"

    if span % SIMD_WIDTH:
        raise _unsupported_span(reduction_type, span)
    if group is None:
        # See reduction_expr: the shared tier needs the in-group thread index.
        raise exc.BackendUnsupported(
            "metal",
            f"{reduction_type} over {span} threads whose thread axis could not "
            "be resolved",
        )
    base = _group_base_expr(span)
    value_buffer = alloc_threadgroup_buffer(value_type)
    index_buffer = alloc_threadgroup_buffer(index_type)
    result = (
        f"{NS}.tg_{reduction_type}("
        f"{value_buffer} + {base}, {index_buffer} + {base}, "
        f"{value}, {index}, {group.index_expr}, {span})"
    )
    return f"(static_cast<{out_type}>({result}))"


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
    accumulator index starts at ``reduction_index_init_expr`` (INT_MAX), so a
    row that never strictly beats the identity -- an all-``-inf`` row under
    argmax, say -- would otherwise never write ``acc_index`` and return the
    sentinel as if it were a position.

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
