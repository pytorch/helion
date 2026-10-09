from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
import operator
import re
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast
import weakref

import sympy
import torch

from .. import exc
from .._compat import shape_env_size_hint
from .._utils import indexing_uses_tensor_descriptor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .ast_read_writes import HELION_VEC_LANE_OF_ATTR
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .cute.access_regions import new_loop_instance
from .cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cute.register_tile_admission import RegisterTileUnsupported
from .cute.scalar_recipe import PURE_DECODE_HELPERS
from .device_function import DeviceFunction
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import Tcgen05PersistentProgramIDs
from .program_id import XYZProgramIDs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..language.memory_ops import CuteTileVecStoreSite
    from ..runtime.config import Config
    from .cute.lane_loop_distribution import LanePlacement
    from .cute.lane_loop_distribution import LaneScope
    from .cute.memory_ops import CuteLaneRelocation
    from .inductor_lowering import CodegenState
    from .pallas.dma import DmaResources

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


class ThreadAxisTracker:
    """Tracks thread axis assignments for block dimensions during codegen."""

    __slots__ = ("sizes", "block_axes")

    def __init__(self) -> None:
        self.sizes: dict[int, int] = {}
        self.block_axes: dict[int, int] = {}

    def record(self, block_idx: int, axis: int, size: int) -> None:
        """Record a thread axis mapping for a single block dimension."""
        self.sizes[axis] = max(self.sizes.get(axis, 1), size)
        self.block_axes[block_idx] = axis

    def record_all(self, block_ids: list[int], axis: int, size: int) -> None:
        """Record the same thread axis mapping for all block dimensions."""
        self.sizes[axis] = size
        for block_id in block_ids:
            self.block_axes[block_id] = axis

    def record_symbolic_axis(self, block_ids: Iterable[int], axis: int) -> None:
        """Record the thread axis of blocks whose extent is only known at launch.

        An argument-sized ``hl.tile`` (``block_size=bsz`` with ``bsz`` an int
        argument, ``static_shapes=False``) lives one element per thread on an
        axis the static shapes hold a one for, so no size is recorded; the
        axis itself still names the leader thread of a tile-uniform atomic
        (``cute.arch.thread_idx()[axis] == 0``), which ran once per thread of
        the axis while the block was missing here.  CuTe only: the other
        backends have no thread layout.
        """
        if CompileEnvironment.current().backend_name != "cute":
            return
        for block_id in block_ids:
            self.block_axes[block_id] = axis


def _lane_loop_iter(extent: int) -> ast.AST:
    # CuTe lane loops carry per-thread scalar state. Emitting them via
    # cutlass.range(_constexpr) miscompiles scalar matmul paths, so keep them
    # as ordinary Python loops.
    return expr_from_string(f"range({extent})")


def _static_lane_loop_extent(loop: ast.For) -> int | None:
    """Trip count of a synthetic lane loop, or ``None`` for any other loop.

    Recognizes the ascending ``range(N)`` form and the descending
    ``range(N - 1, -1, -1)`` form produced by :func:`_reverse_lane_loop_iter`
    for reverse scans; both visit the same ``N`` lanes.
    """
    iterator = loop.iter
    if not isinstance(iterator, ast.Call) or iterator.keywords:
        return None
    function = ast.unparse(iterator.func)
    if function not in ("range", "cutlass.range_constexpr"):
        return None
    args = iterator.args
    if not all(
        isinstance(arg, ast.Constant) and isinstance(arg.value, int) for arg in args
    ):
        return None
    values = [cast("int", cast("ast.Constant", arg).value) for arg in args]
    if len(values) == 1:
        extent = values[0]
    elif len(values) == 3 and values[1] == -1 and values[2] == -1:
        extent = values[0] + 1
    else:
        return None
    if extent <= 1:
        return None
    return extent


def _ascending_lane_loop_form(loop: ast.For) -> tuple[str, int] | None:
    """``(range function, extent)`` of a plain ascending synthetic lane loop.

    Matches ``range(N)`` / ``cutlass.range_constexpr(N)`` with a literal
    ``N >= 1``; anything else (already reversed, keyword arguments, dynamic
    bounds) yields ``None``.
    """
    iterator = loop.iter
    if (
        not isinstance(iterator, ast.Call)
        or len(iterator.args) != 1
        or iterator.keywords
    ):
        return None
    function = ast.unparse(iterator.func)
    extent = iterator.args[0]
    if (
        function not in ("range", "cutlass.range_constexpr")
        or not isinstance(extent, ast.Constant)
        or not isinstance(extent.value, int)
        or extent.value < 1
    ):
        return None
    return function, extent.value


def _lane_loop_reversible(loop: ast.For) -> bool:
    """Whether :func:`_reverse_lane_loop_iter` would succeed on ``loop``.

    A pure check: callers use it to decide *before* mutating anything.
    """
    return _ascending_lane_loop_form(loop) is not None


def _reverse_lane_loop_iter(loop: ast.For) -> bool:
    """Rewrite an ascending lane loop to visit its lanes in descending order.

    ``range(N)`` / ``cutlass.range_constexpr(N)`` become the ``(N - 1, -1, -1)``
    forms.  Lane bodies are order-independent except for a reverse
    ``hl.associative_scan`` carried across the lanes, which requests this.
    Returns ``False`` (leaving the loop untouched) when the iterator is not the
    plain ascending form, e.g. because it was already reversed.
    """
    form = _ascending_lane_loop_form(loop)
    if form is None:
        return False
    function, extent = form
    reversed_iter = expr_from_string(f"{function}({extent - 1}, -1, -1)")
    assert isinstance(reversed_iter, ast.expr)
    loop.iter = reversed_iter
    return True


def _create_lane_loop(
    lane_var: str, extent: int, body: list[ast.AST], *, constexpr: bool = False
) -> ast.For:
    """Build a synthetic lane loop.

    ``constexpr`` emits ``cutlass.range_constexpr`` so the lanes unroll at
    trace time: a register-tile reduction (see
    ``cute/register_tile_reductions.py``) needs every lane's loads issued
    before any lane's store and keeps its per-lane values in trace-time Python
    lists, which a rolled ``scf.for`` could not carry.
    """
    iterator = (
        expr_from_string(f"cutlass.range_constexpr({extent})")
        if constexpr
        else _lane_loop_iter(extent)
    )
    loop = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=iterator,
        body=body,
        orelse=[],
        type_comment=None,
    )
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return loop


def _encloses(loop: ast.For, statements: list[ast.AST]) -> bool:
    """Whether ``statements`` is the body of ``loop`` or of a loop nested in it."""
    return loop.body is statements or any(
        isinstance(child, ast.For) and _encloses(child, statements)
        for child in loop.body
    )


def _is_constexpr_lane_iter(loop: ast.For) -> bool:
    """Whether a synthetic lane loop unrolls at trace time."""
    iterator = loop.iter
    return (
        isinstance(iterator, ast.Call)
        and ast.unparse(iterator.func) == "cutlass.range_constexpr"
        and len(iterator.args) == 1
        and not iterator.keywords
    )


@dataclasses.dataclass(frozen=True)
class CuteLaneAxis:
    """How a CuTe per-thread strategy distributes one block axis of a tile.

    ``threads`` CUDA threads split the ``extent`` elements of the axis; each
    thread walks its remaining elements in ``lane_steps`` iterations of the
    ``lane_var`` lane loop, optionally as ``vec_width``-wide vectors whose
    elements the constexpr ``vec_lane_var`` loop (``vloop``) enumerates.
    ``strided`` lane steps are ``threads``-wide chunks of consecutive elements
    (``index = thread + step * threads``); blocked steps give every thread one
    contiguous slice.  Returned by the strategies' ``cute_lane_axis`` so
    consumers such as the scan lowering need not read strategy internals.
    """

    extent: int
    threads: int
    lane_var: str | None
    lane_steps: int
    vec_lane_var: str | None
    vec_width: int
    strided: bool


def _clone_lane_loop_with_body(loop: ast.For, body: list[ast.AST]) -> ast.For:
    """Clone a synthetic lane loop while preserving its iterator form.

    Most lane loops use ordinary ``range``.  A persistent reduction whose
    complete per-thread slice is one vector instead uses
    ``cutlass.range_constexpr(V)`` so the load/store hoist machinery can form
    one native vector transaction.  Reduction splitting must retain that
    constexpr iterator rather than silently rebuilding it as ``range``.
    """
    assert isinstance(loop.target, ast.Name)
    target = create(ast.Name, id=loop.target.id, ctx=ast.Store())
    iterator = ast.parse(ast.unparse(loop.iter), mode="eval").body
    cloned = create(
        ast.For,
        target=target,
        iter=iterator,
        body=body,
        orelse=[],
        type_comment=None,
    )
    lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
    if isinstance(lane_var, str):
        setattr(cloned, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return cloned


# Marker call emitted by reduction strategies when a reduction over a
# lane-distributed block is generated inside a single-pass lane loop.  The
# ``split_lane_loop_reductions`` post-pass recognizes these markers and
# rewrites the enclosing lane loop into a two-pass structure:
#
#   (phase 1) accumulate the per-lane reduction inputs across the lane loop,
#             then combine across the live thread axis (``threads_in_group``)
#             into the final scalar;
#   (finalize) define the reduced scalar between the two passes;
#   (phase 2) re-iterate the lanes to apply any lane-varying consumers (e.g.
#             the broadcast normalize / store) using the finalized scalar.
#
# The marker never reaches the emitted kernel — the post-pass strips every
# marker it processes.
_HELION_LANE_REDUCE_MARKER = "_helion_lane_reduce"
_CUTE_UNIFORM_GLOBAL_NAMES = frozenset(
    {
        "abs",
        "bool",
        "cutlass",
        "cute",
        "float",
        "int",
        "len",
        "math",
        "max",
        "min",
        "operator",
        "range",
    }
)
_CUTE_THREAD_VARYING_INTRINSICS = ("thread_idx", "lane_idx", "warp_idx")


def _lane_reduce_marker_expr(
    input_name: str,
    reduction_type: str,
    identity_expr: str,
    threads_in_group: int,
    *,
    group_pre: int = 1,
    group_span: int = 0,
    group_lane_expr: str = "",
    group_count: int = 1,
    group_cluster_n: int = 1,
    owner_lane: str | None = None,
    matmul_contribution: bool = False,
    strided_restore: bool = False,
    shared_lane_expr: str = "",
) -> str:
    # ``group_*`` (optional) carry the parameters of a strided grouped
    # reduction. They are required when the reduction's live thread axis is
    # interleaved with an unrelated sibling thread axis. ``group_lane_expr`` is
    # base64-free but may contain commas/parens, so it is passed as a string
    # literal that the post-pass re-parses.
    #
    # When ``group_span <= 32`` the de-interleaving fits in a single warp and
    # the finalize uses ``_cute_grouped_reduce_warp``. When ``group_span > 32``
    # (and a multiple of 32) the reduction group is spread across warps, so the
    # finalize uses the cross-warp ``_cute_grouped_reduce_shared_two_stage``;
    # ``group_count`` (the number of independent groups) is needed only by
    # that two-stage helper.
    #
    # ``group_lane_expr`` must stay a static linear combination of
    # ``thread_idx()`` coordinates: the post-pass parses it to recover the
    # reduce axis and to decide which consume stores need an owner predicate.
    # ``shared_lane_expr`` (optional) is a different lane expression that only
    # keys the two-stage helper's shared memory, e.g. the full runtime thread
    # id when a redundant thread axis may still be mapped later in codegen;
    # ``group_count`` then counts the groups of that keying.
    # Trailing positional arguments, each filled when a later one is present:
    # owner_lane, matmul_contribution, strided_restore, shared_lane_expr.
    owner = f", {owner_lane!r}" if owner_lane is not None else ""
    if matmul_contribution:
        assert owner_lane is not None and reduction_type == "sum"
    if matmul_contribution or strided_restore or shared_lane_expr:
        assert owner_lane is not None
        owner += f", {matmul_contribution!s}"
    if strided_restore or shared_lane_expr:
        # strided_restore: the loop body keeps the per-element strided
        # semantics, so a loop the two-pass split declines may finalize this
        # lane's raw input across the thread group in place -- complete when
        # the result feeds lane carries only; otherwise the per-lane shares
        # are totalled over the lanes (``_restore_lane_markers``).
        owner += f", {strided_restore!s}"
    if shared_lane_expr:
        owner += f", {shared_lane_expr!r}"
    return (
        f"{_HELION_LANE_REDUCE_MARKER}({input_name}, {reduction_type!r}, "
        f"{identity_expr}, {threads_in_group}, {group_pre}, {group_span}, "
        f"{group_lane_expr!r}, {group_count}, {group_cluster_n}{owner})"
    )


@dataclasses.dataclass
class _LaneReduceMarker:
    result_var: str
    input_name: str
    reduction_type: str
    identity_expr: str
    threads_in_group: int
    # The original RHS expression with the marker call replaced by the string
    # ``{finalized}``; ``finalize_expr(x)`` substitutes ``x`` to re-apply any
    # surrounding dtype cast / reshape to the finalized reduced scalar.
    wrap_template: str
    # Optional strided grouped reduction (the reduction's live thread axis
    # shares a warp / CTA with an unrelated sibling axis). When ``group_span``
    # > 0 the finalize uses a grouped reduction keyed on ``group_lane_expr``
    # instead of a plain consecutive-lane ``cute.arch.warp_reduction_*``:
    # ``_cute_grouped_reduce_warp`` when ``group_span <= 32`` (single warp),
    # ``_cute_grouped_reduce_shared_two_stage`` when ``group_span`` is a
    # multiple of 32 greater than 32 (cross-warp, ``group_count`` groups).
    group_pre: int = 1
    group_span: int = 0
    group_lane_expr: str = ""
    group_count: int = 1
    # > 1 when the reduction group is additionally split across the CTAs of
    # a thread-block cluster; the finalize then uses the DSM cluster reduce.
    group_cluster_n: int = 1
    # An explicit marker argument survives the text-based AST cloning below.
    # Physical thread groups and equal loop extents do not identify a serial
    # lane: an unrelated nested tile can have either in common with this one.
    owner_lane: str | None = None
    # Emitted only for the product side of a scalar matmul contraction. The
    # accumulator/rescale is deliberately outside this complete sum.
    matmul_contribution: bool = False
    # True when the loop body still carries the per-element strided semantics,
    # so a lane loop the two-pass split declines may finalize this lane's raw
    # input across the thread group in place: complete as it stands when the
    # result feeds lane carries only (each lane adds its share), otherwise the
    # shares are totalled over the lanes for a lane-invariant tail, and a
    # consumer varying with the lane is declined (``_restore_lane_markers``).
    strided_restore: bool = False
    # Optional lane expression that keys the cross-warp two-stage helper's
    # shared memory instead of ``group_lane_expr`` (which stays the static
    # expression the ownership analysis parses). Used to key on the full
    # runtime thread id so redundant thread axes get their own slots.
    shared_lane_expr: str = ""

    def finalize_expr(self, reduced: str) -> str:
        return self.wrap_template.replace("__HELION_FINALIZED__", f"({reduced})")


def _find_lane_reduce_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == _HELION_LANE_REDUCE_MARKER
        ):
            return sub
    return None


class _ReplaceLaneReduceCall(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == _HELION_LANE_REDUCE_MARKER
        ):
            return ast.copy_location(
                create(ast.Name, id="__HELION_FINALIZED__", ctx=ast.Load()), node
            )
        return node


def _is_lane_reduce_marker_assign(stmt: ast.AST) -> _LaneReduceMarker | None:
    """If ``stmt`` assigns an expression containing a single
    ``_helion_lane_reduce(IN, TYPE, ID, T)`` marker call, return a
    :class:`_LaneReduceMarker`; otherwise return ``None``.

    The marker may be nested inside a surrounding cast/reshape (e.g.
    ``R = cutlass.Float32(_helion_lane_reduce(...))``); the wrapping is
    captured so it can be re-applied to the finalized reduced scalar.
    """
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    call = _find_lane_reduce_call(stmt.value)
    if call is None or len(call.args) not in (8, 9, 10, 11, 12, 13):
        return None
    (
        input_node,
        type_node,
        identity_node,
        threads_node,
        group_pre_node,
        group_span_node,
        group_lane_node,
        group_count_node,
        *rest,
    ) = call.args
    group_cluster_n = int(ast.literal_eval(rest[0])) if rest else 1
    owner_lane = ast.literal_eval(rest[1]) if len(rest) >= 2 else None
    if owner_lane is not None and (not isinstance(owner_lane, str) or not owner_lane):
        raise exc.BackendUnsupported("cute", "invalid reduction lane owner")
    input_name = ast.unparse(input_node)
    reduction_type = ast.literal_eval(type_node)
    matmul_contribution = ast.literal_eval(rest[2]) if len(rest) >= 3 else False
    if type(matmul_contribution) is not bool or (
        matmul_contribution and (owner_lane is None or reduction_type != "sum")
    ):
        raise exc.BackendUnsupported("cute", "invalid matmul contribution marker")
    strided_restore = ast.literal_eval(rest[3]) if len(rest) >= 4 else False
    if type(strided_restore) is not bool or (strided_restore and owner_lane is None):
        raise exc.BackendUnsupported("cute", "invalid strided restore marker")
    shared_lane_expr = ast.literal_eval(rest[4]) if len(rest) == 5 else ""
    if not isinstance(shared_lane_expr, str):
        raise exc.BackendUnsupported("cute", "invalid reduction shared lane")
    identity_expr = ast.unparse(identity_node)
    threads_in_group = int(ast.literal_eval(threads_node))
    group_pre = int(ast.literal_eval(group_pre_node))
    group_span = int(ast.literal_eval(group_span_node))
    group_lane_expr = ast.literal_eval(group_lane_node)
    group_count = int(ast.literal_eval(group_count_node))
    # Build the wrap template by replacing the marker call with a sentinel.
    wrapped = _ReplaceLaneReduceCall().visit(ast.parse(ast.unparse(stmt.value)).body[0])
    assert isinstance(wrapped, ast.Expr)
    wrap_template = ast.unparse(wrapped.value)
    return _LaneReduceMarker(
        result_var=target.id,
        input_name=input_name,
        reduction_type=reduction_type,
        identity_expr=identity_expr,
        threads_in_group=threads_in_group,
        wrap_template=wrap_template,
        group_pre=group_pre,
        group_span=group_span,
        group_lane_expr=group_lane_expr,
        group_count=group_count,
        group_cluster_n=group_cluster_n,
        owner_lane=owner_lane,
        matmul_contribution=matmul_contribution,
        strided_restore=strided_restore,
        shared_lane_expr=shared_lane_expr,
    )


def validate_lane_reduce_owners(body: list[ast.AST]) -> None:
    """Reject markers inside a different serial lane before any rewriting.

    The compiler emitters attach the actual strategy's lane name. Older
    unowned markers remain supported by standalone AST helpers/tests; they are
    not emitted by production lowering. Serial non-lane loops are left to the
    existing interchange proof. Residual owned markers must not be restored to
    incomplete scalar inputs by the final safety net.
    """

    def visit(node: ast.AST, lanes: tuple[tuple[str, bool], ...]) -> None:
        lane = getattr(node, HELION_LANE_LOOP_VAR_ATTR, None)
        if isinstance(node, ast.For) and lane is not None:
            lanes = (*lanes, (lane, _is_constexpr_lane_iter(node)))
        marker = _is_lane_reduce_marker_assign(node)
        if marker is not None and marker.owner_lane is not None:
            # The owner is normally the innermost lane.  A register-tile
            # reduction nests trace-time tile lane loops (one-vector wrappers)
            # under a constexpr owner: those inner loops enumerate a thread's
            # own elements and do not redistribute the reduction, so the
            # owner may sit above them (``cute/register_tile_reductions.py``).
            owners = [
                index
                for index, (name, _) in enumerate(lanes)
                if name == marker.owner_lane
            ]
            if not owners or not all(
                constexpr for _, constexpr in lanes[owners[-1] + 1 :]
            ):
                raise exc.BackendUnsupported(
                    "cute", "reduction marker is nested in a different lane owner"
                )
        for child in ast.iter_child_nodes(node):
            visit(child, lanes)

    for statement in body:
        visit(statement, ())


def _combine_expr(reduction_type: str, acc: str, val: str) -> str:
    if reduction_type == "sum":
        return f"({acc}) + ({val})"
    if reduction_type == "prod":
        return f"({acc}) * ({val})"
    if reduction_type == "max":
        return f"({acc}) if ({acc}) > ({val}) else ({val})"
    if reduction_type == "min":
        return f"({acc}) if ({acc}) < ({val}) else ({val})"
    raise NotImplementedError(f"lane reduce combine {reduction_type!r}")


def _dtype_ctor_from_identity(identity_expr: str) -> str | None:
    """Extract the dtype constructor (e.g. ``cutlass.Float32``) from an identity
    expression like ``cutlass.Float32(0)`` so the per-lane input can be cast to
    the accumulator's dtype before combining."""
    try:
        node = ast.parse(identity_expr, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return None


def _grouped_warp_reduce_expr(
    reduction_type: str,
    acc: str,
    identity_expr: str,
    lane_expr: str,
    *,
    pre: int,
    group_span: int,
) -> str:
    """Strided grouped warp reduction over a single warp.

    Reduces ``acc`` across the ``group_span`` lanes that share the same
    ``lane % pre`` within each ``group_span``-lane block, so an interleaved
    sibling thread axis (occupying the low ``pre`` strides) stays distinct.
    """
    return (
        "_cute_grouped_reduce_warp("
        f"{acc}, {reduction_type!r}, {identity_expr}, {lane_expr}, "
        f"pre={pre}, group_span={group_span})"
    )


def _grouped_two_stage_reduce_stmts(
    result_acc: str,
    reduction_type: str,
    acc: str,
    identity_expr: str,
    lane_expr: str,
    *,
    pre: int,
    group_span: int,
    group_count: int,
) -> list[ast.AST]:
    """Cross-warp grouped reduction over ``group_span`` (> 32) lanes.

    Mirrors ``BlockReductionStrategy._strided_thread_reduction_expr``'s
    ``group_span > 32`` branch: the reduce group is spread across warps, so a
    single ``cute.arch.warp_reduction_*`` cannot fold it. The two-stage shared
    helper reduces each warp, stages the per-warp partials in shared memory, and
    combines them, keeping the ``pre`` interleaved sibling lanes distinct.

    Unlike that reference emitter this path has no shared-memory-budget fallback,
    but it does not need one: it only fires for a block-resident reduced tile
    whose live thread count is bounded by ``MAX_THREADS_PER_BLOCK`` (<= 1024, so
    <= 32 staged per-warp partials), which cannot overflow the reduction SMEM
    budget.

    Returns the (lane-index setup + reduce) statements that define
    ``result_acc`` (used in place of the single-shuffle ``reduced`` scalar).
    """
    lane_var = f"{result_acc}_lane"
    lane_in_group_var = f"{result_acc}_lane_in_group"
    lane_mod_pre_var = f"{result_acc}_lane_mod_pre"
    return [
        statement_from_string(f"{lane_var} = {lane_expr}"),
        statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}"),
        statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}"),
        statement_from_string(
            f"{result_acc} = _cute_grouped_reduce_shared_two_stage("
            f"{acc}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"pre={pre}, group_span={group_span}, group_count={group_count})"
        ),
    ]


def _finalize_lane_reduce_marker(m: _LaneReduceMarker, acc_var: str) -> list[ast.AST]:
    """Combine a marker's per-lane accumulator ``acc_var`` across the live
    thread axis and assign the finalized scalar to ``m.result_var``.

    Picks the cross-thread combine that matches the marker's thread layout:

    * a cross-warp two-stage shared reduction when the reduce group spans more
      than one warp (``group_span`` a multiple of 32 > 32);
    * a single-warp strided grouped reduction when the reduce axis shares a warp
      with an unrelated sibling axis (``1 < group_span <= 32`` with ``pre`` > 1);
    * a plain consecutive-lane warp reduction otherwise;
    * the accumulator unchanged when there is no live thread axis to combine.
    """
    if m.group_cluster_n > 1:
        # The two-pass marker finalize has no access to the preamble
        # buffer/mbarrier plumbing the DSM cluster reduce needs (only the
        # strided-thread-reduction path emits that), so a cluster-split
        # reduce landing here cannot be combined across the cluster —
        # reject the config loudly.
        from .. import exc

        raise exc.BackendUnsupported(
            "cute",
            "cute_cluster_n > 1 is not supported for the two-pass lane "
            "reduction finalize; only the strided cross-warp reduce path "
            "can perform the cluster combine",
        )
    if m.group_span > 32 and m.group_span % 32 == 0 and m.group_lane_expr:
        # Cross-warp: the reduce group is spread across warps, so fold the
        # per-lane accumulator with the two-stage shared-memory reduction.
        # The helper keys its shared memory on ``shared_lane_expr`` when the
        # emitter provided one (the full runtime thread id); the ownership
        # analysis above kept using the static ``group_lane_expr``.
        stmts = _grouped_two_stage_reduce_stmts(
            f"{acc_var}_reduced",
            m.reduction_type,
            acc_var,
            m.identity_expr,
            m.shared_lane_expr or m.group_lane_expr,
            pre=m.group_pre,
            group_span=m.group_span,
            group_count=m.group_count,
        )
        stmts.append(
            statement_from_string(
                f"{m.result_var} = {m.finalize_expr(f'{acc_var}_reduced')}"
            )
        )
        return stmts
    if m.group_span > 1 and m.group_pre > 1 and m.group_lane_expr:
        # Strided grouped reduction: the reduce axis shares a warp with an
        # unrelated sibling axis, so combine only the lanes that share the
        # current lane's sibling coordinate.
        reduced = _grouped_warp_reduce_expr(
            m.reduction_type,
            acc_var,
            m.identity_expr,
            m.group_lane_expr,
            pre=m.group_pre,
            group_span=m.group_span,
        )
    elif m.threads_in_group > 1:
        if m.threads_in_group > 32:
            # ``warp_reduction_*`` shuffles within one warp only; a wider
            # consecutive-lane group must arrive here with cross-warp group
            # params (``group_span`` a multiple of 32) instead of silently
            # summing only the first warp.
            from .. import exc

            raise exc.BackendUnsupported(
                "cute",
                "lane reduction across more than one warp requires the "
                f"cross-warp grouped finalize (threads={m.threads_in_group})",
            )
        reduced = _warp_reduce_expr(m.reduction_type, acc_var, m.threads_in_group)
    else:
        reduced = acc_var
    return [statement_from_string(f"{m.result_var} = {m.finalize_expr(reduced)}")]


def _warp_reduce_expr(reduction_type: str, acc: str, threads_in_group: int) -> str:
    tg = f", threads_in_group={threads_in_group}"
    if reduction_type == "sum":
        return f"cute.arch.warp_reduction_sum({acc}{tg})"
    if reduction_type == "max":
        return f"cute.arch.warp_reduction_max({acc}{tg})"
    if reduction_type == "min":
        return f"cute.arch.warp_reduction(({acc}), lambda a, b: a if a < b else b{tg})"
    if reduction_type == "prod":
        return f"cute.arch.warp_reduction(({acc}), lambda a, b: (a * b){tg})"
    raise NotImplementedError(f"lane warp reduce {reduction_type!r}")


def _backward_slice(
    body: list[ast.AST],
    roots: set[str],
    rename_groups: Mapping[str, str] | None = None,
) -> tuple[list[int], set[str]]:
    """Return the indices of the statements in ``body`` that (transitively)
    produce any name in ``roots``, plus the set of all names those statements
    write (as spelled).  Statements are scanned in reverse so a producer is
    included once a later consumer (already selected) reads its output.

    Names are compared through ``rename_groups``, the device function's
    aliases of each carried value: a device loop of ``body`` that
    accumulates a value writes it under its loop-output name (``v_8 =
    row_sums_copy_0 + sum_1`` for ``row_sums``), which only the final rename
    pass folds into the accumulator, so a slice over the names as spelled
    would take the accumulator's initializer for its only producer and
    leave the loop that accumulates it out.
    """
    from .ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    needed = canonical(roots)
    selected: list[int] = []
    written: set[str] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        if canonical(rw.writes) & needed:
            selected.append(idx)
            written |= set(rw.writes)
            needed |= canonical(rw.reads)
    selected.reverse()
    return selected, written


def split_lane_loop_reductions(
    body: list[ast.AST],
    *,
    uniform_names: set[str] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
    thread_axis_names: dict[str, frozenset[int]] | None = None,
    scalar_definitions: dict[str, ast.AST] | None = None,
    rename_groups: dict[str, str] | None = None,
    running_sums: set[str] | None = None,
) -> list[ast.AST]:
    """Rewrite single-pass lane loops that contain ``_helion_lane_reduce``
    markers into the two-pass accumulate / finalize / consume structure.

    Operates bottom-up so nested lane loops are handled before their parents.
    Lane loops without markers are returned unchanged (their inner statements
    are still recursed into so nested markers are processed).
    """
    if not any(_find_lane_reduce_call(stmt) is not None for stmt in body):
        return body

    proven_uniform = set(_CUTE_UNIFORM_GLOBAL_NAMES)
    if uniform_names is not None:
        proven_uniform.update(uniform_names)
    proven_thread_axes = dict(thread_axis_names or {})
    proven_scalar_definitions = dict(scalar_definitions or {})
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(
            _split_stmt_lane_reductions(
                stmt,
                proven_uniform,
                proven_disjoint_tensor_pairs or set(),
                proven_tensor_stride_values or {},
                proven_thread_axes,
                proven_scalar_definitions,
                rename_groups or {},
                running_sums or set(),
            )
        )
        _update_proven_uniform_names(stmt, proven_uniform)
        _update_thread_axis_names(stmt, proven_thread_axes)
        _update_scalar_definitions(stmt, proven_scalar_definitions)
    return new_body


def _restore_stmt_lane_reduce_markers(stmt: ast.AST) -> ast.AST:
    """Reject unlowered owned markers; restore only legacy unowned markers."""
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, [_restore_stmt_lane_reduce_markers(s) for s in old])
    if isinstance(stmt, ast.Assign):
        m = _is_lane_reduce_marker_assign(stmt)
        if m is not None:
            if m.owner_lane is not None:
                raise exc.BackendUnsupported(
                    "cute", "reduction marker has no proved lane lowering"
                )
            return statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
    return stmt


def restore_unprocessed_lane_reduce_markers(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Reject production markers whose lane lowering has not been proved.

    The proved interchange path removes its markers itself. A surviving owned
    marker has no reduction-preserving raw-input fallback. Unowned markers
    retain the legacy restore for standalone AST callers; production reduction
    emitters always attach an owner.

    Recurses only into statement-bearing fields (``body``/``orelse``/
    ``finalbody``) instead of using ``ast.NodeTransformer``; markers are always
    statement-level assignments, so this avoids the transformer's in-place
    mutation of expression list fields, which fails when an AST node carries a
    ``torch.fx`` ``immutable_list`` (e.g. multi-output ``inline_asm_elementwise``).
    """
    return [_restore_stmt_lane_reduce_markers(stmt) for stmt in body]


def _split_stmt_lane_reductions(
    stmt: ast.AST,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None = None,
) -> list[ast.AST]:
    # Recurse into any statement-list-bearing fields first so nested lane
    # loops are rewritten before the enclosing one.
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(
                stmt,
                field,
                split_lane_loop_reductions(
                    old,
                    uniform_names=set(uniform_names),
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                    proven_tensor_stride_values=proven_tensor_stride_values,
                    thread_axis_names=dict(thread_axis_names),
                    scalar_definitions=dict(scalar_definitions),
                    rename_groups=rename_groups,
                    running_sums=running_sums,
                ),
            )
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _split_one_lane_loop(
        stmt,
        lane_var,
        uniform_names,
        proven_disjoint_tensor_pairs,
        proven_tensor_stride_values,
        thread_axis_names,
        scalar_definitions,
        rename_groups,
        running_sums,
    )


@dataclasses.dataclass(frozen=True)
class _LaneRunningSum:
    """One compiler-created scalar self-add kept in the full first pass."""

    name: str
    update_index: int
    input_name: str
    phase1_indices: tuple[int, ...]
    iterator: str


def _certify_lane_running_sum(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _LaneRunningSum | None:
    """Prove the narrow mixed-matmul/owned-reduction composition.

    A name spelling is not provenance. Only the scalar accumulators recorded
    by matmul fallback may enter this path, and that provenance is necessary
    but insufficient: one unconditional self-add must have a complete, pure
    input slice independent of every marker and all other carried bindings.
    The marker inputs must also be independent of this running sum. Multiple
    definitions, forward/conditional definitions, collectives and unknown
    effects require another schedule and remain unsupported here.
    """
    from .ast_read_writes import ReadWrites
    from .ast_read_writes import ast_rename

    if not running_sums or any(marker.owner_lane is None for _, marker in markers):
        return None
    body = [_clone_stmt(statement) for statement in loop.body]
    for statement in body:
        ast_rename(statement, rename_groups)
    writes = [
        {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        for statement in body
    ]
    all_writes = set().union(*writes)
    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", body))
    candidates = (
        {_canonical_name(name, rename_groups) for name in running_sums}
        & effects.reads_before_write
        & all_writes
    )
    if not candidates:
        return None
    if _static_lane_loop_extent(loop) is None or loop.orelse:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires a complete static lane loop"
        )
    if len(candidates) != 1:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one scalar accumulator"
        )
    name = next(iter(candidates))
    positions = [index for index, names in enumerate(writes) if name in names]
    if len(positions) != 1:
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one unconditional self-add"
        )
    update_index = positions[0]
    update = body[update_index]
    addend = _self_addend(update, name)
    raw_update = loop.body[update_index]
    raw_addend = _self_addend(raw_update, name)
    if (
        _plain_assignment_name(update) != name
        or writes[update_index] != {name}
        or _plain_assignment_name(raw_update) != name
        or not isinstance(addend, ast.Name)
        or not isinstance(raw_addend, ast.Name)
        or addend.id not in all_writes
        or any(name in ReadWrites.from_ast(stmt).reads for stmt in body[:update_index])
    ):
        raise exc.BackendUnsupported(
            "cute", "certified lane running sum requires one unconditional self-add"
        )

    marker_indices = {index for index, _ in markers}
    marker_results = {
        _canonical_name(marker.result_var, rename_groups) for _, marker in markers
    }
    for index, statement in enumerate(body):
        if index in marker_indices:
            continue
        if _is_proven_relocatable_assignment(statement, allow_load=True):
            continue
        if _validated_idempotent_store(statement) is None:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has an unproved body effect"
            )

    selected: set[int] = {update_index}

    def require_input_slice(root: str, before: int) -> None:
        if root == name or root in marker_results:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum input depends on a carry or marker"
            )
        if root == lane_var or root not in all_writes:
            return
        producers = [index for index, names in enumerate(writes) if root in names]
        if len(producers) != 1:
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has no complete input slice"
            )
        index = producers[0]
        statement = body[index]
        if (
            index >= before
            or index in marker_indices
            or writes[index] != {root}
            or _plain_assignment_name(statement) != root
            or sum(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                for node in ast.walk(statement)
            )
            != 1
            or not _is_proven_relocatable_assignment(statement, allow_load=True)
            or _contains_unduplicatable_op(statement)
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has no complete input slice"
            )
        if index in selected:
            return
        selected.add(index)
        for dependency in ReadWrites.from_ast(statement).reads:
            require_input_slice(dependency, index)

    require_input_slice(addend.id, update_index)
    for index, marker in markers:
        require_input_slice(_canonical_name(marker.input_name, rename_groups), index)
    return _LaneRunningSum(
        name=name,
        update_index=update_index,
        input_name=raw_addend.id,
        phase1_indices=tuple(sorted(selected)),
        iterator=ast.dump(loop.iter),
    )


def _lane_invariant_tail_after_consume(
    statements: list[ast.AST],
    *,
    invariant: set[int],
    accumulated: set[int],
    consumed: set[int],
    rename_groups: dict[str, str],
) -> set[int]:
    """The lane-invariant tail statements that must follow the consume pass.

    A lane split runs the lane-varying statements of ``statements`` inside
    its passes (``accumulated`` in an accumulate pass, ``consumed`` in the
    final consume pass) and each ``invariant`` statement once, between the
    passes.  That keeps the body's order only where the passes' accesses are
    independent of the invariant statement.  One that must stay after a
    consumed statement (a store into the tensor a per-lane store fills, a
    load or an atomic of that tensor) is emitted after the consume pass
    instead, together with every later invariant statement ordered after
    it; one that must also stay before a consumed statement, or before an
    accumulated one, has no position and the split is rejected.  The order
    is the lane-loop distribution's (``cute/lane_loop_distribution.py``):
    register and tensor dependencies, with unknown effects and non-relaxed
    atomics ordered against every memory access.
    """
    from .ast_read_writes import ReadWrites
    from .cute.lane_loop_distribution import _analyze
    from .cute.lane_loop_distribution import _depends

    analyzed = {
        index: _analyze(index, statements[index], rename_groups)
        for index in invariant | accumulated | consumed
    }
    # A statement with unknown effects may mutate in place any local name it
    # reads or writes; a register statement sharing such a name with it keeps
    # its side of it (``_prepare`` of the distribution does the same).
    defined = {
        rename_groups.get(name, name)
        for statement in statements
        for name in ReadWrites.from_ast(statement).writes
    }
    for statement in analyzed.values():
        if statement.pinned:
            statement.touched = (statement.reads | statement.writes) & defined
    after: set[int] = set()
    for index in sorted(invariant):
        statement = analyzed[index]
        source = ast.unparse(statements[index]).partition("\n")[0]
        # A statement the accumulate pass re-runs is ordered inside that pass;
        # only its tail copy is placed here.
        if index not in accumulated and any(
            _depends(statement, analyzed[later])
            for later in accumulated
            if later > index
        ):
            raise exc.BackendUnsupported(
                "cute",
                "a lane-invariant statement is ordered before the accumulate pass "
                f"of a lane reduction: {source}",
            )
        follows = any(
            _depends(analyzed[earlier], statement)
            for earlier in consumed | after
            if earlier < index
        )
        precedes = any(
            _depends(statement, analyzed[later]) for later in consumed if later > index
        )
        if follows and precedes:
            raise exc.BackendUnsupported(
                "cute",
                "a lane-invariant statement is ordered between per-lane statements "
                f"of a lane reduction's consume pass: {source}",
            )
        if follows:
            after.add(index)
    return after


def _split_one_lane_loop(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None = None,
) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    markers: list[tuple[int, _LaneReduceMarker]] = []
    for idx, stmt in enumerate(body):
        parsed = _is_lane_reduce_marker_assign(stmt)
        if parsed is not None:
            markers.append((idx, parsed))
    if not markers:
        nested_owners = {
            marker.owner_lane
            for stmt in body
            for node in ast.walk(stmt)
            if (marker := _is_lane_reduce_marker_assign(node)) is not None
        }
        register_tile_failure: exc.BackendUnsupported | None = None
        if nested_owners and _is_constexpr_lane_iter(loop):
            if nested_owners == {lane_var}:
                # Markers nested in trace-time tile element loops under this
                # constexpr owner: the register-tile lowering schedules every
                # lane's loads before the first store and reduces each tile
                # element once.  A body outside its shape may still be a
                # lane-invariant guard around one (lifted below).
                from .cute.register_tile_reductions import lower_register_tile_lane_loop

                try:
                    return lower_register_tile_lane_loop(
                        loop,
                        lane_var,
                        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                        proven_tensor_stride_values=proven_tensor_stride_values,
                        thread_axis_names=thread_axis_names,
                        scalar_definitions=scalar_definitions,
                        rename_groups=rename_groups,
                    )
                except RegisterTileUnsupported as failure:
                    register_tile_failure = failure
            elif lane_var not in nested_owners:
                # A trace-time tile lane loop below a register-tile owner: the
                # enclosing owner lowers these markers together.
                return [loop]
        # A dynamic guard that does not depend on the synthetic lane may hide
        # reduction markers in one of its branches.  Move that guard outside
        # the lane loop so each branch owns an ordinary lane loop that can be
        # split below.  This occurs in packed recurrent kernels which guard a
        # state slot before performing several reductions over a free arange.
        lifted = _lift_lane_invariant_if(loop, lane_var, uniform_names, rename_groups)
        if lifted is not None:
            return split_lane_loop_reductions(
                lifted,
                uniform_names=set(uniform_names),
                proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                proven_tensor_stride_values=proven_tensor_stride_values,
                thread_axis_names=dict(thread_axis_names),
                scalar_definitions=dict(scalar_definitions),
                rename_groups=rename_groups,
                running_sums=running_sums,
            )
        if register_tile_failure is not None:
            # ``generate_ast`` regenerates the kernel with the rolled lane
            # nesting; the reason stays in its debug log.
            raise register_tile_failure
        if any(_find_lane_reduce_call(node) is not None for node in ast.walk(loop)):
            folded = _split_lane_loop_around_vector_lane(
                loop,
                lane_var,
                uniform_names,
                proven_disjoint_tensor_pairs,
                proven_tensor_stride_values,
                thread_axis_names,
                scalar_definitions,
                rename_groups,
                running_sums,
            )
            if folded is not None:
                return folded
            if any(
                isinstance(node, ast.For)
                and not _is_serial_for(node)
                and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) is None
                and any(_find_lane_reduce_call(s) is not None for s in node.body)
                for node in ast.walk(loop)
            ):
                # A ``cutlass.range_constexpr`` vector lane (``cute_vector_widths``
                # > 1) holds the marker and the body is outside the shape the
                # fold above proves (a collective or a matmul contribution in
                # the vector lane, a marker input independent of the vector
                # element, statements after the vector lane, a memory write
                # anywhere in the loop, a constexpr loop that is not this
                # lane's own vector lane): its complete reduction would need
                # the vector unroll folded together with this lane loop.
                raise exc.BackendUnsupported(
                    "cute",
                    "lane reduction nested in a constexpr vector lane "
                    "(cute_vector_widths > 1) has no proved lowering",
                )
            raise exc.BackendUnsupported(
                "cute",
                "lane reduction under a guard whose thread uniformity cannot be proven",
            )
        return [loop]

    marker_indices = {i for i, _ in markers}
    if any(
        marker.owner_lane is not None and marker.owner_lane != lane_var
        for _, marker in markers
    ):
        raise exc.BackendUnsupported(
            "cute", "reduction marker is nested in a different lane owner"
        )
    running_sum = _certify_lane_running_sum(
        loop, lane_var, markers, rename_groups, running_sums
    )
    # A certified matmul self-add stays in the complete accumulation pass.
    # Only its completed value is available to the remaining carry updates;
    # keep the existing independent-carry check for every other binding.
    tail_indices = [
        index
        for index in range(len(body))
        if running_sum is None or index != running_sum.update_index
    ]
    carry_body = [body[index] for index in tail_indices]
    carry_markers = {
        index
        for index, original in enumerate(tail_indices)
        if original in marker_indices
    }
    _validate_renamed_lane_carries(carry_body, lane_var, carry_markers, rename_groups)

    if _lane_split_reorders_aliasing_memory(
        body,
        markers,
        proven_disjoint_tensor_pairs,
        lane_var,
        _static_lane_loop_extent(loop),
        proven_tensor_stride_values,
        additional_inputs=(
            [(running_sum.update_index, running_sum.input_name)]
            if running_sum is not None
            else None
        ),
        rename_groups=rename_groups,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "synthetic-lane reduction would reorder a potentially aliasing write",
        )

    if any(marker.matmul_contribution for _, marker in markers):
        staged = _split_staged_matmul_lane_reductions(
            loop,
            lane_var,
            markers,
            thread_axis_names=thread_axis_names,
            scalar_definitions=scalar_definitions,
            rename_groups=rename_groups,
        )
        if staged is None:
            raise exc.BackendUnsupported(
                "cute", "matmul contribution has no complete staged lane schedule"
            )
        _validate_owned_lane_carry_schedule(
            body, staged, lane_var, marker_indices, rename_groups
        )
        return staged

    # A matmul whose *output* is reduced over a lane-distributed axis (e.g.
    # matmul_layernorm's ``acc.sum(-1)`` over the synthetic-lane N output) cannot
    # be handled by the per-lane / two-pass paths below: each lane owns a
    # distinct output column, so the reduction must combine DIFFERENT lanes, and
    # the matmul (an unduplicatable cross-thread shared-memory reduction) cannot
    # be re-run in a second lane pass.  The register-stash lowering runs the
    # matmul once, stashes each lane's output in a per-thread fragment, and
    # re-derives every downstream reduction / consumer from the stash.
    #
    # This is gated narrowly so it does NOT disturb kernels the existing paths
    # already handle correctly:
    #   * an unduplicatable op must feed the reduction, and
    #   * the marker results must NOT be consumed by a cross-lane loop-carried
    #     accumulator. A phi copy only rules out this stash path; it does not
    #     prove that an owned full-lane reduction can restore its raw input.
    if (
        running_sum is None
        and any(_contains_unduplicatable_op(stmt) for stmt in body)
        and not _markers_feed_cross_lane_carry(body, lane_var, markers)
    ):
        stashed = _split_lane_loop_with_register_stash(
            loop, lane_var, markers, rename_groups
        )
        if stashed is not None:
            _validate_owned_lane_carry_schedule(
                body, stashed, lane_var, marker_indices, rename_groups
            )
            return stashed

    # Safety: the two-pass split is only valid when the reduction marker is the
    # ONLY unproved cross-lane carried value in this lane loop. Another
    # loop-carried accumulator across the lanes (e.g. an uncertified matmul sum or a
    # plain ``extra += per_lane`` sum that already accumulates over the lanes),
    # splitting would drop or double-count it. Only legacy unowned markers can
    # use the original raw-input fallback; owned markers require a complete
    # reduction and are rejected before selecting a split subpath.
    if _has_extra_cross_lane_carry(carry_body, lane_var, carry_markers):
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    # Phase 1: the backward slices that produce the reduction inputs, read
    # through the rename groups: a device loop of the body that accumulates
    # an input writes it under its loop-output name, which only the final
    # rename pass folds into the accumulator, and the loop belongs to the
    # accumulate pass beside the initializer (the dynamic-shape nested-row
    # seeds of jagged_layer_norm reduced the fresh zero in a pass ahead of
    # the loop otherwise, and normalized by a zero mean and variance).  Each
    # input is sliced over the statements before its own marker: a later
    # statement that rewrites the input's group (a loop that keeps
    # accumulating the reduced value, an if-join) consumes the reduction's
    # input rather than producing it, and a slice over the whole body took
    # it for a producer, so the fold read the value behind it.
    if running_sum is not None:
        phase1_indices = list(running_sum.phase1_indices)
    else:
        phase1_index_union: set[int] = set()
        for marker_index, marker in markers:
            indices, _written = _backward_slice(
                body[:marker_index], {marker.input_name}, rename_groups
            )
            phase1_index_union.update(indices)
        phase1_indices = sorted(phase1_index_union)

    # A loop-output name the rename groups do not map (a body split on its
    # own, a carried value the device function has not registered) still
    # hides the update from the slice, which then selects the initializer
    # alone.  A serial loop before the marker that is not in the slice and
    # reads the marker input, or snapshots a name the slice defines in a phi
    # copy (``row_sums_copy = row_sums``: the loop carries that name and
    # rewrites it under its output name), is the sign of it: splitting that
    # shape would reduce the initializer (often zero) instead of the
    # completed accumulator.  Decline an owned reduction without a proved
    # slice; retain the legacy unowned per-lane behavior.
    phase1_index_set = set(phase1_indices)
    phase1_written = {
        name
        for index in phase1_indices
        for name in ReadWrites.from_ast(body[index]).writes
    }
    for marker_index, marker in markers:
        if any(
            index not in phase1_index_set
            and isinstance(stmt, (ast.For, ast.While))
            and (
                marker.input_name in ReadWrites.from_ast(stmt).reads
                or _carries_a_name(stmt, phase1_written)
            )
            for index, stmt in enumerate(body[:marker_index])
        ):
            return _restore_lane_markers(
                loop,
                lane_var,
                markers,
                rename_groups,
                thread_axis_names,
                scalar_definitions,
            )

    # Sequentially-dependent reductions (one marker's input depends on another
    # marker's result) require one accumulate/finalize pass per dependency
    # level.  The common single-pass path below remains preferable when all
    # markers are independent because it walks the lane extent only once.
    if set(phase1_indices) & marker_indices:
        dependent = _split_dependent_lane_reductions(
            loop,
            lane_var,
            markers,
            thread_axis_names=thread_axis_names,
            scalar_definitions=scalar_definitions,
            rename_groups=rename_groups,
        )
        if dependent is not None:
            _validate_owned_lane_carry_schedule(
                body, dependent, lane_var, marker_indices, rename_groups
            )
            return dependent
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    # The phase-1 (accumulate) and phase-2 (consume) passes both re-run the
    # reduction-input producers. That is only safe for side-effect-free
    # producers. A matmul / collective in the slice (cross-thread shared-memory
    # reductions, ``cute.gemm``, ``dot``) cannot be duplicated without racing on
    # shared memory. If the stash path did not prove a complete reduction,
    # decline owned markers rather than restoring an incomplete raw input.
    if any(_contains_unduplicatable_op(body[i]) for i in phase1_indices):
        return _restore_lane_markers(
            loop,
            lane_var,
            markers,
            rename_groups,
            thread_axis_names,
            scalar_definitions,
        )

    prefix: list[ast.AST] = []  # acc init statements (outside the lane loops)
    accumulate_body: list[ast.AST] = [body[i] for i in phase1_indices]
    marker_updates: dict[int, ast.AST] = {}
    finalize: list[ast.AST] = []
    for marker_index, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        prefix.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Cast the per-lane input to the accumulator dtype before combining so
        # the CUTLASS DSL's strict ternary type check (max/min emit a Python
        # ``a if a > b else b``) does not see mixed fp32/bf16 operands.
        ctor = _dtype_ctor_from_identity(m.identity_expr)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        update = statement_from_string(
            f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
        )
        accumulate_body.append(update)
        marker_updates[marker_index] = update
        finalize.extend(_finalize_lane_reduce_marker(m, acc_var))
    if running_sum is not None:
        # Keep the original producer/self-add order, interleaving each marker's
        # accumulation at its original position. Never regroup the matmul sum
        # or initialize it again at a synthetic-lane boundary.
        accumulate_body = [
            marker_updates[index] if index in marker_updates else body[index]
            for index in sorted(phase1_index_set | marker_indices)
        ]

    # Phase 2: everything except the marker assignments themselves; the
    # reduced scalar is already finalized so consumers read it directly.
    phase2_indices = [index for index in tail_indices if index not in marker_indices]
    phase2_body = [body[index] for index in phase2_indices]

    # A statement is lane-varying if it (transitively) reads the lane var, or
    # writes a name a lane-varying statement writes under any spelling of its
    # rename group (the initializer of an accumulator a lane-varying device
    # loop rewrites under its loop-output name: hoisted once between the
    # passes, every lane after the first would continue from the previous
    # lane's final value).  Statements that only depend on the finalized
    # scalar(s) are lane-invariant and run once after the lane loops;
    # lane-varying consumers run in a second lane loop, but only those that
    # contribute to a side effect (a store, an if-with-store, an in-place
    # write). Pure lane-varying producers that fed only the (now-removed)
    # reduction markers are dropped.  A lane-invariant statement runs between
    # the two loops unless the consume loop's accesses order it after them
    # (``_lane_invariant_tail_after_consume``).
    lane_varying_names = _lane_varying_names(phase2_body, lane_var, rename_groups)
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body,
        thread_axis_names,
        scalar_definitions,
    )

    def is_lane_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = {rename_groups.get(name, name) for name in rw.reads}
        writes = {rename_groups.get(name, name) for name in rw.writes}
        return (
            lane_var in reads
            or bool(reads & lane_varying_names)
            or bool(writes & lane_varying_names)
        )

    if running_sum is not None:
        # Compiler provenance identifies a completed matmul result, not a
        # prefix-sum API. An observable per-lane/conditional use of that value
        # needs a different proof; admit only pure lane-invariant finalizers.
        derived = _forward_live_names(phase2_body, {running_sum.name})
        if any(
            set(ReadWrites.from_ast(stmt).reads) & derived
            and (
                is_lane_varying(stmt)
                or not _is_proven_relocatable_assignment(stmt, allow_load=False)
            )
            for stmt in phase2_body
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum has a non-invariant consumer"
            )

    keep_indices = _live_phase2_indices(phase2_body)
    varying_indices = {i for i, s in enumerate(phase2_body) if is_lane_varying(s)}
    after_consume = _lane_invariant_tail_after_consume(
        phase2_body,
        invariant=set(range(len(phase2_body))) - varying_indices,
        accumulated={
            i for i, index in enumerate(phase2_indices) if index in phase1_index_set
        },
        consumed=varying_indices & keep_indices,
        rename_groups=rename_groups,
    )
    lane_invariant_tail: list[ast.AST] = []
    lane_invariant_after_consume: list[ast.AST] = []
    lane_varying_tail: list[ast.AST] = []
    for i, s in enumerate(phase2_body):
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            s,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(s), thread_axis_names),
            scalar_defs_before.get(id(s), {}),
        )
        if i in varying_indices:
            if i in keep_indices:
                lane_varying_tail.append(
                    _guard_stmt_with_owner(s, owner_exprs) if owner_exprs else s
                )
        else:
            if owner_exprs:
                predicate = " and ".join(
                    f"({owner_expr})" for owner_expr in owner_exprs
                )
                s = _guard_stmt_with_owner(s, [predicate])
            if i in after_consume:
                lane_invariant_after_consume.append(s)
            else:
                lane_invariant_tail.append(s)

    result: list[ast.AST] = []
    result.extend(prefix)
    result.append(_clone_lane_loop_with_body(loop, accumulate_body))
    result.extend(finalize)
    result.extend(lane_invariant_tail)
    if lane_varying_tail:
        result.append(_clone_lane_loop_with_body(loop, lane_varying_tail))
    result.extend(lane_invariant_after_consume)
    _validate_owned_lane_carry_schedule(
        body,
        result,
        lane_var,
        marker_indices,
        rename_groups,
        running_sum=running_sum,
    )
    return result


def _vector_lane_fold_shape(
    loop: ast.For, lane_var: str
) -> tuple[list[ast.AST], ast.For] | None:
    """``(prefix, vector lane)`` when every reduction marker of ``loop`` sits
    at the top level of one trailing constexpr vector lane that
    :func:`_split_lane_loop_around_vector_lane` can fold.

    The shape is the per-element body of a lane-looped tile axis with
    ``cute_vector_widths > 1``: ``prefix`` holds the lane base and the vector
    loads hoisted above the ``cutlass.range_constexpr(V)`` loop (plain
    assignments without a store, a collective or a marker), and the vector
    lane walks the elements of each packet.  The vector lane must be the
    wrapper loop the strategies build for ``lane_var`` (``vec_lane_{N}`` for
    ``lane_{N}``, see ``VecLaneWrapper``); any other constexpr loop is a
    serial unroll whose iterations the fold may not treat as lane
    coordinates.  Every marker must be a plain reduction owned by
    ``lane_var`` (no matmul contribution, no strided restore); the loop may
    not hold a collective, a nested loop or a marker anywhere else, and
    nothing may follow the vector lane.

    Nothing in the loop may write memory.  The split proves its passes
    reorder no aliasing access on the flat body, where the ``vec = lane``
    sentinel is a definition of ``vec``: the alias expansion would rewrite a
    packet-shifted address such as ``base + vec`` to ``lane * V + lane`` and
    reason in the wrong iteration space, so a store in the vector lane could
    be hoisted past a load of another lane's packet that the rolled loop
    orders after it.  With no write, the passes only re-run loads and the
    proof is trivially sound.
    """
    body: list[ast.AST] = list(loop.body)
    if not body:
        return None
    vloop = body[-1]
    if not (
        isinstance(vloop, ast.For)
        and _is_constexpr_lane_iter(vloop)
        and getattr(vloop, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and isinstance(vloop.target, ast.Name)
        and getattr(vloop, HELION_VEC_LANE_OF_ATTR, None) == lane_var
        and not vloop.orelse
    ):
        return None
    prefix = body[:-1]
    if any(
        not isinstance(stmt, ast.Assign)
        or _has_side_effect(stmt)
        or _find_lane_reduce_call(stmt) is not None
        for stmt in prefix
    ):
        return None
    if any(_has_observable_memory_write(stmt) for stmt in body):
        return None
    if _contains_unduplicatable_op(loop):
        return None
    markers = [
        marker
        for stmt in vloop.body
        if (marker := _is_lane_reduce_marker_assign(stmt)) is not None
    ]
    if not markers or any(
        marker.owner_lane != lane_var
        or marker.matmul_contribution
        or marker.strided_restore
        for marker in markers
    ):
        return None
    nested_markers = sum(
        _is_lane_reduce_marker_assign(node) is not None for node in ast.walk(loop)
    )
    if nested_markers != len(markers):
        return None
    if any(
        isinstance(node, (ast.For, ast.While))
        for stmt in vloop.body
        for node in ast.walk(stmt)
    ):
        return None
    return prefix, vloop


def _split_lane_loop_around_vector_lane(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> list[ast.AST] | None:
    """Split a lane loop whose reduction markers sit in its constexpr vector lane.

    ``cute_vector_widths > 1`` nests the per-element body of a lane-looped
    tile axis in ``for vec in cutlass.range_constexpr(V)`` below the vector
    load of each lane step, so a reduction over that axis reduces over the
    lane steps AND the vector elements.  The vector lane is a second lane
    coordinate rather than a serial loop: the two-pass / dependent split of
    :func:`_split_one_lane_loop` applies unchanged to the flat body
    ``prefix + [vec = lane] + elements``.  The sentinel assignment makes every
    element-level value lane-varying and marks where the vector lane begins;
    every marker input must depend on it, so each accumulate pass of the
    split keeps the vector lane and folds all ``lanes x V`` elements.  Each
    lane loop the split emits is re-nested at the sentinel: the prefix stays
    per lane step and the element statements (with the accumulator updates
    that follow them) go back under the constexpr loop, while the split's
    lane-invariant tail already runs once outside both.  Returns ``None``
    when the loop is outside the shape; a split decline propagates.
    """
    shape = _vector_lane_fold_shape(loop, lane_var)
    if shape is None:
        return None
    prefix, vloop = shape
    vec_var = cast("ast.Name", vloop.target).id
    sentinel_index = len(prefix)
    flat: list[ast.AST] = [
        *prefix,
        statement_from_string(f"{vec_var} = {lane_var}"),
        *vloop.body,
    ]
    for index, stmt in enumerate(flat):
        marker = _is_lane_reduce_marker_assign(stmt)
        if marker is None:
            continue
        indices, _ = _backward_slice(flat[:index], {marker.input_name}, rename_groups)
        if sentinel_index not in indices:
            return None
    lowered = _split_one_lane_loop(
        _clone_lane_loop_with_body(loop, flat),
        lane_var,
        uniform_names,
        proven_disjoint_tensor_pairs,
        proven_tensor_stride_values,
        thread_axis_names,
        scalar_definitions,
        rename_groups,
        running_sums,
    )
    return _renest_vector_lane(lowered, lane_var, vloop, vec_var)


def _renest_vector_lane(
    statements: list[ast.AST], lane_var: str, vloop: ast.For, vec_var: str
) -> list[ast.AST]:
    """Move the statements after the ``vec = lane`` sentinel of every emitted
    ``lane_var`` loop back under a copy of the constexpr vector lane."""

    def is_sentinel(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == vec_var
            and isinstance(stmt.value, ast.Name)
            and stmt.value.id == lane_var
        )

    def renest(stmt: ast.AST) -> ast.AST:
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(stmt, field, [renest(s) for s in old])
        if (
            isinstance(stmt, ast.For)
            and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) == lane_var
        ):
            positions = [i for i, s in enumerate(stmt.body) if is_sentinel(s)]
            if len(positions) == 1:
                split_at = positions[0]
                elements = stmt.body[split_at + 1 :]
                nested: list[ast.stmt] = []
                if elements:
                    nested.append(
                        create(
                            ast.For,
                            target=create(ast.Name, id=vec_var, ctx=ast.Store()),
                            iter=ast.parse(ast.unparse(vloop.iter), mode="eval").body,
                            body=elements,
                            orelse=[],
                            type_comment=None,
                        )
                    )
                stmt.body = [*stmt.body[:split_at], *nested]
        return stmt

    result = [renest(stmt) for stmt in statements]
    if any(is_sentinel(node) for stmt in result for node in ast.walk(stmt)):
        raise exc.BackendUnsupported(
            "cute",
            "lane reduction nested in a constexpr vector lane left the vector "
            "lane outside its lane loop",
        )
    return result


def _validate_owned_lane_carry_schedule(
    body: list[ast.AST],
    lowered: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    rename_groups: dict[str, str],
    *,
    running_sum: _LaneRunningSum | None = None,
) -> None:
    """Require each allowed carry update once after the full-lane reductions.

    The early variation check proves that a carry can be uniform after marker
    finalization; it does not prove that every split subpath preserves it.
    In particular, the register stash drops outside-only updates and repeats
    inside-observed updates in its consume loop. Admit a carry only when the
    selected schedule retains its exact normalized assignment once, outside
    every loop and after all marker results. The explicit matmul running-sum
    certificate is the sole exception: its original self-add and full input
    slice must remain once per lane, in order, inside the original iterator.
    Compound or multiple updates remain conservatively declined.
    """
    from .ast_read_writes import ast_rename

    if not any(
        marker.owner_lane is not None
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(body[index])) is not None
    ):
        return

    def normalize(statements: list[ast.AST]) -> list[ast.AST]:
        result = [_clone_stmt(statement) for statement in statements]
        for statement in result:
            ast_rename(statement, rename_groups)
        return result

    def binding_writes(statement: ast.AST) -> set[str]:
        # ReadWrites also models store(value) as an in-place memory write.
        # This proof counts assignments to scalar bindings, not stored values.
        return {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    original = normalize(body)
    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", original))
    original_writes = [binding_writes(statement) for statement in original]
    carried = (effects.reads_before_write & set().union(*original_writes)) - {lane_var}
    if not carried:
        return
    selected = normalize(lowered)
    marker_results = {
        marker.result_var
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(original[index])) is not None
    }
    selected_writes = [binding_writes(statement) for statement in selected]
    if running_sum is not None:
        positions = [
            index
            for index, writes in enumerate(selected_writes)
            if running_sum.name in writes
        ]
        if len(positions) != 1 or running_sum.name not in carried:
            raise exc.BackendUnsupported(
                "cute", "lane reduction cannot prove the certified per-lane update"
            )
        boundary = positions[0]
        phase1 = selected[boundary]
        expected: list[ast.AST] = []
        for index in sorted(set(running_sum.phase1_indices) | marker_indices):
            if index not in marker_indices:
                expected.append(body[index])
                continue
            marker = _is_lane_reduce_marker_assign(body[index])
            assert marker is not None
            accumulator = f"{marker.result_var}_lane_acc"
            ctor = _dtype_ctor_from_identity(marker.identity_expr)
            value = (
                f"{ctor}({marker.input_name})"
                if ctor is not None
                else marker.input_name
            )
            expected.append(
                statement_from_string(
                    f"{accumulator} = "
                    f"{_combine_expr(marker.reduction_type, accumulator, value)}"
                )
            )
        if not (
            isinstance(phase1, ast.For)
            and isinstance(phase1.target, ast.Name)
            and phase1.target.id == lane_var
            and ast.dump(phase1.iter) == running_sum.iterator
            and not phase1.orelse
            and [ast.dump(stmt) for stmt in phase1.body]
            == [ast.dump(stmt) for stmt in normalize(expected)]
        ):
            raise exc.BackendUnsupported(
                "cute", "lane reduction cannot prove the certified per-lane update"
            )
        # Every reduction must finalize after that complete lane sweep, before
        # a tail can observe the completed matmul accumulator.
        finalized_at: set[int] = set()
        for marker_result in marker_results:
            definitions = [
                index
                for index, writes in enumerate(selected_writes)
                if marker_result in writes
            ]
            if (
                len(definitions) != 1
                or definitions[0] <= boundary
                or _plain_assignment_name(selected[definitions[0]]) != marker_result
            ):
                raise exc.BackendUnsupported(
                    "cute", "certified lane update precedes no complete finalization"
                )
            finalized_at.add(definitions[0])
        from .ast_read_writes import ReadWrites

        if any(
            index != boundary
            and index <= max(finalized_at)
            and running_sum.name in ReadWrites.from_ast(statement).reads
            for index, statement in enumerate(selected)
        ):
            raise exc.BackendUnsupported(
                "cute", "certified lane running sum is consumed before finalization"
            )
    for name in carried:
        if running_sum is not None and name == running_sum.name:
            continue
        updates = [
            statement
            for statement, writes in zip(original, original_writes, strict=True)
            if name in writes
        ]
        positions = [
            index for index, writes in enumerate(selected_writes) if name in writes
        ]
        if (
            len(updates) == 1
            and isinstance(update := updates[0], ast.Assign)
            and len(update.targets) == 1
            and isinstance(update.targets[0], ast.Name)
            and update.targets[0].id == name
            and len(positions) == 1
            and ast.dump(selected[positions[0]]) == ast.dump(update)
        ):
            boundary = positions[0]
            finalized = set().union(*selected_writes[:boundary])
            if marker_results <= finalized and not any(
                writes & marker_results for writes in selected_writes[boundary:]
            ):
                continue
        raise exc.BackendUnsupported(
            "cute", "lane reduction cannot prove a once-per-tile carried update"
        )


def _validate_renamed_lane_carries(
    body: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    rename_groups: dict[str, str],
) -> None:
    """Check every independent carry before any stash or split can discard it.

    Always normalize production names, even when a raw-name carry already
    exists. That carry neither protects a second renamed update from the
    earlier stash path nor proves a complete per-lane reduction. An owned
    marker needs a full reduction over its lane; no subpath below proves that
    reduction together with an independent lane-varying carry. Decline the
    composition rather than losing either observable update or the reduction.

    Unowned standalone AST markers retain their legacy raw-carry behavior;
    all production emitters attach an owner.
    """
    from .ast_read_writes import ast_rename

    owned = any(
        marker.owner_lane is not None
        for index in marker_indices
        if (marker := _is_lane_reduce_marker_assign(body[index])) is not None
    )
    if not owned and (
        not rename_groups or _has_extra_cross_lane_carry(body, lane_var, marker_indices)
    ):
        return
    normalized = [_clone_stmt(statement) for statement in body]
    for statement in normalized:
        ast_rename(statement, rename_groups)
    if _has_extra_cross_lane_carry(
        normalized, lane_var, marker_indices, finalized_markers=owned
    ):
        raise exc.BackendUnsupported(
            "cute", "lane reduction cannot preserve an independent loop-carried value"
        )


def _lift_lane_invariant_if(
    loop: ast.For,
    lane_var: str,
    uniform_names: set[str],
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.AST] | None:
    """Lift one lane-invariant guard that hides reduction markers.

    Synthetic reduction lanes wrap the complete grid body.  Consequently a
    uniform runtime guard such as ``if state_index < 0`` can sit between the
    lane loop and its reduction markers, while :func:`_split_one_lane_loop`
    intentionally only rewrites direct marker assignments.  Lift the guard
    when its complete condition slice is independent of both the sequential
    lane and CUDA thread coordinates.  Each branch then owns a lane loop and
    can be processed by the normal reduction splitter.

    This is deliberately narrow: exactly one top-level guarded tail, no
    statements after it, and no side effects in the condition slice.  The
    slice is read through ``rename_groups``, so a loop updating a value of
    the condition under its loop-output name is in it; as a side effect it
    declines the lift, and the caller then rejects the markers under the
    guard ("lane reduction under a guard whose thread uniformity cannot be
    proven") or re-raises the register-tile failure, rather than restoring
    them.
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = [*loop.body]
    candidates = [
        (idx, stmt)
        for idx, stmt in enumerate(body)
        if isinstance(stmt, ast.If)
        and any(_find_lane_reduce_call(child) is not None for child in ast.walk(stmt))
    ]
    if len(candidates) != 1:
        return None
    if_index, branch = candidates[0]
    if if_index != len(body) - 1:
        return None

    prefix: list[ast.AST] = body[:if_index]
    condition_reads = set(ReadWrites.from_ast(branch.test).reads)
    condition_indices, condition_writes = _backward_slice(
        prefix, condition_reads, rename_groups
    )
    condition_nodes = [prefix[idx] for idx in condition_indices]
    external_reads = set(condition_reads)
    for stmt in condition_nodes:
        external_reads.update(ReadWrites.from_ast(stmt).reads)
    external_reads.difference_update(condition_writes)
    # Every dependency from outside the condition slice must be a kernel
    # argument, generated constexpr, module global, or an enclosing assignment
    # already proven independent of CUDA thread coordinates.
    if not all(_is_proven_uniform_name(name, uniform_names) for name in external_reads):
        return None
    condition_source = "\n".join(
        [ast.unparse(branch.test), *(ast.unparse(stmt) for stmt in condition_nodes)]
    )
    if lane_var in condition_source:
        return None
    if any(name in condition_source for name in _CUTE_THREAD_VARYING_INTRINSICS):
        return None
    if any(_has_side_effect(stmt) for stmt in condition_nodes):
        return None

    condition_index_set = set(condition_indices)
    lane_prefix = [
        _clone_stmt(stmt)
        for idx, stmt in enumerate(prefix)
        if idx not in condition_index_set
    ]

    def branch_loop(statements: list[ast.AST]) -> list[ast.stmt]:
        branch_body = [
            *(_clone_stmt(stmt) for stmt in lane_prefix),
            *statements,
        ]
        if not branch_body:
            return [ast.Pass()]
        return [
            cast(
                "ast.stmt",
                _clone_lane_loop_with_body(loop, branch_body),
            )
        ]

    lifted_if = create(
        ast.If,
        test=_clone_expr(branch.test),
        body=branch_loop([_clone_stmt(stmt) for stmt in branch.body]),
        orelse=branch_loop([_clone_stmt(stmt) for stmt in branch.orelse]),
    )
    return [*(_clone_stmt(stmt) for stmt in condition_nodes), lifted_if]


def _is_proven_uniform_name(name: str, uniform_names: set[str]) -> bool:
    return name in uniform_names or name.startswith(("_BLOCK_SIZE_", "_RDIM_SIZE_"))


def _update_proven_uniform_names(stmt: ast.AST, uniform_names: set[str]) -> None:
    """Track simple assignments proven uniform across the CUDA thread block."""
    from .ast_read_writes import ReadWrites

    rw = ReadWrites.from_ast(stmt)
    writes = set(rw.writes)
    if not writes:
        return
    source = ast.unparse(stmt)
    if not isinstance(stmt, ast.Assign) or any(
        name in source for name in _CUTE_THREAD_VARYING_INTRINSICS
    ):
        uniform_names.difference_update(writes)
        return
    reads = set(rw.reads)
    if all(_is_proven_uniform_name(name, uniform_names) for name in reads):
        uniform_names.update(writes)
    else:
        uniform_names.difference_update(writes)


def _thread_axes_read_by(
    node: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
) -> set[int]:
    """Physical thread axes that can affect an expression or statement."""
    axes: set[int] = set()

    class ThreadAxisVisitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                axes.update(thread_axis_names.get(node.id, ()))

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "lane_idx":
                # ``lane_idx`` is a linear warp coordinate, not CUDA axis 0.
                # In a multidimensional CTA it can depend on every physical
                # thread axis, and it repeats across warps.  This provenance is
                # intentionally a conservative superset so a lane-based value
                # or predicate cannot be mistaken for axis-0-only ownership.
                axes.update((0, 1, 2))
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "thread_idx"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                axes.add(node.slice.value)
            self.generic_visit(node)

    ThreadAxisVisitor().visit(node)
    return axes


def _update_thread_axis_names(
    stmt: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
) -> None:
    """Track physical-thread provenance through generated scalar assignments.

    The lane-reduction post-pass runs after code generation, when logical block
    provenance has otherwise been erased.  Keeping this small dataflow map lets
    the pass distinguish a real sibling output coordinate from a redundant
    launch coordinate before it emits a memory write.
    """
    from .ast_read_writes import ReadWrites

    rw = ReadWrites.from_ast(stmt)
    writes = set(rw.writes)
    if not writes:
        return
    axes = _thread_axes_read_by(stmt, thread_axis_names)
    marker = _is_lane_reduce_marker_assign(stmt)
    if marker is not None:
        reduce_axis = _lane_reduce_axis(marker)
        if reduce_axis is not None:
            axes.discard(reduce_axis)
    frozen_axes = frozenset(axes)
    for name in writes:
        thread_axis_names[name] = frozen_axes


def _update_scalar_definitions(
    stmt: ast.AST,
    scalar_definitions: dict[str, ast.AST],
) -> None:
    """Track simple generated scalar assignments for predicate proofs."""
    from .ast_read_writes import ReadWrites

    writes = set(ReadWrites.from_ast(stmt).writes)
    invalidated = set(writes)
    changed = True
    while changed:
        changed = False
        for name, expression in list(scalar_definitions.items()):
            if (
                name in invalidated
                or set(ReadWrites.from_ast(expression).reads) & invalidated
            ):
                scalar_definitions.pop(name)
                if name not in invalidated:
                    invalidated.add(name)
                    changed = True
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        scalar_definitions[stmt.targets[0].id] = stmt.value


def _statement_provenance_before(
    body: list[ast.AST],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> tuple[
    dict[int, dict[str, frozenset[int]]],
    dict[int, dict[str, ast.AST]],
]:
    """Snapshot physical-thread and scalar provenance before each statement."""
    thread_axes_before: dict[int, dict[str, frozenset[int]]] = {}
    scalar_defs_before: dict[int, dict[str, ast.AST]] = {}
    local_thread_axes = dict(thread_axis_names)
    local_scalar_defs = dict(scalar_definitions)
    for stmt in body:
        thread_axes_before[id(stmt)] = dict(local_thread_axes)
        scalar_defs_before[id(stmt)] = dict(local_scalar_defs)
        _update_thread_axis_names(stmt, local_thread_axes)
        _update_scalar_definitions(stmt, local_scalar_defs)
    return thread_axes_before, scalar_defs_before


def _lane_reduction_owner_exprs_for_statement(
    stmt: ast.AST,
    markers: list[tuple[int, _LaneReduceMarker]],
    marker_dependencies: list[tuple[_LaneReduceMarker, set[str]]],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[str]:
    """Return proven owner predicates for one reduction-consume statement.

    Calling ``_store_thread_axes`` even when no predicate is ultimately needed
    is intentional: it rejects unguardable memory writes and unknown side
    effects instead of letting a reduction-broadcast consumer race.
    """
    from .ast_read_writes import ReadWrites

    store_axes = _store_thread_axes(stmt, thread_axis_names, scalar_definitions)
    owner_exprs = _redundant_thread_axis_owner_exprs(markers, store_axes)
    if store_axes is None:
        return owner_exprs
    reads = set(ReadWrites.from_ast(stmt).reads)
    for marker, dependencies in marker_dependencies:
        owner_expr = _lane_reduce_owner_expr(marker, store_axes)
        if owner_expr is not None and reads & dependencies:
            owner_exprs.append(owner_expr)
    return list(dict.fromkeys(owner_exprs))


def _split_staged_matmul_lane_reductions(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
) -> list[ast.AST] | None:
    """Stash a collective prefix, then finalize proved product reductions.

    This path accepts only owned markers and straight-line scalar assignments.
    Each FP32 collective in the prefix executes once per lane; later passes
    read its register fragment. The existing dependent-reduction scheduler
    keeps final, lane-invariant recurrence updates outside every lane sweep.
    Both carry validators still apply to the original body and final schedule.
    """
    if any(marker.owner_lane != lane_var for _, marker in markers):
        return None
    extent = _static_lane_loop_extent(loop)
    if extent is None or not 1 < extent <= 256:
        return None
    body = list(loop.body)
    marker_indices = {index for index, _ in markers}
    names = [_plain_assignment_name(statement) for statement in body]
    if None in names or len(set(names)) != len(names):
        return None
    for index, statement in enumerate(body):
        if index not in marker_indices and not _is_proven_relocatable_assignment(
            statement, allow_load=True, allow_reduction=True
        ):
            return None
    collective_indices = [
        index
        for index, statement in enumerate(body)
        if _contains_unduplicatable_op(statement)
    ]
    if collective_indices and min(marker_indices) <= max(collective_indices):
        return None
    transformed = [_clone_stmt(statement) for statement in body]
    prefix: list[ast.AST] = []
    if collective_indices:
        uid = next(_LANE_STASH_COUNTER)
        compute = [
            _clone_stmt(statement) for statement in body[: max(collective_indices) + 1]
        ]
        for index in collective_indices:
            statement = body[index]
            assert isinstance(statement, ast.Assign)
            call = statement.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in _CHUNK_REDUCTION_HELPERS
                and len(call.args) >= 3
                and _dtype_ctor_from_identity(ast.unparse(call.args[2]))
                == "cutlass.Float32"
            ):
                return None
            name = names[index]
            fragment = f"_lane_stash_{uid}_{name}"
            prefix.append(
                statement_from_string(
                    f"{fragment} = cute.make_rmem_tensor({extent}, cutlass.Float32)"
                )
            )
            compute.append(statement_from_string(f"{fragment}[{lane_var}] = {name}"))
            transformed[index] = statement_from_string(
                f"{name} = {fragment}[{lane_var}]"
            )
        prefix.append(_clone_lane_loop_with_body(loop, compute))
    staged_loop = _clone_lane_loop_with_body(loop, transformed)
    assert isinstance(staged_loop, ast.For)
    scheduled = _split_dependent_lane_reductions(
        staged_loop,
        lane_var,
        markers,
        thread_axis_names=thread_axis_names,
        scalar_definitions=scalar_definitions,
        rename_groups=rename_groups,
    )
    return [*prefix, *scheduled] if scheduled is not None else None


def _split_dependent_lane_reductions(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
) -> list[ast.AST] | None:
    """Split an ordered chain of lane reductions into multiple passes.

    Each marker is accumulated and finalized before the next marker's input is
    recomputed.  This handles normalization/update/project chains such as
    ``sum(k*k) -> normalized_k -> sum(state*k) -> ...`` while keeping the
    existing one-pass rewrite for independent reductions.
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = [*loop.body]
    marker_indices = {idx for idx, _ in markers}
    marker_results = {marker.result_var for _, marker in markers}
    finalized: set[str] = set()
    result: list[ast.AST] = []
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body,
        thread_axis_names,
        scalar_definitions,
    )

    # Every stage indexes a prefix of the marker-free body below.
    accumulated: set[int] = set()
    for marker_index, marker in markers:
        available_body: list[ast.AST] = [
            stmt
            for idx, stmt in enumerate(body[:marker_index])
            if idx not in marker_indices
        ]
        stage_indices, _ = _backward_slice(
            available_body, {marker.input_name}, rename_groups
        )
        stage = [available_body[idx] for idx in stage_indices]
        accumulated.update(stage_indices)
        stage_reads = {
            name for stmt in stage for name in ReadWrites.from_ast(stmt).reads
        }
        if stage_reads & (marker_results - finalized):
            return None
        if any(_contains_unduplicatable_op(stmt) for stmt in stage):
            return None
        if any(_has_side_effect(stmt) for stmt in stage):
            return None

        acc_var = f"{marker.result_var}_lane_acc"
        result.append(statement_from_string(f"{acc_var} = {marker.identity_expr}"))
        acc_body = [_clone_stmt(stmt) for stmt in stage]
        ctor = _dtype_ctor_from_identity(marker.identity_expr)
        combine_val = (
            f"{ctor}({marker.input_name})" if ctor is not None else marker.input_name
        )
        acc_body.append(
            statement_from_string(
                f"{acc_var} = "
                f"{_combine_expr(marker.reduction_type, acc_var, combine_val)}"
            )
        )
        result.append(_clone_lane_loop_with_body(loop, acc_body))
        result.extend(_finalize_lane_reduce_marker(marker, acc_var))
        finalized.add(marker.result_var)

    # Re-run the lane-invariant tail once with every reduction finalized, and
    # only the dependency closure of observable lane-varying side effects in a
    # final lane pass.  The invariant tail must not be pruned to memory side
    # effects: generated SSA assignments can be loop-carried outputs whose
    # physical names are restored by the later rename pass (for example the
    # ``acc_cnt``/``acc_mean``/``acc_m2`` updates in Welford).
    consume_candidates: list[ast.AST] = [
        stmt for idx, stmt in enumerate(body) if idx not in marker_indices
    ]
    keep_indices = _live_phase2_indices(consume_candidates)
    lane_varying = _lane_varying_names(consume_candidates, lane_var, rename_groups)

    def is_lane_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = {rename_groups.get(name, name) for name in rw.reads}
        writes = {rename_groups.get(name, name) for name in rw.writes}
        return (
            lane_var in reads
            or bool(reads & lane_varying)
            or bool(writes & lane_varying)
        )

    varying_indices = {
        idx for idx, stmt in enumerate(consume_candidates) if is_lane_varying(stmt)
    }
    after_consume = _lane_invariant_tail_after_consume(
        consume_candidates,
        invariant=set(range(len(consume_candidates))) - varying_indices,
        accumulated=accumulated,
        consumed=varying_indices & keep_indices,
        rename_groups=rename_groups,
    )
    invariant: list[ast.AST] = []
    invariant_after_consume: list[ast.AST] = []
    varying: list[ast.AST] = []
    for idx, stmt in enumerate(consume_candidates):
        lane_varying_stmt = idx in varying_indices
        if lane_varying_stmt and idx not in keep_indices:
            continue
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            stmt,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(stmt), thread_axis_names),
            scalar_defs_before.get(id(stmt), {}),
        )
        cloned = _clone_stmt(stmt)
        if owner_exprs:
            cloned = _guard_stmt_with_owner(cloned, owner_exprs)
        if lane_varying_stmt:
            varying.append(cloned)
        elif idx in after_consume:
            invariant_after_consume.append(cloned)
        else:
            invariant.append(cloned)
    result.extend(invariant)
    if varying:
        result.append(_clone_lane_loop_with_body(loop, varying))
    result.extend(invariant_after_consume)
    return result


_LANE_STASH_COUNTER = itertools.count()


def _stash_dtype_for_value(
    value_name: str, body: list[ast.AST], markers: list[tuple[int, _LaneReduceMarker]]
) -> str:
    """Pick a CuTe scalar dtype constructor for a stashed lane value.

    Prefer the accumulator dtype of a marker whose input transitively reads the
    stashed value (matmul outputs feed an fp32 ``sum``), then any cast that
    wraps the value's defining assignment, then ``cutlass.Float32``.
    """
    from .ast_read_writes import ReadWrites

    for _idx, m in markers:
        slice_indices, _ = _backward_slice(body, {m.input_name})
        reads_value = any(
            value_name in ReadWrites.from_ast(body[i]).reads for i in slice_indices
        )
        if reads_value:
            ctor = _dtype_ctor_from_identity(m.identity_expr)
            if ctor is not None:
                return ctor
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == value_name
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "cutlass"
        ):
            return f"cutlass.{stmt.value.func.attr}"
    return "cutlass.Float32"


def _strip_ssa_suffix(name: str) -> str:
    """Strip Helion's SSA / loop-carry suffixes from a variable name.

    ``acc_1`` / ``acc_copy`` / ``acc_copy_0`` all collapse to ``acc`` so a
    loop-carried accumulator can be matched to its per-iteration rewrites.
    """
    base = re.sub(r"(_copy)(_\d+)*$", "", name)
    return re.sub(r"(_\d+)+$", "", base)


def _carries_a_name(loop: ast.AST, names: set[str]) -> bool:
    """Whether ``loop`` snapshots one of ``names`` in a phi copy (``X_copy = X``).

    Helion carries a value through a device loop with such a copy at the top
    of the loop's body and rewrites the value under the loop-output name,
    which only the final rename pass folds back into ``X``: a copy of a name
    inside the loop is the loop accumulating that name.
    """
    return any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Name)
        and node.value.id in names
        and node.targets[0].id != node.value.id
        and _strip_ssa_suffix(node.targets[0].id) == _strip_ssa_suffix(node.value.id)
        for node in ast.walk(loop)
    )


def _assigns_simple_name(stmt: ast.AST, names: set[str]) -> bool:
    """Return True when ``stmt`` is ``X = ...`` for some ``X`` in ``names``."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in names
    )


def _undup_stmt_rederives(stmt: ast.AST, name: str) -> bool:
    """Return True when ``stmt`` (an unduplicatable statement, typically the
    matmul K ``for`` loop) re-derives the loop-carried accumulator ``name``.

    The statement re-derives ``name`` when it both reads ``name`` (the live-in
    accumulator) and writes a value transitively dependent on ``name`` whose
    SSA-stripped name equals ``name`` (e.g. ``acc_1 = acc_copy_0 + ...``).  A
    plain input the matmul only reads (``indices_2``) is not re-derived.
    """
    from .ast_read_writes import ReadWrites

    if not isinstance(stmt, ast.For):
        rw = ReadWrites.from_ast(stmt)
        return name in rw.reads and any(_strip_ssa_suffix(w) == name for w in rw.writes)
    if name not in ReadWrites.from_ast(stmt).reads:
        return False
    # Forward slice from ``name`` within the loop body; True when it reaches a
    # write whose stripped name is ``name``.
    inner = list(stmt.body)
    tainted = {name}
    changed = True
    while changed:
        changed = False
        for s in inner:
            rw = ReadWrites.from_ast(s)
            if set(rw.reads) & tainted:
                for w in rw.writes:
                    if w not in tainted:
                        tainted.add(w)
                        changed = True
    return any(_strip_ssa_suffix(w) == name for w in tainted if w != name)


def _store_value_and_addr_reads(stmt: ast.AST) -> tuple[set[str], set[str]] | None:
    """For a statement containing a single ``(ADDR).store(VALUE)`` call, return
    ``(value_reads, addr_reads)`` — the names read in the stored VALUE and the
    names read in the ADDRESS expression (plus any guarding condition).

    Returns ``None`` when the statement has no ``.store(...)`` call (or has more
    than one, which this analysis does not attempt to characterize)."""
    stores = [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and len(node.args) == 1
    ]
    if len(stores) != 1:
        return None
    store = stores[0]
    value_reads = {n.id for n in ast.walk(store.args[0]) if isinstance(n, ast.Name)}
    # Everything read in the statement that is not part of the stored value
    # belongs to the address expression / guard (e.g. an enclosing ``if mask:``).
    all_reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    addr_reads = all_reads - value_reads
    return value_reads, addr_reads


def _lane_axis_wrongly_collapsed(
    tail: list[ast.AST],
    lane_var: str,
    lane_varying: set[str],
    stash_set: set[str],
    marker_result_names: set[str],
) -> bool:
    """Return True when the register-stash lowering would collapse the lane axis
    incorrectly.

    The stash lowering reduces each marker over the lane axis, which is only
    correct when the lane-distributed axis IS the reduced axis (matmul_layernorm:
    the reduced output free dim is broadcast back into a per-lane store whose
    VALUE still depends on the per-lane stash output).  When a side-effecting
    store writes to a lane-varying ADDRESS but its stored VALUE depends only on
    the reduced marker scalars (never on a per-lane stash output), each lane is a
    distinct, preserved output element that the lane reduction wrongly collapses
    (e.g. ``baddbmm(...).sum(-1)`` with the reduced dim folded into the matmul).

    ``lane_varying`` is the set of lane-derived names over the FULL lane-loop body
    (the per-lane output index, e.g. ``indices_2``, lives in the compute region
    rather than the tail, so it must be supplied by the caller).
    """
    # Names in the tail that (transitively) depend on a stashed per-lane value,
    # WITHOUT crossing a marker reduction.  A reduction marker collapses the lane
    # axis: its result is a cross-lane reduced scalar, no longer a per-lane value,
    # so the taint must STOP at marker results (a store of the reduced scalar is
    # exactly the collapse-bug signature, and must not count as stash-dependent).
    stash_tainted = _forward_taint_excluding_markers(
        tail, stash_set, marker_result_names
    )
    for stmt in tail:
        parsed = _store_value_and_addr_reads(stmt)
        if parsed is None:
            continue
        value_reads, addr_reads = parsed
        addr_lane_varying = lane_var in addr_reads or bool(addr_reads & lane_varying)
        value_depends_on_stash = bool(value_reads & stash_tainted)
        value_depends_on_marker = bool(value_reads & marker_result_names)
        if addr_lane_varying and value_depends_on_marker and not value_depends_on_stash:
            return True
    return False


def _forward_taint_excluding_markers(
    body: list[ast.AST], roots: set[str], marker_result_names: set[str]
) -> set[str]:
    """Forward-slice taint of ``roots`` through ``body`` that does NOT propagate
    across a lane-reduce marker.

    A marker assigns a cross-lane reduced scalar (``R = ..._helion_lane_reduce``);
    its result no longer depends on a single lane's value, so a statement that
    only writes a marker result must not inherit the taint even when its reduction
    input was tainted.
    """
    from .ast_read_writes import ReadWrites

    tainted = set(roots)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if not (set(rw.reads) & tainted):
                continue
            for w in rw.writes:
                # A marker result is a reduction boundary: never taint it.
                if w in marker_result_names:
                    continue
                if w not in tainted:
                    tainted.add(w)
                    changed = True
    return tainted


def _split_lane_loop_with_register_stash(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: Mapping[str, str] | None = None,
) -> list[ast.AST] | None:
    """Lower a lane loop whose reduction inputs depend on an unduplicatable op.

    The standard two-pass split re-runs the reduction-input producers in a
    second pass, which is unsafe when a matmul / cross-thread reduction is in
    the slice.  Instead, run the matmul-bearing *compute region* (the prefix of
    the body up to and including the last unduplicatable statement) exactly
    once, stashing each lane's unduplicatable-derived live-out values into
    per-thread register fragments.  Every downstream reduction marker and
    consumer is then re-derived from the stash (a plain register read, safely
    duplicatable), one accumulate/finalize pass per marker in dependency order,
    plus a final consume pass for the side-effecting statements.

    Returns the replacement statement list, or ``None`` when the pattern does
    not apply (caller then falls back to the per-lane single-pass behavior).
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    marker_indices = {i for i, _ in markers}
    extent = _lane_loop_extent(loop)

    # Compute region: prefix of the body up to (and including) the last
    # unduplicatable statement.  Everything after it must be free of
    # unduplicatable ops so it can be re-derived from the stash.
    undup_indices = [
        i for i in range(len(body)) if _contains_unduplicatable_op(body[i])
    ]
    if not undup_indices:
        return None
    region_end = max(undup_indices)
    region = body[: region_end + 1]
    tail = body[region_end + 1 :]
    tail_offset = region_end + 1
    if any(_contains_unduplicatable_op(s) for s in tail):
        return None
    # A marker inside the compute region cannot be re-derived from the stash
    # (its reduction would have to run before the matmul finishes).
    if any(i <= region_end for i in marker_indices):
        return None

    if extent <= 0 or extent > 256:
        return None

    lane_varying = _lane_varying_names(body, lane_var)

    tail_reads: set[str] = set()
    for stmt in tail:
        tail_reads |= set(ReadWrites.from_ast(stmt).reads)

    # Identify the loop-carried accumulators produced by the unduplicatable
    # statements — values whose post-region magnitude depends on the matmul and
    # therefore cannot be recomputed.  These are exactly the names that MUST be
    # stashed.  Helion emits a matmul K loop as
    # ``acc = 0; for k: acc_copy = acc; ...; acc_<n> = acc_copy_0 + reduce`` and
    # an implicit phi makes the post-loop ``acc`` equal the loop output
    # ``acc_<n>`` (collapsed by a later rename pass).  A name X is carried when:
    #   * X is written by a non-unduplicatable region statement (its ``acc = 0``
    #     seed) and read after the region (lane-varying), and
    #   * an unduplicatable statement's body re-derives X — its forward slice
    #     from X reaches a write whose SSA-stripped name equals X (``acc_1`` /
    #     ``acc_copy`` -> ``acc``).
    # The forward-slice condition distinguishes a true accumulator (``acc``,
    # rewritten each K step) from a plain input that the matmul merely reads
    # (``indices_2``, unchanged through the loop).
    seed_writes: set[str] = set()
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        seed_writes |= set(ReadWrites.from_ast(stmt).writes)
    # The carried accumulator's *name* (``acc`` from ``acc = 0``) is not itself
    # lane-varying — only the loop output alias (``acc_1``) is — so do NOT filter
    # candidates by ``lane_varying`` here.  The re-derives check confirms the
    # matmul transforms the accumulator, and below we require the re-deriving
    # statement to be lane-varying so a genuinely lane-invariant accumulator is
    # left alone.
    candidate_carried = seed_writes & tail_reads
    carried: set[str] = set()
    for name in candidate_carried:
        for i in undup_indices:
            if not _undup_stmt_rederives(body[i], name):
                continue
            stmt_reads = set(ReadWrites.from_ast(body[i]).reads)
            if lane_var in stmt_reads or bool(stmt_reads & lane_varying):
                carried.add(name)
                break
    # Also stash any value DIRECTLY produced by an unduplicatable statement that
    # is read by the tail (e.g. a matmul whose output is a fresh name rather than
    # an accumulator phi).
    undup_writes: set[str] = set()
    for i in undup_indices:
        undup_writes |= set(ReadWrites.from_ast(body[i]).writes)
    direct = undup_writes & lane_varying & tail_reads
    stash_names = sorted(carried | direct)
    if not stash_names:
        return None

    stash_set = set(stash_names)

    # Correctness gate: the register-stash lowering reduces each marker OVER THE
    # LANE axis (stash per lane, then sum lanes + warp-reduce).  That is only
    # valid when the lane-distributed axis IS the reduced axis — i.e. the genuine
    # matmul_layernorm pattern, where the matmul OUTPUT free dim is split across
    # the lane and the layernorm reduces exactly that dim, then broadcasts the
    # reduced scalar back into a *per-lane* normalize+store (each lane keeps a
    # distinct, stash-derived output value).
    #
    # A different pattern — e.g. ``baddbmm(...).sum(-1)`` over a small static dim
    # — folds the reduced axis into the matmul itself, leaving each lane holding a
    # COMPLETE, distinct output element.  There the lane axis is a *preserved*
    # output dim, the spurious marker reduces over the wrong (lane) axis, and the
    # store writes the single reduced scalar to lane-varying addresses (every lane
    # storing the same collapsed value).  Detect that here and bail to the
    # correct per-lane path: a side-effecting store whose ADDRESS depends on the
    # lane var but whose stored VALUE depends only on reduced marker results (no
    # per-lane stash output) means the lane axis was wrongly collapsed.
    marker_result_names = {m.result_var for _, m in markers}
    if _lane_axis_wrongly_collapsed(
        tail, lane_var, lane_varying, stash_set, marker_result_names
    ):
        return None
    # Region statements that are *duplicatable* and can be recomputed cheaply in
    # the later passes (e.g. ``indices_2 = thread_idx[0] + lane * 4``).  Drop the
    # unduplicatable statements and any statement that produces a stashed name.
    recompute: list[ast.AST] = []
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        writes = set(ReadWrites.from_ast(stmt).writes)
        if writes & stash_set:
            continue
        recompute.append(stmt)
    # Keep only the recompute statements that (transitively) feed the tail.
    recompute_keep_idx, _ = _backward_slice(recompute, tail_reads, rename_groups)
    recompute_kept = [recompute[i] for i in recompute_keep_idx]
    # The recompute slice must not pull in an unduplicatable op or reference a
    # stashed name (which is only available from the fragment, not recomputable).
    for s in recompute_kept:
        if _contains_unduplicatable_op(s):
            return None

    # Allocate one register fragment per stashed value.
    uid = next(_LANE_STASH_COUNTER)
    frag_by_name: dict[str, str] = {}
    decls: list[ast.AST] = []
    for name in stash_names:
        frag = f"_lane_stash_{uid}_{name}"
        frag_by_name[name] = frag
        dtype = _stash_dtype_for_value(name, body, markers)
        decls.append(
            statement_from_string(f"{frag} = cute.make_rmem_tensor({extent}, {dtype})")
        )

    def read_stash_stmts() -> list[ast.AST]:
        return [
            statement_from_string(f"{name} = {frag_by_name[name]}[{lane_var}]")
            for name in stash_names
        ]

    # Phase 0: run the compute region once and stash the live-out values.
    phase0_body: list[ast.AST] = list(region)
    for name in stash_names:
        phase0_body.append(
            statement_from_string(f"{frag_by_name[name]}[{lane_var}] = {name}")
        )

    # The marker assignments within the tail produce already-finalized scalars
    # (computed once after each reduction pass), so they must NOT be re-run as
    # per-lane passthroughs inside any later pass.  Each marker's input is
    # re-derived from the non-marker tail statements before it (a later
    # rewrite of the input's group consumes the reduction's input), with the
    # finalized marker result vars as pre-defined boundaries of the slices.
    marker_result_vars = {m.result_var for _, m in markers}

    result: list[ast.AST] = []
    result.extend(decls)
    result.append(_clone_lane_loop_with_body(loop, phase0_body))

    # Process each marker in source order (they are sequentially dependent: a
    # later marker's input may read an earlier marker's finalized scalar).
    for marker_index, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        result.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Accumulate pass: recompute cheap region producers, read the stash,
        # then re-derive this marker's input and fold it into the accumulator.
        # The slice runs over the re-derivable tail before the marker only;
        # references to other markers' results stop at those finalized scalars.
        rederivable_before = [
            s
            for index, s in enumerate(tail, tail_offset)
            if index < marker_index and _is_lane_reduce_marker_assign(s) is None
        ]
        input_slice_idx, _ = _backward_slice(
            rederivable_before, {m.input_name}, rename_groups
        )
        input_stmts = [
            rederivable_before[i]
            for i in input_slice_idx
            if not _assigns_simple_name(rederivable_before[i], marker_result_vars)
        ]
        acc_body: list[ast.AST] = []
        acc_body.extend(_clone_stmt(s) for s in recompute_kept)
        acc_body.extend(read_stash_stmts())
        acc_body.extend(_clone_stmt(s) for s in input_stmts)
        ctor = _dtype_ctor_from_identity(m.identity_expr)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        acc_body.append(
            statement_from_string(
                f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
            )
        )
        result.append(_clone_lane_loop_with_body(loop, acc_body))
        result.extend(_finalize_lane_reduce_marker(m, acc_var))

    # Final consume pass: everything in the tail except the marker assignments,
    # re-derived from the stash + finalized scalars.  Only keep statements that
    # feed a side effect (a store / in-place write).
    consume_candidates = [s for i, s in enumerate(body) if i >= tail_offset]
    consume_candidates = [
        s for s in consume_candidates if _is_lane_reduce_marker_assign(s) is None
    ]
    keep_idx = _live_phase2_indices(consume_candidates)
    consume_kept = [s for i, s in enumerate(consume_candidates) if i in keep_idx]
    if consume_kept:
        consume_body: list[ast.AST] = []
        consume_body.extend(_clone_stmt(s) for s in recompute_kept)
        consume_body.extend(read_stash_stmts())
        consume_body.extend(_clone_stmt(s) for s in consume_kept)
        result.append(_clone_lane_loop_with_body(loop, consume_body))
    return result


def _markers_feed_cross_lane_carry(
    body: list[ast.AST],
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
) -> bool:
    """Return True when the lane loop carries an accumulator across the lanes
    that a marker result feeds (online-softmax attention's ``m_i`` / ``l_i`` /
    ``acc`` recurrence).

    Helion represents a loop-carried value with a phi ``X_copy = X`` read at the
    TOP of the loop body (before the value is rewritten) and a corresponding
    output assignment renamed back to ``X`` by a later pass. Such a copy
    conservatively excludes the register-stash path, which assumes each marker
    result is consumed only by per-lane / store consumers. It does not prove
    completeness for a raw-input restore; owned markers still require a
    complete reduction lowering.

    matmul_layernorm's N-output lane loop has no such top-level ``X_copy = X``
    carry (its ``acc`` is the matmul accumulator, carried by the *inner* K loop,
    not across the lane iterations), so this returns False and the stash path is
    free to run.
    """
    from .ast_read_writes import ReadWrites

    if not markers:
        return False

    # Live-in names of the lane body (read before written), excluding the lane
    # var.  A loop-carried accumulator phi is live-in.
    written_so_far: set[str] = set()
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written_so_far:
                live_in.add(name)
        written_so_far |= set(rw.writes)

    # Detect top-level phi copies ``X_copy = X`` of a live-in accumulator.
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            src = stmt.value.id
            dst = stmt.targets[0].id
            if src in live_in and _strip_ssa_suffix(dst) == _strip_ssa_suffix(src):
                return True
    return False


def _has_extra_cross_lane_carry(
    body: list[ast.AST],
    lane_var: str,
    marker_indices: set[int],
    *,
    finalized_markers: bool = False,
) -> bool:
    """Return True when ``body`` contains a loop-carried accumulator across the
    lanes that is INDEPENDENT of the reduction markers.

    Two-pass splitting is correct whenever every cross-lane carried value
    transitively consumes a marker result (e.g. an online-softmax
    ``mi = max(mi, local_amax)`` or ``di = di + sum``): after the split the
    carried update runs once per outer iteration on the fully-reduced scalar.
    But an *independent* carried accumulator — one that does not depend on any
    marker result, such as a matmul ``dot_acc += dot_product`` — must keep
    accumulating once per lane, so the split would drop or corrupt it. In that
    case the caller falls back to the single-pass per-lane behavior.

    For owned-marker legality, ``finalized_markers`` checks variation after
    treating every marker result as a completed, lane-invariant reduction.
    Merely depending on a marker is insufficient: ``carry + partial + reduced``
    still contains an independent lane-varying contribution. Only an update
    that becomes lane-invariant after finalization can move to the tile tail.
    The default retains the legacy unowned-marker predicate.

    A carried value is detected as a name that is *live-in* to the lane body
    (read before it is written within the body, directly or through a
    ``X_copy = X`` alias) and also written within the body.
    """
    from .ast_read_writes import ReadWrites

    definitely_written_so_far: set[str] = set()
    aliases: dict[str, str] = {}  # copy_var -> original carried name
    live_in: set[str] = set()
    for stmt in body:
        effects = _ordered_statement_reads_writes(stmt)
        for name in effects.reads_before_write:
            if name != lane_var and name not in definitely_written_so_far:
                live_in.add(name)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            aliases[stmt.targets[0].id] = stmt.value.id
        definitely_written_so_far.update(effects.definite_writes)

    def root(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    # Names that (transitively) depend on a marker result. Carried values that
    # only depend on these are fine under the two-pass split.
    marker_results = {
        m.result_var
        for i, stmt in enumerate(body)
        if i in marker_indices
        and (m := _is_lane_reduce_marker_assign(stmt)) is not None
    }
    marker_tainted = set(marker_results)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & marker_tainted:
                for w in rw.writes:
                    if w not in marker_tainted:
                        marker_tainted.add(w)
                        changed = True

    # Names that (transitively) depend on the lane var. A carried accumulator
    # whose per-iteration update is lane-varying (e.g. a matmul
    # ``dot_acc += dot_product(lane)``) cannot move to the once-per-tile tail,
    # so the two-pass split would break it. A lane-invariant update (e.g.
    # welford's ``acc_cnt += block_size``) is fine in the tail.
    variation_body = (
        [
            statement
            for index, statement in enumerate(body)
            if index not in marker_indices
        ]
        if finalized_markers
        else body
    )
    lane_varying = _lane_varying_names(variation_body, lane_var)

    # Bail if some carried accumulator is independent of every marker AND its
    # update consumes a lane-varying value.
    for idx, stmt in enumerate(body):
        if idx in marker_indices:
            continue
        rw = ReadWrites.from_ast(stmt)
        reads = set(rw.reads)
        update_is_lane_varying = lane_var in reads or bool(reads & lane_varying)
        if not update_is_lane_varying:
            continue
        for w in rw.writes:
            if root(w) in live_in and (
                finalized_markers or root(w) not in marker_tainted
            ):
                return True
    return False


class _OrderedReadWrites(NamedTuple):
    reads_before_write: set[str]
    may_writes: set[str]
    definite_writes: set[str]


def _ordered_block_reads_writes(body: list[ast.stmt]) -> _OrderedReadWrites:
    reads_before_write: set[str] = set()
    may_writes: set[str] = set()
    definitely_written_so_far: set[str] = set()
    for stmt in body:
        effects = _ordered_statement_reads_writes(stmt)
        reads_before_write.update(
            effects.reads_before_write - definitely_written_so_far
        )
        may_writes.update(effects.may_writes)
        definitely_written_so_far.update(effects.definite_writes)
    return _OrderedReadWrites(
        reads_before_write,
        may_writes,
        definitely_written_so_far,
    )


def _literal_int_expr(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _literal_int_expr(node.operand)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if is_builtin_int or is_cutlass_int:
            return _literal_int_expr(node.args[0])
    return None


def _for_is_provably_nonempty(stmt: ast.For) -> bool:
    if any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
        for node in ast.walk(stmt)
    ):
        return False
    iterator = stmt.iter
    if not (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
        and not iterator.keywords
        and 1 <= len(iterator.args) <= 3
    ):
        return False
    values = [_literal_int_expr(arg) for arg in iterator.args]
    if any(value is None for value in values):
        return False
    integers = cast("list[int]", values)
    try:
        return len(range(*integers)) > 0
    except ValueError:
        return False


def _ordered_statement_reads_writes(stmt: ast.AST) -> _OrderedReadWrites:
    """Return ordered reads plus may-write and definitely-write sets.

    :class:`ReadWrites` intentionally summarizes a compound statement as an
    unordered set.  That is too conservative for cross-*outer*-lane carry
    detection: locals assigned near the top of a generated serial loop and
    read later in that same loop look like live-ins, causing a valid reduction
    marker to be replaced by its per-lane value.

    A conditional write, however, cannot suppress a later read unless every
    branch definitely writes the name.  Likewise, writes in a loop only become
    definite when its generated ``range`` is statically non-empty.  Keeping
    may-writes separate preserves both properties.
    """
    from .ast_read_writes import ReadWrites

    if isinstance(stmt, ast.If):
        test_rw = ReadWrites.from_ast(stmt.test)
        body = _ordered_block_reads_writes(stmt.body)
        orelse = _ordered_block_reads_writes(stmt.orelse)
        return _OrderedReadWrites(
            set(test_rw.reads) | body.reads_before_write | orelse.reads_before_write,
            set(test_rw.writes) | body.may_writes | orelse.may_writes,
            set(test_rw.writes) | (body.definite_writes & orelse.definite_writes),
        )

    if not isinstance(stmt, ast.For):
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        # Other compound statements (while/try/with/match) may skip or branch
        # around writes. Treat their writes as possible only; generated plain
        # assignments and expression statements execute unconditionally.
        definite_writes = (
            set()
            if any(
                isinstance(stmt, kind)
                for kind in (ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
            )
            else writes
        )
        return _OrderedReadWrites(set(rw.reads), writes, definite_writes)

    iter_rw = ReadWrites.from_ast(stmt.iter)
    target_rw = ReadWrites.from_ast(stmt.target)
    target_writes = set(target_rw.writes)
    body = _ordered_block_reads_writes(stmt.body)
    orelse = _ordered_block_reads_writes(stmt.orelse)
    reads_before_write = set(iter_rw.reads)
    reads_before_write.update(body.reads_before_write - target_writes)
    # A ``for ... else`` can execute its else suite without one body iteration,
    # so no body/target write is available to suppress these reads.
    reads_before_write.update(orelse.reads_before_write)
    may_writes = (
        set(iter_rw.writes) | target_writes | body.may_writes | orelse.may_writes
    )
    definite_writes = set(iter_rw.writes)
    if _for_is_provably_nonempty(stmt):
        definite_writes.update(target_writes)
        definite_writes.update(body.definite_writes)
    return _OrderedReadWrites(reads_before_write, may_writes, definite_writes)


def _lane_reduce_owner_expr(
    marker: _LaneReduceMarker,
    store_axes: _StoreThreadAxes | None = None,
) -> str | None:
    """Predicate selecting one writer for a broadcast reduction result."""
    reduce_axis = _lane_reduce_axis(marker)
    if store_axes is not None and reduce_axis is not None:
        if reduce_axis in store_axes.address:
            return None
        if reduce_axis in store_axes.unique_predicate:
            return None
        if reduce_axis in store_axes.predicate or reduce_axis in store_axes.value:
            raise exc.BackendUnsupported(
                "cute",
                "cannot infer unique ownership for a thread-varying store",
            )
    if marker.group_span > 1 and marker.group_lane_expr:
        # ``group_pre`` is the product of sibling coordinates below the
        # reduction axis.  The first ``group_pre`` lanes in each group are the
        # reduction-coordinate-zero owners, one for each sibling element.
        return (
            f"(({marker.group_lane_expr}) % {marker.group_span}) < {marker.group_pre}"
        )
    if marker.threads_in_group > 1:
        return f"(cutlass.Int32(cute.arch.lane_idx()) % {marker.threads_in_group}) == 0"
    return None


def _lane_reduce_axis(marker: _LaneReduceMarker) -> int | None:
    if marker.group_span > 1 and marker.group_lane_expr:
        strides = _linear_thread_axis_strides(marker.group_lane_expr)
        if strides is None:
            return None
        matches = [
            axis for axis, stride in strides.items() if stride == marker.group_pre
        ]
        return matches[0] if len(matches) == 1 else None
    if marker.threads_in_group > 1:
        return 0
    return None


def _linear_thread_axis_strides(expr: str) -> dict[int, int] | None:
    """Recover ``thread_idx`` coefficients from a flattened lane expression."""
    try:
        root = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def merge(
        left: tuple[dict[int, int], int],
        right: tuple[dict[int, int], int],
        sign: int = 1,
    ) -> tuple[dict[int, int], int]:
        coefficients = dict(left[0])
        for axis, coefficient in right[0].items():
            coefficients[axis] = coefficients.get(axis, 0) + sign * coefficient
        return coefficients, left[1] + sign * right[1]

    def scale(
        value: tuple[dict[int, int], int], factor: int
    ) -> tuple[dict[int, int], int]:
        return (
            {axis: coefficient * factor for axis, coefficient in value[0].items()},
            value[1] * factor,
        )

    def visit(node: ast.AST) -> tuple[dict[int, int], int] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {}, node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            return None if value is None else scale(value, -1)
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            # Generated lane expressions wrap both coordinates and constants in
            # the configured index dtype constructor.
            return visit(node.args[0])
        if isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "thread_idx"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
            ):
                return {node.slice.value: 1}, 0
            return None
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return merge(left, right)
            if isinstance(node.op, ast.Sub):
                return merge(left, right, -1)
            if isinstance(node.op, ast.Mult):
                if not left[0]:
                    return scale(right, left[1])
                if not right[0]:
                    return scale(left, right[1])
        return None

    result = visit(root)
    if result is None or result[1] != 0:
        return None
    return result[0]


class _StoreThreadAxes(NamedTuple):
    value: set[int]
    address: set[int]
    predicate: set[int]
    unique_predicate: set[int]
    zero_safe_predicate: set[int]


def _contains_sync_threads(stmt: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "sync_threads"
            or isinstance(node.func, ast.Name)
            and node.func.id == "sync_threads"
        )
        for node in ast.walk(stmt)
    )


def _memory_write_calls(stmt: ast.AST) -> list[ast.Call]:
    helper_stores = {
        "_cute_store_u16_vec",
        "_cute_store_u32_vec",
        "_helion_persistent_branch_vec_store",
    }
    helper_atomics = {
        "_cute_atomic_max_float32",
        "_cute_atomic_min_float32",
        "_cute_red_add_f32_vec",
    }
    result: list[ast.Call] = []
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and (
                node.func.attr in ("store", "__setitem__")
                or node.func.attr.startswith("atomic_")
            )
            or isinstance(node.func, ast.Name)
            and (
                node.func.id in helper_stores
                or node.func.id in helper_atomics
                or node.func.id.startswith("atomic_")
            )
        ):
            result.append(node)
    return result


def _is_persistent_branch_vec_store(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "_helion_persistent_branch_vec_store"
        and len(call.args) == 6
        and not call.keywords
    )


def _is_vector_flush_store(call: ast.Call) -> bool:
    """``_cute_store_u16_vec(ptr, values)`` / ``_cute_store_u32_vec(ptr, values)``.

    The flush of a V-loop's collected per-element values into one vector
    store; its destination is the first argument and its value the second.
    """
    return (
        isinstance(call.func, ast.Name)
        and call.func.id in ("_cute_store_u16_vec", "_cute_store_u32_vec")
        and len(call.args) == 2
        and not call.keywords
    )


def _atomic_pointer_and_value(call: ast.Call) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(call.func, ast.Name) and (
        call.func.id.startswith("atomic_")
        or call.func.id
        in (
            "_cute_atomic_max_float32",
            "_cute_atomic_min_float32",
            "_cute_red_add_f32_vec",
        )
    ):
        return (call.args[0], call.args[1]) if len(call.args) >= 2 else None
    if not (
        isinstance(call.func, ast.Attribute) and call.func.attr.startswith("atomic_")
    ):
        return None
    # ``cute.arch.atomic_add(ptr, value)`` carries its pointer as the first
    # argument; pointer-method atomics carry it in the callee receiver.
    if ast.unparse(call.func.value) == "cute.arch":
        if len(call.args) >= 2:
            return (call.args[0], call.args[1])
        # The emitted form spells the operand by its parameter name,
        # ``cute.arch.atomic_add(ptr, val=value, sem='relaxed')``; a
        # compare-and-swap carries two operands and is not a plain update.
        values = [keyword.value for keyword in call.keywords if keyword.arg != "sem"]
        if len(call.args) == 1 and len(values) == 1:
            return (call.args[0], values[0])
        return None
    return (call.func.value, call.args[0]) if call.args else None


def _is_isolated_store_statement(stmt: ast.AST, store: ast.Call) -> bool:
    """Whether wrapping *stmt* predicates only one otherwise isolated store."""
    if any(isinstance(node, ast.NamedExpr) for node in ast.walk(stmt)):
        return False
    for node in ast.walk(stmt):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, (ast.If, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and node.value is store:
            continue
        return False
    return True


def _direct_thread_coordinate(node: ast.AST) -> int | None:
    """Return an axis for a direct thread coordinate, allowing scalar casts."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if is_builtin_int or is_cutlass_int:
            return _direct_thread_coordinate(node.args[0])
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "thread_idx"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        return None
    return node.slice.value


def _uniquely_predicated_thread_axes(predicate: ast.AST) -> set[int]:
    """Axes for which *predicate* fixes one exact physical thread coordinate."""
    if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
        result: set[int] = set()
        for value in predicate.values:
            result.update(_uniquely_predicated_thread_axes(value))
        return result
    if not (
        isinstance(predicate, ast.Compare)
        and len(predicate.ops) == 1
        and isinstance(predicate.ops[0], ast.Eq)
        and len(predicate.comparators) == 1
    ):
        return set()
    lhs_axis = _direct_thread_coordinate(predicate.left)
    rhs_axis = _direct_thread_coordinate(predicate.comparators[0])
    lhs_literal = _literal_int_expr(predicate.left)
    rhs_literal = _literal_int_expr(predicate.comparators[0])
    if lhs_axis is not None and rhs_literal is not None:
        return {lhs_axis}
    if rhs_axis is not None and lhs_literal is not None:
        return {rhs_axis}
    return set()


def _resolve_scalar_definition(
    node: ast.AST,
    scalar_definitions: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> ast.AST:
    """Resolve generated scalar aliases without expanding arbitrary syntax."""
    if not isinstance(node, ast.Name) or node.id not in scalar_definitions:
        return node
    if seen is None:
        seen = set()
    if node.id in seen:
        return node
    return _resolve_scalar_definition(
        scalar_definitions[node.id],
        scalar_definitions,
        {*seen, node.id},
    )


def _thread_axis_linear_coefficient(
    node: ast.AST,
    axis: int,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    seen: set[str] | None = None,
) -> int | None:
    """Return the integer coefficient of one thread axis, when affine."""
    if axis not in _thread_axes_read_by(node, thread_axis_names):
        return 0
    if isinstance(node, ast.Name):
        if node.id not in scalar_definitions:
            return None
        if seen is None:
            seen = set()
        if node.id in seen:
            return None
        return _thread_axis_linear_coefficient(
            scalar_definitions[node.id],
            axis,
            thread_axis_names,
            scalar_definitions,
            {*seen, node.id},
        )
    direct_axis = _direct_thread_coordinate(node)
    if direct_axis is not None:
        return int(direct_axis == axis)
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if is_builtin_int or is_cutlass_int:
            return _thread_axis_linear_coefficient(
                node.args[0],
                axis,
                thread_axis_names,
                scalar_definitions,
                seen,
            )
    if isinstance(node, ast.UnaryOp):
        coefficient = _thread_axis_linear_coefficient(
            node.operand,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        if coefficient is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return coefficient
        if isinstance(node.op, ast.USub):
            return -coefficient
    if isinstance(node, ast.BinOp):
        left = _thread_axis_linear_coefficient(
            node.left,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        right = _thread_axis_linear_coefficient(
            node.right,
            axis,
            thread_axis_names,
            scalar_definitions,
            seen,
        )
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            left_literal = _literal_int_expr(node.left)
            right_literal = _literal_int_expr(node.right)
            if left_literal is not None:
                return left_literal * right
            if right_literal is not None:
                return right_literal * left
    return None


def _is_generated_block_extent(
    node: ast.AST,
    scalar_definitions: dict[str, ast.AST],
) -> bool:
    """Whether *node* is a generated, statically positive block extent."""
    node = _resolve_scalar_definition(node, scalar_definitions)
    while isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        is_builtin_int = isinstance(node.func, ast.Name) and node.func.id == "int"
        is_cutlass_int = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
        )
        if not (is_builtin_int or is_cutlass_int):
            return False
        node = _resolve_scalar_definition(node.args[0], scalar_definitions)
    return isinstance(node, ast.Name) and node.id.startswith("_BLOCK_SIZE_")


def _predicate_accepts_zero_thread_axis(
    predicate: ast.AST,
    axis: int,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> bool:
    """Prove a generated bound remains true after selecting axis lane zero.

    A ``<``/``<=`` bound whose left side has a positive affine coefficient for
    the nonnegative CUDA thread coordinate can only become easier to satisfy
    when that coordinate is replaced by zero. Other predicate forms fail
    closed rather than risk adding an owner condition that makes them empty.
    """
    if axis not in _thread_axes_read_by(predicate, thread_axis_names):
        return True
    resolved = _resolve_scalar_definition(predicate, scalar_definitions)
    if resolved is not predicate:
        return _predicate_accepts_zero_thread_axis(
            resolved,
            axis,
            thread_axis_names,
            scalar_definitions,
        )
    if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
        return all(
            _predicate_accepts_zero_thread_axis(
                value,
                axis,
                thread_axis_names,
                scalar_definitions,
            )
            for value in predicate.values
        )
    if not (
        isinstance(predicate, ast.Compare)
        and len(predicate.ops) == 1
        and isinstance(predicate.ops[0], (ast.Lt, ast.LtE))
        and len(predicate.comparators) == 1
    ):
        return False
    lhs = _resolve_scalar_definition(predicate.left, scalar_definitions)
    rhs = _resolve_scalar_definition(predicate.comparators[0], scalar_definitions)
    return (
        axis not in _thread_axes_read_by(rhs, thread_axis_names)
        and _is_generated_block_extent(rhs, scalar_definitions)
        and (
            coefficient := _thread_axis_linear_coefficient(
                lhs,
                axis,
                thread_axis_names,
                scalar_definitions,
            )
        )
        is not None
        and coefficient > 0
    )


def _store_enclosing_predicates(stmt: ast.AST, store: ast.Call) -> list[ast.AST]:
    """Control predicates on the unique path from *stmt* to *store*."""

    def find(node: ast.AST, predicates: list[ast.AST]) -> list[ast.AST] | None:
        if node is store:
            return predicates
        if isinstance(node, ast.If):
            for child in node.body:
                found = find(child, [*predicates, node.test])
                if found is not None:
                    return found
            for child in node.orelse:
                # The else path is guarded by the negation. It can still carry
                # axis provenance, but an equality in the positive test does
                # not prove that the else selects a unique lane.
                found = find(
                    child,
                    [*predicates, create(ast.UnaryOp, op=ast.Not(), operand=node.test)],
                )
                if found is not None:
                    return found
            return None
        for child in ast.iter_child_nodes(node):
            found = find(child, predicates)
            if found is not None:
                return found
        return None

    return find(stmt, []) or []


def _store_thread_axes(
    stmt: ast.AST,
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> _StoreThreadAxes | None:
    """Validate one guardable store and return its thread-axis provenance."""
    memory_calls = _memory_write_calls(stmt)
    atomic_parts = (
        _atomic_pointer_and_value(memory_calls[0])
        if len(memory_calls) == 1
        and _is_isolated_store_statement(stmt, memory_calls[0])
        else None
    )
    if atomic_parts is not None:
        if _contains_sync_threads(stmt):
            raise exc.BackendUnsupported(
                "cute", "cannot predicate a statement containing sync_threads"
            )
        store = memory_calls[0]
        pointer, value = atomic_parts
    else:
        store = _validated_owner_store(stmt)
        if store is None:
            return None
        if _is_persistent_branch_vec_store(store):
            pointer = store.args[3]
            value = store.args[4]
        elif _is_vector_flush_store(store):
            pointer = store.args[0]
            value = store.args[1]
        else:
            store_func = cast("ast.Attribute", store.func)
            pointer = store_func.value
            value = store.args[0]
    predicates = _store_enclosing_predicates(stmt, store)
    if _is_persistent_branch_vec_store(store):
        marker_mask = store.args[5]
        if not (isinstance(marker_mask, ast.Constant) and marker_mask.value is None):
            predicates.append(marker_mask)
    value_axes = _thread_axes_read_by(value, thread_axis_names)
    predicate_axes = [
        _thread_axes_read_by(predicate, thread_axis_names) for predicate in predicates
    ]
    all_predicate_axes = set().union(*predicate_axes)
    return _StoreThreadAxes(
        value_axes,
        _thread_axes_read_by(pointer, thread_axis_names),
        all_predicate_axes,
        set().union(
            *(_uniquely_predicated_thread_axes(predicate) for predicate in predicates)
        ),
        {
            axis
            for axis in all_predicate_axes
            if all(
                axis not in axes
                or _predicate_accepts_zero_thread_axis(
                    predicate,
                    axis,
                    thread_axis_names,
                    scalar_definitions,
                )
                for predicate, axes in zip(predicates, predicate_axes, strict=True)
            )
        },
    )


def _validated_owner_store(stmt: ast.AST) -> ast.Call | None:
    """Return the only isolated store, rejecting unsafe ownership rewrites."""
    if _contains_sync_threads(stmt):
        raise exc.BackendUnsupported(
            "cute", "cannot predicate a statement containing sync_threads"
        )
    # A call used as a statement discards its result, so it exists for an
    # effect.  Unknown callees must not pass through as if they were pure: the
    # finalized reduction is broadcast to every reduction thread, and an
    # unguarded custom writer would therefore race.  Recognized memory calls
    # continue through the stricter unique-store validation below.
    memory_calls = _memory_write_calls(stmt)
    memory_call_ids = {id(call) for call in memory_calls}
    if any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and id(node.value) not in memory_call_ids
        for node in ast.walk(stmt)
    ):
        raise exc.BackendUnsupported(
            "cute", "cannot infer whether a standalone call has side effects"
        )
    if not _has_observable_memory_write(stmt):
        return None
    stores = [
        node
        for node in memory_calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "store"
            and len(node.args) == 1
            or _is_persistent_branch_vec_store(node)
            or _is_vector_flush_store(node)
        )
    ]
    if len(memory_calls) != 1 or len(stores) != 1:
        raise exc.BackendUnsupported(
            "cute", "cannot infer unique ownership for this memory write"
        )
    store = stores[0]
    if not _is_isolated_store_statement(stmt, store):
        raise exc.BackendUnsupported(
            "cute", "cannot predicate a compound statement containing a store"
        )
    return store


def _redundant_thread_axis_owner_exprs(
    markers: list[tuple[int, _LaneReduceMarker]],
    store_axes: _StoreThreadAxes | None,
) -> list[str]:
    """Select one writer along launch axes duplicated by a lane reduction.

    A scan feeding a reduction can expose the same logical free coordinate via
    two physical axes: one chosen by the scan/store layout and one introduced
    by the reduction layout.  The grouped reduction correctly preserves every
    sibling coordinate, but a store whose value, address, and enclosing control
    flow use none of those coordinates is otherwise emitted by every redundant
    thread. Gate only an axis proven absent from the complete statement. If the
    stored value varies on an unaddressed axis, reject instead of silently
    choosing one conflicting writer.
    """
    redundant_axes: set[int] = set()
    for _, marker in markers:
        if marker.group_span <= 1 or not marker.group_lane_expr:
            continue
        strides = _linear_thread_axis_strides(marker.group_lane_expr)
        if strides is None:
            continue
        reduce_axis = _lane_reduce_axis(marker)
        if reduce_axis is None:
            continue
        for axis in sorted(strides):
            if axis != reduce_axis:
                redundant_axes.add(axis)
    return _thread_axis_owner_exprs(redundant_axes, store_axes)


def _thread_axis_owner_exprs(
    redundant_axes: set[int],
    store_axes: _StoreThreadAxes | None,
) -> list[str]:
    """Choose one writer along physical axes absent from a logical store."""
    if store_axes is None:
        return []
    predicates: list[str] = []
    for axis in sorted(redundant_axes):
        if axis in store_axes.address or axis in store_axes.unique_predicate:
            continue
        if axis in store_axes.value:
            raise exc.BackendUnsupported(
                "cute",
                "lane reduction store aliases a value-varying thread axis",
            )
        if axis in store_axes.predicate and axis not in store_axes.zero_safe_predicate:
            raise exc.BackendUnsupported(
                "cute", "cannot infer unique ownership from a thread predicate"
            )
        predicates.append(f"cutlass.Int32(cute.arch.thread_idx()[{axis}]) == 0")
    return predicates


def _guard_stmt_with_owner(stmt: ast.AST, predicates: list[str]) -> ast.AST:
    if not predicates:
        return stmt
    store = _validated_owner_store(stmt)
    if store is None:
        raise exc.BackendUnsupported("cute", "owner predicate requires one store")
    predicate = " and ".join(f"({expr})" for expr in dict.fromkeys(predicates))
    if _is_persistent_branch_vec_store(store):
        mask = store.args[5]
        mask_source = (
            None
            if isinstance(mask, ast.Constant) and mask.value is None
            else ast.unparse(mask)
        )
        guarded = _clone_stmt(stmt)
        assert isinstance(guarded, ast.Expr)
        guarded_store = cast("ast.Call", guarded.value)
        guarded_mask = expr_from_string(
            predicate if mask_source is None else f"({mask_source}) and ({predicate})"
        )
        assert isinstance(guarded_mask, ast.expr)
        guarded_store.args[5] = guarded_mask
        return guarded
    return statement_from_string(
        f"if {predicate}:\n"
        + "\n".join(f"    {line}" for line in ast.unparse(stmt).splitlines())
    )


def _identity_cast_operand(node: ast.AST, ctor: str | None) -> ast.AST:
    """``node`` below any casts to the accumulator dtype (``cutlass.Float32(x)``)."""
    while (
        ctor is not None
        and isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) == ctor
    ):
        node = node.args[0]
    return node


def _lane_carry_combine_operands(
    value: ast.AST, reduction_type: str, ctor: str | None
) -> tuple[str, str] | None:
    """The two names ``value`` combines the way the marker's reduction does.

    ``a + b`` for a sum, ``a * b`` for a product, ``cute.math.max(a, b,
    propagate_nan=...)``, ``max(a, b)`` or the ternary ``a if a > b else b``
    for a max (``min`` likewise, and a ternary spelling the other extremum
    is not a carry of this marker), each operand under any cast to the
    accumulator dtype ``ctor``; ``None`` for any other expression.
    """
    value = _identity_cast_operand(value, ctor)

    def names(left: ast.AST, right: ast.AST) -> tuple[str, str] | None:
        left = _identity_cast_operand(left, ctor)
        right = _identity_cast_operand(right, ctor)
        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
            return left.id, right.id
        return None

    if reduction_type in ("sum", "prod"):
        combine = ast.Add if reduction_type == "sum" else ast.Mult
        if isinstance(value, ast.BinOp) and isinstance(value.op, combine):
            return names(value.left, value.right)
        return None
    if reduction_type not in ("max", "min"):
        return None
    if isinstance(value, ast.Call):
        if (
            ast.unparse(value.func) in (reduction_type, f"cute.math.{reduction_type}")
            and len(value.args) == 2
            and all(keyword.arg == "propagate_nan" for keyword in value.keywords)
        ):
            return names(value.args[0], value.args[1])
        return None
    if not (
        isinstance(value, ast.IfExp)
        and isinstance(value.test, ast.Compare)
        and len(value.test.ops) == 1
    ):
        return None
    compared = names(value.test.left, value.test.comparators[0])
    selected = names(value.body, value.orelse)
    if (
        compared is None
        or selected is None
        or len(set(compared)) != 2
        or set(compared) != set(selected)
    ):
        return None
    (op,) = value.test.ops
    if isinstance(op, (ast.Gt, ast.GtE)):
        body_selected_when_larger = True
    elif isinstance(op, (ast.Lt, ast.LtE)):
        body_selected_when_larger = False
    else:
        return None
    # The body is taken when the test holds: the left operand is the larger
    # one under ``>`` and the smaller under ``<``, so the ternary computes
    # the max exactly when the body names the operand the test ranks larger.
    body_is_left = selected[0] == compared[0]
    computes_max = body_is_left == body_selected_when_larger
    return compared if computes_max == (reduction_type == "max") else None


def _lane_carry_update_shape(
    stmt: ast.AST,
) -> tuple[str, list[ast.expr], set[str]] | None:
    """``(target, assigned values, names the guard reads)`` of a carry update.

    A plain assignment, or an if-join assigning one name in both branches
    (``if c: T = C + r`` / ``else: T = C``), whose test's reads are returned
    so the caller can require the join to be the same in every lane.
    """
    from .ast_read_writes import ReadWrites

    target = _plain_assignment_name(stmt)
    if target is not None:
        assert isinstance(stmt, ast.Assign)
        return target, [stmt.value], set()
    if not (isinstance(stmt, ast.If) and len(stmt.body) == 1 and len(stmt.orelse) == 1):
        return None
    branches = [stmt.body[0], stmt.orelse[0]]
    targets = {_plain_assignment_name(branch) for branch in branches}
    if len(targets) != 1 or None in targets:
        return None
    (target,) = targets
    assert target is not None
    values = [cast("ast.Assign", branch).value for branch in branches]
    return target, values, set(ReadWrites.from_ast(stmt.test).reads)


def _restored_marker_consumers_are_lane_carries(
    body: Sequence[ast.AST],
    marker: _LaneReduceMarker,
    lane_var: str,
    rename_groups: Mapping[str, str] | None = None,
) -> bool:
    """Whether every reader of ``marker``'s result folds it into a lane carry.

    A ``strided_restore`` marker finalized per lane yields the lane's share
    of the reduction: its thread group's elements of this lane.  That share
    is a complete lowering only through a carry across the lanes: a reader
    ``T = C + result`` (``*`` for a product, ``cute.math.max`` / ``max`` /
    the ternary for a max or min, the operands under casts to the
    accumulator dtype) whose other operand ``C`` is the phi copy ``C = X``
    of a value live into the body, where ``T`` is ``X`` under the rename
    groups, and no other statement reads or writes the carried value, its
    copy or its loop-output spelling.  The update may sit in an if-join
    whose other branch keeps the carry (``T = C``) and whose condition is
    the same in every lane (it reads no lane-varying name).  Once the loop
    ends the carry holds the shares of every lane, which is the reduction
    over the tile (jagged_layer_norm's ``mean_acc = mean_acc +
    row_sums.sum(dim=1)``; the max over the lanes of the lanes' maxes).
    Any other reader -- a store of the result, a product with the per-lane
    values, a chain into a further reduction -- would observe one lane's
    share.  A result nothing reads is complete.
    """
    from .ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    ctor = _dtype_ctor_from_identity(marker.identity_expr)
    live_in = canonical(
        _ordered_block_reads_writes(cast("list[ast.stmt]", body)).reads_before_write
    )
    varying = _lane_varying_names(
        [stmt for stmt in body if _is_lane_reduce_marker_assign(stmt) is None],
        lane_var,
        renames,
    )
    result = marker.result_var
    for index, stmt in enumerate(body):
        rw = ReadWrites.from_ast(stmt)
        if result not in rw.reads:
            continue
        shape = _lane_carry_update_shape(stmt)
        if shape is None:
            return False
        target, values, guard_reads = shape
        if set(rw.writes) != {target} or guard_reads & ({result, lane_var} | varying):
            return False
        copy_names: set[str] = set()
        for value in values:
            operands = _lane_carry_combine_operands(value, marker.reduction_type, ctor)
            if operands is None:
                # The branch of an if-join that keeps the carry.
                kept = _identity_cast_operand(value, ctor)
                if len(values) == 1 or not isinstance(kept, ast.Name):
                    return False
                copy_names.add(kept.id)
                continue
            if result not in operands or len(set(operands)) != 2:
                return False
            (copy_name,) = set(operands) - {result}
            copy_names.add(copy_name)
        if len(copy_names) != 1:
            return False
        (copy_name,) = copy_names
        # The phi copies of the carried value (``mean_acc_copy = mean_acc``,
        # ``mean_acc_copy_0 = mean_acc_copy``) lead back to a name the body
        # does not define.
        chain: list[ast.AST] = []
        names = {copy_name}
        name = copy_name
        while True:
            definitions = [
                candidate
                for candidate in body[:index]
                if _plain_assignment_name(candidate) == name
            ]
            if not definitions:
                break
            if len(definitions) != 1:
                return False
            (copy,) = definitions
            assert isinstance(copy, ast.Assign)
            if not isinstance(copy.value, ast.Name) or copy.value.id in names:
                return False
            chain.append(copy)
            name = copy.value.id
            names.add(name)
        carried = canonical({name})
        if carried != canonical({target}) or not carried <= live_in:
            return False
        group = carried | names
        if canonical(guard_reads) & group:
            return False
        for other in body:
            if other is stmt or any(other is copy for copy in chain):
                continue
            other_rw = ReadWrites.from_ast(other)
            if (canonical(other_rw.reads) | canonical(other_rw.writes)) & group:
                return False
    return True


def _restore_lane_markers(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[ast.AST]:
    """Lower a lane loop the two-pass split declined, keeping it whole.

    The per-lane restore (:func:`_restore_per_lane_markers`) is complete when
    every strided marker's result feeds a lane carry.  Otherwise the compiler
    supplies the carry itself: each lane's share is totalled across the lanes
    and the lane-invariant tail consuming the results runs once after the
    loop (:func:`_rereduce_restored_lane_markers`).  A body that needs a
    result inside the lanes (a consumer varying with the lane) has no
    whole-loop lowering and is declined.
    """
    body: list[ast.AST] = list(loop.body)
    if all(
        marker.owner_lane is None
        or (
            marker.strided_restore
            and _restored_marker_consumers_are_lane_carries(
                body, marker, lane_var, rename_groups
            )
        )
        for _, marker in markers
    ):
        return [_restore_per_lane_markers(loop, markers, rename_groups)]
    rereduced = _rereduce_restored_lane_markers(
        loop,
        lane_var,
        markers,
        rename_groups,
        thread_axis_names,
        scalar_definitions,
    )
    if rereduced is not None:
        return rereduced
    raise exc.BackendUnsupported(
        "cute", "owned lane reduction has no proved complete per-lane restore"
    )


def _rereduce_restored_lane_markers(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: dict[str, str],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
) -> list[ast.AST] | None:
    """Finalize the strided markers per lane and total the shares over the lanes.

    The loop runs whole, as the restore runs it (its carries, collectives
    and serial loops in place), but each marker's per-lane result is folded
    into a total the loop carries, and the body's tail -- every statement
    after the first marker, which must be lane-invariant once the results
    are final -- runs once after the loop on the totals, as the two-pass
    split runs its invariant tail (stores owner-guarded).  A pure assignment
    between the markers that reads only prefix values (the masked input of a
    second reduction of one accumulator under dynamic shapes, ``_mask_to_2 =
    row_sums if mask else -inf``) joins the prefix, so both reductions read
    inputs produced before the first marker.  ``None`` when the tail is not
    of that shape: a consumer varying with the lane (``out = row_sums *
    total[:, None]`` needs every lane's values after the total, which only
    the two-pass and stash lowerings provide), a carry update (its
    accumulation over the lanes would run once), a marker input produced
    after the first marker by a statement that cannot join the prefix, or a
    tail reading a prefix value no pure lane-invariant assignment re-derives
    after the loop.
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    marker_by_index = dict(markers)
    if any(
        marker.owner_lane is None or not marker.strided_restore
        for marker in marker_by_index.values()
    ):
        return None

    def canonical(names: Iterable[str]) -> set[str]:
        return {rename_groups.get(name, name) for name in names}

    first = min(marker_by_index)
    results = {marker.result_var for marker in marker_by_index.values()}
    hoisted = _prefix_only_statements_between_markers(
        body, marker_by_index, canonical, results
    )
    for index, marker in markers:
        produced_between = {
            name
            for between in range(first, index)
            if between not in marker_by_index and between not in hoisted
            for name in ReadWrites.from_ast(body[between]).writes
        }
        if marker.input_name in results or (
            canonical({marker.input_name}) & canonical(produced_between)
        ):
            return None
    non_markers = [
        stmt for index, stmt in enumerate(body) if index not in marker_by_index
    ]
    varying = _lane_varying_names(non_markers, lane_var, rename_groups)

    def is_varying(stmt: ast.AST) -> bool:
        rw = ReadWrites.from_ast(stmt)
        reads = canonical(rw.reads)
        writes = canonical(rw.writes)
        return lane_var in reads or bool(reads & varying) or bool(writes & varying)

    effects = _ordered_block_reads_writes(cast("list[ast.stmt]", body))
    carried = (
        canonical(effects.reads_before_write) & canonical(effects.may_writes)
    ) - {lane_var}
    tail = [
        body[index]
        for index in range(first, len(body))
        if index not in marker_by_index and index not in hoisted
    ]
    tail_reads: set[str] = set()
    for stmt in tail:
        rw = ReadWrites.from_ast(stmt)
        if is_varying(stmt) or (canonical(rw.reads) | canonical(rw.writes)) & carried:
            return None
        tail_reads |= set(rw.reads)
    prefix = [*body[:first], *(body[index] for index in sorted(hoisted))]
    rederived_indices, _ = _backward_slice(prefix, tail_reads - results, rename_groups)
    rederived = [prefix[index] for index in rederived_indices]
    for stmt in rederived:
        rw = ReadWrites.from_ast(stmt)
        if (
            is_varying(stmt)
            or (canonical(rw.reads) | canonical(rw.writes)) & carried
            or not _is_proven_relocatable_assignment(stmt, allow_load=False)
        ):
            return None

    totals: list[ast.AST] = []
    loop_body: list[ast.AST] = list(prefix)
    finalized: list[ast.AST] = []
    for index in range(first, len(body)):
        marker = marker_by_index.get(index)
        if marker is None:
            continue
        share = f"{marker.result_var}_lane_share"
        total = f"{marker.result_var}_lane_total"
        totals.append(statement_from_string(f"{total} = {marker.identity_expr}"))
        # The per-lane finalize of the restore, assigned to the share instead
        # of the result (the wrap, a cast or reshape, is applied to the total).
        per_lane = dataclasses.replace(
            marker, result_var=share, wrap_template="__HELION_FINALIZED__"
        )
        loop_body.extend(_finalize_lane_reduce_marker(per_lane, marker.input_name))
        ctor = _dtype_ctor_from_identity(marker.identity_expr)
        value = f"{ctor}({share})" if ctor is not None else share
        loop_body.append(
            statement_from_string(
                f"{total} = {_combine_expr(marker.reduction_type, total, value)}"
            )
        )
        finalized.append(
            statement_from_string(
                f"{marker.result_var} = {marker.finalize_expr(total)}"
            )
        )
    result: list[ast.AST] = [*totals, _clone_lane_loop_with_body(loop, loop_body)]
    result.extend(finalized)
    result.extend(_clone_stmt(stmt) for stmt in rederived)
    marker_dependencies = [
        (marker, _forward_live_names(body, {marker.result_var}))
        for _, marker in markers
    ]
    thread_axes_before, scalar_defs_before = _statement_provenance_before(
        body, thread_axis_names, scalar_definitions
    )
    for stmt in tail:
        owner_exprs = _lane_reduction_owner_exprs_for_statement(
            stmt,
            markers,
            marker_dependencies,
            thread_axes_before.get(id(stmt), thread_axis_names),
            scalar_defs_before.get(id(stmt), {}),
        )
        if owner_exprs:
            predicate = " and ".join(f"({owner_expr})" for owner_expr in owner_exprs)
            stmt = _guard_stmt_with_owner(stmt, [predicate])
        result.append(stmt)
    return result


def _prefix_only_statements_between_markers(
    body: Sequence[ast.AST],
    marker_by_index: Mapping[int, _LaneReduceMarker],
    canonical: Callable[[Iterable[str]], set[str]],
    results: set[str],
) -> set[int]:
    """Indices of the statements between the markers that may join the prefix.

    A statement moves before the first marker when it is a pure assignment
    (:func:`_is_proven_relocatable_assignment` without loads) that reads no
    marker result and nothing a statement staying between the markers
    writes, and writes nothing such a statement reads or writes; the
    statements that stay keep their order, so each decision only compares
    against the markers and the stayers before it.
    """
    from .ast_read_writes import ReadWrites

    first, last = min(marker_by_index), max(marker_by_index)
    hoisted: set[int] = set()
    staying_reads: set[str] = set()
    staying_writes: set[str] = canonical(results)
    for index in range(first, last + 1):
        stmt = body[index]
        rw = ReadWrites.from_ast(stmt)
        reads, writes = canonical(rw.reads), canonical(rw.writes)
        if (
            index not in marker_by_index
            and _is_proven_relocatable_assignment(stmt, allow_load=False)
            and not reads & staying_writes
            and not writes & (staying_reads | staying_writes)
        ):
            hoisted.add(index)
            continue
        staying_reads |= reads
        staying_writes |= writes
    return hoisted


def _restore_per_lane_markers(
    loop: ast.For,
    markers: list[tuple[int, _LaneReduceMarker]],
    rename_groups: Mapping[str, str] | None = None,
    *,
    lane_var: str | None = None,
) -> ast.For:
    """Keep the lane loop whole when a two-pass split is unsafe.

    A carry or unduplicatable producer explains why a split is unsafe, not why
    its reduction can be omitted. A legacy marker's input is finalized in
    place. A production marker denotes a full reduction over the owning serial
    lane and physical thread group, which a raw per-lane input cannot complete
    -- except for a ``strided_restore`` marker whose loop body kept the
    per-element strided semantics and whose result feeds lane carries only
    (:func:`_restored_marker_consumers_are_lane_carries`: a sum or product
    update, a running max or min): finalizing this lane's raw input across
    the thread group (the strided form the split was going to replace)
    yields the lane's share, and the carry accumulates the shares of every
    lane into the reduction over the tile.  Any other consumer of a strided
    marker would read one lane's share (a stored total, a product with the
    per-lane values, an atomic add of the total), so such a marker is
    declined here; :func:`_restore_lane_markers` totals the shares across
    the lanes for a lane-invariant tail instead.  The interchange proof removes its
    already-materialized marker consumers separately.

    ``loop`` is the lane loop itself, or (``lane_var`` given) a serial loop
    nested in the lane loop ``lane_var`` whose body holds the markers.
    """
    body = list(loop.body)
    if lane_var is None:
        assert isinstance(loop.target, ast.Name)
        lane_var = loop.target.id
    if any(
        marker.owner_lane is not None
        and (
            not marker.strided_restore
            or not _restored_marker_consumers_are_lane_carries(
                body, marker, lane_var, rename_groups
            )
        )
        for _, marker in markers
    ):
        raise exc.BackendUnsupported(
            "cute", "owned lane reduction has no proved complete per-lane restore"
        )
    for idx, m in sorted(markers, key=operator.itemgetter(0), reverse=True):
        if m.owner_lane is None:
            body[idx] = statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
        else:
            body[idx : idx + 1] = cast(
                "list[ast.stmt]", _finalize_lane_reduce_marker(m, m.input_name)
            )
    loop.body = body
    return loop


_UNDUPLICATABLE_CALLS = (
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared_tree",
    "_cute_grouped_reduce_warp",
    "cute.gemm",
    "warp_reduction",
)


def _contains_unduplicatable_op(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` (or any nested statement) contains a matmul /
    collective whose shared-memory side effects make it unsafe to re-run in a
    second lane pass."""
    src = ast.unparse(stmt)
    return any(call in src for call in _UNDUPLICATABLE_CALLS)


def _lane_split_reorders_aliasing_memory(
    body: list[ast.AST],
    markers: list[tuple[int, _LaneReduceMarker]],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    lane_var: str,
    lane_extent: int | None,
    proven_tensor_stride_values: dict[tuple[str, int], int],
    *,
    additional_inputs: list[tuple[int, str]] | None = None,
    rename_groups: Mapping[str, str] | None = None,
) -> bool:
    """Whether splitting the repeated lane loop can reorder an aliasing write.

    A multi-pass lane split omits stores from its accumulation passes.  Writes
    on either side of a producer load are barriers: a lexically later write
    executes before the next iteration's load.  The narrow exception is a later
    exact load/store pair with the same injective lane mapping.  Distinct
    generated tensor names do not establish disjointness because two user
    arguments may be the same tensor or overlapping views.
    """
    from .ast_read_writes import ReadWrites
    from .cute.fuse_two_pass_loads import _contains_arch_attribute
    from .cute.fuse_two_pass_loads import _is_store_call
    from .cute.fuse_two_pass_loads import _store_tensor_roots
    from .cute.fuse_two_pass_loads import _tensor_arg_roots
    from .cute.persistent_branch_vec import _definition_snapshots
    from .cute.persistent_branch_vec import _generated_access_pointer
    from .cute.persistent_branch_vec import _lane_accesses_are_iteration_independent
    from .cute.persistent_branch_vec import _memory_load_calls

    tensor_names = {
        node.value.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }

    def roots_may_alias(
        left: frozenset[str] | None,
        right: frozenset[str] | None,
    ) -> bool:
        if left is None or right is None:
            return True
        return any(
            left_name == right_name
            or frozenset((left_name, right_name)) not in proven_disjoint_tensor_pairs
            for left_name in left
            for right_name in right
        )

    writes: list[tuple[int, ast.Call, frozenset[str] | None]] = []
    for write_index, stmt in enumerate(body):
        for call in (
            node
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call) and _is_store_call(node)
        ):
            writes.append((write_index, call, _store_tensor_roots(call, tensor_names)))

    address_snapshots: dict[frozenset[str], list[dict[str, ast.expr]] | None] = {}
    definitely_written: set[str] = set()
    live_in: set[str] = set()
    may_writes: set[str] = set()
    for statement in body:
        effects = ReadWrites.from_ast(statement)
        live_in.update(set(effects.reads) - definitely_written - {lane_var})
        may_writes.update(effects.writes)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            definitely_written.update(effects.writes)
    loop_carried_names = live_in & may_writes

    inputs = [(index, marker.input_name) for index, marker in markers]
    inputs.extend(additional_inputs or ())
    for input_index, input_name in inputs:
        stage_indices, _ = _backward_slice(
            body[:input_index], {input_name}, rename_groups
        )
        for load_index in stage_indices:
            for load_call in _memory_load_calls(body[load_index]):
                load_marker = (
                    isinstance(load_call.func, ast.Name)
                    and load_call.func.id == "_helion_persistent_branch_vec_load"
                    and len(load_call.args) == 6
                )
                if load_marker:
                    pointer = load_call.args[4]
                else:
                    load_access = _generated_access_pointer(load_call)
                    func = load_call.func
                    if load_access is not None:
                        pointer = load_access[0]
                    elif isinstance(func, ast.Attribute) and not (
                        load_call.args and _contains_arch_attribute(func.value)
                    ):
                        pointer = func.value
                    else:
                        # A byte-packed word load or an unknown helper: the
                        # element count is not visible, so no reordering.
                        return True
                load_roots = _tensor_arg_roots(pointer, tensor_names)
                for write_index, store_call, write_roots in writes:
                    if not roots_may_alias(write_roots, load_roots):
                        continue
                    if _is_persistent_branch_vec_store(store_call):
                        store_pointer: ast.AST | None = store_call.args[3]
                    else:
                        store_access = _generated_access_pointer(store_call)
                        store_pointer = (
                            None if store_access is None else store_access[0]
                        )
                    if write_index <= load_index or store_pointer is None:
                        return True
                    # Freeze only the two addresses and their complete
                    # dependency closure. Large value arithmetic cannot
                    # invalidate an otherwise compact address proof.
                    address_names = frozenset(
                        ReadWrites.from_ast(pointer).reads
                        | ReadWrites.from_ast(store_pointer).reads
                    )
                    if address_names not in address_snapshots:
                        address_snapshots[address_names] = _definition_snapshots(
                            cast("list[ast.stmt]", body),
                            required_names=set(address_names),
                        )
                    definition_snapshots = address_snapshots[address_names]
                    if definition_snapshots is None:
                        return True
                    unstable_address_names = set(loop_carried_names)
                    for crossed_statement in body[load_index + 1 : write_index + 1]:
                        unstable_address_names.update(
                            ReadWrites.from_ast(crossed_statement).writes
                        )
                    # A write before this producer load is an intra-iteration
                    # dependence and cannot move. A later exact store may cross
                    # the loop backedge only when both accesses have the same
                    # injective lane mapping.
                    if _lane_accesses_are_iteration_independent(
                        load_call,
                        store_call,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        load_definitions=definition_snapshots[load_index],
                        store_definitions=definition_snapshots[write_index],
                        proven_tensor_stride_values=proven_tensor_stride_values,
                        loop_carried_names=unstable_address_names,
                    ):
                        continue
                    return True
    return False


def _has_side_effect(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` produces an observable side effect (a store,
    an in-place / atomic write, or any non-plain-assignment statement such as
    an ``if mask: tensor[...].store(...)``)."""
    from .ast_read_writes import ReadWrites

    if isinstance(stmt, ast.Assign):
        return bool(ReadWrites.from_ast(stmt).inplace_writes)
    # Conservatively treat structured / expression statements as
    # side-effecting (store calls live inside ``if`` blocks / bare exprs).
    return True


def _has_observable_memory_write(stmt: ast.AST) -> bool:
    """True for generated tensor stores/atomics, including guarded stores."""
    from .ast_read_writes import ReadWrites

    if ReadWrites.from_ast(stmt).inplace_writes:
        return True
    return bool(_memory_write_calls(stmt))


def _live_phase2_indices(body: list[ast.AST]) -> set[int]:
    """Indices of statements in ``body`` that contribute to a side effect
    (directly, or by feeding a later side-effecting statement)."""
    from .ast_read_writes import ReadWrites

    needed_names: set[str] = set()
    keep: set[int] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        if _has_side_effect(stmt) or (writes & needed_names):
            keep.add(idx)
            needed_names |= set(rw.reads)
    return keep


def _lane_loop_extent(loop: ast.For) -> int:
    call = loop.iter
    assert isinstance(call, ast.Call)
    assert len(call.args) == 1
    return int(ast.literal_eval(call.args[0]))


def _lane_varying_names(
    body: list[ast.AST],
    lane_var: str,
    rename_groups: Mapping[str, str] | None = None,
) -> set[str]:
    """Names whose values (transitively) depend on the lane var within ``body``.

    Names are compared through ``rename_groups`` and returned under their
    canonical spelling: a carried value a lane-varying statement rewrites
    under its loop-output name (``v_8`` for ``acc``) is lane-varying under
    every spelling, so its initializer ``acc = 0`` is one of its lane-varying
    writes and runs in every pass of a lane split.
    """
    from .ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    varying = {lane_var}
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if canonical(rw.reads) & varying:
                written = canonical(rw.writes) - varying
                if written:
                    varying |= written
                    changed = True
    varying.discard(lane_var)
    return varying


def _is_serial_for(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` is an ordinary serial ``for`` loop (a device
    serial loop), NOT a per-thread lane loop.

    A ``cutlass.range_constexpr`` unroll (the constexpr vector lane of a
    ``cute_vector_widths`` partition) is a lane dimension as well, never a
    serial device loop, so markers nested in it are left to the lane split.
    """
    return (
        isinstance(stmt, ast.For)
        and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and not _is_constexpr_lane_iter(stmt)
    )


def _clone_stmt(stmt: ast.AST) -> ast.AST:
    """Return an independent copy of ``stmt`` via unparse + reparse.

    ``interchange_lane_outside_serial_reductions`` emits two loop nests that
    both re-run the shared (side-effect-free) producers. Splicing the same node
    objects into two places in the tree breaks AST walking, so each reused
    statement is rebuilt from its source text into a fresh ExtendedAST node.
    """
    return statement_from_string(ast.unparse(stmt))


def _clone_expr(node: ast.AST) -> ast.AST:
    return expr_from_string(ast.unparse(node))


def _forward_live_names(body: list[ast.AST], roots: set[str]) -> set[str]:
    """Names produced by the forward slice that (transitively) consumes any
    name in ``roots`` within ``body``."""
    from .ast_read_writes import ReadWrites

    tainted = set(roots)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & tainted:
                for w in rw.writes:
                    if w not in tainted:
                        tainted.add(w)
                        changed = True
    return tainted


def interchange_lane_outside_serial_reductions(
    body: list[ast.AST],
    *,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    protected_names: set[str] | None = None,
) -> list[ast.AST]:
    """Interchange a ``for LANE: ... for MB: ...`` nest whose inner serial loop
    contains ``_helion_lane_reduce`` markers.

    layer_norm_bwd / rms_norm_bwd compute, inside a serial ``mb`` loop, BOTH a
    per-feature accumulator that must keep the lane loop OUTSIDE the ``mb`` loop
    (``grad_w_acc += ...``) AND a feature reduction whose result is broadcast
    back into a per-row store (``grad_x``), which needs every lane summed *per*
    ``mb`` iteration (lane INSIDE ``mb``). A single lane loop cannot satisfy
    both nestings, so emit two specialized loop nests:

    * Nest B (grad_w): the original ``for LANE: ... for MB: ...`` loop. Remove
      reduction-consuming stores and their dead producers when aliasing and
      exact overwrite coverage are proven; otherwise retain the partial stores.
      A reduction-consuming atomic is always removed (Nest A applies the full
      reduction once; the lanes' partials must not be added on top), and the
      nest is declined when it cannot be; Nest B itself goes when that leaves
      it empty.
    * Nest A (grad_x): a ``for MB: ... for LANE: ...`` loop carrying only the
      lane reduction and its broadcast consumer. Its inner lane loop still holds
      the markers so the subsequent ``split_lane_loop_reductions`` pass produces
      the per-``mb`` accumulate -> warp-combine -> consume structure.

    Returns ``body`` unchanged when no such pattern is present.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(
            _interchange_stmt(
                stmt, proven_disjoint_tensor_pairs or set(), protected_names or set()
            )
        )
    return new_body


def _interchange_stmt(
    stmt: ast.AST,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
) -> list[ast.AST]:
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(
                stmt,
                field,
                interchange_lane_outside_serial_reductions(
                    old,
                    proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
                    protected_names=protected_names,
                ),
            )
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _interchange_one_lane_loop(
        stmt, lane_var, proven_disjoint_tensor_pairs, protected_names
    )


def _interchange_one_lane_loop(
    loop: ast.For,
    lane_var: str,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    # Find the single inner serial ``for`` loop that carries lane-reduce
    # markers of THIS lane (or legacy unowned markers).  A vector tile's
    # constexpr V-loop under a register-tile owner also holds markers, but
    # they belong to the enclosing reduction lane and are lowered there
    # (``cute/register_tile_reductions.py``).
    mb_index: int | None = None
    for idx, stmt in enumerate(body):
        if _is_serial_for(stmt) and any(
            (marker := _is_lane_reduce_marker_assign(s)) is not None
            and marker.owner_lane in (None, lane_var)
            for s in cast("ast.For", stmt).body
        ):
            if mb_index is not None:
                # More than one candidate serial loop: not the simple pattern.
                return [loop]
            mb_index = idx
    if mb_index is None:
        return [loop]
    mb_loop = cast("ast.For", body[mb_index])
    lane_prefix = body[:mb_index]
    lane_suffix = body[mb_index + 1 :]
    mb_body: list[ast.AST] = list(mb_loop.body)

    markers = [
        (i, m)
        for i, s in enumerate(mb_body)
        if (m := _is_lane_reduce_marker_assign(s)) is not None
    ]
    if not markers:
        return [loop]
    marker_indices = {i for i, _ in markers}
    marker_results = {m.result_var for _, m in markers}

    # The mb body's only side effect that consumes a marker result is the
    # broadcast-reduction store (e.g. ``grad_x[mb] = ...``). Everything else in
    # the mb body / suffix (the per-feature accumulators and their stores) is
    # independent of the reduction and is handled correctly by Nest B alone.
    grad_x_seed = {m.input_name for _, m in markers} | marker_results
    grad_x_live = _forward_live_names(mb_body, grad_x_seed)

    def is_reduction_store(stmt: ast.AST) -> bool:
        return _has_side_effect(stmt) and bool(
            set(ReadWrites.from_ast(stmt).reads) & grad_x_live
        )

    if not any(is_reduction_store(s) for s in mb_body):
        # The lane reduction inside the serial loop is not consumed by a
        # broadcast store, so the interchange does not apply. Markers nested in
        # a serial loop are not reachable by ``split_lane_loop_reductions`` (it
        # only rewrites top-level lane loops), so finalize them in place like
        # an unsplittable lane loop would: a legacy unowned marker restores
        # its raw per-lane input and a ``strided_restore`` marker whose
        # consumers are lane carries folds it across its thread group, while
        # any other owned marker denotes a full reduction over the lane that
        # a raw per-lane input cannot complete and declines loudly.
        _restore_per_lane_markers(mb_loop, markers, lane_var=lane_var)
        return [loop]

    # A name is lane-varying if it (transitively) depends on the lane var across
    # the whole lane-loop body; lane-invariant names (mb bounds, masks, ...) are
    # recomputed once rather than per lane.
    lane_varying = _lane_varying_names([*lane_prefix, *mb_body, *lane_suffix], lane_var)

    def reads_lane(stmt: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(stmt).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # --- Nest A (grad_x): for MB: for LANE: <reduction + broadcast store> ------
    # Backward slice from the reduction-broadcast stores and the marker inputs,
    # across both the mb body and the lane prefix. The per-feature accumulators
    # feed only the suffix stores (never the reduction store), so they are
    # naturally excluded — no carry analysis is required.
    mb_bound_names = set(ReadWrites.from_ast(mb_loop.iter).reads)
    needed = {m.input_name for _, m in markers} | mb_bound_names
    keep_mb_a: list[ast.AST] = []
    for idx in range(len(mb_body) - 1, -1, -1):
        stmt = mb_body[idx]
        rw = ReadWrites.from_ast(stmt)
        if idx in marker_indices:
            keep_mb_a.append(stmt)
            needed |= set(rw.reads) - marker_results
            continue
        if is_reduction_store(stmt) or (set(rw.writes) & needed):
            keep_mb_a.append(stmt)
            needed |= set(rw.reads)
    keep_mb_a.reverse()
    keep_prefix_a: list[ast.AST] = []
    for stmt in reversed(lane_prefix):
        rw = ReadWrites.from_ast(stmt)
        if set(rw.writes) & needed:
            keep_prefix_a.append(stmt)
            needed |= set(rw.reads)
    keep_prefix_a.reverse()

    # Partition kept statements into lane-invariant (run once per mb iteration)
    # vs lane-varying (recomputed per lane inside the inner lane loop).
    prefix_invariant_a = [_clone_stmt(s) for s in keep_prefix_a if not reads_lane(s)]
    prefix_varying_a = [_clone_stmt(s) for s in keep_prefix_a if reads_lane(s)]
    mb_head_a = [_clone_stmt(s) for s in keep_mb_a if not reads_lane(s)]
    mb_varying_a = [_clone_stmt(s) for s in keep_mb_a if reads_lane(s)]

    inner_lane_loop_a = _clone_lane_loop_with_body(
        loop, [*prefix_varying_a, *mb_varying_a]
    )
    mb_loop_a = create(
        ast.For,
        target=_clone_expr(mb_loop.target),
        iter=_clone_expr(mb_loop.iter),
        body=[*mb_head_a, inner_lane_loop_a],
        orelse=[],
        type_comment=None,
    )
    nest_a: list[ast.AST] = [*prefix_invariant_a, mb_loop_a]

    # --- Nest B (grad_w): the original lane loop with markers reverted to their
    # raw per-lane inputs. Its per-feature accumulators (lane-outside-mb) are
    # already correct; its reduction-broadcast store writes a partial (per-lane)
    # value that Nest A re-stores with the full reduction afterwards. Prune the
    # first store only after proving it is unobservable and fully overwritten.
    # An atomic consumer has no such overwrite: Nest A applies the full
    # reduction once, so Nest B must not apply the lanes' partials at all.
    restored_mb_body = list(mb_loop.body)
    for idx, m in markers:
        restored_mb_body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
        )
    mb_loop.body = restored_mb_body
    nest_b = loop

    from .cute.interchanged_store_dce import atomic_write_calls
    from .cute.interchanged_store_dce import eliminate_interchanged_atomics
    from .cute.interchanged_store_dce import eliminate_interchanged_stores

    reduction_consumers = _forward_live_names(mb_body, marker_results)
    consumer_indices = {
        index
        for index, statement in enumerate(mb_body)
        if _has_side_effect(statement)
        and set(ReadWrites.from_ast(statement).reads) & reduction_consumers
    }
    atomic_consumers = [
        mb_body[index]
        for index in sorted(consumer_indices)
        if atomic_write_calls(mb_body[index])
    ]
    eliminate_interchanged_stores(
        nest_b,
        mb_loop,
        nest_a,
        consumer_indices,
        proven_disjoint_tensor_pairs,
        protected_names,
    )
    if atomic_consumers:
        dropped = eliminate_interchanged_atomics(
            nest_b, mb_loop, nest_a, atomic_consumers, protected_names
        )
        if dropped != len(atomic_consumers):
            raise exc.BackendUnsupported(
                "cute",
                "interchanged lane reduction has an atomic consumer the first "
                "pass cannot drop",
            )
        if not nest_b.body:
            return nest_a

    return [nest_b, *nest_a]


# ---------------------------------------------------------------------------
# Chunked-recurrence (GDN) lane-invariant accumulator hoist.
#
# A chunked recurrence such as gdn_fwd_h carries an accumulator ``b_h`` across a
# SERIAL chunk loop and, inside each chunk, contracts a matmul over the
# within-chunk position ``c`` (lowered as an inner lane loop).  The scalar
# matmul fallback emits a running sum because the per-chunk rescale is
# lane-invariant.  Depending on how the source spelled the update, its relevant
# dataflow inside the lane loop is either direct or separated by temporaries:
#
#       base = <lane-invariant rescale of b_h>       # e.g. b_h * decay
#       dot_acc = dot_acc + <product(c)>             # accumulate over c
#       reduced = <cross-thread reduction>(dot_acc)
#       b_h = base + reduced                          # WRONG: per-lane reassign
#
# with ``dot_acc = <identity>`` reset OUTSIDE the chunk loop.  Reassigning ``b_h``
# every lane iteration corrupts the recurrence (and any lane-invariant op that
# must read the chunk-ENTRY ``b_h``, e.g. the store of ``b_h`` and the matmul
# operand ``c_h = b_h``).  This pass restructures the nest to apply the rescale
# and the final add once per chunk:
#
#   for chunk:
#       dot_acc = <identity>                  # reset per chunk
#       <lane-invariant chunk-entry stores using b_h>
#       base = <rescale of frozen b_h>
#       for lane:
#           <producers; dot_acc = dot_acc + product(c)>
#       reduced = <cross-thread reduction>(dot_acc)
#       b_h = base + reduced                  # once per chunk
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ChunkRecurrence:
    """Dataflow description of one lane-folded, chunk-carried matmul."""

    dot_acc_var: str
    lane_idx: int
    lane_loop: ast.For
    lane_var: str
    sum_idx: int
    base_indices: tuple[int, ...]
    finalize_indices: tuple[int, ...]
    final_idx: int
    state_name: str


def _plain_assignment_name(stmt: ast.AST) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


_PURE_RELOCATABLE_NAMES = frozenset(
    {
        "abs",
        "bool",
        "float",
        "int",
        "len",
        "max",
        "min",
    }
)
_PURE_OPERATOR_CALLS = frozenset(
    {
        "add",
        "and_",
        "eq",
        "floordiv",
        "ge",
        "getitem",
        "gt",
        "invert",
        "le",
        "lshift",
        "lt",
        "mod",
        "mul",
        "ne",
        "neg",
        "not_",
        "or_",
        "pos",
        "pow",
        "rshift",
        "sub",
        "truediv",
        "xor",
    }
)
_CHUNK_REDUCTION_HELPERS = frozenset(
    {
        "_cute_grouped_reduce_shared_tree",
        "_cute_grouped_reduce_shared_two_stage",
        "_cute_grouped_reduce_warp",
    }
)


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


def _is_proven_relocatable_call(
    call: ast.Call,
    *,
    allow_load: bool,
    allow_reduction: bool = False,
) -> bool:
    """Whether evaluating ``call`` is known not to have observable effects."""
    if isinstance(call.func, ast.Attribute):
        if call.func.attr == "load":
            return allow_load
        if call.func.attr == "bitcast":
            return True
    name = _qualified_name(call.func)
    if name is None:
        return False
    if name in _PURE_RELOCATABLE_NAMES or name in PURE_DECODE_HELPERS:
        return True
    if name.startswith("cutlass."):
        member = name.rsplit(".", 1)[-1]
        return bool(member) and member[0].isupper()
    if name.startswith(("math.", "cute.math.")):
        return True
    if name == "ir.VectorType.get":
        # The MLIR vector type of a hoisted packet load: a pure type value.
        return True
    if name.startswith("operator."):
        return name.rsplit(".", 1)[-1] in _PURE_OPERATOR_CALLS
    if name in {
        "cute.arch.block_dim",
        "cute.arch.block_idx",
        "cute.arch.grid_dim",
        "cute.arch.lane_idx",
        "cute.arch.thread_idx",
        "cute.arch.warp_idx",
    }:
        return True
    if name == "cute.arch.load" or name in _CUTE_CACHE_LOAD_HELPER_NAMES:
        # The packet load and its L2-policy helper forms, the load-side twin
        # of the ``_cute_store_*`` convention ``_is_store_call`` relies on.
        # Only the listed helpers: the memory passes collect and measure
        # exactly these, so any other ``_cute_load_*`` stays unproven.
        return allow_load
    return allow_reduction and (
        name in _CHUNK_REDUCTION_HELPERS or name.startswith("cute.arch.warp_reduction")
    )


def _is_proven_relocatable_assignment(
    stmt: ast.AST,
    *,
    allow_load: bool,
    allow_reduction: bool = False,
) -> bool:
    """Whether a simple assignment may safely change lane-loop scope."""
    if _plain_assignment_name(stmt) is None:
        return False
    return all(
        _is_proven_relocatable_call(
            node,
            allow_load=allow_load,
            allow_reduction=allow_reduction,
        )
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
    )


def _validated_idempotent_store(stmt: ast.AST) -> ast.Call | None:
    """Return a single ordinary store whose surrounding expressions are pure.

    Atomics, helper writes, compound effects, synchronization, and calls with
    unknown purity are deliberately rejected: changing any of those from once
    per synthetic lane to once per chunk is not semantics-preserving.
    """
    store = _validated_owner_store(stmt)
    if store is None:
        return None
    if any(
        node is not store
        and isinstance(node, ast.Call)
        and not _is_proven_relocatable_call(node, allow_load=False)
        for node in ast.walk(stmt)
    ):
        raise exc.BackendUnsupported(
            "cute", "chunk recurrence store contains a call with unknown purity"
        )
    return store


def _store_iterator_roots(store: ast.Call) -> set[str]:
    """Generated tensor iterator names that determine a store's allocation."""
    if not isinstance(store.func, ast.Attribute):
        return set()
    return {
        node.value.id
        for node in ast.walk(store.func.value)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }


def _self_addend(stmt: ast.AST, target: str) -> ast.AST | None:
    """Return the non-self operand of ``target = target + value``."""
    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.BinOp):
        return None
    if not isinstance(stmt.value.op, ast.Add):
        return None
    if isinstance(stmt.value.left, ast.Name) and stmt.value.left.id == target:
        return stmt.value.right
    if isinstance(stmt.value.right, ast.Name) and stmt.value.right.id == target:
        return stmt.value.left
    return None


def _additive_operands(expr: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Find the add below any generated one-argument dtype conversions."""
    while isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords:
        expr = expr.args[0]
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return expr.left, expr.right
    return None


def _canonical_name(name: str, rename_groups: dict[str, str]) -> str:
    return rename_groups.get(name, name)


def _dependency_names(body: list[ast.AST], roots: set[str]) -> set[str]:
    """Return every name in the backward slice producing ``roots``."""
    from .ast_read_writes import ReadWrites

    indices, _ = _backward_slice(body, roots)
    names = set(roots)
    for idx in indices:
        rw = ReadWrites.from_ast(body[idx])
        names.update(rw.reads)
        names.update(rw.writes)
    return names


def _find_dot_acc_recurrence(
    lane_loop: ast.For,
    lane_var: str,
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _ChunkRecurrence | None:
    """Find a lane-folded matmul that updates a chunk-carried accumulator.

    Matmul and DCE temporary names are deliberately not part of the contract.
    The recognizable structure is a scalar self-add over lane-varying input,
    followed by an additive combine of its finalized value with a
    lane-invariant value derived from the chunk-entry state.  The combine's
    result must alias that state according to the compiler's SSA rename groups.

    The dot-derived side may contain arbitrary plain-assignment setup and a
    cross-thread reduction.  This covers both ``dot(..., acc=state)`` and the
    equivalent ``state = state * decay; state = state + dot(...)`` spelling.
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(lane_loop.body)
    matches: list[_ChunkRecurrence] = []
    for sum_idx, sum_stmt in enumerate(body):
        dot_acc_var = _plain_assignment_name(sum_stmt)
        if dot_acc_var is None:
            continue
        if running_sums is not None and dot_acc_var not in running_sums:
            continue
        product = _self_addend(sum_stmt, dot_acc_var)
        if product is None:
            continue

        # The update must actually fold values across this synthetic lane.
        prefix_varying = _lane_varying_names(body[: sum_idx + 1], lane_var)
        product_reads = set(ReadWrites.from_ast(product).reads)
        if lane_var not in product_reads and not (product_reads & prefix_varying):
            continue

        dot_derived = {dot_acc_var}
        for final_idx in range(sum_idx + 1, len(body)):
            candidate = body[final_idx]
            candidate_name = _plain_assignment_name(candidate)
            if candidate_name is not None and isinstance(candidate, ast.Assign):
                operands = _additive_operands(candidate.value)
                if operands is not None:
                    left_reads = set(ReadWrites.from_ast(operands[0]).reads)
                    right_reads = set(ReadWrites.from_ast(operands[1]).reads)
                    left_is_dot = bool(left_reads & dot_derived)
                    right_is_dot = bool(right_reads & dot_derived)
                    if left_is_dot != right_is_dot:
                        dot_roots = left_reads if left_is_dot else right_reads
                        base_roots = right_reads if left_is_dot else left_reads
                        base_indices, _ = _backward_slice(
                            body[:sum_idx], set(base_roots)
                        )
                        base_dependencies = _dependency_names(
                            body[:sum_idx], set(base_roots)
                        )
                        base_external_varying = base_dependencies & (
                            {lane_var} | prefix_varying
                        )
                        state_name = _canonical_name(candidate_name, rename_groups)
                        carries_state = any(
                            _canonical_name(name, rename_groups) == state_name
                            for name in base_dependencies
                        )
                        base_is_invariant = all(
                            lane_var not in set(ReadWrites.from_ast(body[idx]).reads)
                            and not (
                                set(ReadWrites.from_ast(body[idx]).reads)
                                & prefix_varying
                            )
                            for idx in base_indices
                        )
                        base_is_pure = all(
                            _is_proven_relocatable_assignment(
                                body[idx], allow_load=True
                            )
                            for idx in base_indices
                        )

                        finalizer_body = body[sum_idx + 1 : final_idx]
                        finalize_indices_rel, _ = _backward_slice(
                            finalizer_body, set(dot_roots)
                        )
                        finalize_indices = tuple(
                            sum_idx + 1 + idx for idx in finalize_indices_rel
                        )
                        finalize_dependencies = _dependency_names(
                            finalizer_body, set(dot_roots)
                        )
                        finalize_external_varying = (
                            finalize_dependencies - {dot_acc_var}
                        ) & ({lane_var} | prefix_varying)
                        finalizer_is_pure = all(
                            _is_proven_relocatable_assignment(
                                body[idx],
                                allow_load=False,
                                allow_reduction=True,
                            )
                            for idx in finalize_indices
                        )
                        reaches_running_sum = dot_acc_var in finalize_dependencies

                        recognized_recurrence = bool(
                            base_roots and carries_state and reaches_running_sum
                        )
                        if recognized_recurrence and (
                            base_external_varying
                            or not base_is_invariant
                            or finalize_external_varying
                        ):
                            raise exc.BackendUnsupported(
                                "cute",
                                "chunk recurrence is not lane-invariant",
                            )
                        if recognized_recurrence and (
                            not base_is_pure
                            or not finalizer_is_pure
                            or not _is_proven_relocatable_assignment(
                                candidate, allow_load=False
                            )
                        ):
                            raise exc.BackendUnsupported(
                                "cute",
                                "chunk recurrence contains a call with unknown purity",
                            )
                        if recognized_recurrence:
                            matches.append(
                                _ChunkRecurrence(
                                    dot_acc_var=dot_acc_var,
                                    lane_idx=-1,
                                    lane_loop=lane_loop,
                                    lane_var=lane_var,
                                    sum_idx=sum_idx,
                                    base_indices=tuple(base_indices),
                                    finalize_indices=finalize_indices,
                                    final_idx=final_idx,
                                    state_name=state_name,
                                )
                            )

            rw = ReadWrites.from_ast(candidate)
            if set(rw.reads) & dot_derived:
                dot_derived.update(rw.writes)

    # Multiple candidates would require coordinated resets and post-lane
    # finalizers.  Fail closed instead of guessing which matmul owns the carry.
    if len(matches) > 1:
        raise exc.BackendUnsupported(
            "cute", "chunk recurrence has multiple candidate state updates"
        )
    if not matches:
        return None
    return matches[0]


def _single_lane_loop_in_body(
    body: list[ast.AST],
) -> tuple[int, ast.For, str] | None:
    """If ``body`` contains exactly one direct lane loop, return its index, the
    loop node, and its lane var; else ``None``."""
    found: tuple[int, ast.For, str] | None = None
    for idx, stmt in enumerate(body):
        lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
        if (
            lane_var is not None
            and isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == lane_var
        ):
            if found is not None:
                return None
            found = (idx, stmt, lane_var)
    return found


def _find_reset_assign(body: list[ast.AST], var: str) -> int | None:
    """Index of the last ``var = <expr>`` plain assignment in ``body`` (the
    ``dot_acc`` reset emitted before the chunk loop), or ``None``."""
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == var
        ):
            return idx
    return None


def _collect_statement_provenance(
    body: list[ast.AST],
) -> tuple[
    dict[int, dict[str, frozenset[int]]],
    dict[int, dict[str, ast.AST]],
]:
    """Capture thread-axis and scalar-definition facts before every statement."""
    thread_axes_before: dict[int, dict[str, frozenset[int]]] = {}
    scalar_defs_before: dict[int, dict[str, ast.AST]] = {}

    def visit_block(
        statements: list[ast.AST],
        inherited_axes: dict[str, frozenset[int]],
        inherited_defs: dict[str, ast.AST],
    ) -> None:
        local_axes = dict(inherited_axes)
        local_defs = dict(inherited_defs)
        for statement in statements:
            thread_axes_before[id(statement)] = dict(local_axes)
            scalar_defs_before[id(statement)] = dict(local_defs)
            for field in ("body", "orelse", "finalbody"):
                child = getattr(statement, field, None)
                if isinstance(child, list) and all(
                    isinstance(child_stmt, ast.stmt) for child_stmt in child
                ):
                    visit_block(child, local_axes, local_defs)
            _update_thread_axis_names(statement, local_axes)
            _update_scalar_definitions(statement, local_defs)

    visit_block(body, {}, {})
    return thread_axes_before, scalar_defs_before


def _call_keyword_int(call: ast.Call, name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _literal_int_expr(keyword.value)
    return None


def _chunk_recurrence_reduction_axes(
    info: _ChunkRecurrence,
    scalar_defs_before: dict[int, dict[str, ast.AST]],
) -> set[int]:
    """Physical thread axes folded by the moved cross-thread finalizer."""
    axes: set[int] = set()
    for idx in info.finalize_indices:
        statement = info.lane_loop.body[idx]
        definitions = scalar_defs_before.get(id(statement), {})
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            name = _qualified_name(call.func)
            if name in _CHUNK_REDUCTION_HELPERS:
                if len(call.args) < 4:
                    raise exc.BackendUnsupported(
                        "cute", "cannot infer chunk recurrence reduction ownership"
                    )
                pre = _call_keyword_int(call, "pre")
                lane_expr = _resolve_scalar_definition(call.args[3], definitions)
                strides = _linear_thread_axis_strides(ast.unparse(lane_expr))
                matches = (
                    []
                    if pre is None or strides is None
                    else [axis for axis, stride in strides.items() if stride == pre]
                )
                if len(matches) != 1:
                    raise exc.BackendUnsupported(
                        "cute", "cannot infer chunk recurrence reduction ownership"
                    )
                axes.add(matches[0])
            elif name is not None and name.startswith("cute.arch.warp_reduction"):
                axes.add(0)
    return axes


def hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
    *,
    rename_groups: dict[str, str] | None = None,
    running_sums: set[str] | None = None,
) -> list[ast.AST]:
    """Restructure ``for chunk: for lane: <dot_acc recurrence>`` nests so the
    lane-invariant rescale, chunk-entry stores, and final accumulator combine
    run once per chunk (see the module comment above).

    Tightly gated: only fires on a serial chunk loop whose single inner lane
    loop has the verified dataflow described by :func:`_find_dot_acc_recurrence`
    and whose running-sum reset sits in the same statement list before it.
    """
    if running_sums is not None and not running_sums:
        return body

    thread_axes_before, scalar_defs_before = _collect_statement_provenance(body)
    return _hoist_lane_invariant_chunk_recurrence(
        body,
        rename_groups or {},
        running_sums,
        thread_axes_before,
        scalar_defs_before,
    )


def _hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
    aliases: dict[str, str],
    running_sums: set[str] | None,
    thread_axes_before: dict[int, dict[str, frozenset[int]]],
    scalar_defs_before: dict[int, dict[str, ast.AST]],
) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    new_body: list[ast.AST] = []
    for stmt in body:
        # Recurse into nested statement-bearing fields first.
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(
                    stmt,
                    field,
                    _hoist_lane_invariant_chunk_recurrence(
                        old,
                        aliases,
                        running_sums,
                        thread_axes_before,
                        scalar_defs_before,
                    ),
                )

        info = _detect_chunk_recurrence(stmt, aliases, running_sums)
        if info is None:
            new_body.append(stmt)
            continue
        reset_idx = _find_reset_assign(new_body, info.dot_acc_var)
        if reset_idx is None:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence has no relocatable accumulator reset"
            )
        reset_stmt = new_body[reset_idx]
        if not _is_proven_relocatable_assignment(reset_stmt, allow_load=False):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset has effects or unknown call purity"
            )
        chunk_writes = {
            _canonical_name(name, aliases) for name in ReadWrites.from_ast(stmt).writes
        }
        reset_reads = {
            _canonical_name(name, aliases)
            for name in ReadWrites.from_ast(reset_stmt).reads
        }
        if reset_reads & chunk_writes:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset depends on chunk-carried state"
            )
        if any(
            info.dot_acc_var
            in {
                *ReadWrites.from_ast(intervening).reads,
                *ReadWrites.from_ast(intervening).writes,
            }
            for intervening in new_body[reset_idx + 1 :]
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset has an intervening consumer"
            )
        if any(
            {
                _canonical_name(name, aliases)
                for name in ReadWrites.from_ast(intervening).writes
            }
            & reset_reads
            for intervening in new_body[reset_idx + 1 :]
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence reset dependency is overwritten"
            )
        new_body.pop(reset_idx)
        assert isinstance(stmt, ast.For)
        new_body.append(
            _rewrite_chunk_recurrence(
                stmt,
                info,
                reset_stmt,
                aliases,
                thread_axes_before,
                scalar_defs_before,
            )
        )
    return new_body


def _detect_chunk_recurrence(
    stmt: ast.AST,
    rename_groups: dict[str, str],
    running_sums: set[str] | None,
) -> _ChunkRecurrence | None:
    """Describe a serial chunk loop carrying one lane-folded recurrence."""
    if not _is_serial_for(stmt) or not isinstance(stmt, ast.For):
        return None
    found = _single_lane_loop_in_body(list(stmt.body))
    if found is None:
        return None
    lane_idx, lane_loop, lane_var = found
    recurrence = _find_dot_acc_recurrence(
        lane_loop,
        lane_var,
        rename_groups,
        running_sums,
    )
    if recurrence is None:
        return None
    return dataclasses.replace(recurrence, lane_idx=lane_idx)


def _rewrite_chunk_recurrence(
    stmt: ast.For,
    info: _ChunkRecurrence,
    reset_stmt: ast.AST,
    rename_groups: dict[str, str],
    thread_axes_before: dict[int, dict[str, frozenset[int]]],
    scalar_defs_before: dict[int, dict[str, ast.AST]],
) -> ast.For:
    """Build the restructured chunk loop (see module comment)."""
    from .ast_read_writes import ReadWrites

    lane_idx = info.lane_idx
    lane_loop = info.lane_loop
    lane_var = info.lane_var
    chunk_body: list[ast.AST] = list(stmt.body)
    lane_body: list[ast.AST] = list(lane_loop.body)
    lane_varying = _lane_varying_names(lane_body[: info.sum_idx + 1], lane_var)

    def reads_lane(s: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(s).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # The base slice was proven lane-invariant by the detector.  It freezes the
    # chunk-entry state and computes its optional per-chunk rescale.
    hoist_set = set(info.base_indices)
    reduction_axes = _chunk_recurrence_reduction_axes(info, scalar_defs_before)
    guarded_statements: dict[int, ast.AST] = {}
    all_lane_writes = {
        name
        for lane_stmt in lane_body
        for name in ReadWrites.from_ast(lane_stmt).writes
    }
    lane_effects = _ordered_block_reads_writes(cast("list[ast.stmt]", lane_body))
    canonical_live_ins = {
        _canonical_name(name, rename_groups)
        for name in lane_effects.reads_before_write
        if name != lane_var
    }
    canonical_lane_writes = {
        _canonical_name(name, rename_groups) for name in lane_effects.may_writes
    }
    loop_carried_names = canonical_live_ins & canonical_lane_writes
    allowed_recurrence_carries = {
        info.state_name,
        _canonical_name(info.dot_acc_var, rename_groups),
    }

    def reject_unrelated_carries(index: int, moved_stmt: ast.AST) -> None:
        dependencies = _dependency_names(
            lane_body[:index], set(ReadWrites.from_ast(moved_stmt).reads)
        )
        dependencies.update(ReadWrites.from_ast(moved_stmt).writes)
        if {_canonical_name(name, rename_groups) for name in dependencies} & (
            loop_carried_names - allowed_recurrence_carries
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence cannot relocate another loop carry"
            )

    def guard_idempotent_store(
        index: int,
        store_stmt: ast.AST,
        *,
        thread_axes: dict[str, frozenset[int]] | None = None,
        scalar_definitions: dict[str, ast.AST] | None = None,
    ) -> None:
        store = _validated_idempotent_store(store_stmt)
        if store is None:
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence can only relocate an ordinary store"
            )
        reject_unrelated_carries(index, store_stmt)
        store_func = cast("ast.Attribute", store.func)
        address_and_predicate_reads = set(ReadWrites.from_ast(store_func.value).reads)
        for predicate in _store_enclosing_predicates(store_stmt, store):
            address_and_predicate_reads.update(ReadWrites.from_ast(predicate).reads)
        address_and_predicate_dependencies = _dependency_names(
            lane_body[:index], address_and_predicate_reads
        )
        if {
            _canonical_name(name, rename_groups)
            for name in address_and_predicate_dependencies
        } & allowed_recurrence_carries:
            raise exc.BackendUnsupported(
                "cute",
                "chunk recurrence store address or predicate is not idempotent",
            )
        store_axes = _store_thread_axes(
            store_stmt,
            (
                thread_axes
                if thread_axes is not None
                else thread_axes_before.get(id(store_stmt), {})
            ),
            (
                scalar_definitions
                if scalar_definitions is not None
                else scalar_defs_before.get(id(store_stmt), {})
            ),
        )
        if store_axes is None:
            raise exc.BackendUnsupported(
                "cute", "cannot infer chunk recurrence store ownership"
            )
        owner_exprs = _thread_axis_owner_exprs(reduction_axes, store_axes)
        guarded_statements[index] = (
            _guard_stmt_with_owner(store_stmt, owner_exprs)
            if owner_exprs
            else store_stmt
        )

    # Lane-invariant side-effecting statements whose backward slice reaches the
    # (frozen) chunk-ENTRY accumulator (the store of ``b_h``) must hoist before
    # the lane loop so they run once per chunk on the frozen value; bring their
    # producers along.  A store that does NOT read the accumulator stays in the
    # lane loop (it is a genuinely per-lane side effect).
    entry_indices: list[int] = []
    for idx, prod in enumerate(lane_body[: info.sum_idx]):
        if idx in hoist_set or reads_lane(prod):
            continue
        slice_indices, _ = _backward_slice(
            lane_body[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        slice_names = _dependency_names(
            lane_body[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        if not any(
            _canonical_name(name, rename_groups) == info.state_name
            for name in slice_names
        ):
            continue
        store = _validated_idempotent_store(prod)
        if store is None:
            if not _is_proven_relocatable_assignment(prod, allow_load=True):
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence state consumer has unknown side effects",
                )
            continue
        # The store + its complete, lane-invariant, side-effect-free producer
        # slice all hoist.  A live-in modified elsewhere in the lane would make
        # the repeated stores observably different and is therefore not
        # idempotent.
        if any(
            reads_lane(lane_body[i])
            or not _is_proven_relocatable_assignment(lane_body[i], allow_load=False)
            for i in slice_indices
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence store does not have a pure complete slice"
            )
        relocated_before_store = hoist_set | set(entry_indices) | set(slice_indices)
        if any(
            earlier_idx not in relocated_before_store
            and not _is_proven_relocatable_assignment(
                lane_body[earlier_idx], allow_load=False
            )
            for earlier_idx in range(idx)
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence cannot move a store across a prior effect"
            )
        entry_store_roots = _store_iterator_roots(store)
        for later_idx in range(idx + 1, info.sum_idx):
            if later_idx in relocated_before_store:
                continue
            later_stmt = lane_body[later_idx]
            try:
                later_store = _validated_idempotent_store(later_stmt)
            except exc.BackendUnsupported as error:
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse a store before a later effect",
                ) from error
            if later_store is not None:
                later_roots = _store_iterator_roots(later_store)
                if (
                    entry_store_roots
                    and later_roots
                    and later_roots.isdisjoint(entry_store_roots)
                ):
                    continue
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse stores to the same allocation",
                )
            if not _is_proven_relocatable_assignment(
                later_stmt,
                allow_load=True,
                allow_reduction=True,
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot collapse a store before a later effect",
                )
        slice_writes = {
            name
            for slice_index in slice_indices
            for name in ReadWrites.from_ast(lane_body[slice_index]).writes
        }
        slice_reads = set(ReadWrites.from_ast(prod).reads)
        for slice_index in slice_indices:
            slice_reads.update(ReadWrites.from_ast(lane_body[slice_index]).reads)
        if any(
            name in all_lane_writes
            and name not in slice_writes
            and _canonical_name(name, rename_groups) != info.state_name
            for name in slice_reads
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence store is not proven idempotent"
            )
        guard_idempotent_store(idx, prod)
        entry_indices.append(idx)
        hoist_set.update(slice_indices)

    hoist_pre = sorted(hoist_set | set(entry_indices))
    for idx in hoist_pre:
        reject_unrelated_carries(idx, lane_body[idx])
    pre_lane = [guarded_statements.get(i, lane_body[i]) for i in hoist_pre]

    # Moving the state update out of the repeated lane requires moving its
    # complete lexical suffix with it.  Otherwise a suffix consumer executes
    # before the newly finalized state exists.  Relocate only lane-invariant,
    # provably pure assignments and isolated idempotent stores; fail closed for
    # atomics, synchronization, compound effects, or calls of unknown purity.
    required_post_indices = {*info.finalize_indices, info.final_idx}
    post_lane_indices = set(range(info.sum_idx + 1, len(lane_body)))
    extra_post_indices = post_lane_indices - required_post_indices
    extra_post_reads = {
        name
        for idx in extra_post_indices
        for name in ReadWrites.from_ast(lane_body[idx]).reads
    }
    suffix_dependency_indices, _ = _backward_slice(
        lane_body[: info.sum_idx + 1], extra_post_reads
    )
    for idx in suffix_dependency_indices:
        if idx in hoist_set or idx == info.sum_idx:
            continue
        dependency = lane_body[idx]
        if reads_lane(dependency) or not _is_proven_relocatable_assignment(
            dependency, allow_load=False
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence suffix does not have a pure complete slice"
            )
    post_varying = set(lane_varying)
    first_post = info.sum_idx + 1
    effective_post_axes = dict(
        thread_axes_before.get(id(lane_body[first_post]), {})
        if first_post < len(lane_body)
        else {}
    )
    effective_post_defs = dict(
        scalar_defs_before.get(id(lane_body[first_post]), {})
        if first_post < len(lane_body)
        else {}
    )
    for idx in sorted(post_lane_indices):
        post_stmt = lane_body[idx]
        post_rw = ReadWrites.from_ast(post_stmt)
        reject_unrelated_carries(idx, post_stmt)
        if idx in required_post_indices:
            # The detector proved these statements turn the lane fold into a
            # lane-invariant final value.  Do not let their original dataflow
            # taint later consumers that are relocated with them.
            post_varying.difference_update(post_rw.writes)
            _update_thread_axis_names(post_stmt, effective_post_axes)
            _update_scalar_definitions(post_stmt, effective_post_defs)
            if any(
                _qualified_name(node.func) in _CHUNK_REDUCTION_HELPERS
                or (
                    (_qualified_name(node.func) or "").startswith(
                        "cute.arch.warp_reduction"
                    )
                )
                for node in ast.walk(post_stmt)
                if isinstance(node, ast.Call)
            ):
                for name in post_rw.writes:
                    effective_post_axes[name] = frozenset(
                        set(effective_post_axes.get(name, ())) - reduction_axes
                    )
            continue
        statement_is_lane_varying = lane_var in post_rw.reads or bool(
            set(post_rw.reads) & post_varying
        )
        if statement_is_lane_varying or any(
            _canonical_name(name, rename_groups) == info.state_name
            for name in post_rw.writes
        ):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence has a lane-varying state suffix"
            )
        store = _validated_idempotent_store(post_stmt)
        if store is not None:
            if idx < info.final_idx:
                raise exc.BackendUnsupported(
                    "cute",
                    "chunk recurrence cannot relocate a store before the finalized state",
                )
            guard_idempotent_store(
                idx,
                post_stmt,
                thread_axes=effective_post_axes,
                scalar_definitions=effective_post_defs,
            )
        elif not _is_proven_relocatable_assignment(post_stmt, allow_load=False):
            raise exc.BackendUnsupported(
                "cute", "chunk recurrence suffix is not proven pure"
            )
        post_varying.difference_update(post_rw.writes)
        _update_thread_axis_names(post_stmt, effective_post_axes)
        _update_scalar_definitions(post_stmt, effective_post_defs)

    lane_kept = [
        lane_body[i]
        for i in range(len(lane_body))
        if i not in hoist_set
        and i not in set(entry_indices)
        and i not in post_lane_indices
    ]

    new_lane_loop = _clone_lane_loop_with_body(lane_loop, lane_kept)
    post_lane = [
        guarded_statements.get(i, lane_body[i]) for i in sorted(post_lane_indices)
    ]
    new_chunk_body: list[ast.AST] = [
        reset_stmt,
        *chunk_body[:lane_idx],
        *pre_lane,
        new_lane_loop,
        *post_lane,
        *chunk_body[lane_idx + 1 :],
    ]
    return create(
        ast.For,
        target=stmt.target,
        iter=stmt.iter,
        body=new_chunk_body,
        orelse=stmt.orelse,
        type_comment=None,
    )


@dataclasses.dataclass
class LoopDimInfo:
    begin_var_name: str | None = None
    begin_expr: sympy.Expr | None = None
    end_var_name: str | None = None
    end_expr: sympy.Expr | None = None
    # True when the generated extent mask checks both the logical begin and end.
    mask_has_lower_bound: bool = False

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        shape_env = CompileEnvironment.current().shape_env
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return shape_env_size_hint(shape_env, expected) == shape_env_size_hint(
            shape_env, self.end_expr
        )


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]
    thread_axis_sizes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    block_thread_axes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    # The serial naming this state's tile begins in the access regions
    # (``cute/access_regions.py``).
    region_instance: int = dataclasses.field(
        default_factory=new_loop_instance, kw_only=True
    )

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        """The per-thread index and mask definitions this loop emits around its
        body, and the tile masks among them (``add_thread_barriers`` reads the
        thread axes through them)."""
        return [], frozenset()

    def lane_loop_headers(self) -> list[ast.For]:
        """The lane loops this loop nests around its body, as ``for`` statements
        for the barrier pass of a loop nested in the body
        (``_thread_barrier_context``)."""
        return []


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # Block ids that this device loop distributes across a per-thread lane
    # loop (CuTe only). A reduction over one of these blocks needs the
    # two-pass lane structure (see ``split_lane_loop_reductions``).
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    # The lane loops themselves (CuTe per-thread strategies), outermost first:
    # the blocks each distributes, the vec partitions among them and the
    # per-lane index / mask definitions at the top of the innermost body.
    # Unlike a grid's (``DeviceGridState.wrap_body``) the nest is built
    # around the body before the body exists and is never redistributed;
    # ``check_lane_loop_nest`` rejects it unless it is the tile program.
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_block_ids: dict[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    vec_lane_wrappers: dict[str, VecLaneWrapper] = dataclasses.field(
        default_factory=dict
    )
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    # The tile masks among those definitions (``LaneScope.masks``).
    tile_masks: frozenset[str] = frozenset()
    # Run once the loop body is complete (CuTe: restore the tile-vector stores
    # whose deferred flush a later access of their tensor would observe).
    body_finalizers: list[Callable[[], None]] = dataclasses.field(default_factory=list)

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        return _lane_index_definitions(
            self.outer_prefix, self.lane_setup_statements, self.vec_lane_wrappers
        ), self.tile_masks

    def lane_loop_headers(self) -> list[ast.For]:
        return _lane_loop_headers(self.lane_loops, self.vec_lane_wrappers)

    def loop_headers(self) -> list[ast.For]:
        """The device loops around ``inner_statements``, outermost first.

        The ``for`` statements on the way from ``for_node`` down to the body
        (a multi-dimensional tile nests one loop per block); the synthetic
        lane loops and constexpr vector loops among them are per-thread
        structure, not iteration.
        """
        loops: list[ast.For] = []
        loop: ast.For | None = self.for_node
        while loop is not None:
            if getattr(
                loop, HELION_LANE_LOOP_VAR_ATTR, None
            ) is None and not _is_constexpr_lane_iter(loop):
                loops.append(loop)
            if loop.body is self.inner_statements:
                break
            loop = next(
                (
                    child
                    for child in loop.body
                    if isinstance(child, ast.For)
                    and _encloses(child, self.inner_statements)
                ),
                None,
            )
        return loops

    def loop_variables(self) -> frozenset[str]:
        """The names the device loops around ``inner_statements`` redefine per iteration."""
        return frozenset(
            node.id
            for loop in self.loop_headers()
            for node in ast.walk(loop.target)
            if isinstance(node, ast.Name)
        )

    def tile_begin_shifts(self) -> dict[sympy.Symbol, sympy.Expr | None]:
        """The begin symbol of each block this loop iterates and its step per iteration.

        The symbols the access regions of the body use for this loop's tiles
        (``tile_begin_symbol``, keyed on this loop instance).  A loop over one
        block advances its begin by the block size each iteration; a nest
        over several blocks steps them in turn, so the next iteration's
        begins are unrelated to the current ones (None).
        """
        from .cute.access_regions import block_size_symbol
        from .cute.access_regions import tile_begin_symbol

        single = len(self.block_ids) == 1
        return {
            tile_begin_symbol(block_id, self): (
                block_size_symbol(self.strategy.fn.block_size_var(block_id))
                if single
                else None
            )
            for block_id in self.block_ids
        }

    def check_lane_loop_nest(self) -> None:
        """Make the lane loop nest around the body the tile program, or reject it.

        Every statement of the body runs inside every lane loop, so one whose
        values ignore a loop repeats once per iteration.  The full-nest check
        of the lane-loop distribution accepts the nest when every repetition
        is idempotent, pins an atomic that is uniform along a loop's tile axis
        to the loop's first lane and rejects the rest
        (``cute/lane_loop_distribution.py``).  Only the live loops are
        checked: a loop none of whose names the body reads (a tile indexed
        only by its ``begin``) is spliced away by the dead-code elimination
        and repeats nothing, as in ``DeviceGridState.wrap_body``, and a pin
        reading its lane variable would keep it alive.  Then, as for a grid
        body, block-wide barriers separate the accesses of one tensor that
        threads differing along a shared thread axis (this loop's or an
        enclosing loop's) could reorder (``add_thread_barriers``), and the
        body's accesses are checked across the loop's iterations too, a
        per-thread store of one iteration against a loop-invariant access of
        the next.
        """
        if CompileEnvironment.current().backend_name != "cute":
            return
        from .ast_read_writes import ReadWrites
        from .cute.lane_loop_distribution import LanePlacement
        from .cute.lane_loop_distribution import add_thread_barriers
        from .cute.lane_loop_distribution import check_full_nest

        setup = {id(statement) for statement in self.lane_setup_statements}
        body = [
            statement
            for statement in self.inner_statements
            if id(statement) not in setup
        ]
        needed, kept_setup, spliced_wrappers = _live_lane_setup(
            body, self.lane_setup_statements, self.vec_lane_wrappers
        )
        live_loops = [
            (lane_var, extent)
            for lane_var, extent in self.lane_loops
            if _lane_loop_is_live(
                lane_var, self.vec_lane_wrappers.get(lane_var), needed, spliced_wrappers
            )
        ]
        coordinates = {
            lane_var: _cute_lane_coordinates(
                lane_var, self.vec_lane_wrappers.get(lane_var)
            )
            for lane_var, _extent in live_loops
        }
        setup_by_lane, _ = _assign_lane_setup(live_loops, coordinates, kept_setup)
        scopes = _cute_lane_scopes(
            live_loops,
            setup_by_lane,
            coordinates,
            {
                lane_var: _vec_wrapper_statements(self.vec_lane_wrappers.get(lane_var))
                for lane_var, _extent in live_loops
            },
            self.vec_lane_wrappers,
            self.tile_masks,
        )
        rename_groups = _current_rename_groups()
        if live_loops:
            check_full_nest(body, scopes, rename_groups=rename_groups)
        # The nest as emitted: the body inside every live loop.
        items: list[ast.AST | LanePlacement] = list(body)
        loops = [LanePlacement(lane_var, items) for lane_var, _extent in live_loops]
        for outer, inner in itertools.pairwise(loops):
            outer.items = [inner]
        axis_sizes, definitions, masks, constants, enclosing = _thread_barrier_context(
            self
        )
        add_thread_barriers(
            [loops[0]] if loops else items,
            scopes,
            rename_groups=rename_groups,
            axis_sizes=axis_sizes,
            definitions=definitions,
            masks=masks,
            constants=constants,
            # A branch condition may vary per thread; a block-wide barrier in
            # a branch some threads skip would deadlock.
            allow_barriers=not _in_divergent_control_flow(),
            loop_variables=self.loop_variables(),
            loop_body=items,
            loop_shifts=self.tile_begin_shifts(),
            # The loops around the body: those around this loop, then its own.
            loop_headers=[*enclosing, *self.loop_headers()],
        )
        for loop in loops:
            _apply_wrapper_barriers(
                self.vec_lane_wrappers.get(loop.lane_var), loop.barriers
            )
        # The innermost list holds the body statements and the barriers added
        # among them, never a loop.
        statements = [item for item in items if isinstance(item, ast.AST)]
        assert len(statements) == len(items)
        # The dead loops' coordinates and the setup computed from them leave
        # the body with the loops (``_splice_lane_loops``).
        live = dict(live_loops)
        dead = {
            lane_var for lane_var, _extent in self.lane_loops if lane_var not in live
        }
        dead_names: set[str] = set()
        for lane_var in dead:
            dead_names |= _cute_lane_coordinates(
                lane_var, self.vec_lane_wrappers.get(lane_var)
            )
        kept: list[ast.AST] = []
        for statement in self.lane_setup_statements:
            rw = ReadWrites.from_ast(statement)
            if set(rw.reads) & dead_names:
                dead_names |= set(rw.writes)
            else:
                kept.append(statement)
        kept_ids = {id(statement) for statement in kept}
        self.inner_statements[:] = [
            *(
                statement
                for statement in self.inner_statements
                if id(statement) in kept_ids
            ),
            *statements,
        ]
        self._splice_lane_loops(dead)

    def _splice_lane_loops(self, lane_vars: set[str]) -> None:
        """Take the lane loops of ``lane_vars`` out of the nest around the body, their bodies in their place.

        The nest is built around the body before the body exists
        (``codegen_device_loop``): a loop none of whose coordinates the body
        reads repeats the body for nothing, or wrongly (a read-modify-write
        nested in it runs once per lane), while ``check_lane_loop_nest``
        described the nest without it; the emitted nest is the checked one.
        A vectorized loop goes with its per-thread base and V-loop.  The
        setup computed from the loops' coordinates is out of the body
        already.
        """
        loop = self.for_node
        while loop.body is not self.inner_statements:
            child = next(
                child
                for child in loop.body
                if isinstance(child, ast.For)
                and _encloses(child, self.inner_statements)
            )
            lane_var = getattr(child, HELION_LANE_LOOP_VAR_ATTR, None)
            if lane_var is None or lane_var not in lane_vars:
                loop = child
                continue
            wrapper = self.vec_lane_wrappers.get(lane_var)
            inner = child.body if wrapper is None else wrapper.vloop.body
            position = loop.body.index(child)
            loop.body[position : position + 1] = inner
            if inner is self.inner_statements:
                self.inner_statements = cast("list[ast.AST]", loop.body)


@dataclasses.dataclass
class EmitPipelineLoopState(DeviceLoopOrGridState):
    """State for emit_pipeline-based loops on TPU (Pallas backend)."""

    body_fn_name: str
    body_fn_def: ast.FunctionDef | None = None
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    pipeline_call: ast.AST | None = None
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    # Per downstream _mask_to node and block id, physical tensor bounds needed
    # to zero lanes left stale by a shortened input DMA.
    _deferred_physical_mask_bounds: dict[torch.fx.Node, dict[int, tuple[str, ...]]] = (
        dataclasses.field(default_factory=dict)
    )
    # Clean-region branches prove these physical bounds all-true for this tile.
    _proven_physical_mask_block_ids: set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class ForiLoopState(DeviceLoopOrGridState):
    """State for fori_loop-based loops on TPU (Pallas backend).

    Uses jax.lax.fori_loop with pltpu.make_async_copy for tensors whose
    inner-block shape passes ``_check_dma_alignment``; tensors that fail
    are kept on their outer BlockSpec and accessed via ``pl.ds`` from the
    body. Per-tensor pipelining membership lives in
    ``_tensor_to_dma_scratch``; input tensors with an overlapped prefetch are
    recorded in ``_prefetched_load_tensors``.
    """

    body_fn_name: str
    loop_var_name: str  # The fori_loop index variable (e.g., "_j")
    iteration_count: str | None = None
    static_unroll: bool = False
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    _tensor_to_sem: dict[str, str] = dataclasses.field(default_factory=dict)
    _prefetched_load_tensors: set[str] = dataclasses.field(default_factory=set)
    _memory_op_to_dma_scratch: dict[torch.fx.Node, DmaResources] = dataclasses.field(
        default_factory=dict
    )


def _cute_epilogue_subtile_active(config: object) -> bool:
    """True when the config requests epilogue subtiling (>= 2)."""
    value = getattr(config, "config", {}).get("epilogue_subtile")
    return isinstance(value, int) and value > 1


@dataclasses.dataclass
class VecLaneWrapper:
    """Pre-built outer x constexpr-V lane structure for a GRID lane loop.

    Grid lane loops are materialized late (``DeviceGridState.wrap_body``),
    but the vec-load/store hoist protocol needs the outer-lane body list and
    the constexpr V-loop node to exist DURING body codegen (memory_ops
    splices hoisted ``cute.arch.load(..., V)`` statements into it).  So
    ``PerThreadNDTileStrategy.codegen_grid`` pre-builds the structure and
    ``wrap_body`` just plugs the user body into ``vloop.body``.
    """

    outer_for: ast.For
    vloop: ast.For
    vec_lane_var: str
    base_index_var: str
    # Persistent reductions whose complete per-thread slice is one vector do
    # not need a separate outer scalar-lane loop.  Keep ``outer_for`` as the
    # mutable container used by the load/store hoist protocol, but splice its
    # body directly into the enclosing scope when this flag is set.
    elide_outer_loop: bool = False

    def __post_init__(self) -> None:
        # The lane reduction split tells this V-loop from any other constexpr
        # loop in the lane body by the lane variable it is nested in.
        lane_target = self.outer_for.target
        assert isinstance(lane_target, ast.Name)
        setattr(self.vloop, HELION_VEC_LANE_OF_ATTR, lane_target.id)


def _cute_lane_coordinates(lane_var: str, wrapper: VecLaneWrapper | None) -> set[str]:
    """A lane loop's own coordinates: its lane variable and, when vectorized,
    its constexpr V-loop variable and per-thread lane base."""
    names = {lane_var}
    if wrapper is not None:
        names.update((wrapper.vec_lane_var, wrapper.base_index_var))
    return names


def _cute_first_lane_predicate(lane_var: str, wrapper: VecLaneWrapper | None) -> str:
    """The first iteration of ``lane_var``'s loop, and of its constexpr V-loop."""
    if wrapper is None or wrapper.vec_lane_var == lane_var:
        return f"{lane_var} == 0"
    return f"{lane_var} == 0 and {wrapper.vec_lane_var} == 0"


def _vec_wrapper_statements(wrapper: VecLaneWrapper | None) -> list[ast.AST]:
    """The statements a vec wrapper emits besides its V-loop: the per-thread
    lane base, the packet loads hoisted above the V-loop, the store flushes
    after it."""
    if wrapper is None:
        return []
    return [stmt for stmt in wrapper.outer_for.body if stmt is not wrapper.vloop]


def _live_lane_setup(
    body: list[ast.AST],
    setup: list[ast.AST],
    wrappers: dict[str, VecLaneWrapper],
) -> tuple[set[str], list[ast.AST], set[str]]:
    """The names ``body`` needs, the lane setup statements defining them and
    the vec wrappers whose lane bodies received splices.

    Setup statements (per-lane index / mask definitions) whose results the
    body never reads are dropped, and a lane loop whose variable is then left
    unreferenced is dead (``_lane_loop_is_live``).  An unused rdim block
    (e.g. one allocated for a ``tile.index`` that codegen serves from the
    tile's own block) would otherwise wrap the whole body in a dead innermost
    loop, and the lane-reduce split pass would then try to split reductions
    on that loop's lane var instead of the loop that actually distributes
    them.
    """
    from .ast_read_writes import ReadWrites

    needed: set[str] = set()
    for stmt in body:
        needed |= set(ReadWrites.from_ast(stmt).reads)
    # A vec wrapper whose lane body received memory_ops splices (hoisted vec
    # loads before the V-loop, store flushes after it) must be kept
    # regardless of what the inner body reads: the flush IS the store.  Its
    # splices also read setup vars (masks, sibling index vars), so fold those
    # reads in before selecting the kept setup.
    spliced_wrappers: set[str] = set()
    for lane_var, wrapper in wrappers.items():
        if len(wrapper.outer_for.body) > 2:  # more than [base, vloop]
            spliced_wrappers.add(lane_var)
            needed |= set(ReadWrites.from_ast(wrapper.outer_for).reads)
    kept_setup: list[ast.AST] = []
    for stmt in reversed(setup):
        rw = ReadWrites.from_ast(stmt)
        if not rw.writes or set(rw.writes) & needed:
            kept_setup.append(stmt)
            needed |= set(rw.reads)
    kept_setup.reverse()
    return needed, kept_setup, spliced_wrappers


def _lane_loop_is_live(
    lane_var: str,
    wrapper: VecLaneWrapper | None,
    needed: set[str],
    spliced_wrappers: set[str],
) -> bool:
    """Whether the lane loop of ``lane_var`` survives the dead-code elimination."""
    if wrapper is None:
        return lane_var in needed
    # The per-element index var reads ``base_index_var`` + ``vec_lane_var``
    # (not ``lane_var`` directly), so test all three before dropping the
    # structure as dead.
    return lane_var in spliced_wrappers or bool(
        {lane_var, wrapper.vec_lane_var, wrapper.base_index_var} & needed
    )


def _current_rename_groups() -> dict[str, str]:
    """Every alias the device function will rename, mapped to its canonical name."""
    try:
        renames = DeviceFunction.current()._variable_renames
    except NoCurrentFunction:
        # Unit tests build the loop states outside a device function.
        return {}
    return {name: aliases[0] for name, aliases in renames.items()}


def _assign_lane_setup(
    lane_loops: list[tuple[str, int]],
    coordinates: dict[str, set[str]],
    setup: list[ast.AST],
) -> tuple[dict[str, list[ast.AST]], list[ast.AST]]:
    """Place each setup statement at the shallowest lane scope that defines
    every lane / base variable it reads.

    Historically all setup statements were placed in the innermost lane loop.
    That is semantically correct for scalar loops, but strands an outer-axis
    index definition *inside* an inner persistent vec loop while that vec
    loop's hoisted load needs the index outside its constexpr V loop.
    Explicit scope placement keeps outer tile coordinates available to nested
    reduction vec hoists and avoids redundantly recomputing them for every
    inner lane.  A statement reading no lane coordinate keeps the old
    innermost placement rather than speculatively widening its scope (a later
    setup statement reading its value must remain at least as deep), or is
    returned separately when there are no lane loops.
    """
    from .ast_read_writes import ReadWrites

    setup_by_lane: dict[str, list[ast.AST]] = {
        lane_var: [] for lane_var, _extent in lane_loops
    }
    fallback: list[ast.AST] = []
    setup_name_scopes: dict[str, int] = {}
    for stmt in setup:
        rw = ReadWrites.from_ast(stmt)
        reads = set(rw.reads)
        matching_depths = [
            depth
            for depth, (lane_var, _extent) in enumerate(lane_loops)
            if reads & coordinates[lane_var]
        ]
        matching_depths.extend(
            setup_name_scopes[name] for name in reads if name in setup_name_scopes
        )
        if matching_depths:
            # A statement depending on multiple lane coordinates belongs to
            # the deepest of those nested scopes.
            depth = max(matching_depths)
        elif lane_loops:
            depth = len(lane_loops) - 1
        else:
            fallback.append(stmt)
            continue
        setup_by_lane[lane_loops[depth][0]].append(stmt)
        for name in rw.writes:
            setup_name_scopes[name] = depth
    return setup_by_lane, fallback


def _cute_lane_scopes(
    live_loops: list[tuple[str, int]],
    setup_by_lane: dict[str, list[ast.AST]],
    coordinates: dict[str, set[str]],
    attached: dict[str, list[ast.AST]],
    wrappers: dict[str, VecLaneWrapper],
    masks: frozenset[str],
) -> list[LaneScope]:
    """The ``LaneScope`` of every live lane loop, outermost first, for the
    lane-loop distribution (``cute/lane_loop_distribution.py``).

    A loop's names are its coordinates and the definitions of its setup and
    attached statements; it requires the loops whose names its setup reads;
    its masks are the tile masks its setup defines; its counts are its trip
    count and its constexpr V-loop's, by variable.
    """
    from .ast_read_writes import ReadWrites
    from .cute.lane_loop_distribution import LaneScope

    def counts(lane_var: str, extent: int) -> dict[str, int]:
        wrapper = wrappers.get(lane_var)
        if wrapper is None:
            # A plain loop runs ``range(extent)``.
            return {lane_var: extent}
        # A vec wrapper's outer loop is pre-built (``extent`` counts the
        # thread's elements, not its iterations); its V-loop is constexpr.
        result: dict[str, int] = {}
        if wrapper.elide_outer_loop:
            result[lane_var] = 1
        elif (form := _ascending_lane_loop_form(wrapper.outer_for)) is not None:
            result[lane_var] = form[1]
        if (form := _ascending_lane_loop_form(wrapper.vloop)) is not None:
            result[wrapper.vec_lane_var] = form[1]
        return result

    setup_writes = {
        lane_var: frozenset().union(
            *(ReadWrites.from_ast(stmt).writes for stmt in setup_by_lane[lane_var])
        )
        for lane_var, _extent in live_loops
    }
    scope_names = {
        lane_var: frozenset(coordinates[lane_var]).union(
            setup_writes[lane_var],
            *(ReadWrites.from_ast(stmt).writes for stmt in attached[lane_var]),
        )
        for lane_var, _extent in live_loops
    }
    return [
        LaneScope(
            lane_var,
            scope_names[lane_var],
            frozenset(
                other
                for other, names in scope_names.items()
                if other != lane_var
                and any(
                    set(ReadWrites.from_ast(stmt).reads) & names
                    for stmt in setup_by_lane[lane_var]
                )
            ),
            tuple(attached[lane_var]),
            frozenset(coordinates[lane_var]),
            tuple(setup_by_lane[lane_var]),
            _cute_first_lane_predicate(lane_var, wrappers.get(lane_var)),
            masks & setup_writes[lane_var],
            _vloop_index(wrappers.get(lane_var)),
            counts(lane_var, extent),
        )
        for lane_var, extent in live_loops
    ]


def _vloop_index(wrapper: VecLaneWrapper | None) -> int:
    """How many of a vec wrapper's statements precede its V-loop (``LaneScope.vloop_index``).

    All of them when the V-loop is not among them (a wrapper built by hand).
    """
    if wrapper is None:
        return 0
    return next(
        (
            index
            for index, statement in enumerate(wrapper.outer_for.body)
            if statement is wrapper.vloop
        ),
        len(wrapper.outer_for.body),
    )


def _apply_wrapper_barriers(
    wrapper: VecLaneWrapper | None, barriers: list[int]
) -> None:
    """Insert the barriers ``add_thread_barriers`` placed among a wrapper's statements."""
    if not barriers:
        return
    assert wrapper is not None
    for position in sorted(barriers, reverse=True):
        wrapper.outer_for.body.insert(
            position, statement_from_string("cute.arch.sync_threads()")
        )


def _thread_barrier_context(
    state: DeviceLoopOrGridState,
) -> tuple[
    dict[int, int], list[ast.AST], frozenset[str], dict[str, int], list[ast.For]
]:
    """What ``add_thread_barriers`` needs to know about the threads running ``state``'s body.

    The thread count along every launch axis (the body runs on every thread
    of the launch, those a loop nested in it addresses included), the
    per-thread index definitions of this loop and of every enclosing one
    (a device loop's body addresses the grid's axes too), the tile masks
    among them, the values of the block-size constants (the bounds of the
    loops inside the body name them) and the loops around this loop: the
    enclosing device loops and the enclosing loops' lane loops (a lane nest
    is materialized after its body, so a loop nested in the body sees the
    lane variables as names), whose variables are one value each
    throughout the body, nonnegative and below their bounds
    (``AddressMaps`` models the loops around a body).
    """
    axis_sizes = dict(state.thread_axis_sizes)
    own, masks = state.lane_index_definitions()
    constants: dict[str, int] = {}
    enclosing_loops: list[ast.For] = []
    try:
        fn = DeviceFunction.current()
    except NoCurrentFunction:
        # Unit tests build the loop states outside a device function.
        return axis_sizes, own, masks, constants, enclosing_loops
    env = CompileEnvironment.current()
    for key, name in fn.block_size_var_cache.items():
        if len(key) == 1:
            value = env.block_sizes[key[0]].from_config(fn.config)
            if isinstance(value, int):
                constants[name] = value
    codegen = fn.codegen
    for axis, size in codegen.launch_thread_axis_sizes().items():
        axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
    # In program order, for the address maps to resolve each name to the
    # definition reaching the body: whatever the enclosing scopes defined so
    # far (a flattened tile's per-thread base, a coordinate a strategy
    # emitted straight into the kernel, an earlier loop over the same block
    # with its own index) may carry a thread coordinate into the body too,
    # then the enclosing loops' index definitions, then this loop's own.
    # The list at the bottom of the stack is the host wrapper's: what it
    # defines (a dynamic size unpacked from a shape) reaches the kernel as
    # an argument, one uniform value, not as a definition the body reads.
    definitions: list[ast.AST] = []
    for statements in codegen.statements_stack[1:]:
        for statement in statements:
            # The loops enclosing the body (this one among them, already in
            # its parent's list) are the body's structure, not definitions
            # reaching it.
            if (
                isinstance(state, DeviceLoopState)
                and isinstance(statement, ast.For)
                and _encloses(statement, state.inner_statements)
            ):
                continue
            definitions.append(statement)
    seen = {id(state)}
    for loops in codegen.active_device_loops.values():
        for enclosing in loops:
            if id(enclosing) not in seen:
                seen.add(id(enclosing))
                enclosing_definitions, enclosing_masks = (
                    enclosing.lane_index_definitions()
                )
                definitions.extend(enclosing_definitions)
                masks |= enclosing_masks
                # An enclosing loop's ``for`` statements join their parent's
                # list once its body is complete, so the loops around this
                # one are read from their states: the device loop's own
                # headers and the lane loops it nests around its body.
                if isinstance(enclosing, DeviceLoopState):
                    enclosing_loops.extend(enclosing.loop_headers())
                enclosing_loops.extend(enclosing.lane_loop_headers())
    definitions.extend(own)
    return axis_sizes, definitions, masks, constants, enclosing_loops


def _in_divergent_control_flow() -> bool:
    """Whether the statements being generated sit in a branch or while loop
    (``GenerateAST.divergent_control_flow``), whose condition a CuTe SIMT
    thread may evaluate differently from its neighbours."""
    try:
        return DeviceFunction.current().codegen.divergent_control_flow_depth > 0
    except NoCurrentFunction:
        return False


def _lane_index_definitions(
    prefix: list[ast.AST],
    setup: list[ast.AST],
    wrappers: dict[str, VecLaneWrapper],
) -> list[ast.AST]:
    """A loop's per-thread index definitions: those hoisted before its lane
    loops, the lane setup and the vec wrappers' own statements."""
    return [
        *prefix,
        *setup,
        *(
            statement
            for wrapper in wrappers.values()
            for statement in _vec_wrapper_statements(wrapper)
        ),
    ]


def _lane_loop_headers(
    lane_loops: list[tuple[str, int]], wrappers: dict[str, VecLaneWrapper]
) -> list[ast.For]:
    """A loop's lane loops as ``for`` statements: ``for lane in range(extent)``
    for each, and the V-loop of each vectorized one.

    A lane nest is materialized after its body (``wrap_body``), so a loop
    nested in the body sees the lane variables as names; the headers say
    what the names are (one value per pass, nonnegative, below the extent)
    to the address maps of the nested loop's own barrier pass.
    """
    headers: list[ast.For] = []
    for lane_var, extent in lane_loops:
        (header,) = ast.parse(f"for {lane_var} in range({extent}):\n    pass\n").body
        assert isinstance(header, ast.For)
        headers.append(header)
    headers.extend(wrapper.vloop for wrapper in wrappers.values())
    return headers


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    # Preserve which logical blocks each synthetic lane variable distributes.
    # The aggregate set above is retained for existing reduction scheduling;
    # late structural lowerings need the exact association to prove coordinate
    # coverage without guessing from generated variable names.
    lane_loop_block_ids: dict[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    # The tile masks among the lane setup's definitions (``LaneScope.masks``).
    tile_masks: frozenset[str] = frozenset()
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # Statement list that will receive ``wrap_body(...)``.  Definitions
    # already emitted there dominate every synthetic lane wrapper; lists
    # pushed after it are branch/loop bodies that the wrapper will enclose.
    hoist_parent_statements: list[ast.AST] | None = None
    # lane_var -> pre-built vec partition (see VecLaneWrapper).  Only grid
    # lane loops whose block has ``cute_vector_widths[block] > 1`` (and a
    # divisible elements-per-thread) get an entry.
    vec_lane_wrappers: dict[str, VecLaneWrapper] = dataclasses.field(
        default_factory=dict
    )
    deferred_vector_ops: list[tuple[ast.For, ast.AST, Callable[[], ast.AST | None]]] = (
        dataclasses.field(default_factory=list)
    )
    # Lane vars whose loops must visit lanes in descending order because a
    # reverse ``hl.associative_scan`` carries its suffix across them (see
    # ``cute/scan_ops.py``).  Applied when ``wrap_body`` materializes the
    # loops; never names a lane with a ``vec_lane_wrappers`` entry.
    reversed_lane_vars: set[str] = dataclasses.field(default_factory=set)
    # Lane vars whose loops a later pass rewrites structurally (resident
    # reductions, register-tile reductions); ``wrap_body`` keeps the full
    # nest around them.
    undistributable_lane_vars: set[str] = dataclasses.field(default_factory=set)
    # Body definitions that packet loads hoisted into a vec wrapper read
    # (``cute/memory_ops.py`` records them while lowering the body).
    # ``wrap_body`` emits them before that lane loop or, when it keeps the
    # full nest, moves them into the wrapper above the V-loop.
    pending_relocations: list[CuteLaneRelocation] = dataclasses.field(
        default_factory=list
    )
    # Synthetic reduction lanes materialized as ``cutlass.range_constexpr``
    # loops OUTSIDE the one-vector tile wrappers (register-tile reductions,
    # see ``nest_reduction_lane_outside_vector_tiles``).
    constexpr_lane_vars: set[str] = dataclasses.field(default_factory=set)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def add_lane_loop(
        self,
        block_id: int,
        lane_var: str,
        extent: int,
        *,
        position: int | None = None,
    ) -> None:
        if position is None:
            self.lane_loops.append((lane_var, extent))
        else:
            self.lane_loops.insert(position, (lane_var, extent))
        self.lane_loop_blocks.add(block_id)
        self.lane_loop_block_ids[lane_var] = self.lane_loop_block_ids.get(
            lane_var, frozenset()
        ) | {block_id}

    def nest_reduction_lane_outside_vector_tiles(
        self,
        block_id: int,
        lane_var: str,
        extent: int,
        *,
        max_unrolled_elements: int,
    ) -> bool:
        """Register a synthetic reduction lane as a trace-time loop OUTSIDE
        every one-vector tile wrapper, so the tile V-loops become the innermost
        element loops (a per-thread register tile).

        With the default nesting the constexpr tile V-loop wraps the rolled
        reduction lane loop, so a load whose row coordinate comes from the
        reduction lane is defined below the V-loop and can never be hoisted as
        one vector transaction; every lane iteration also waits for its own
        load before the next lane's load issues.  Placing the reduction lane
        outside the wrappers makes the row coordinate available above the
        V-loop (one LDG.128 per lane) and, because both loops unroll at trace
        time, lets ``split_lane_loop_reductions`` schedule every lane's loads
        before the first store.

        Applies only when every tile lane loop of this grid is a vector
        wrapper whose outer loop has exactly one trip (each thread owns one
        V-wide fragment per tile axis) and the unrolled per-thread element
        count stays within ``max_unrolled_elements``; otherwise the caller
        keeps the established rolled nesting.  Returns whether the lane was
        registered.
        """
        if not self.lane_loops or extent <= 1:
            return False
        unrolled = extent
        for tile_lane_var, _tile_extent in self.lane_loops:
            wrapper = self.vec_lane_wrappers.get(tile_lane_var)
            if wrapper is None or wrapper.elide_outer_loop:
                return False
            if _lane_loop_extent(wrapper.outer_for) != 1:
                return False
            unrolled *= _lane_loop_extent(wrapper.vloop)
        if unrolled > max_unrolled_elements:
            return False
        for tile_lane_var, _tile_extent in self.lane_loops:
            wrapper = self.vec_lane_wrappers[tile_lane_var]
            constexpr_iter = expr_from_string("cutlass.range_constexpr(1)")
            assert isinstance(constexpr_iter, ast.expr)
            wrapper.outer_for.iter = constexpr_iter
        self.constexpr_lane_vars.add(lane_var)
        # The register-tile lowering (``cute/register_tile_reductions.py``)
        # rewrites this nest as a whole, so ``wrap_body`` keeps it intact.
        self.undistributable_lane_vars.add(lane_var)
        self.add_lane_loop(block_id, lane_var, extent, position=0)
        return True

    def _live_lane_setup(
        self, body: list[ast.AST]
    ) -> tuple[set[str], list[ast.AST], set[str]]:
        return _live_lane_setup(
            body, self.lane_setup_statements, self.vec_lane_wrappers
        )

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        return _lane_index_definitions(
            self.outer_prefix, self.lane_setup_statements, self.vec_lane_wrappers
        ), self.tile_masks

    def lane_loop_headers(self) -> list[ast.For]:
        return _lane_loop_headers(self.lane_loops, self.vec_lane_wrappers)

    def add_body_barriers(self, body: list[ast.AST]) -> None:
        """Order across threads the accesses of a root body without lane loops.

        Such a body is emitted as it is (``wrap_body`` is for the lane loops),
        but its threads share the tile axes all the same: the barriers
        ``wrap_body`` would place separate its racing accesses
        (``add_thread_barriers``, in place).
        """
        if CompileEnvironment.current().backend_name != "cute":
            return
        from .cute.lane_loop_distribution import add_thread_barriers

        axis_sizes, definitions, masks, constants, enclosing = _thread_barrier_context(
            self
        )
        add_thread_barriers(
            body,  # pyrefly: ignore [bad-argument-type]
            [],
            rename_groups=_current_rename_groups(),
            axis_sizes=axis_sizes,
            definitions=definitions,
            masks=masks,
            constants=constants,
            loop_headers=enclosing,
        )

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:

        needed, kept_setup, spliced_wrappers = self._live_lane_setup(body)
        if self.deferred_vector_ops:
            has_barrier = any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.arch.sync_threads"
                for stmt in body
                for node in ast.walk(stmt)
            )
            # Root codegen can register synthetic reduction lanes that are
            # subsequently unused (for example aliases of tile.index). Decide
            # the innermost live lane only after the entire body is available.
            innermost = None
            for lane_var, _extent in self.lane_loops:
                wrapper = self.vec_lane_wrappers.get(lane_var)
                names = {lane_var}
                if wrapper is not None:
                    names.update((wrapper.vec_lane_var, wrapper.base_index_var))
                if names & needed or lane_var in spliced_wrappers:
                    innermost = wrapper
            replacements = {}
            for vloop, scalar, emit in self.deferred_vector_ops:
                # Hoisting a load or delaying a store across a RAW barrier
                # changes its value even when the lane scope is unchanged.
                if (
                    not has_barrier
                    and innermost is not None
                    and innermost.vloop is vloop
                ):
                    vector = emit()
                    if vector is not None:
                        replacements[id(scalar)] = vector
            self.deferred_vector_ops.clear()

            class ReplaceVectorOps(ast.NodeTransformer):
                def visit(self, node: ast.AST) -> ast.AST:
                    if replacement := replacements.get(id(node)):
                        return replacement
                    return super().visit(node)

            replace = ReplaceVectorOps()
            body = [replace.visit(stmt) for stmt in body]
            # Accepted operations added vector-load/store splices, whose
            # address and mask setup must participate in final liveness too.
            needed, kept_setup, spliced_wrappers = self._live_lane_setup(body)
        # Place each setup at the shallowest lane scope that defines every
        # lane/base variable it reads (``_assign_lane_setup``).
        lane_scope_names: dict[str, set[str]] = {
            lane_var: _cute_lane_coordinates(
                lane_var, self.vec_lane_wrappers.get(lane_var)
            )
            for lane_var, _extent in self.lane_loops
        }
        setup_by_lane, fallback_setup = _assign_lane_setup(
            self.lane_loops, lane_scope_names, kept_setup
        )

        live_loops = [
            (lane_var, extent)
            for lane_var, extent in self.lane_loops
            if self._lane_loop_is_live(lane_var, needed, spliced_wrappers)
        ]
        from .cute.lane_loop_distribution import LanePlacement
        from .cute.lane_loop_distribution import add_thread_barriers
        from .cute.lane_loop_distribution import check_full_nest
        from .cute.lane_loop_distribution import definitions_precede_loop
        from .cute.lane_loop_distribution import distribute_lane_loops

        # A statement that reads none of a lane loop's coordinates runs once
        # per iteration of that loop for nothing; place each statement inside
        # only the loops it depends on (``cute/lane_loop_distribution.py``).
        # The full nest stays when every statement depends on every loop or
        # the redistribution cannot be proven safe, and for loops other passes
        # rewrite structurally (resident reductions, reverse scans).  Either
        # nest then gets the barriers ordering the accesses threads sharing a
        # tile axis could race on (``add_thread_barriers``).
        # A vec wrapper's outer lane body carries statements of its own
        # besides the V-loop: the per-thread lane base, packet loads
        # hoisted above the V-loop and store flushes after it.  Their
        # definitions are the loop's, and the body statements that move
        # past the loop are ordered against their memory accesses.
        attached = {
            lane_var: _vec_wrapper_statements(self.vec_lane_wrappers.get(lane_var))
            for lane_var, _extent in live_loops
        }
        scopes = _cute_lane_scopes(
            live_loops,
            setup_by_lane,
            lane_scope_names,
            attached,
            self.vec_lane_wrappers,
            self.tile_masks,
        )
        rename_groups = _current_rename_groups()
        placement = None
        if (
            live_loops
            and not self.reversed_lane_vars
            and not self.undistributable_lane_vars & {lane for lane, _ in live_loops}
            and not any(
                setup_by_lane[lane_var]
                for lane_var, _extent in self.lane_loops
                if lane_var not in dict(live_loops)
            )
        ):
            placement = distribute_lane_loops(body, scopes, rename_groups=rename_groups)
            # The definitions a hoisted packet load reads stay body statements
            # here; they must come out before the loop that hoisted the load.
            # Otherwise the full nest stays, if it is exact.
            if placement is not None and not all(
                definitions_precede_loop(
                    placement,
                    self._vec_wrapper_lane_var(relocation.vloop),
                    relocation.statements,
                )
                for relocation in self.pending_relocations
            ):
                check_full_nest(body, scopes, rename_groups=rename_groups)
                placement = None
        extents = dict(live_loops)
        # Pass order.  The deferred vector operations were emitted on the
        # scalar-form body and the body is now placed; next the collected
        # stores a later statement of their loop observes return to their
        # scalar form (``demote_reordered_tile_vec_stores``).  A demotion
        # changes the body and the wrapper statements the scopes describe, so
        # the body is placed again from the start.  Only the final,
        # demotion-free pass reaches ``add_thread_barriers``: the barrier pass
        # classifies the statements as they are emitted (a restored scalar
        # store is a plain store of its tensor) and no barrier it records
        # among a wrapper's statements is moved or dropped afterwards.
        if placement is None:
            # The full nest keeps every statement inside every loop: a site's
            # later accesses are the body statements after it.
            if self._demote_reordered_tile_vec_stores(body, None):
                return self.wrap_body(body)
            for relocation in self.pending_relocations:
                relocation.apply(body)
            placement = self._full_nest(body, setup_by_lane, extents)
        elif self._demote_reordered_tile_vec_stores(body, placement):
            return self.wrap_body(body)
        # Later passes rewrite the nest around an undistributable lane as a
        # whole; a body with compiler markers is left alone by the pass.
        # Metal's rolled reductions run this wrapper too; the barrier pass
        # and its ``cute.arch.sync_threads()`` are the CuTe backend's (a
        # wrapper built without a compile environment keeps the pass).
        cute_backend = (
            not CompileEnvironment.has_current()
            or CompileEnvironment.current().backend_name == "cute"
        )
        if cute_backend and not self.undistributable_lane_vars & set(extents):
            axis_sizes, definitions, masks, constants, enclosing = (
                _thread_barrier_context(self)
            )
            add_thread_barriers(
                placement,
                scopes,
                rename_groups=rename_groups,
                axis_sizes=axis_sizes,
                definitions=definitions,
                masks=masks,
                constants=constants,
                loop_headers=enclosing,
            )

        def materialize(items: list[ast.AST | LanePlacement]) -> list[ast.AST]:
            result: list[ast.AST] = []
            for item in items:
                if isinstance(item, LanePlacement):
                    result.extend(
                        self._materialize_lane_loop(
                            item.lane_var,
                            extents[item.lane_var],
                            [
                                *setup_by_lane[item.lane_var],
                                *materialize(item.items),
                            ],
                            barriers=item.barriers,
                        )
                    )
                else:
                    result.append(item)
            return result

        return [*fallback_setup, *materialize(placement)]

    def _full_nest(
        self,
        body: list[ast.AST],
        setup_by_lane: dict[str, list[ast.AST]],
        extents: dict[str, int],
    ) -> list[ast.AST | LanePlacement]:
        """The original nest, ``body`` inside every lane loop, as placement items.

        A live loop (in ``extents``) is a ``LanePlacement``; a dead loop is
        spliced away and only its setup statements stay, where the loop
        would have been.
        """
        from .cute.lane_loop_distribution import LanePlacement

        items: list[ast.AST | LanePlacement] = list(body)
        for lane_var, _extent in reversed(self.lane_loops):
            if lane_var in extents:
                items = [LanePlacement(lane_var, items)]
            else:
                items = [*setup_by_lane[lane_var], *items]
        return items

    def _vec_wrapper_lane_var(self, vloop: ast.For) -> str:
        """The lane var whose vec wrapper owns the constexpr V-loop ``vloop``."""
        for lane_var, wrapper in self.vec_lane_wrappers.items():
            if wrapper.vloop is vloop:
                return lane_var
        raise AssertionError("relocation recorded for an unknown V-loop")

    def _demote_reordered_tile_vec_stores(
        self,
        body: list[ast.AST],
        placement: list[ast.AST | LanePlacement] | None,
    ) -> bool:
        """Restore the collected stores a later statement of their loop observes.

        ``placement`` distributes ``body`` around the lane loops, or is None
        when the full nest is kept.  Only the statements that stay inside a
        store's lane loop run before its flush: a lane-invariant statement
        placed after the loop follows the flush and one placed before it
        precedes the store's site, both ordered by ``distribute_lane_loops``.
        True when a site was demoted; its flush left the loop and its scalar
        store joined the body, so the caller places the body again.  Only the
        per-thread lane strategies collect vector stores.
        """
        from .cute.lane_loop_distribution import statements_inside_loop
        from .cute.memory_ops import demote_reordered_tile_vec_stores

        strategy = self.strategy
        if not isinstance(
            strategy, (PerThreadNDTileStrategy, PerThreadFlattenedTileStrategy)
        ):
            # Only the per-thread lane strategies collect vector stores; the
            # other strategies (and the unit tests' stand-ins) have none.
            return False
        demoted = False
        for block_id, sites in strategy._cute_lane_vec_stores_by_block.items():
            if not sites:
                continue
            inside = None
            if placement is not None:
                lane_var = self._vec_wrapper_lane_var(
                    strategy._cute_lane_vloop_by_block[block_id]
                )
                inside = statements_inside_loop(placement, lane_var)
            if demote_reordered_tile_vec_stores(
                strategy, strategy.fn, block_id, body, inside=inside
            ):
                demoted = True
        return demoted

    def _lane_loop_is_live(
        self, lane_var: str, needed: set[str], spliced_wrappers: set[str]
    ) -> bool:
        return _lane_loop_is_live(
            lane_var, self.vec_lane_wrappers.get(lane_var), needed, spliced_wrappers
        )

    def _materialize_lane_loop(
        self,
        lane_var: str,
        extent: int,
        wrapped: list[ast.AST],
        *,
        barriers: list[int] | None = None,
    ) -> list[ast.AST]:
        """Wrap ``wrapped`` in the (pre-built or plain) loop of ``lane_var``.

        ``barriers`` are the positions among a vec wrapper's statements where
        ``add_thread_barriers`` put a block-wide barrier.
        """
        wrapper = self.vec_lane_wrappers.get(lane_var)
        _apply_wrapper_barriers(wrapper, barriers or [])
        if wrapper is not None:
            wrapper.vloop.body = wrapped  # type: ignore[assignment]
            if lane_var in self.reversed_lane_vars:
                # The vector store protocol appends the V results of the
                # constexpr vector loop in iteration order, so this
                # partition cannot run backwards; the scan declines such
                # shapes before requesting a reversal.
                raise exc.BackendUnsupported(
                    "cute", "reverse scan over a vectorised lane loop"
                )
            return (
                list(wrapper.outer_for.body)
                if wrapper.elide_outer_loop
                else [wrapper.outer_for]
            )
        lane_loop = _create_lane_loop(
            lane_var,
            extent,
            wrapped,
            constexpr=lane_var in self.constexpr_lane_vars,
        )
        if lane_var in self.reversed_lane_vars and not _reverse_lane_loop_iter(
            lane_loop
        ):
            raise exc.BackendUnsupported(
                "cute", "reverse scan lane loop is not reversible"
            )
        return [lane_loop]


@dataclasses.dataclass
class PersistentReductionState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [_create_lane_loop(lane_var, extent, wrapped)]
        return wrapped


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_ids: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        # CuTe DSL preprocessor counter collision: the preprocessor's
        # negative-step machinery (``_handle_negative_step`` in
        # ``cutlass.base_dsl.ast_preprocessor.DSLPreprocessor``) emits
        # ``offset_<counter>`` / ``start_<counter>`` / ``stop_<counter>`` /
        # ``step_<counter>`` / ``isNegative_<counter>`` helpers at the enclosing
        # scope of every for-loop whose step is not a positive Python literal.
        # Helion's tile-offset names share the same ``offset_<n>`` namespace —
        # Python's name-binding rule sees the late preprocessor assignment and
        # treats the variable as local for the whole function body, turning
        # earlier reads into ``UnboundLocalError``. The ``tile_`` prefix moves
        # Helion's names out of the reserved CuTe DSL namespace. Of the five
        # reserved suffixes, only ``offset_`` and ``step_`` are emitted by
        # Helion (``offset_<bid>`` here; ``step_<n>`` via ``codegen.lift(...,
        # prefix='step')`` in ``codegen_grid_loops`` / ``codegen_lane_loops``);
        # both are renamed on cute. ``start_/stop_/isNegative_`` collisions are
        # not currently emitted by Helion. Non-CuTe backends keep the
        # historical short name to preserve existing goldens — this is a
        # deliberate trade-off (see ``cute_plan.md`` §7.6.5.2 for the trade-off
        # rationale; search "CuTe DSL preprocessor counter collision" for the
        # diagnosis).
        env = CompileEnvironment.current()
        offset_prefix = "tile_offset" if env.backend.name == "cute" else "offset"
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"{offset_prefix}_{block_idx}", dce=True)
            for block_idx in block_ids
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def tile_begin_var(self, block_idx: int) -> str:
        """Uniform first index of the current tile along ``block_idx``.

        ``tile.begin`` / ``tile.end`` / ``tile.id`` render through this. It is
        the loop offset for every strategy whose ``offset_var`` names the tile
        start; strategies whose offset is already the per-element index (the
        CuTe per-thread flattened tile) override it with the tile base.
        """
        return self.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def supports_index_rank_expansion(self) -> bool:
        """Whether index expressions produced by this strategy are tensor-shaped."""
        return True

    def thread_axes_used(self) -> int:
        return 0

    def thread_block_sizes(self) -> list[int]:
        """Return the thread block size for each thread axis this strategy uses."""
        return []

    def thread_block_size_exprs(self) -> list[str]:
        """Return per-axis thread block sizes as launch-time expressions."""
        return [str(size) for size in self.thread_block_sizes()]

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            config.range_warp_specializes, block_idx, None
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if indexing_uses_tensor_descriptor(
            config.indexing
        ) or indexing_uses_tensor_descriptor(config.atomic_indexing):
            # Device-created tensor descriptors combined with multi-stage
            # range pipelines tend to cause CUDA "misaligned address" or
            # "unspecified launch failure" errors. Host-created descriptors
            # do not carry that per-program construction through the range.
            if range_num_stages > 0 and not config.host_tensor_descriptors:
                range_num_stages = 0
            # Host construction has not established that descriptor indexing
            # is safe with both unrolling and a separate kernel-level pipeline.
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            if env.backend.effective_num_warps(config) == 1:
                # One-warp kernels have no cross-warp layout hazard. Keep the
                # pipeline within the number of complete unrolled iterations;
                # using only the final tile's remainder would collapse every
                # exactly divisible reduction to one stage.
                loop_iterations = int(math.ceil(loop_numel / block_size))
                unrolled_iterations = int(
                    math.ceil(loop_iterations / range_unroll_factor)
                )
                range_num_stages = min(max(1, unrolled_iterations), range_num_stages)
            else:
                # Multi-warp unrolling plus pipelining can cause CUDA IMA. Keep
                # the conservative legacy bound for those kernels.
                step = range_unroll_factor * block_size
                last_offset = ((loop_numel - 1) // block_size) * block_size
                remainder = loop_numel - last_offset
                range_num_stages = min(
                    max(1, int(math.ceil(remainder / step))), range_num_stages
                )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            kwargs.append(f"data_partition_factor={dpf_value}")

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()

        # Allow backend to override the range expression entirely
        backend_range = env.backend.range_str(begin, end, step)
        if backend_range is not None:
            return backend_range

        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self,
        state: CodegenState,
        use_proxy_ends: bool = False,
        ends_override: list[object] | None = None,
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds.

        Args:
            state: The codegen state
            use_proxy_ends: If True, use proxy_ends from state.proxy_args (for device loops)
            ends_override: If provided, use these ends instead of block_sizes.numel (for data-dependent bounds)
        """
        env = CompileEnvironment.current()
        block_id_to_info = {}

        def begin_to_ast(value: object) -> ast.AST:
            if isinstance(value, ast.AST):
                return value
            if isinstance(value, int):
                return expr_from_string(repr(value))
            if isinstance(value, sympy.Expr):
                return expr_from_string(DeviceFunction.current().sympy_expr(value))
            if isinstance(value, torch.SymInt):
                return begin_to_ast(value._sympy_())
            if isinstance(value, torch.Tensor):
                tensor_arg = DeviceFunction.current().tensor_arg(value)
                return expr_from_string(env.backend.scalar_load_expr(tensor_arg.name))
            raise NotImplementedError(f"{type(value)} is not implemented.")

        def normalize_dim_values(value: object) -> list[object]:
            if isinstance(value, (list, tuple, torch.Size)):
                return list(value)
            return [value]

        begin_values: list[object] | None = None
        proxy_begins: list[object] | None = None
        if isinstance(state.ast_args, (list, tuple)):
            if len(state.ast_args) >= 2 and isinstance(state.ast_args[1], list):
                begin_values = state.ast_args[1]
        if isinstance(state.proxy_args, (list, tuple)):
            if len(state.proxy_args) >= 2 and isinstance(
                state.proxy_args[1], (list, tuple, torch.Size)
            ):
                proxy_begins = normalize_dim_values(state.proxy_args[1])
                if begin_values is None:
                    begin_values = proxy_begins
            elif len(state.proxy_args) >= 2:
                begin_arg, end_arg = state.proxy_args[:2]
                if end_arg is None:
                    proxy_begins = [0] * len(normalize_dim_values(begin_arg))
                else:
                    proxy_begins = normalize_dim_values(begin_arg)
                if begin_values is None:
                    begin_values = proxy_begins

        if use_proxy_ends:
            _, _, proxy_ends, _, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for idx, (block_idx, end) in enumerate(
                zip(self.block_ids, proxy_ends, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=None,
                    end_expr=end_expr,
                )
        elif ends_override is not None:
            # Data-dependent bounds: use the provided ends
            for idx, (block_id, end) in enumerate(
                zip(self.block_ids, ends_override, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                    end_var_name = state.sympy_expr(end_expr)
                else:
                    # Tensor (data-dependent) - end_expr is None, but we still need end_var
                    end_expr = None
                    end_var_name = None
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )
        else:
            for idx, block_id in enumerate(self.block_ids):
                block_size_info = env.block_sizes[block_id]
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if block_size_info.size is None:
                    # Data-dependent bound - skip numel, it will be handled elsewhere
                    end_expr = None
                    end_var_name = None
                else:
                    end_expr = block_size_info.numel
                    end_var_name = state.sympy_expr(end_expr)
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )

        return block_id_to_info

    def _setup_block_size_constexpr(
        self, state: CodegenState, block_size_var: str, block_size: SymIntLike
    ) -> None:
        """Helper to setup constexpr block size variable on host."""
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def _get_data_dependent_numel(
        self, state: CodegenState, end: object, begin: object
    ) -> sympy.Expr | str:
        """Get numel for data-dependent bounds using the tensor end value.

        When the tile bound is a tensor (data-dependent), we need to pass
        the tensor to the kernel and use it to compute the number of elements.
        Returns either a sympy.Expr or a string expression.
        """
        from .device_function import DeviceFunction

        device_function = DeviceFunction.current()

        if isinstance(end, torch.Tensor):
            # For tensor bounds, we need to add it as a kernel argument
            # and load the scalar value
            tensor_arg = device_function.tensor_arg(end)
            end_expr = CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        elif isinstance(end, (int, torch.SymInt)):
            end_expr = device_function.sympy_expr(_to_sympy(end))
        else:
            raise NotImplementedError(f"Unsupported end type: {type(end)}")

        if begin == 0:
            # Simple case: numel = end
            return end_expr  # type: ignore[return-value]
        if isinstance(begin, torch.Tensor):
            begin_arg = device_function.tensor_arg(begin)
            begin_expr = CompileEnvironment.current().backend.scalar_load_expr(
                begin_arg.name
            )
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        if isinstance(begin, (int, torch.SymInt)):
            begin_expr = device_function.sympy_expr(_to_sympy(begin))
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        raise NotImplementedError(f"Unsupported begin type: {type(begin)}")

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := state.device_function.resolved_block_size(
                    block_id
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end

    def _compute_thread_axis_offset(
        self,
        active_device_loops: dict[int, list[DeviceLoopOrGridState]],
    ) -> int:
        """Compute the starting thread axis for the next strategy.

        Counts axes already claimed by active device loops, reserving at
        least one axis for reduction strategies when the backend places
        reductions first.

        When a ``CuTeGridExecutionPlan`` with ``block_axis_priority`` is
        in scope for this strategy's blocks, the offset is instead
        derived from ``thread_axis_for_strategy`` so the M/N axis order
        is dictated by the plan (e.g. the warp-per-row layout swaps the
        outer M-grid and inner N-tile axes so each warp owns one row).
        """
        from .reduction_strategy import ReductionStrategy

        env = CompileEnvironment.current()

        # Plan-driven path: honor ``block_axis_priority`` so the outer
        # grid loop can reserve an axis for a lower-priority inner tile
        # loop even when that inner loop has not yet entered
        # ``active_device_loops``.  Used by the warp-per-row layout where
        # the outer M-grid must take a HIGHER thread-axis index than the
        # inner N-tile so 32 contiguous threads on axis 0 form one warp
        # per row.
        plan = self.fn.tile_strategy.current_cute_grid_execution_plan(
            block_ids=self.block_ids
        )
        if plan is not None and any(
            plan.priority_for_block(block_id) is not None for block_id in self.block_ids
        ):
            offset = self.fn.tile_strategy.thread_axis_for_strategy(self)
            if offset is not None:
                return offset

        if (
            env.backend.name == "cute"
            and len(HostFunction.current().device_ir.phases) > 1
        ):
            # All barrier phases share the launch layout. Use the dispatcher's
            # kernel-wide reservation, including axes absent from this phase,
            # so the body's tile indices agree with the launch block dimensions.
            offset = self.fn.tile_strategy.thread_axis_for_strategy(self)
            if offset is not None:
                return offset

        seen: set[int] = set()
        active_reduction_axes = 0
        active_non_reduction_axes = 0
        for loops in active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                axes = loop_state.strategy.thread_axes_used()
                if env.backend.reduction_axis_first() and isinstance(
                    loop_state.strategy, ReductionStrategy
                ):
                    active_reduction_axes += axes
                else:
                    active_non_reduction_axes += axes

        if not env.backend.reduction_axis_first():
            return active_non_reduction_axes + active_reduction_axes

        # Reserve through the highest axis assigned to a coexecuting
        # multi-thread reduction or full slice. Counting strategies can miss
        # gaps in those assignments and make tile indices overlap a reduction
        # axis. Single-thread reductions use constant zero and need no axis.
        reduction_strategies = [
            strategy
            for strategy in self.fn.tile_strategy.strategies
            if isinstance(strategy, ReductionStrategy)
        ]
        planned_reduction_axes = max(
            max(
                (
                    axis + strategy.thread_axes_used()
                    for strategy in reduction_strategies
                    if strategy._reduction_thread_count() > 1
                    and self.fn.tile_strategy.strategies_can_coexecute(self, strategy)
                    and (
                        axis := self.fn.tile_strategy.thread_axis_for_strategy(strategy)
                    )
                    is not None
                ),
                default=0,
            ),
            1
            if any(strategy.thread_axes_used() > 0 for strategy in reduction_strategies)
            else 0,
        )
        if plan is not None and any(
            plan.disables_reduction_axis_reservation(block_id)
            for block_id in self.block_ids
        ):
            return active_non_reduction_axes + active_reduction_axes
        reserved_reduction_axes = max(planned_reduction_axes, active_reduction_axes)
        offset = reserved_reduction_axes + active_non_reduction_axes
        tile_axes = set(range(offset, offset + self.thread_axes_used()))
        for strategy in self.fn.tile_strategy.strategies:
            if not isinstance(strategy, ReductionStrategy):
                continue
            if not self.fn.tile_strategy.strategies_can_coexecute(self, strategy):
                continue
            if not any(size > 1 for size in strategy.thread_block_sizes()):
                continue
            reduction_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduction_axis is None:
                continue
            reduction_axes = set(
                range(reduction_axis, reduction_axis + strategy.thread_axes_used())
            )
            if collision := tile_axes & reduction_axes:
                axes = ", ".join(map(str, sorted(collision)))
                raise exc.BackendUnsupported(
                    env.backend.name,
                    "thread-axis collision: tile blocks "
                    f"{self.block_ids} and reduction/slice block "
                    f"{strategy.block_index} both require axis {axes}",
                )
        return offset

    def select_pid_strategy(self) -> ProgramIDs:
        env = CompileEnvironment.current()
        if env.compact_worklist_plan is not None:
            # Compact worklist: the owner hl.grid becomes the work-item grid.
            from .program_id import WorklistProgramIDs

            return WorklistProgramIDs(upper_expr=str(env.compact_worklist_upper))
        backend_name = env.backend.name
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        use_tcgen05_scheduler = self._use_tcgen05_persistent_scheduler(
            pid_type, backend_name
        )
        if pid_type == "persistent_blocked":
            if use_tcgen05_scheduler:
                return Tcgen05PersistentProgramIDs(is_blocked=True)
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            if use_tcgen05_scheduler:
                return Tcgen05PersistentProgramIDs(is_blocked=False)
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()

    def _use_tcgen05_persistent_scheduler(
        self, pid_type: str, backend_name: str
    ) -> bool:
        if backend_name != "cute" or not pid_type.startswith("persistent"):
            return False
        from .backend import _kernel_specialized_mma_impl
        from .cute.cute_warp_mma_gemm import MATMUL_FAMILY_WARP_MMA
        from .cute.cute_warp_mma_gemm import WARP_MMA_FAMILY_KEY

        # The register-MMA family owns the whole body (one CTA per tile); the
        # tcgen05 persistent scheduler would need a tcgen05 plan it never gets.
        if self.fn.config.get(WARP_MMA_FAMILY_KEY) == MATMUL_FAMILY_WARP_MMA:
            return False
        return _kernel_specialized_mma_impl(self.fn, config=self.fn.config) == "tcgen05"


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    # pyrefly: ignore [bad-override]
    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        self._mask_elidable = not env.backend.force_tile_mask() and env.known_multiple(
            functools.reduce(  # pyrefly: ignore[incompatible-overload-residual]
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        )
        # Elision also requires that no sibling section launches this
        # strategy's thread axis wider than the tile, which is unknowable
        # until every strategy is registered in the dispatch — ``mask_var``
        # decides lazily on first use during codegen.
        self._mask_elision_decided = False
        self._mask_var: str | None = self.new_var("mask", dce=True)
        self._offsets_var = self.new_var("offsets", dce=True)

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        if not self._mask_elision_decided:
            self._mask_elision_decided = True
            if self._mask_elidable and not (
                CompileEnvironment.current().backend.launches_surplus_tile_threads()
                and self.fn.tile_strategy.has_surplus_threads_for_strategy(self)
            ):
                self._mask_var = None
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def thread_axes_used(self) -> int:
        return int(self._uses_thread_axis())

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis() or not isinstance(self.block_size, int):
            return []
        return [self.block_size]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        if isinstance(self.block_size, int):
            return [str(self.block_size)]
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        return [bs_var]

    def _uses_thread_axis(self) -> bool:
        return not (isinstance(self.block_size, int) and self.block_size == 1)

    def _numel_str(self, state: CodegenState, value: sympy.Expr | str) -> str:
        if isinstance(value, str):
            return value
        return state.sympy_expr(value)

    def _range_trip_count(
        self,
        begin: object,
        end: object,
        step: object | None,
    ) -> sympy.Expr | str:
        return self._range_numel_expr(begin, end, step)

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        if isinstance(value, torch.Tensor):
            tensor_arg = DeviceFunction.current().tensor_arg(value)
            return CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        if isinstance(value, str):
            return value
        raise NotImplementedError(f"{type(value)} is not implemented.")

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _extract_root_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        assert len(state.proxy_args) == 3
        if state.proxy_args[1] is None:
            begins: list[object] = [0] * len(self.block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins_arg = state.proxy_args[0]
            begins = (
                list(begins_arg)
                if isinstance(begins_arg, (list, tuple))
                else [begins_arg]
            )
            ends_arg = state.proxy_args[1]
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _extract_device_loop_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        if len(state.ast_args) == 5:
            _, begins_arg, ends_arg, _, steps_arg = state.ast_args
        else:
            _, begins_arg, ends_arg, _ = state.ast_args
            steps_arg = None
        begins = (
            list(begins_arg) if isinstance(begins_arg, (list, tuple)) else [begins_arg]
        )
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(steps_arg, len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _codegen_common(
        self,
        state: CodegenState,
        *,
        begins: list[object] | None = None,
        ends: list[object] | None = None,
        steps: list[object | None] | None = None,
    ) -> tuple[str, str, sympy.Expr | str, list[ast.AST]]:
        offsets_var = self._offsets_var
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        if begins is None:
            begins = [0] * len(block_ids)
        if ends is None:
            ends = [env.block_sizes[block_id].numel for block_id in block_ids]
        if steps is None:
            steps = [None] * len(block_ids)
        total_numel: sympy.Expr | str = sympy.S.One
        statements = []

        # pyrefly: ignore [bad-assignment]
        for i, (block_idx, begin, end, step) in enumerate(
            self._reorder([*zip(block_ids, begins, ends, steps, strict=True)])
        ):
            cute_scalar_tile = (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
                and step not in (None, 1)
            )
            numel = (
                self._range_numel_expr(begin, end, None)
                if cute_scalar_tile
                else self._range_trip_count(begin, end, step)
            )
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({self._numel_str(state, total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({self._numel_str(state, numel)})"
            step_expr = self._expr_str(step) if step not in (None, 1) else None
            if step_expr is not None and not (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
            ):
                expr = f"({expr}) * ({step_expr})"
            if begin != 0:
                expr = f"({self._expr_str(begin)}) + ({expr})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            if isinstance(total_numel, str) or isinstance(numel, str):
                total_numel = (
                    f"({self._numel_str(state, total_numel)})"
                    f" * ({self._numel_str(state, numel)})"
                )
            else:
                assert isinstance(total_numel, sympy.Expr)
                assert isinstance(numel, sympy.Expr)
                total_numel = sympy.Mul(total_numel, numel)

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            mask_terms = [f"{offsets_var} < ({self._numel_str(state, total_numel)})"]
            # Skip the ``thread_idx[axis] < block_size`` term for a CuTe
            # block-size-1 axis (see ``codegen_grid``): the axis is not a thread
            # axis, so this term would otherwise pin the launch dim to 1 and
            # block a synthetic free-``hl.arange`` axis from reusing it.
            if not (env.backend.name == "cute" and not self._uses_thread_axis()):
                thread_mask = env.backend.thread_in_tile_mask_expr(
                    block_size_var, axis=self._flat_thread_axis()
                )
                if thread_mask is not None:
                    mask_terms.insert(0, f"({thread_mask})")
            mask_expr = " and ".join(mask_terms)
            statements.append(statement_from_string(f"{mask_var} = {mask_expr}"))
        # pyrefly: ignore [bad-return]
        return block_size_var, offsets_var, total_numel, statements

    def _flat_thread_axis(self) -> int:
        """Compute the thread axis for this flattened strategy.

        For CuTe, reduction strategies occupy earlier axes.
        """
        return self._compute_thread_axis_offset(self.fn.codegen.active_device_loops)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        assert state.ast_args is None

        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        scalar_grid_loop = False
        if isinstance(type_info, IterType):
            inner = (
                type_info.inner.unpack()
                if isinstance(type_info.inner, SequenceType)
                else [type_info.inner]
            )
            scalar_grid_loop = len(inner) == 1 and isinstance(inner[0], GridIndexType)

        if (
            scalar_grid_loop
            and len(self.block_ids) == 1
            and len(state.proxy_args) == 3
            and not isinstance(state.proxy_args[0], (list, tuple))
            and (
                state.proxy_args[1] is None
                or not isinstance(state.proxy_args[1], (list, tuple))
            )
            and not isinstance(state.proxy_args[2], (list, tuple))
        ):

            def _range_bound_to_sympy(value: object) -> sympy.Expr:
                assert isinstance(value, (int, torch.SymInt, sympy.Expr))
                return _to_sympy(value)

            step = state.proxy_args[2]
            if step not in (None, 1):
                block_id = self.block_ids[0]
                if state.proxy_args[1] is None:
                    begin = 0
                    end = state.proxy_args[0]
                else:
                    begin = state.proxy_args[0]
                    end = state.proxy_args[1]
                    if isinstance(begin, (list, tuple)):
                        assert len(begin) == 1
                        begin = begin[0]
                    if isinstance(end, (list, tuple)):
                        assert len(end) == 1
                        end = end[0]
                begin_expr = _range_bound_to_sympy(begin)
                end_expr = _range_bound_to_sympy(end)
                step_expr = _range_bound_to_sympy(step)
                trip_count = (
                    f"(({state.sympy_expr(end_expr)}) - ({state.sympy_expr(begin_expr)}) + "
                    f"({state.sympy_expr(step_expr)}) - 1) // ({state.sympy_expr(step_expr)})"
                )

                env = CompileEnvironment.current()
                dtype = env.index_type()
                pid_var = state.device_function.new_var("pid_flat", dce=True)
                offsets_var = self._offsets_var
                block_size_var = self.block_size_var(-1)
                self._setup_block_size_constexpr(state, block_size_var, self.block_size)
                pids = self.select_pid_strategy()
                if isinstance(state.device_function.pid, ForEachProgramID):
                    pids.shared_pid_var = state.device_function.pid.shared_pid_var
                pids.append(PIDInfo(pid_var, block_size_var, trip_count, block_id))
                state.add_statement(
                    env.backend.arange_expr(
                        offsets_var,
                        pid_var,
                        block_size_var,
                        dtype,
                        axis=self._flat_thread_axis(),
                    )
                )
                index_var = self.index_var(block_id)
                state.add_statement(
                    f"{index_var} = ({state.sympy_expr(begin_expr)}) + ({offsets_var}) * ({state.sympy_expr(step_expr)})"
                )
                mask_var = self.mask_var(-1)
                if mask_var is not None:
                    mask_terms = [f"{offsets_var} < ({trip_count})"]
                    thread_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=self._flat_thread_axis()
                    )
                    if thread_mask is not None:
                        mask_terms.insert(0, f"({thread_mask})")
                    state.add_statement(
                        statement_from_string(
                            f"{mask_var} = {' and '.join(mask_terms)}"
                        )
                    )
                pids.codegen(state)
                if isinstance(state.device_function.pid, ForEachProgramID):
                    shared_pid = state.device_function.pid
                    shared_pid.cases.append(pids)
                    shared_pid.codegen(state)
                else:
                    state.device_function.set_pid(pids)
                tracker = ThreadAxisTracker()
                if self._uses_thread_axis() and isinstance(self.block_size, int):
                    tracker.record_all(
                        self.block_ids, self._flat_thread_axis(), self.block_size
                    )
                elif self._uses_thread_axis():
                    tracker.record_symbolic_axis(
                        self.block_ids, self._flat_thread_axis()
                    )
                return DeviceGridState(
                    self,
                    block_id_to_info=self._create_block_id_info_dict(
                        state, ends_override=[end]
                    ),
                    thread_axis_sizes=tracker.sizes,
                    block_thread_axes=tracker.block_axes,
                )
        begins, ends, steps = self._extract_root_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))

        # A CuTe grid whose block size is 1 does not claim a thread axis: its
        # ``offsets = pid * 1 + thread_idx[axis]`` term is always 0 (launch dim
        # for the axis is 1). Emit ``offsets = pid * 1`` instead so the axis is
        # genuinely free for a synthetic free-``hl.arange`` thread axis to reuse
        # without the grid's ``thread_idx[axis] < 1`` mask filtering its lanes.
        if env.backend.name == "cute" and not self._uses_thread_axis():
            state.add_statement(
                statement_from_string(
                    f"{offsets_var} = ({pid_var}) * ({block_size_var})"
                )
            )
        else:
            state.add_statement(
                env.backend.arange_expr(
                    offsets_var,
                    pid_var,
                    block_size_var,
                    dtype,
                    axis=self._flat_thread_axis(),
                )
            )
        state.codegen.statements_stack[-1].extend(statements)

        pids.codegen(state)

        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
            else:
                tracker.record_symbolic_axis(self.block_ids, self._flat_thread_axis())
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        begins, ends, steps = self._extract_device_loop_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()
        lid = self.new_var("lid")
        numel_str = self._numel_str(state, total_numel)
        end_var = env.backend.cdiv_expr(numel_str, block_size_var, is_device=True)
        # Mirror ``codegen_grid``: a CuTe block-size-1 loop axis does not claim a
        # thread axis, so drop the always-zero ``+ thread_idx[axis]`` term so the
        # axis stays free (and consistent with the mask emitted by
        # ``_codegen_common``, which also drops its thread term for this case).
        if env.backend.name == "cute" and not self._uses_thread_axis():
            arange_expr = f"{offsets_var} = ({lid}) * ({block_size_var})"
        else:
            arange_expr = env.backend.arange_expr(
                offsets_var, lid, block_size_var, dtype, axis=self._flat_thread_axis()
            )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=(
                body := [
                    statement_from_string(arange_expr),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
            else:
                tracker.record_symbolic_axis(self.block_ids, self._flat_thread_axis())
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            disable = not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            )
            if not disable:
                for i, j in itertools.pairwise(block_ids):
                    if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                        # The block indices must be contiguous
                        disable = True
                        break
            if disable:
                flatten_loops.disable_block_id(block_ids[0])
                if env.backend_name == "cute":
                    # This gate exists for vector-model backends, where a
                    # partial-block access (``bias[tile_n]``) cannot
                    # broadcast against a flattened [BS] value vector.  The
                    # cute per-thread SCALAR model recomputes per-dim index
                    # vars from the flat offset per element, so a PURE
                    # pointwise kernel can flatten safely — record the spec
                    # and let device-IR analysis re-register it once the
                    # PointwiseElementwiseFact (no reductions / matmuls /
                    # accumulators) is known.
                    cands = env.config_spec.cute_reflatten_candidates
                    if all(c.block_ids != block_ids for c in cands):
                        cands.append(spec)

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # Keep axis structure intact for multi-phase kernels (e.g., barrier) to
        # avoid mismatched ranks in downstream reductions.
        if len(HostFunction.current().device_ir.root_ids) > 1:
            return shapes

        env = CompileEnvironment.current()
        # Filter out unit-sized blocks that don't need compacting
        compact_block_ids = [
            block_id
            for block_id in self.block_ids
            if not (
                isinstance(env.block_sizes[block_id].size, int)
                and env.block_sizes[block_id].size == 1
            )
        ]
        if not compact_block_ids:
            return shapes

        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            # Check if this starts our flattened sequence
            if len(shape.block_ids) != 1 or shape.block_ids[0] != compact_block_ids[0]:
                output.append(shape)
                continue

            # Try to collect the full sequence
            group_shapes = [shape]
            found_complete_sequence = True
            for expected in compact_block_ids[1:]:
                if (
                    shape_queue
                    and len(shape_queue[0].block_ids) == 1
                    and shape_queue[0].block_ids[0] == expected
                ):
                    group_shapes.append(shape_queue.popleft())
                else:
                    # Partial match - don't combine
                    found_complete_sequence = False
                    output.extend(group_shapes)
                    break

            if found_complete_sequence:
                # Full match - combine into one
                for s in group_shapes[1:]:
                    shape = shape.combine(s)
                output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    # pyrefly: ignore [bad-override]
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def _uses_thread_axis(self, block_size: SymIntLike) -> bool:
        return not (isinstance(block_size, int) and block_size == 1)

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        """Hook: does ``block_id`` claim a CUDA thread axis under this strategy?

        Defaults to ``_uses_thread_axis(block_size)``. Subclasses that
        track per-block-id state (e.g. ``PerThreadNDTileStrategy``'s
        ``inactive_block_ids``) override this to return False for
        block_ids that don't claim an axis so the grid / device-loop
        codegen does not emit ``thread_idx[axis]`` for them.
        """
        return self._uses_thread_axis(block_size)

    def thread_axes_used(self) -> int:
        return sum(
            1 for block_size in self.block_size if self._uses_thread_axis(block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if self._uses_thread_axis(bs) and isinstance(bs, int):
                sizes.append(bs)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis(bs):
                continue
            if isinstance(bs, int):
                exprs.append(str(bs))
            else:
                bs_var = self.block_size_var(block_id)
                if bs_var is None:
                    return []
                exprs.append(bs_var)
        return exprs

    def _thread_axis_offset(self, state: CodegenState) -> int:
        return self._compute_thread_axis_offset(state.codegen.active_device_loops)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis(block_size_by_id[block_id]):
                axis += 1
        return mapping

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _root_grid_steps(self, state: CodegenState) -> list[object | None]:
        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        assert isinstance(type_info, IterType)
        inner = (
            type_info.inner.unpack()
            if isinstance(type_info.inner, SequenceType)
            else [type_info.inner]
        )
        if not all(isinstance(value, GridIndexType) for value in inner):
            return [None] * len(self.block_ids)
        return self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        return ast.unparse(self._to_ast(value))

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        elif (
            isinstance(pids, FlatProgramIDs)
            and env.backend.name == "pallas"
            and len(block_ids) >= 2
        ):
            pids = XYZProgramIDs()

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — rename our lifted step var to
                # avoid the same UnboundLocalError that drove the offset rename.
                step_prefix = "tile_step" if env.backend.name == "cute" else "step"
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
            axis = thread_axis_offset + thread_axis_map[block_idx]
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); without the polymorphic
            # ``_uses_thread_axis_for_block`` hook the grid would emit
            # ``thread_idx[axis]`` for them and collide with the inner
            # device-loop on the same axis.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.grid_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            elif uses_thread_axis:
                tracker.record_symbolic_axis([block_idx], axis)
            state.add_statement(f"{index_var} = {idx_expr}")
            if (
                uses_thread_axis
                and isinstance(block_size, int)
                and env.backend_name == "cute"
                and env.config_spec.pointwise_facts
                and not _cute_epilogue_subtile_active(self.fn.config)
            ):
                # The non-lane ND path must retain the same producer-axis
                # evidence as PerThreadNDTileStrategy's lane-loop path.
                self.fn.cute_state.grid_thread_extents[index_var] = (axis, block_size)
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if uses_thread_axis else None,
                block_size_var=bs if uses_thread_axis else None,
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        # Only use ends_override if there are data-dependent (tensor) bounds
        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                cast_expr = CompileEnvironment.current().backend.ast_to_dtype_expr(
                    "{value}", to_dtype
                )
                return expr_from_string(cast_expr, value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        if isinstance(x, torch.Tensor):
            # Handle tensor values (for data-dependent bounds)
            # For scalar tensors, we need to load the value using tl.load
            from .device_function import DeviceFunction

            tensor_arg = DeviceFunction.current().tensor_arg(x)
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_load_expr(tensor_arg.name)
            )
        if isinstance(x, str):
            # Already a string expression (for data-dependent numel)
            return expr_from_string(x)
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); see ``codegen_grid`` above for the
            # collision this guards against.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.loop_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            elif uses_thread_axis:
                tracker.record_symbolic_axis([block_idx], axis)
            extra_body = [
                statement_from_string(f"{index_var} = {idx_expr}"),
            ]
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if uses_thread_axis else None,
                block_size_var=bs if uses_thread_axis else None,
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            # pyrefly: ignore [unsupported-operation]
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
        *,
        thread_axis: int | None = None,
        block_size_var: str | None = None,
    ) -> ast.stmt | None:
        """Build the bounds mask for one tile axis."""
        env = CompileEnvironment.current()
        if (
            not env.backend.force_tile_mask()
            and env.block_sizes[block_idx].known_multiple(block_size)
            and not env.is_jagged_tile(block_idx)
            and not (
                env.backend.launches_surplus_tile_threads()
                and self.fn.tile_strategy.has_surplus_threads_for_block_id(block_idx)
            )
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )

        if env.is_jagged_tile(block_idx):
            jagged_tile_parents_ast = state.ast_args[3]
            jagged_tile_parents_proxy = state.proxy_args[3]
            assert isinstance(jagged_tile_parents_ast, list)
            assert isinstance(jagged_tile_parents_proxy, list)
            # We guarantee the first lifted loop input is the jagged_tile parent tensor.
            jagged_tile_parent = jagged_tile_parents_ast[0]
            jagged_tile_block_size = env.block_sizes[block_idx].var
            jagged_tile_parent_proxy = jagged_tile_parents_proxy[0]
            assert isinstance(jagged_tile_parent_proxy, torch.Tensor)
            parent_dims: list[torch.SymInt] = []
            for d in jagged_tile_parent_proxy.size():
                assert isinstance(d, torch.SymInt)
                parent_dims.append(d)
            assert len(parent_dims) >= 1
            env.jagged_tile_mask_shapes[block_idx] = [
                *parent_dims,
                jagged_tile_block_size,
            ]
            if not self.supports_index_rank_expansion():
                tile_mask = None
                if (
                    thread_axis is not None
                    and block_size_var is not None
                    and env.backend.launches_surplus_tile_threads()
                    and self.fn.tile_strategy.has_surplus_threads_for_block_id(
                        block_idx
                    )
                ):
                    tile_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=thread_axis
                    )
                prefix = f"({tile_mask}) and " if tile_mask is not None else ""
                return statement_from_string(
                    f"{mask_var} = {prefix}({index_var}) < {{parent}}",
                    parent=self._to_ast(jagged_tile_parent),
                )
            k = len(parent_dims)
            child_expand = "[" + ", ".join(["None"] * k + [":"]) + "]"
            parent_expand = "[" + ", ".join([":"] * k + ["None"]) + "]"
            return statement_from_string(
                f"{mask_var} = ({index_var}){child_expand} < {{parent}}{parent_expand}",
                parent=self._to_ast(jagged_tile_parent),
            )

        mask_terms = [f"({index_var}) < {{end}}"]
        if (
            thread_axis is not None
            and block_size_var is not None
            and env.backend.launches_surplus_tile_threads()
        ):
            thread_mask = env.backend.thread_in_tile_mask_expr(
                block_size_var, axis=thread_axis
            )
            if thread_mask is not None:
                mask_terms.insert(0, f"({thread_mask})")
        return statement_from_string(
            f"{mask_var} = {' and '.join(mask_terms)}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class PerThreadNDTileStrategy(NDTileStrategy):
    """N-D tiling for backends that give each thread one element of the tile.

    Adds, on top of :class:`NDTileStrategy`, the per-axis ``num_threads`` split
    and the lane loop it implies: an axis served by fewer threads than its
    block size has each thread walk ``block_size // num_threads`` elements.
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
        num_threads: list[int] | None = None,
        mma_mode: bool = False,
        inactive_block_ids: set[int] | None = None,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order, l2_grouping)
        assert isinstance(block_size, list)
        if num_threads is None:
            num_threads = [0 for _ in block_ids]
        assert len(num_threads) == len(block_ids)
        self.num_threads = num_threads
        self._shared_thread_extents: dict[int, int] = {}
        self.mma_mode = mma_mode
        self.inactive_block_ids = inactive_block_ids or set()
        self._lane_var_by_block: dict[int, str] = {}
        # Per-block vec width for the lane loop (1 = scalar).  Populated
        # from the autotuner-selected ``cute_vector_widths`` config when
        # the block has a lane loop and its ``elements_per_thread`` is
        # divisible by the picked V.  When > 1, ``codegen_device_loop``
        # partitions the lane loop into outer (epT/V) x inner constexpr V
        # so memory_ops can hoist a single ``cute.arch.load(..., V)`` per
        # outer-lane iter (LDG.64 / LDG.128).
        self._cute_lane_vec_width_by_block: dict[int, int] = {}
        # Per-block constexpr V-loop var (only set when the lane loop is
        # vec-partitioned). Used by memory_ops to find the inner loop's
        # target var when emitting per-lane bitcasts.
        self._cute_vec_lane_var_by_block: dict[int, str] = {}
        # Per-block lane-base index var (the per-thread base of a V-wide
        # contiguous chunk).  Set when lane vec is in play; used by
        # memory_ops to compute the vec load pointer once per outer-lane
        # iter.
        self._cute_lane_base_index_var_by_block: dict[int, str] = {}
        # Per-block lane body (list of AST statements inside the outer
        # lane loop, ending in the constexpr V-loop). memory_ops uses
        # ``insert(len(lane_body)-1, hoist_stmt)`` to splice the vec
        # load just before the inner V-loop.
        self._cute_lane_body_by_block: dict[int, list] = {}
        # Per-block constexpr V-loop AST node.  memory_ops locates it in
        # ``lane_body`` to splice vec-load hoists BEFORE it and vec-store
        # flushes AFTER it (position-independent, so both can coexist).
        self._cute_lane_vloop_by_block: dict[int, ast.For] = {}
        # Per-block collected vec-store sites (``CuteTileVecStoreSite``) in
        # source order; their flushes follow the V-loop in that order.
        self._cute_lane_vec_stores_by_block: dict[int, list[CuteTileVecStoreSite]] = {}
        # Per-block lane layout ("blocked" | "strided") from the
        # ``cute_lane_layouts`` config knob.  Only differs from blocked
        # when the lane loop has more than one iteration.
        self._cute_lane_layout_by_block: dict[int, str] = {}
        # Lazy one-shot guard for ``_maybe_apply_cute_cluster``.
        self._cute_cluster_checked: bool = False
        # Per-block thread-block-cluster split (from ``cute_cluster_n``):
        # the axis's elements are divided across ``cl`` cluster CTAs in
        # addition to the CTA's own threads.  Applied only to a single
        # lane-looped axis when the tile covers the full extent (single
        # trip, so cluster reduces run exactly once per call site) and all
        # sibling axes have thread extent 1.
        self._cute_cluster_by_block: dict[int, int] = {}
        # Shared per-block hoist cache: (tensor_name, base_ptr_expr) ->
        # (hoist_var, dtype).  Same shape as
        # ``LoopedReductionStrategy._cute_lane_vec_loads``.
        self._cute_lane_vec_loads_by_block: dict[int, dict] = {}
        if not mma_mode:
            env_local = CompileEnvironment.current()
            cute_vec_widths_cfg = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
            )
            cute_lane_layouts_cfg = cast(
                "list[str]",
                fn.config.config.get("cute_lane_layouts", []) or [],
            )
            for block_id, nt, bs in zip(
                block_ids, num_threads, block_size, strict=True
            ):
                if block_id in self.inactive_block_ids:
                    continue
                static_bs = self._configured_block_size_int(bs)
                if (
                    nt > 0
                    and static_bs is not None
                    and static_bs > nt
                    and static_bs % nt == 0
                ):
                    self._lane_var_by_block[block_id] = self.fn.new_var(
                        f"lane_{block_id}"
                    )
                    if (
                        block_id
                        in env_local.config_spec.cute_lane_layouts.valid_block_ids()
                    ):
                        layout = env_local.config_spec.cute_lane_layouts.config_get(
                            cute_lane_layouts_cfg, block_id, "blocked"
                        )
                        if isinstance(layout, str):
                            self._cute_lane_layout_by_block[block_id] = layout
                    elements_per_thread = static_bs // nt
                    # Vec slot is registered eagerly in device-IR analysis; read
                    # the tuned V.  Never append here — growing the spec during
                    # codegen breaks the autotuner's fixed-width unflatten.
                    if (
                        block_id
                        in env_local.config_spec.cute_vector_widths.valid_block_ids()
                    ):
                        vec_width = env_local.config_spec.cute_vector_widths.config_get(
                            cute_vec_widths_cfg,
                            block_id,
                            1,
                        )
                        if (
                            isinstance(vec_width, int)
                            and vec_width > 1
                            and elements_per_thread % vec_width == 0
                        ):
                            self._cute_lane_vec_width_by_block[block_id] = vec_width
                    else:
                        # Metal shares this strategy without vec-width tuning,
                        # and non-static sizes are eager-skipped — both run
                        # scalar.  A missing static slot on cute is a bug.
                        assert env_local.backend_name != "cute" or not isinstance(
                            env_local.block_sizes[block_id].size,
                            (int, torch.SymInt),
                        ), (
                            f"cute_vector_widths slot missing for static-size "
                            f"block_id={block_id}; it must be registered during "
                            f"device-IR analysis, not lazily during codegen"
                        )

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        if isinstance(block_size, int):
            return block_size
        env = CompileEnvironment.current()
        resolved_block_id = env.resolve_block_id(block_size)
        if resolved_block_id is not None:
            configured_size = self.fn.resolved_block_size(resolved_block_id)
            if isinstance(configured_size, int):
                return configured_size
        block_size_expr = _to_sympy(block_size)
        block_size_expr = env.specialize_expr(block_size_expr)
        if getattr(block_size_expr, "free_symbols", None):
            return None
        return int(block_size_expr)

    def use_shared_thread_extents(self, extents: dict[int, int]) -> None:
        """Fit SIMT lane loops to the physical extents shared by sibling loops."""
        for block_id, threads in extents.items():
            idx = self.block_ids.index(block_id)
            size = self._configured_block_size_int(self.block_size[idx])
            if size is None or (size > threads and size % threads != 0):
                raise exc.BackendUnsupported(
                    "cute",
                    "a shared tile thread axis requires an evenly partitioned tile",
                )
            self._shared_thread_extents[block_id] = threads
            elements_per_thread = max(1, size // threads)
            if elements_per_thread == 1:
                self._lane_var_by_block.pop(block_id, None)
            vec = self._cute_lane_vec_width_by_block.get(block_id, 1)
            if elements_per_thread % vec:
                self._cute_lane_vec_width_by_block.pop(block_id, None)

    def thread_extent_for_masking(self, block_id: int, thread_extent: int) -> int:
        """Tile lanes that can own an element when a sibling widens the launch."""
        if block_id in self._shared_thread_extents:
            size = self._configured_block_size_int(
                self.block_size[self.block_ids.index(block_id)]
            )
            assert size is not None
            return min(thread_extent, size)
        return thread_extent

    def _demote_blocks_beyond_thread_axes(self, state: CodegenState) -> None:
        """Serve blocks that would land on CUDA thread axis >= 3 with lanes.

        A launch only has x/y/z thread axes. Once the enclosing loops and
        reductions have claimed them (``_thread_axis_offset``), every further
        block of this strategy is demoted to what ``num_threads=1`` means:
        thread extent 1 and a single thread's lane loop walking the whole
        tile. This only changes the thread layout, never the elements the tile
        covers, so it is safe to decide per call site at codegen time.
        """
        if self.mma_mode or CompileEnvironment.current().backend.name != "cute":
            return
        axis = self._thread_axis_offset(state)
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            block_size = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, block_size):
                continue
            if axis < 3:
                axis += 1
                continue
            size = self._configured_block_size_int(block_size)
            if size is None:
                raise exc.BackendUnsupported(
                    "cute",
                    f"thread axis {axis}: block {block_id} needs a static block "
                    "size to run as a lane loop",
                )
            self._shared_thread_extents[block_id] = 1
            if size > 1 and block_id not in self._lane_var_by_block:
                self._lane_var_by_block[block_id] = self.fn.new_var(f"lane_{block_id}")
            vec = self._cute_lane_vec_width_by_block.get(block_id, 1)
            if size % vec:
                self._cute_lane_vec_width_by_block.pop(block_id, None)

    def _maybe_apply_cute_cluster(
        self, env: CompileEnvironment, state: CodegenState
    ) -> None:
        """Decide whether the ``cute_cluster_n`` knob applies to this loop
        (invoked lazily on the first ``codegen_device_loop`` call so the
        active grid state is visible).

        See ``_cute_cluster_by_block``.  When applied, the owning
        ``DeviceFunction``'s ``cute_state.simt_cluster_n`` is set so the
        host-side call emits the extra grid dim + cluster launch shape.
        """
        if self._cute_cluster_checked:
            return
        self._cute_cluster_checked = True
        if env.backend_name != "cute" or self.mma_mode:
            return
        cl = self.fn.config.config.get("cute_cluster_n", 1)
        if not isinstance(cl, int) or cl <= 1:
            return
        if getattr(self.fn.cute_state, "simt_cluster_n", 1) > 1:
            # Another device loop's strategy already claimed the cluster
            # rank; splitting a second axis on the same rank would leave
            # only the diagonal rank x rank blocks covered.
            return
        if len(self._lane_var_by_block) != 1:
            return
        # Statements outside the cluster-split loop execute once per
        # cluster CTA — benign for plain (idempotent) stores, but a
        # read-modify-write would repeat ``cluster_n`` times.
        from .host_function import HostFunction

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
        [block_id] = self._lane_var_by_block.keys()
        vec = self._cute_lane_vec_width_by_block.get(block_id, 1)
        idx = self.block_ids.index(block_id)
        nt = self._shared_thread_extents.get(block_id, self.num_threads[idx])
        static_bs = self._configured_block_size_int(self.block_size[idx])
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
        for other_id, other_bs in zip(self.block_ids, self.block_size, strict=True):
            if other_id == block_id:
                continue
            extent = self._static_thread_extent_for_block(other_id, other_bs)
            if extent is not None and extent != 1:
                return
        self._cute_cluster_by_block[block_id] = cl
        self.fn.cute_state.simt_cluster_n = cl

    def _elements_per_thread_for_block(self, block_id: int) -> int:
        """Elements per thread for *block_id* (derived from num_threads and
        the cluster split)."""
        if block_id in self.inactive_block_ids:
            return 1
        idx = self.block_ids.index(block_id)
        nt = self._shared_thread_extents.get(block_id, self.num_threads[idx])
        if nt == 0:
            return 1
        bs = self._configured_block_size_int(self.block_size[idx])
        assert isinstance(bs, int)  # validated by _thread_extent_for_axis
        return max(1, bs // nt) // self._cute_cluster_by_block.get(block_id, 1)

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        if block_id in self.inactive_block_ids:
            return 1
        if self.mma_mode:
            return 1  # MMA handles element distribution, no CUDA threads needed
        idx = self.block_ids.index(block_id)
        nt = self._shared_thread_extents.get(block_id, self.num_threads[idx])
        if nt == 0:
            return block_size
        backend_name = CompileEnvironment.current().backend.name
        resolved_block_size = block_size
        if not isinstance(resolved_block_size, int):
            static_block_size = self._configured_block_size_int(resolved_block_size)
            if static_block_size is None:
                raise exc.BackendUnsupported(
                    backend_name,
                    f"num_threads requires static ND block sizes for {backend_name}",
                )
            resolved_block_size = static_block_size
        if resolved_block_size % nt != 0 and not (
            (block_id in self._shared_thread_extents and resolved_block_size < nt)
            or CompileEnvironment.current().backend.collective_owns_tile(
                self.fn, block_id
            )
        ):
            raise exc.BackendUnsupported(
                backend_name,
                (
                    f"block size must be divisible by num_threads for "
                    f"{backend_name} axis "
                    f"{block_id}: {resolved_block_size} is not divisible by {nt}"
                ),
            )
        return nt

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        if block_id in self.inactive_block_ids:
            return False
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                axis += 1
        return mapping

    def thread_axes_used(self) -> int:
        return sum(
            1
            for block_idx, block_size in zip(
                self.block_ids, self.block_size, strict=True
            )
            if self._uses_thread_axis_for_block(block_idx, block_size)
        )

    def _static_thread_extent_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> int | None:
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        if isinstance(thread_extent, int):
            return thread_extent
        return self._configured_block_size_int(thread_extent)

    def cute_tile_base_expr(self, block_id: int) -> str | None:
        """Uniform expression of the tile's first index along ``block_id``.

        The per-element index is ``base + <thread / lane partition>``, so
        ``index - base`` is the block-local position of the current element.
        """
        if self.mma_mode or block_id not in self.block_ids:
            return None
        return self.offset_var(block_id)

    def cute_lane_axis(self, block_id: int) -> CuteLaneAxis | None:
        """Static thread / lane distribution of ``block_id``, or ``None``.

        ``None`` when the axis has no plain per-thread partition: MMA mode, an
        inactive or cluster-split block, a dynamic extent, or a thread count
        that does not tile the extent exactly.
        """
        if (
            self.mma_mode
            or block_id not in self.block_ids
            or block_id in self.inactive_block_ids
            or block_id in self._cute_cluster_by_block
        ):
            return None
        block_size = self.block_size[self.block_ids.index(block_id)]
        extent = self._configured_block_size_int(block_size)
        threads = self._static_thread_extent_for_block(block_id, block_size)
        if extent is None or threads is None or threads <= 0:
            return None
        elements_per_thread = self._elements_per_thread_for_block(block_id)
        if threads * elements_per_thread != extent:
            return None
        lane_var = self._lane_var_by_block.get(block_id)
        vec_lane_var = self._cute_vec_lane_var_by_block.get(block_id)
        vec_width = (
            self._cute_lane_vec_width_by_block.get(block_id, 1)
            if vec_lane_var is not None
            else 1
        )
        if lane_var is None:
            if elements_per_thread != 1:
                return None
            lane_steps = 1
        elif elements_per_thread % vec_width:
            return None
        else:
            lane_steps = elements_per_thread // vec_width
        return CuteLaneAxis(
            extent=extent,
            threads=threads,
            lane_var=lane_var,
            lane_steps=lane_steps,
            vec_lane_var=vec_lane_var,
            vec_width=vec_width,
            strided=(
                lane_var is not None
                and threads > 1
                and self._cute_lane_layout_by_block.get(block_id, "blocked")
                == "strided"
            ),
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            thread_extent = self._thread_extent_for_axis(
                block_id, block_size_by_id[block_id]
            )
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                static_extent = thread_extent
                if not isinstance(static_extent, int):
                    static_extent = self._configured_block_size_int(static_extent)
                if isinstance(static_extent, int):
                    sizes.append(static_extent)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, bs):
                continue
            thread_extent = self._thread_extent_for_axis(block_id, bs)
            if isinstance(thread_extent, int):
                exprs.append(str(thread_extent))
                continue
            if not isinstance(bs, torch.SymInt):
                return []
            bs_var = self.block_size_var(block_id)
            if bs_var is None:
                return []
            elements_per_thread = self._elements_per_thread_for_block(block_id)
            if elements_per_thread == 1:
                exprs.append(bs_var)
            else:
                exprs.append(f"({bs_var}) // {elements_per_thread}")
        return exprs

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        self._demote_blocks_beyond_thread_axes(state)
        if not self._lane_var_by_block and not self._shared_thread_extents:
            return super().codegen_grid(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        lane_setup_statements: list[ast.AST] = []
        outer_setup_statements: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — rename our lifted step var to
                # avoid the same UnboundLocalError that drove the offset rename.
                step_prefix = "tile_step" if env.backend.name == "cute" else "step"
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")

            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            static_extent: int | None = None
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
                else:
                    tracker.record_symbolic_axis([block_idx], axis)
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                vec_width = self._cute_lane_vec_width_by_block.get(block_idx, 1)
                lane_strided = (
                    self._cute_lane_layout_by_block.get(block_idx, "blocked")
                    == "strided"
                    and uses_thread_axis
                    and isinstance(static_extent, int)
                )
                if (
                    vec_width > 1
                    and uses_thread_axis
                    and elements_per_thread % vec_width == 0
                    # A matmul fallback may suppress root lane loops AFTER
                    # loads already hoisted into the wrapper (cute_mma's
                    # request_root_lane_loop_suppression), which would strand
                    # the hoists; keep grid vec to matmul-free kernels.
                    and (
                        not env.config_spec.matmul_facts
                        or set(block_ids).issubset(
                            env.config_spec.cute_pointwise_region_block_ids
                        )
                    )
                    # Epilogue subtiling stages stores through smem with a
                    # sync_threads INSIDE the per-element pipeline, which the
                    # vec store-collection protocol silently corrupts (50%
                    # wrong output measured); keep such configs on the
                    # (correct) scalar form.
                    and not _cute_epilogue_subtile_active(self.fn.config)
                ):
                    # Same outer x constexpr-V lane partition as
                    # ``codegen_device_loop`` (see the comments there), so
                    # the memory_ops vec-load/store hoist protocol applies
                    # to GRID lane loops too.  The loops themselves are
                    # materialized later by ``DeviceGridState.wrap_body``
                    # from the pre-built ``VecLaneWrapper``.
                    vec_lane_var = self.fn.new_var(f"vec_lane_{block_idx}", dce=False)
                    self._cute_vec_lane_var_by_block[block_idx] = vec_lane_var
                    inner_for = cast(
                        "ast.For",
                        ast.parse(
                            f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                            f"    pass"
                        ).body[0],
                    )
                    lane_body: list[ast.AST] = [inner_for]
                    self._cute_lane_body_by_block[block_idx] = lane_body
                    self._cute_lane_vloop_by_block[block_idx] = inner_for
                    base_index_var = self.fn.new_var(
                        f"lane_base_{block_idx}", dce=False
                    )
                    self._cute_lane_base_index_var_by_block[block_idx] = base_index_var
                    if isinstance(static_extent, int) and env.backend_name == "cute":
                        self.fn.cute_state.grid_thread_extents[base_index_var] = (
                            axis,
                            static_extent,
                        )
                    if lane_strided:
                        # ``base = offset + (outer*NT + tid) * V``
                        base_expr = (
                            f"{offset_var} + "
                            f"({env.backend.lane_offset_expr(lane_var)} "
                            f"* {static_extent} + "
                            f"{env.backend.thread_index_expr(axis=axis)}) "
                            f"* {vec_width}"
                        )
                    else:
                        # ``base = offset + tid*EPT + outer*V``
                        base_expr = (
                            f"{idx_expr} + "
                            f"{env.backend.lane_offset_expr(lane_var)} "
                            f"* {vec_width}"
                        )
                    lane_body.insert(
                        0,
                        statement_from_string(f"{base_index_var} = {base_expr}"),
                    )
                    outer_for = _create_lane_loop(
                        lane_var, elements_per_thread // vec_width, lane_body
                    )
                    vec_wrappers[lane_var] = VecLaneWrapper(
                        outer_for=outer_for,
                        vloop=inner_for,
                        vec_lane_var=vec_lane_var,
                        base_index_var=base_index_var,
                    )
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                else:
                    if (
                        lane_strided
                        and not env.config_spec.matmul_facts
                        and not _cute_epilogue_subtile_active(self.fn.config)
                    ):
                        # Collective matmul/subtile lowerings may replace the
                        # lane body later; retain their established mapping.
                        idx_expr = (
                            f"{offset_var} + {env.backend.thread_index_expr(axis=axis)}"
                            f" + {env.backend.lane_offset_expr(lane_var)} * {static_extent}"
                        )
                    else:
                        idx_expr = (
                            f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
                        )
                target = lane_setup_statements
            else:
                # Setup that does not depend on a lane variable can be hoisted
                # out of the lane loops. This avoids reassignments inside the
                # lane-loop body that confuse the CuTe DSL preprocessor when
                # its internal negative-step machinery emits identifiers like
                # ``offset_<n>`` that collide with helion's tile offsets.
                target = outer_setup_statements
            target.append(statement_from_string(f"{index_var} = {idx_expr}"))

            if (
                isinstance(static_extent, int)
                and env.backend_name == "cute"
                and not env.config_spec.matmul_facts
                and not _cute_epilogue_subtile_active(self.fn.config)
            ):
                self.fn.cute_state.grid_thread_extents[index_var] = (
                    axis,
                    static_extent,
                )

            # Bound by the *thread* extent rather than the block size: this
            # axis advances ``elements_per_thread`` per thread, so the threads
            # belonging to the tile number block_size / that stride.
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if isinstance(static_extent, int) else None,
                block_size_var=(
                    str(self.thread_extent_for_masking(block_idx, static_extent))
                    if isinstance(static_extent, int)
                    else None
                ),
            )
            if mask_statement is not None:
                target.append(mask_statement)
            wrapper = vec_wrappers.get(self._lane_var_by_block.get(block_idx, ""))
            if wrapper is not None and env.backend_name == "cute":
                from .cute.device_state import CuteVloopWrapperFact

                vec_width = self._cute_lane_vec_width_by_block.get(block_idx, 1)
                self.fn.cute_state.vloop_sink_wrappers[wrapper.vec_lane_var] = (
                    CuteVloopWrapperFact(
                        block_id=block_idx,
                        vec_width=vec_width,
                        lane_var=self._lane_var_by_block[block_idx],
                        vec_lane_var=wrapper.vec_lane_var,
                        base_index_var=wrapper.base_index_var,
                        index_var=index_var,
                        mask_var=self.mask_vars.get(block_idx),
                        # ``base = begin + pid * block + ... * V`` is V-aligned
                        # only from a zero origin, and a V-wide chunk cannot
                        # straddle an extent that is a multiple of V.
                        uniform_vector_mask=isinstance(begin, int)
                        and begin == 0
                        and env.specialized_multiple(numel, vec_width),
                    )
                )
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = [
            (
                self._lane_var_by_block[block_id],
                self._elements_per_thread_for_block(block_id),
            )
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_loop_block_ids={
                lane_var: frozenset(
                    block_id
                    for block_id, candidate in self._lane_var_by_block.items()
                    if candidate == lane_var
                )
                for lane_var, _extent in lane_loops
            },
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset(
                mask
                for block_id in block_ids
                if (mask := self.mask_vars.get(block_id)) is not None
            ),
            outer_prefix=outer_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            vec_lane_wrappers=vec_wrappers,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        self._demote_blocks_beyond_thread_axes(state)
        if (
            not self._lane_var_by_block
            and not self._shared_thread_extents
            and not self.mma_mode
        ):
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        self._maybe_apply_cute_cluster(env, state)
        dtype = env.index_type()
        block_sizes = self.block_size
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        # Capture per-block (lane_var, full_extent, vec_width).  When
        # vec_width > 1, the outer lane runs (full_extent // vec_width)
        # iters; the inner constexpr-V loop handles V elements per outer
        # iter.  The memory_ops vec-load dispatcher splices a single
        # ``cute.arch.load(..., V)`` between the outer lane setup and the
        # inner constexpr loop, so per-thread bytes-per-load grow from
        # ``sizeof(dtype)`` to ``V * sizeof(dtype)`` (LDG.64 / LDG.128).
        lane_loops_meta: list[tuple[int, str, int, int]] = []
        for block_id in (self.block_ids[i] for i in self.loop_order):
            if block_id not in self._lane_var_by_block:
                continue
            lane_var = self._lane_var_by_block[block_id]
            extent = self._elements_per_thread_for_block(block_id)
            vec_width = self._cute_lane_vec_width_by_block.get(block_id, 1)
            lane_loops_meta.append((block_id, lane_var, extent, vec_width))
        outer_for_by_block: dict[int, ast.For] = {}
        for block_id, lane_var, extent, vec_width in reversed(lane_loops_meta):
            if vec_width > 1 and extent > 0 and extent % vec_width == 0:
                # Partition the lane loop into outer x inner constexpr V.
                # The inner constexpr-V loop's body re-runs the user body
                # for each of the V lanes (the user body's per-lane
                # ``index_var = ...`` setup keys off the COMPOSITE lane =
                # outer*V + inner so the per-element index is correct).
                vec_lane_var = self.fn.new_var(f"vec_lane_{block_id}", dce=False)
                self._cute_vec_lane_var_by_block[block_id] = vec_lane_var
                inner_for = cast(
                    "ast.For",
                    ast.parse(
                        f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                        f"    pass"
                    ).body[0],
                )
                inner_for.body = body  # type: ignore[assignment]
                # ``lane_body`` is what's INSIDE the outer lane loop:
                # statements above the constexpr V-loop, plus the loop
                # itself (last entry).  memory_ops.py splices a hoisted
                # ``cute.arch.load(..., V)`` into ``lane_body[-1:]`` via
                # the same protocol ``LoopedReductionStrategy`` uses.
                lane_body: list[ast.AST] = [inner_for]
                self._cute_lane_body_by_block[block_id] = lane_body
                self._cute_lane_vloop_by_block[block_id] = inner_for
                outer_extent = extent // vec_width
                # Always emit the outer lane loop even when ``outer_extent
                # == 1`` (i.e. EPT == V): the lane-base index expression
                # references ``lane_var``, which must be defined in scope.
                # The CuTe DSL constant-folds the 1-iter loop away.
                outer_for = _create_lane_loop(lane_var, outer_extent, lane_body)
                outer_for_by_block[block_id] = outer_for
                body = [outer_for]
            else:
                lane_for = _create_lane_loop(lane_var, extent, body)
                body = [lane_for]
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        index_setup: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            static_extent: int | None = None
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
                else:
                    tracker.record_symbolic_axis([block_idx], axis)
            else:
                idx_expr = offset_var
            block_vec_width = self._cute_lane_vec_width_by_block.get(block_idx, 1)
            vec_lane_var = self._cute_vec_lane_var_by_block.get(block_idx)
            # Strided lane layout: thread ``t`` owns V-wide chunks strided by
            # the thread count instead of one contiguous EPT-element span, so
            # each warp instruction touches consecutive chunks (coalesced).
            lane_strided = (
                self._cute_lane_layout_by_block.get(block_idx, "blocked") == "strided"
                and uses_thread_axis
                and isinstance(static_extent, int)
            )
            # Cluster split: each of the ``cluster_n`` CTAs in the cluster
            # owns a contiguous ``block/cluster_n`` slice of the tile (the
            # launch adds a grid dim of ``cluster_n`` CTAs per outer index).
            # The slice offset folds into the tile offset; the per-CTA lane
            # layout applies within the slice unchanged
            # (``_elements_per_thread_for_block`` already divides by the
            # cluster width).
            lane_cluster_n = self._cute_cluster_by_block.get(block_idx, 1)
            if lane_cluster_n > 1:
                assert uses_thread_axis and isinstance(static_extent, int)
                cluster_block = self._configured_block_size_int(block_size)
                assert isinstance(cluster_block, int)
                slice_len = cluster_block // lane_cluster_n
                offset_var = (
                    f"({offset_var} + "
                    f"{env.backend.program_id_expr(1, index_dtype=dtype)}"
                    f" * {slice_len})"
                )
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
            if lane_var := self._lane_var_by_block.get(block_idx):
                if block_vec_width > 1 and vec_lane_var is not None:
                    # Composite per-element lane index = outer*V + inner.
                    # outer (``lane_var``) ranges [0, EPT/V); inner
                    # (``vec_lane_var``) ranges [0, V).  Per-thread base
                    # (the start of the V-wide chunk this thread owns
                    # for this outer iter) is stashed in
                    # ``_cute_lane_base_index_var_by_block`` so the vec
                    # load can use it directly (mirrors the
                    # ``LoopedReductionStrategy`` unroll path).
                    base_index_var = self.fn.new_var(
                        f"lane_base_{block_idx}", dce=False
                    )
                    self._cute_lane_base_index_var_by_block[block_idx] = base_index_var
                    # ``base = offset + tid*EPT + outer*V``  (per-thread
                    # V-aligned base) — emitted INSIDE the outer lane
                    # loop's body (above the constexpr V-loop) so a
                    # single ``cute.arch.load(..., V)`` can be hoisted
                    # at the same level by memory_ops.
                    lane_body_list = self._cute_lane_body_by_block.get(block_idx)
                    if lane_body_list is not None:
                        if lane_strided:
                            # ``base = offset + (outer*NT + tid) * V``
                            base_expr = (
                                f"{offset_var} + "
                                f"({env.backend.lane_offset_expr(lane_var)} "
                                f"* {static_extent} + "
                                f"{env.backend.thread_index_expr(axis=axis)}) "
                                f"* {block_vec_width}"
                            )
                        else:
                            base_expr = (
                                f"{idx_expr} + "
                                f"{env.backend.lane_offset_expr(lane_var)} "
                                f"* {block_vec_width}"
                            )
                        lane_body_list.insert(
                            0,
                            statement_from_string(f"{base_index_var} = {base_expr}"),
                        )
                        vec_wrappers[lane_var] = VecLaneWrapper(
                            outer_for=outer_for_by_block[block_idx],
                            vloop=self._cute_lane_vloop_by_block[block_idx],
                            vec_lane_var=vec_lane_var,
                            base_index_var=base_index_var,
                        )
                    # The user-body's per-element index uses the base +
                    # the inner constexpr-V var so the existing scalar
                    # pipeline (mask + cast + reduce-or-store) keeps
                    # working unchanged.
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                elif lane_strided:
                    # ``idx = offset + tid + lane * NT``
                    idx_expr = (
                        f"{offset_var} + {env.backend.thread_index_expr(axis=axis)}"
                        f" + {env.backend.lane_offset_expr(lane_var)}"
                        f" * {static_extent}"
                    )
                else:
                    idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            # Same thread-extent bound as the grid lane path: a device loop is
            # equally able to be launched wider than the tile it indexes.
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if isinstance(static_extent, int) else None,
                block_size_var=(
                    str(self.thread_extent_for_masking(block_idx, static_extent))
                    if isinstance(static_extent, int)
                    else None
                ),
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        device_loop = DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_loops=[
                (lane_var, extent)
                for _block_id, lane_var, extent, _vec_width in lane_loops_meta
            ],
            lane_loop_block_ids={
                lane_var: frozenset({block_id})
                for block_id, lane_var, _extent, _vec_width in lane_loops_meta
            },
            vec_lane_wrappers=vec_wrappers,
            lane_setup_statements=index_setup,
            tile_masks=frozenset(
                mask
                for block_id in block_ids
                if (mask := self.mask_vars.get(block_id)) is not None
            ),
        )
        if self._cute_lane_vloop_by_block:
            device_loop.body_finalizers.append(self._finalize_cute_tile_vec_stores)
        return device_loop

    def _finalize_cute_tile_vec_stores(self) -> None:
        """Restore tile-vector stores a later access of their tensor would observe.

        A device-loop store site is collected while its V-loop body is still
        being built, so the statements that follow it are only known once the
        loop body is complete.
        """
        from .cute.memory_ops import demote_reordered_tile_vec_stores

        for block_id, vloop in self._cute_lane_vloop_by_block.items():
            demote_reordered_tile_vec_stores(self, self.fn, block_id, vloop.body)

    def supports_index_rank_expansion(self) -> bool:
        return False


class PerThreadFlattenedTileStrategy(FlattenedTileStrategy):
    """Flattened tiling with one scalar index per thread.

    The flattened counterpart of :class:`PerThreadNDTileStrategy`: a single
    ``num_threads`` splits the flattened tile, and a lane loop covers the rest
    when it is narrower than the block size.
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        num_threads: int = 0,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self._num_threads = num_threads
        self._lane_var: str | None = None
        if num_threads > 0 and isinstance(block_size, int) and num_threads < block_size:
            self._lane_var = self.new_var("lane", dce=False)
        # Vec-partition state for the memory_ops ``tile_unroll`` hoist
        # protocol (same shape as ``PerThreadNDTileStrategy``).  The vec
        # slot lives on the LAST (stride-1) block; for the multi-block
        # (flattened N-D) form ``_cute_flat_multi`` is set and the hoist
        # emits FLAT base pointers (``t.iterator + lane_base``) — valid
        # only for contiguous tensors covering the whole iteration space,
        # which ``_cute_vector_load_ctx`` gates per tensor.
        self._cute_lane_layout: str = "blocked"
        self._cute_flat_multi: bool = len(block_ids) > 1
        # ``pid * BLOCK`` of the lane-looped grid: the flattened index var
        # aliases its offset var, so the tile base is recorded separately.
        self._cute_tile_base_expr: str | None = None
        self._cute_lane_vec_width_by_block: dict[int, int] = {}
        self._cute_vec_lane_var_by_block: dict[int, str] = {}
        self._cute_lane_base_index_var_by_block: dict[int, str] = {}
        self._cute_lane_body_by_block: dict[int, list] = {}
        self._cute_lane_vloop_by_block: dict[int, ast.For] = {}
        self._cute_lane_vec_stores_by_block: dict[int, list[CuteTileVecStoreSite]] = {}
        self._cute_lane_vec_loads_by_block: dict[int, dict] = {}
        if self._lane_var is not None:
            env = CompileEnvironment.current()
            block_id = block_ids[-1]
            cfg = fn.config.config
            if block_id in env.config_spec.cute_lane_layouts.valid_block_ids():
                layout = env.config_spec.cute_lane_layouts.config_get(
                    cast("list[str]", cfg.get("cute_lane_layouts", []) or []),
                    block_id,
                    "blocked",
                )
                if isinstance(layout, str):
                    self._cute_lane_layout = layout
            if block_id in env.config_spec.cute_vector_widths.valid_block_ids():
                vec_width = env.config_spec.cute_vector_widths.config_get(
                    cast("list[int]", cfg.get("cute_vector_widths", []) or []),
                    block_id,
                    1,
                )
                assert isinstance(block_size, int)
                elements_per_thread = block_size // num_threads
                if (
                    isinstance(vec_width, int)
                    and vec_width > 1
                    and elements_per_thread % vec_width == 0
                ):
                    self._cute_lane_vec_width_by_block[block_id] = vec_width

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        # Shared with the tile_unroll hoist protocol (see the identically
        # named method on PerThreadNDTileStrategy); the flattened block size
        # is already a plain int whenever the lane path is active.
        return block_size if isinstance(block_size, int) else None

    @property
    def _elements_per_thread(self) -> int:
        """Elements per thread (derived from num_threads and block_size)."""
        if self._num_threads == 0:
            return 1
        assert isinstance(self.block_size, int)
        return self.block_size // self._num_threads

    def _thread_extent(self) -> SymIntLike:
        if self._num_threads == 0:
            return self.block_size
        backend_name = CompileEnvironment.current().backend.name
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                backend_name,
                f"num_threads requires static flattened block sizes for {backend_name}",
            )
        if self.block_size % self._num_threads != 0:
            raise exc.BackendUnsupported(
                backend_name,
                (
                    f"block size must be divisible by num_threads for "
                    f"{backend_name}: "
                    f"{self.block_size} is not divisible by {self._num_threads}"
                ),
            )
        return self._num_threads

    def cute_tile_base_expr(self, block_id: int) -> str | None:
        """Uniform ``pid * BLOCK`` of a single lane-looped flattened block.

        ``offset_var`` is the per-element index here (``indices = offsets``),
        so the tile base is recorded by ``codegen_grid`` instead.  ``None``
        for the multi-block form and for grids without a lane loop.
        """
        if self.block_ids != [block_id]:
            return None
        return self._cute_tile_base_expr

    def tile_begin_var(self, block_idx: int) -> str:
        # ``offset_var`` is the per-element index on this strategy, so
        # ``tile.begin`` must not fall back to it (it would broadcast
        # ``w[tile.begin]`` per element). A single lane-looped block records
        # its uniform ``pid * BLOCK`` base; otherwise (multi-block flattened
        # tiles, grids without a lane loop) derive the tile start from the
        # per-element index: tiles are aligned to their static block size, so
        # ``index - index % BLOCK`` is the same value for every element of the
        # tile even though it is rendered per element.
        base = self.cute_tile_base_expr(block_idx)
        if base is not None:
            return base
        if self.block_ids != [block_idx]:
            # A flattened multi-block tile is a contiguous range of the
            # flattened iteration space, not a rectangle, so its
            # per-dimension begin is the per-element coordinate (as on the
            # Triton flattened strategy).
            return self.offset_var(block_idx)
        extent = self._configured_block_size_int(self.block_size)
        if extent is None:
            raise exc.BackendUnsupported(
                "cute",
                "tile.begin / tile.end / tile.id of a flattened per-thread tile "
                "needs a static block size",
            )
        index = self.index_var(block_idx)
        if extent == 1:
            return index
        return f"(({index}) - (({index}) % {extent}))"

    def cute_lane_axis(self, block_id: int) -> CuteLaneAxis | None:
        """Static thread / lane distribution of a single flattened block.

        A flattened multi-block tile walks one merged index whose lanes do not
        follow any single block axis, so only the one-block form qualifies.
        """
        if self.block_ids != [block_id] or not isinstance(self.block_size, int):
            return None
        extent = self.block_size
        threads = self._num_threads if self._num_threads > 0 else extent
        elements_per_thread = self._elements_per_thread
        if threads <= 0 or threads * elements_per_thread != extent:
            return None
        lane_var = self._lane_var
        vec_lane_var = self._cute_vec_lane_var_by_block.get(block_id)
        vec_width = (
            self._cute_lane_vec_width_by_block.get(block_id, 1)
            if vec_lane_var is not None
            else 1
        )
        if lane_var is None:
            if elements_per_thread != 1:
                return None
            lane_steps = 1
        elif elements_per_thread % vec_width:
            return None
        else:
            lane_steps = elements_per_thread // vec_width
        return CuteLaneAxis(
            extent=extent,
            threads=threads,
            lane_var=lane_var,
            lane_steps=lane_steps,
            vec_lane_var=vec_lane_var,
            vec_width=vec_width,
            strided=(
                lane_var is not None
                and threads > 1
                and self._cute_lane_layout == "strided"
            ),
        )

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if not isinstance(thread_extent, int):
            return []
        return [thread_extent]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if isinstance(thread_extent, int):
            return [str(thread_extent)]
        if not isinstance(self.block_size, torch.SymInt):
            return []
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        if self._num_threads == 0:
            return [bs_var]
        return [f"({bs_var}) // {self._elements_per_thread}"]

    def _uses_thread_axis(self) -> bool:
        thread_extent = self._thread_extent()
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if self._lane_var is None:
            return super().codegen_grid(state)

        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        axis = self._flat_thread_axis()
        thread_extent = self._thread_extent()
        vec_width = self._cute_lane_vec_width_by_block.get(block_ids[-1], 1)
        lane_strided = self._cute_lane_layout == "strided" and isinstance(
            thread_extent, int
        )

        if vec_width > 1 and (
            (
                env.config_spec.matmul_facts
                and not set(block_ids).issubset(
                    env.config_spec.cute_pointwise_region_block_ids
                )
            )
            or _cute_epilogue_subtile_active(self.fn.config)
        ):
            # See the matmul-fallback lane-loop-suppression and the
            # epilogue-subtile smem-staging notes in
            # ``PerThreadNDTileStrategy.codegen_grid``.
            vec_width = 1
        if vec_width > 1:
            # Same outer x constexpr-V lane partition as
            # ``PerThreadNDTileStrategy.codegen_grid`` so the memory_ops
            # tile_unroll vec-hoist protocol applies to flattened grid
            # tiles too (the multi-block form via flat base pointers).
            block_id = block_ids[-1]
            vec_lane_var = self.new_var("vec_lane", dce=False)
            self._cute_vec_lane_var_by_block[block_id] = vec_lane_var
            inner_for = cast(
                "ast.For",
                ast.parse(
                    f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                    f"    pass"
                ).body[0],
            )
            lane_body: list[ast.AST] = [inner_for]
            self._cute_lane_body_by_block[block_id] = lane_body
            self._cute_lane_vloop_by_block[block_id] = inner_for
            base_index_var = self.new_var("lane_base", dce=False)
            self._cute_lane_base_index_var_by_block[block_id] = base_index_var
            if lane_strided:
                # ``base = pid*BS + (outer*NT + tid) * V``
                base_expr = (
                    f"{offsets_base_var} + "
                    f"({env.backend.lane_offset_expr(self._lane_var)} "
                    f"* {thread_extent} + "
                    f"{env.backend.thread_index_expr(axis=axis)}) "
                    f"* {vec_width}"
                )
            else:
                # ``base = pid*BS + tid*EPT + outer*V``
                base_expr = (
                    f"{offsets_base_var} + "
                    f"{env.backend.lane_offset_expr(self._lane_var)} "
                    f"* {vec_width}"
                )
            lane_body.insert(
                0, statement_from_string(f"{base_index_var} = {base_expr}")
            )
            outer_for = _create_lane_loop(
                self._lane_var, self._elements_per_thread // vec_width, lane_body
            )
            vec_wrappers[self._lane_var] = VecLaneWrapper(
                outer_for=outer_for,
                vloop=inner_for,
                vec_lane_var=vec_lane_var,
                base_index_var=base_index_var,
            )
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {base_index_var} + cutlass.Int32({vec_lane_var})"
                )
            )
        elif lane_strided:
            # ``offsets = pid*BS + tid + lane * NT`` — consecutive threads
            # touch consecutive elements each lane iter (coalesced even
            # without vectorization).
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {offsets_base_var} + "
                    f"{env.backend.lane_offset_expr(self._lane_var)}"
                    f" * {thread_extent}"
                )
            )
        else:
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
                )
            )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        self._cute_tile_base_expr = f"({pid_var}) * ({block_size_var})"
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        if vec_width > 1 and lane_strided:
            # The strided vec base folds the thread index in itself.
            state.add_statement(
                f"{offsets_base_var} = ({pid_var}) * ({block_size_var})"
            )
        elif lane_strided:
            # ``offsets_base = pid*BS + tid`` (per-lane term adds lane*NT).
            state.add_statement(
                f"{offsets_base_var} = {env.backend.lane_index_expr(f'({pid_var}) * ({block_size_var})', 1, axis=axis)}"
            )
        else:
            state.add_statement(
                f"{offsets_base_var} = {env.backend.lane_index_expr(f'({pid_var}) * ({block_size_var})', self._elements_per_thread, axis=axis)}"
            )
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)
        block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = []
        if self._lane_var is not None:
            lane_loops = [(self._lane_var, self._elements_per_thread)]
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        elif self._uses_thread_axis():
            tracker.record_symbolic_axis(self.block_ids, axis)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self.block_ids) if lane_loops else set(),
            lane_loop_block_ids=(
                {self._lane_var: frozenset(self.block_ids)}
                if self._lane_var is not None
                else {}
            ),
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset() if mask_var is None else frozenset({mask_var}),
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            vec_lane_wrappers=vec_wrappers,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if self._lane_var is None:
            return super().codegen_device_loop(state)

        env = CompileEnvironment.current()
        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        lid = self.new_var("lid")
        end_var = env.backend.cdiv_expr(
            state.sympy_expr(total_numel), block_size_var, is_device=True
        )
        axis = self._flat_thread_axis()
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        user_body[:0] = lane_setup_statements
        if self._lane_var is not None:
            lane_for = _create_lane_loop(
                self._lane_var,
                self._elements_per_thread,
                body,
            )
            body = [lane_for]
        body[:0] = [
            statement_from_string(
                f"{offsets_base_var} = {env.backend.lane_index_expr(f'{lid} * ({block_size_var})', self._elements_per_thread, axis=axis)}"
            )
        ]
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        elif self._uses_thread_axis():
            tracker.record_symbolic_axis(self.block_ids, axis)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loops=[(self._lane_var, self._elements_per_thread)],
            lane_loop_block_ids={self._lane_var: frozenset(self.block_ids)},
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset() if mask_var is None else frozenset({mask_var}),
        )

    def offset_var(self, block_idx: int) -> str:
        return self._offsets_var

    def supports_index_rank_expansion(self) -> bool:
        return False


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
