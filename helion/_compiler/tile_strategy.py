from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast
import weakref

import sympy
import torch

from .. import exc
from .._compat import shape_env_size_hint
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .device_function import DeviceFunction
from .host_function import HostFunction
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
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .inductor_lowering import CodegenState

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


def _lane_loop_iter(extent: int) -> ast.AST:
    # CuTe lane loops carry per-thread scalar state. Emitting them via
    # cutlass.range(_constexpr) miscompiles scalar matmul paths, so keep them
    # as ordinary Python loops.
    return expr_from_string(f"range({extent})")


def _create_lane_loop(lane_var: str, extent: int, body: list[ast.AST]) -> ast.For:
    loop = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=_lane_loop_iter(extent),
        body=body,
        orelse=[],
        type_comment=None,
    )
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return loop


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


def _lane_reduce_marker_expr(
    input_name: str,
    reduction_type: str,
    identity_expr: str,
    threads_in_group: int,
    *,
    group_pre: int = 1,
    group_span: int = 0,
    group_lane_expr: str = "",
) -> str:
    # ``group_*`` (optional) carry the parameters of a strided grouped warp
    # reduction. They are required when the reduction's live thread axis is
    # interleaved with an unrelated sibling thread axis on the same warp (a
    # reshape-merged reduction). ``group_lane_expr`` is base64-free but may
    # contain commas/parens, so it is passed as a string literal that the
    # post-pass re-parses.
    return (
        f"{_HELION_LANE_REDUCE_MARKER}({input_name}, {reduction_type!r}, "
        f"{identity_expr}, {threads_in_group}, {group_pre}, {group_span}, "
        f"{group_lane_expr!r})"
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
    # Optional strided grouped warp reduction (reshape-merged reductions whose
    # live thread axis shares a warp with an unrelated sibling axis). When
    # ``group_span`` > 0 the finalize uses ``_cute_grouped_reduce_warp`` with
    # ``group_pre``/``group_span``/``group_lane_expr`` instead of a plain
    # consecutive-lane ``cute.arch.warp_reduction_*``.
    group_pre: int = 1
    group_span: int = 0
    group_lane_expr: str = ""

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
    if call is None or len(call.args) != 7:
        return None
    (
        input_node,
        type_node,
        identity_node,
        threads_node,
        group_pre_node,
        group_span_node,
        group_lane_node,
    ) = call.args
    input_name = ast.unparse(input_node)
    reduction_type = ast.literal_eval(type_node)
    identity_expr = ast.unparse(identity_node)
    threads_in_group = int(ast.literal_eval(threads_node))
    group_pre = int(ast.literal_eval(group_pre_node))
    group_span = int(ast.literal_eval(group_span_node))
    group_lane_expr = ast.literal_eval(group_lane_node)
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
    )


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


def _backward_slice(body: list[ast.AST], roots: set[str]) -> tuple[list[int], set[str]]:
    """Return the indices of the statements in ``body`` that (transitively)
    produce any name in ``roots``, plus the set of all names those statements
    write.  Statements are scanned in reverse so a producer is included once a
    later consumer (already selected) reads its output.
    """
    from .ast_read_writes import ReadWrites

    needed = set(roots)
    selected: list[int] = []
    written: set[str] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        if writes & needed:
            selected.append(idx)
            written |= writes
            needed |= set(rw.reads)
    selected.reverse()
    return selected, written


def split_lane_loop_reductions(body: list[ast.AST]) -> list[ast.AST]:
    """Rewrite single-pass lane loops that contain ``_helion_lane_reduce``
    markers into the two-pass accumulate / finalize / consume structure.

    Operates bottom-up so nested lane loops are handled before their parents.
    Lane loops without markers are returned unchanged (their inner statements
    are still recursed into so nested markers are processed).
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(_split_stmt_lane_reductions(stmt))
    return new_body


def _restore_stmt_lane_reduce_markers(stmt: ast.AST) -> ast.AST:
    """Recurse into statement-list fields and replace any surviving
    ``R = ..._helion_lane_reduce(IN, TYPE, ID, T)...`` assignment with
    ``R = ...IN...`` (the raw per-lane input)."""
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, [_restore_stmt_lane_reduce_markers(s) for s in old])
    if isinstance(stmt, ast.Assign):
        m = _is_lane_reduce_marker_assign(stmt)
        if m is not None:
            return statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
    return stmt


def restore_unprocessed_lane_reduce_markers(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Replace any surviving ``R = ..._helion_lane_reduce(IN, TYPE, ID, T)...``
    assignment with ``R = ...IN...`` (the raw per-lane input).

    A safety net: ``split_lane_loop_reductions`` only rewrites markers it can
    place in a two-pass lane structure. A marker emitted in a context neither
    that pass nor ``interchange_lane_outside_serial_reductions`` handles would
    otherwise leak the ``_helion_lane_reduce`` call into the emitted kernel.
    Reverting to the per-lane input keeps the kernel compilable (it falls back
    to the original single-pass per-lane reduction behavior).

    Recurses only into statement-bearing fields (``body``/``orelse``/
    ``finalbody``) instead of using ``ast.NodeTransformer``; markers are always
    statement-level assignments, so this avoids the transformer's in-place
    mutation of expression list fields, which fails when an AST node carries a
    ``torch.fx`` ``immutable_list`` (e.g. multi-output ``inline_asm_elementwise``).
    """
    return [_restore_stmt_lane_reduce_markers(stmt) for stmt in body]


def _split_stmt_lane_reductions(stmt: ast.AST) -> list[ast.AST]:
    # Recurse into any statement-list-bearing fields first so nested lane
    # loops are rewritten before the enclosing one.
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, split_lane_loop_reductions(old))
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _split_one_lane_loop(stmt, lane_var)


def _split_one_lane_loop(loop: ast.For, lane_var: str) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    markers: list[tuple[int, _LaneReduceMarker]] = []
    for idx, stmt in enumerate(body):
        parsed = _is_lane_reduce_marker_assign(stmt)
        if parsed is not None:
            markers.append((idx, parsed))
    if not markers:
        return [loop]

    # Safety: the two-pass split is only valid when the reduction marker is the
    # ONLY cross-lane carried value in this lane loop. If the body has another
    # loop-carried accumulator across the lanes (e.g. a matmul ``dot_acc`` or a
    # plain ``extra += per_lane`` sum that already accumulates over the lanes),
    # splitting would drop or double-count it. Fall back to the original
    # single-pass per-lane behavior by replacing each marker with its raw input.
    marker_indices = {i for i, _ in markers}
    if _has_extra_cross_lane_carry(body, lane_var, marker_indices):
        return [_restore_per_lane_markers(loop, markers)]

    input_roots = {m.input_name for _, m in markers}

    # Phase 1: the backward slice that produces all reduction inputs.
    phase1_indices, _phase1_written = _backward_slice(body, input_roots)

    # Sequentially-dependent reductions (one marker's input depends on another
    # marker's result, e.g. online softmax's sum-of-exp needing the max first)
    # would require a multi-pass split. The single-pass phase-1/finalize/phase-2
    # structure can't express that, so fall back to per-lane behavior.
    if set(phase1_indices) & marker_indices:
        return [_restore_per_lane_markers(loop, markers)]

    # The phase-1 (accumulate) and phase-2 (consume) passes both re-run the
    # reduction-input producers. That is only safe for side-effect-free
    # producers. A matmul / collective in the slice (cross-thread shared-memory
    # reductions, ``cute.gemm``, ``dot``) cannot be duplicated without racing on
    # shared memory, so fall back to per-lane behavior in that case.
    if any(_contains_unduplicatable_op(body[i]) for i in phase1_indices):
        return [_restore_per_lane_markers(loop, markers)]

    extent = _lane_loop_extent(loop)

    prefix: list[ast.AST] = []  # acc init statements (outside the lane loops)
    accumulate_body: list[ast.AST] = [body[i] for i in phase1_indices]
    finalize: list[ast.AST] = []
    for _, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        prefix.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Cast the per-lane input to the accumulator dtype before combining so
        # the CUTLASS DSL's strict ternary type check (max/min emit a Python
        # ``a if a > b else b``) does not see mixed fp32/bf16 operands.
        ctor = _dtype_ctor_from_identity(m.identity_expr)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        accumulate_body.append(
            statement_from_string(
                f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
            )
        )
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
            reduced = _warp_reduce_expr(m.reduction_type, acc_var, m.threads_in_group)
        else:
            reduced = acc_var
        finalize.append(
            statement_from_string(f"{m.result_var} = {m.finalize_expr(reduced)}")
        )

    # Phase 2: everything except the marker assignments themselves; the
    # reduced scalar is already finalized so consumers read it directly.
    phase2_body = [s for i, s in enumerate(body) if i not in marker_indices]

    # A statement is lane-varying if it (transitively) reads the lane var.
    # Statements that only depend on the finalized scalar(s) are lane-invariant
    # and run once after the lane loops; lane-varying consumers run in a second
    # lane loop, but only those that contribute to a side effect (a store, an
    # if-with-store, an in-place write). Pure lane-varying producers that fed
    # only the (now-removed) reduction markers are dropped.
    lane_varying_names = _lane_varying_names(phase2_body, lane_var)

    def is_lane_varying(stmt: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(stmt).reads)
        return lane_var in reads or bool(reads & lane_varying_names)

    keep_indices = _live_phase2_indices(phase2_body)
    lane_invariant_tail: list[ast.AST] = []
    lane_varying_tail: list[ast.AST] = []
    for i, s in enumerate(phase2_body):
        if is_lane_varying(s):
            if i in keep_indices:
                lane_varying_tail.append(s)
        else:
            lane_invariant_tail.append(s)

    result: list[ast.AST] = []
    result.extend(prefix)
    result.append(_create_lane_loop(lane_var, extent, accumulate_body))
    result.extend(finalize)
    result.extend(lane_invariant_tail)
    if lane_varying_tail:
        result.append(_create_lane_loop(lane_var, extent, lane_varying_tail))
    return result


def _has_extra_cross_lane_carry(
    body: list[ast.AST], lane_var: str, marker_indices: set[int]
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

    A carried value is detected as a name that is *live-in* to the lane body
    (read before it is written within the body, directly or through a
    ``X_copy = X`` alias) and also written within the body.
    """
    from .ast_read_writes import ReadWrites

    written_so_far: set[str] = set()
    aliases: dict[str, str] = {}  # copy_var -> original carried name
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written_so_far:
                live_in.add(name)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            aliases[stmt.targets[0].id] = stmt.value.id
        written_so_far |= set(rw.writes)

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
    lane_varying = _lane_varying_names(body, lane_var)

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
            if root(w) in live_in and root(w) not in marker_tainted:
                return True
    return False


def _restore_per_lane_markers(
    loop: ast.For, markers: list[tuple[int, _LaneReduceMarker]]
) -> ast.For:
    """Replace each ``_helion_lane_reduce`` marker in ``loop`` with its raw
    per-lane input, restoring the original single-pass behavior (used when the
    two-pass split is unsafe)."""
    body = list(loop.body)
    for idx, m in markers:
        body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
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


def _lane_varying_names(body: list[ast.AST], lane_var: str) -> set[str]:
    """Names whose values (transitively) depend on the lane var within ``body``."""
    from .ast_read_writes import ReadWrites

    varying = {lane_var}
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & varying:
                for w in rw.writes:
                    if w not in varying:
                        varying.add(w)
                        changed = True
    varying.discard(lane_var)
    return varying


def _lane_body_live_in(body: list[ast.AST], lane_var: str) -> set[str]:
    """Names read in ``body`` before they are written (live-in), excluding the
    lane var.  A loop-carried accumulator phi is live-in to the lane body."""
    from .ast_read_writes import ReadWrites

    written: set[str] = set()
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written:
                live_in.add(name)
        written |= set(rw.writes)
    return live_in


def _is_serial_for(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` is an ordinary serial ``for`` loop (a device
    serial loop), NOT a per-thread lane loop."""
    return (
        isinstance(stmt, ast.For)
        and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) is None
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
) -> list[ast.AST]:
    """Interchange a ``for LANE: ... for MB: ...`` nest whose inner serial loop
    contains ``_helion_lane_reduce`` markers.

    layer_norm_bwd / rms_norm_bwd compute, inside a serial ``mb`` loop, BOTH a
    per-feature accumulator that must keep the lane loop OUTSIDE the ``mb`` loop
    (``grad_w_acc += ...``) AND a feature reduction whose result is broadcast
    back into a per-row store (``grad_x``), which needs every lane summed *per*
    ``mb`` iteration (lane INSIDE ``mb``). A single lane loop cannot satisfy
    both nestings, so emit two specialized loop nests:

    * Nest B (grad_w): the original ``for LANE: ... for MB: ...`` loop with the
      lane-reduce markers and the reduction-consuming side effects removed —
      keeping only the per-feature accumulators and their stores.
    * Nest A (grad_x): a ``for MB: ... for LANE: ...`` loop carrying only the
      lane reduction and its broadcast consumer. Its inner lane loop still holds
      the markers so the subsequent ``split_lane_loop_reductions`` pass produces
      the per-``mb`` accumulate -> warp-combine -> consume structure.

    Returns ``body`` unchanged when no such pattern is present.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(_interchange_stmt(stmt))
    return new_body


def _interchange_stmt(stmt: ast.AST) -> list[ast.AST]:
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, interchange_lane_outside_serial_reductions(old))
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _interchange_one_lane_loop(stmt, lane_var)


def _interchange_one_lane_loop(loop: ast.For, lane_var: str) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    # Find the single inner serial ``for`` loop that carries lane-reduce markers.
    mb_index: int | None = None
    for idx, stmt in enumerate(body):
        if _is_serial_for(stmt) and any(
            _is_lane_reduce_marker_assign(s) is not None
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
        # only rewrites top-level lane loops), so restore them to their raw
        # per-lane inputs to avoid leaving an unprocessed marker behind.
        restored: list[ast.stmt] = [cast("ast.stmt", s) for s in mb_body]
        for idx, m in markers:
            restored[idx] = statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
        mb_loop.body = restored
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

    extent = _lane_loop_extent(loop)
    inner_lane_loop_a = _create_lane_loop(
        lane_var, extent, [*prefix_varying_a, *mb_varying_a]
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
    # value that Nest A re-stores with the full reduction afterwards.
    restored_mb_body = list(mb_loop.body)
    for idx, m in markers:
        restored_mb_body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
        )
    mb_loop.body = restored_mb_body
    nest_b = loop

    return [nest_b, *nest_a]


# ---------------------------------------------------------------------------
# Chunked-recurrence (GDN) lane-invariant accumulator hoist.
#
# A chunked recurrence such as gdn_fwd_h carries an accumulator ``b_h`` across a
# SERIAL chunk loop and, inside each chunk, contracts a matmul over the
# within-chunk position ``c`` (lowered as an inner lane loop).  ``matmul_fallback``
# emits the running-sum ``dot_acc`` form for this matmul (because the per-chunk
# rescale is lane-invariant), producing inside the lane loop:
#
#       dot_acc_base = <lane-invariant rescale of b_h>      # e.g. b_h * decay
#       dot_acc = dot_acc + <product(c)>                    # accumulate over c
#       b_h = dot_acc_base + dot_acc                        # WRONG: per-lane reassign
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
#       dot_acc_base = <rescale of frozen b_h>
#       for lane:
#           <producers; dot_acc = dot_acc + product(c)>
#       b_h = dot_acc_base + dot_acc          # once per chunk
# ---------------------------------------------------------------------------


def _find_dot_acc_recurrence(
    lane_loop: ast.For, lane_var: str
) -> tuple[str, str, str] | None:
    """If ``lane_loop`` ends with the chunked-recurrence ``dot_acc`` triple,
    return ``(acc_var, dot_acc_base_var, dot_acc_var)``; else ``None``.

    The triple (emitted by ``_emit_cute_matmul``'s lane-invariant ``dot_acc``
    path) is, as the LAST three statements of the lane-loop body:

        dot_acc_base = <expr>             # lane-invariant rescale of acc
        dot_acc = dot_acc + <product>    # running sum over the lane axis
        acc = dot_acc_base + dot_acc     # final combine
    """
    body = lane_loop.body
    if len(body) < 3:
        return None
    base_stmt, sum_stmt, final_stmt = body[-3], body[-2], body[-1]

    def assign_name(stmt: ast.AST) -> str | None:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.targets[0].id
        return None

    base_var = assign_name(base_stmt)
    dot_acc_var = assign_name(sum_stmt)
    acc_var = assign_name(final_stmt)
    if base_var is None or dot_acc_var is None or acc_var is None:
        return None
    if not (base_var.startswith("dot_acc_base") and dot_acc_var.startswith("dot_acc")):
        return None
    # ``dot_acc = dot_acc + product``: a running self-sum over the lane axis.
    assert isinstance(sum_stmt, ast.Assign)
    if not (
        isinstance(sum_stmt.value, ast.BinOp)
        and isinstance(sum_stmt.value.op, ast.Add)
        and isinstance(sum_stmt.value.left, ast.Name)
        and sum_stmt.value.left.id == dot_acc_var
    ):
        return None
    # ``acc = dot_acc_base + dot_acc``: the final per-chunk combine.
    assert isinstance(final_stmt, ast.Assign)
    final_reads = {n.id for n in ast.walk(final_stmt.value) if isinstance(n, ast.Name)}
    if final_reads != {base_var, dot_acc_var}:
        return None
    # The rescale (``dot_acc_base``) must be lane-INVARIANT: it must not read the
    # lane var nor any value derived from it within the lane body.
    lane_body: list[ast.AST] = list(body)
    lane_varying = _lane_varying_names(lane_body, lane_var)
    from .ast_read_writes import ReadWrites

    base_reads = set(ReadWrites.from_ast(base_stmt).reads)
    if lane_var in base_reads or (base_reads & lane_varying):
        return None
    return acc_var, base_var, dot_acc_var


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


def hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Restructure ``for chunk: for lane: <dot_acc recurrence>`` nests so the
    lane-invariant rescale, chunk-entry stores, and final accumulator combine
    run once per chunk (see the module comment above).

    Tightly gated: only fires on a serial chunk loop whose single inner lane
    loop ends with the ``dot_acc`` triple and whose ``dot_acc`` reset sits in
    the same statement list before the chunk loop.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        # Recurse into nested statement-bearing fields first.
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(stmt, field, hoist_lane_invariant_chunk_recurrence(old))

        info = _detect_chunk_recurrence(stmt)
        if info is None:
            new_body.append(stmt)
            continue
        dot_acc_var = info[0]
        reset_idx = _find_reset_assign(new_body, dot_acc_var)
        if reset_idx is None:
            # No relocatable reset found: bail out (leave nest unchanged).
            new_body.append(stmt)
            continue
        reset_stmt = new_body.pop(reset_idx)
        assert isinstance(stmt, ast.For)
        new_body.append(_rewrite_chunk_recurrence(stmt, info, reset_stmt))
    return new_body


def _detect_chunk_recurrence(
    stmt: ast.AST,
) -> tuple[str, int, ast.For, str, str, str, str] | None:
    """Return ``(dot_acc_var, lane_idx, lane_loop, lane_var, acc_var, base_var,
    dot_acc_var)`` when ``stmt`` is a serial chunk loop carrying the ``dot_acc``
    recurrence, else ``None``."""
    if not _is_serial_for(stmt) or not isinstance(stmt, ast.For):
        return None
    found = _single_lane_loop_in_body(list(stmt.body))
    if found is None:
        return None
    lane_idx, lane_loop, lane_var = found
    triple = _find_dot_acc_recurrence(lane_loop, lane_var)
    if triple is None:
        return None
    acc_var, base_var, dot_acc_var = triple
    return dot_acc_var, lane_idx, lane_loop, lane_var, acc_var, base_var, dot_acc_var


def _rewrite_chunk_recurrence(
    stmt: ast.For,
    info: tuple[str, int, ast.For, str, str, str, str],
    reset_stmt: ast.AST,
) -> ast.For:
    """Build the restructured chunk loop (see module comment)."""
    from .ast_read_writes import ReadWrites

    _dot_acc, lane_idx, lane_loop, lane_var, acc_var, _base_var, _dot_acc2 = info
    chunk_body: list[ast.AST] = list(stmt.body)
    lane_body: list[ast.AST] = list(lane_loop.body)
    base_stmt = lane_body[-3]
    sum_stmt = lane_body[-2]
    final_stmt = lane_body[-1]
    producers: list[ast.AST] = lane_body[:-3]

    lane_varying = _lane_varying_names(lane_body, lane_var)

    def reads_lane(s: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(s).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # The "accumulator family": the names that all hold the chunk-ENTRY
    # accumulator value.  Helion may capture the loop-carried phi through a chain
    # of plain copy-aliases (``b_h_copy = b_h``; ``b_h_copy_0 = b_h_copy``) and
    # the rescale / chunk-entry store read those copies, not ``acc_var`` (which
    # is the chunk-EXIT result).  Build the copy-alias closure over the chunk
    # prefix and the lane producers, seeded from the loop-carried phi: a name
    # that is live-in to the lane body (read before written) and feeds the
    # lane-invariant rescale ``base_stmt``.
    chunk_prefix = chunk_body[:lane_idx]
    copy_src: dict[str, str] = {}
    for s in (*chunk_prefix, *producers):
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Name)
        ):
            copy_src[s.targets[0].id] = s.value.id

    def copy_root(name: str) -> str:
        seen: set[str] = set()
        while name in copy_src and name not in seen:
            seen.add(name)
            name = copy_src[name]
        return name

    base_slice_indices, _ = _backward_slice(
        producers, set(ReadWrites.from_ast(base_stmt).reads)
    )
    base_slice_reads = set(ReadWrites.from_ast(base_stmt).reads)
    for i in base_slice_indices:
        base_slice_reads |= set(ReadWrites.from_ast(producers[i]).reads)
    lane_live_in = _lane_body_live_in(producers, lane_var)
    phi_roots = {
        copy_root(name) for name in base_slice_reads if copy_root(name) in lane_live_in
    }
    acc_family = set(phi_roots)
    for name in (*copy_src.keys(),):
        if copy_root(name) in phi_roots:
            acc_family.add(name)

    # Backward slice feeding the lane-invariant rescale ``base_stmt``: those
    # producers are lane-invariant and hoist before the lane loop with it.
    base_seed = set(ReadWrites.from_ast(base_stmt).reads)
    hoist_indices, _ = _backward_slice(producers, base_seed)
    hoist_set = set(hoist_indices)

    # Lane-invariant side-effecting statements whose backward slice reaches the
    # (frozen) chunk-ENTRY accumulator (the store of ``b_h``) must hoist before
    # the lane loop so they run once per chunk on the frozen value; bring their
    # producers along.  A store that does NOT read the accumulator stays in the
    # lane loop (it is a genuinely per-lane side effect).
    entry_indices: list[int] = []
    for idx, prod in enumerate(producers):
        if idx in hoist_set or reads_lane(prod) or not _has_side_effect(prod):
            continue
        slice_indices, _ = _backward_slice(
            producers[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        slice_reads = set(ReadWrites.from_ast(prod).reads)
        for i in slice_indices:
            slice_reads |= set(ReadWrites.from_ast(producers[i]).reads)
        if not (acc_family & slice_reads):
            continue
        # The store + its lane-invariant producer slice all hoist.
        if any(reads_lane(producers[i]) for i in slice_indices):
            continue
        entry_indices.append(idx)
        hoist_set.update(slice_indices)

    hoist_pre = sorted(hoist_set | set(entry_indices))
    pre_lane = [producers[i] for i in hoist_pre]
    lane_kept = [
        producers[i]
        for i in range(len(producers))
        if i not in hoist_set and i not in set(entry_indices)
    ]

    new_lane_loop = _create_lane_loop(
        lane_var, _lane_loop_extent(lane_loop), [*lane_kept, sum_stmt]
    )
    new_chunk_body: list[ast.AST] = [
        *chunk_body[:lane_idx],
        reset_stmt,
        *pre_lane,
        base_stmt,
        new_lane_loop,
        final_stmt,
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

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids


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


@dataclasses.dataclass
class ForiLoopState(DeviceLoopOrGridState):
    """State for fori_loop-based loops on TPU (Pallas backend).

    Uses jax.lax.fori_loop with pltpu.make_async_copy for tensors whose
    inner-block shape passes ``_check_dma_alignment``; tensors that fail
    are kept on their outer BlockSpec and accessed via ``pl.ds`` from the
    body.  Per-tensor pipelining membership lives in ``_tensor_to_dma_scratch``.
    """

    body_fn_name: str
    loop_var_name: str  # The fori_loop index variable (e.g., "_j")
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    _tensor_to_sem: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def add_lane_loop(self, block_id: int, lane_var: str, extent: int) -> None:
        self.lane_loops.append((lane_var, extent))
        self.lane_loop_blocks.add(block_id)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [_create_lane_loop(lane_var, extent, wrapped)]
        return wrapped


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

        if "tensor_descriptor" in config.indexing:
            # Tensor descriptor + multi-stage pipelines in addition to unrolling tend to cause
            # CUDA "misaligned address" or "unspecified launch failure" errors.
            if range_num_stages > 0:
                range_num_stages = 0
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            # Unrolling can cause CUDA IMA with pipelining
            # We want to ensure new step size + pipeline is within bounds
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
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

        has_reduction_strategy = any(
            isinstance(strategy, ReductionStrategy) and strategy.thread_axes_used() > 0
            for strategy in self.fn.tile_strategy.strategies
        )
        if plan is not None and any(
            plan.disables_reduction_axis_reservation(block_id)
            for block_id in self.block_ids
        ):
            return active_non_reduction_axes + active_reduction_axes
        reserved_reduction_axes = max(
            1 if has_reduction_strategy else 0, active_reduction_axes
        )
        return reserved_reduction_axes + active_non_reduction_axes

    def select_pid_strategy(self) -> ProgramIDs:
        backend_name = CompileEnvironment.current().backend.name
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
        if not env.backend.force_tile_mask() and env.known_multiple(
            functools.reduce(
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        ):
            self._mask_var = None
        else:
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
            if not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            ):
                flatten_loops.disable_block_id(block_ids[0])
                continue
            for i, j in itertools.pairwise(block_ids):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    flatten_loops.disable_block_id(block_ids[0])
                    break

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
        track per-block-id state (e.g. ``CuteNDTileStrategy``'s
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
            state.add_statement(f"{index_var} = {idx_expr}")
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
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
            extra_body = [
                statement_from_string(f"{index_var} = {idx_expr}"),
            ]
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
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
    ) -> ast.stmt | None:
        env = CompileEnvironment.current()
        if (
            not env.backend.force_tile_mask()
            and env.block_sizes[block_idx].known_multiple(block_size)
            and not env.is_jagged_tile(block_idx)
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
                return statement_from_string(
                    f"{mask_var} = ({index_var}) < {{parent}}",
                    parent=self._to_ast(jagged_tile_parent),
                )
            k = len(parent_dims)
            child_expand = "[" + ", ".join(["None"] * k + [":"]) + "]"
            parent_expand = "[" + ", ".join([":"] * k + ["None"]) + "]"
            return statement_from_string(
                f"{mask_var} = ({index_var}){child_expand} < {{parent}}{parent_expand}",
                parent=self._to_ast(jagged_tile_parent),
            )

        return statement_from_string(
            f"{mask_var} = ({index_var}) < {{end}}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class CuteNDTileStrategy(NDTileStrategy):
    """CuTe N-D tile strategy using the standard tile pipeline."""

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
        # Shared per-block hoist cache: (tensor_name, base_ptr_expr) ->
        # (hoist_var, dtype).  Same shape as
        # ``LoopedReductionStrategy._cute_lane_vec_loads``.
        self._cute_lane_vec_loads_by_block: dict[int, dict] = {}
        if not mma_mode:
            from ..autotuner.config_spec import CuteVectorWidthSpec

            env_local = CompileEnvironment.current()
            cute_vec_widths_cfg = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
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
                    # Register a cute_vector_widths slot for this lane
                    # block if not already present (rolled reductions
                    # register their own via device_ir).  We add idempotently
                    # so the autotuner can explore V in {1,2,4,8}.
                    if (
                        block_id
                        not in env_local.config_spec.cute_vector_widths.valid_block_ids()
                    ):
                        import contextlib

                        # Silently skip when append fails (e.g.
                        # config_spec frozen) — the lane loop just
                        # runs in scalar mode in that case.
                        with contextlib.suppress(Exception):
                            env_local.config_spec.cute_vector_widths.append(
                                CuteVectorWidthSpec(
                                    block_id=block_id,
                                    size_hint=static_bs,
                                )
                            )
                    elements_per_thread = static_bs // nt
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

    def _elements_per_thread_for_block(self, block_id: int) -> int:
        """Elements per thread for *block_id* (derived from num_threads)."""
        if block_id in self.inactive_block_ids:
            return 1
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return 1
        bs = self._configured_block_size_int(self.block_size[idx])
        assert isinstance(bs, int)  # validated by _thread_extent_for_axis
        return bs // nt

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        if block_id in self.inactive_block_ids:
            return 1
        if self.mma_mode:
            return 1  # MMA handles element distribution, no CUDA threads needed
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return block_size
        resolved_block_size = block_size
        if not isinstance(resolved_block_size, int):
            static_block_size = self._configured_block_size_int(resolved_block_size)
            if static_block_size is None:
                raise exc.BackendUnsupported(
                    "cute",
                    "num_threads requires static ND block sizes for cute",
                )
            resolved_block_size = static_block_size
        if resolved_block_size % nt != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute axis "
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
        if not self._lane_var_by_block:
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
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
                target = lane_setup_statements
            else:
                # Setup that does not depend on a lane variable can be hoisted
                # out of the lane loops. This avoids reassignments inside the
                # lane-loop body that confuse the CuTe DSL preprocessor when
                # its internal negative-step machinery emits identifiers like
                # ``offset_<n>`` that collide with helion's tile offsets.
                target = outer_setup_statements
            target.append(statement_from_string(f"{index_var} = {idx_expr}"))

            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                target.append(mask_statement)
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
            lane_setup_statements=lane_setup_statements,
            outer_prefix=outer_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if not self._lane_var_by_block and not self.mma_mode:
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
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
                outer_extent = extent // vec_width
                # Always emit the outer lane loop even when ``outer_extent
                # == 1`` (i.e. EPT == V): the lane-base index expression
                # references ``lane_var``, which must be defined in scope.
                # The CuTe DSL constant-folds the 1-iter loop away.
                outer_for = _create_lane_loop(lane_var, outer_extent, lane_body)
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
        index_setup: list[ast.stmt] = []
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
                idx_expr = offset_var
            block_vec_width = self._cute_lane_vec_width_by_block.get(block_idx, 1)
            vec_lane_var = self._cute_vec_lane_var_by_block.get(block_idx)
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
                        base_expr = (
                            f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)} "
                            f"* {block_vec_width}"
                        )
                        lane_body_list.insert(
                            0,
                            statement_from_string(f"{base_index_var} = {base_expr}"),
                        )
                    # The user-body's per-element index uses the base +
                    # the inner constexpr-V var so the existing scalar
                    # pipeline (mask + cast + reduce-or-store) keeps
                    # working unchanged.
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                else:
                    idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loop_blocks=set(self._lane_var_by_block),
        )

    def supports_index_rank_expansion(self) -> bool:
        return False


class CuteFlattenedTileStrategy(FlattenedTileStrategy):
    """Flattened CuTe strategy: scalar index per thread over a flattened tile."""

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
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                "cute",
                "num_threads requires static flattened block sizes for cute",
            )
        if self.block_size % self._num_threads != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute: "
                    f"{self.block_size} is not divisible by {self._num_threads}"
                ),
            )
        return self._num_threads

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
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        axis = self._flat_thread_axis()
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
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self.block_ids) if lane_loops else set(),
            lane_setup_statements=lane_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
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
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
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
