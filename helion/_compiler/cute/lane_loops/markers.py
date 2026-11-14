"""Lane-loop forms and the ``_helion_lane_reduce`` marker protocol (CuTe).

Role: general-purpose emitter (the lane-loop forms) and the analysis that
recognizes them.
A CuTe lane loop (``for lane in range(N)`` carrying ``HELION_LANE_LOOP_VAR_ATTR``)
iterates the per-thread elements of a lane-distributed block; the reduction
strategies emit a ``_helion_lane_reduce(...)`` marker inside it and the lane
split (``split.py``) lowers the marker into the two-pass accumulate /
finalize / consume structure.  This module owns the loop constructors, the
marker expression and its parser (``_LaneReduceMarker``), the finalize
emitters (warp / grouped / two-stage combines) and the validators that reject
an owned marker without a proved lowering.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from .... import exc
from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..reduce_call import ReduceCall
from ..reduce_call import ReduceScope

if TYPE_CHECKING:
    from collections.abc import Sequence


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
HELION_VECTOR_REDUCTION_OWNER_ATTR = "_helion_vector_reduction_owner"
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
    broadcast_owner: bool = False,
    matmul_contribution: bool = False,
    strided_restore: bool = False,
    shared_lane_expr: str = "",
) -> str:
    """The marker call text of a lane reduction (see :class:`_LaneReduceMarker`
    for the fields); ``_LaneReduceMarker.to_call`` renders it."""
    if matmul_contribution:
        assert owner_lane is not None and reduction_type == "sum"
    if strided_restore or shared_lane_expr or broadcast_owner:
        assert owner_lane is not None
    return _LaneReduceMarker(
        result_var="",
        input_name=input_name,
        reduction_type=reduction_type,
        identity_expr=identity_expr,
        threads_in_group=threads_in_group,
        wrap_template="",
        group_pre=group_pre,
        group_span=group_span,
        group_lane_expr=group_lane_expr,
        group_count=group_count,
        group_cluster_n=group_cluster_n,
        owner_lane=owner_lane,
        broadcast_owner=broadcast_owner,
        matmul_contribution=matmul_contribution,
        strided_restore=strided_restore,
        shared_lane_expr=shared_lane_expr,
    ).to_call()


@dataclasses.dataclass
class _LaneReduceMarker:
    """A ``_helion_lane_reduce`` marker: the cross-lane reduction a lane loop
    body owes once the lane split decides how its threads combine.

    The marker is a call the emitters write into the body
    (:meth:`to_call`) and the lane passes read back (:meth:`from_call`):
    the input, op and identity are positional expressions, every other field
    a keyword (defaults omitted).  ``result_var`` and ``wrap_template``
    describe the assignment the call sits in.  The marker is stripped before
    the kernel is emitted; ``restore_unprocessed_lane_reduce_markers``
    rejects any owned marker that survives.
    """

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
    # An explicitly replicated input needs every synthetic lane even when
    # its value never reads the coordinate. Other owners still need their
    # ordinary data dependency or an independently proved reshape mapping.
    broadcast_owner: bool = False
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

    def to_call(self) -> str:
        """The marker call: ``threads_in_group`` always, the other keyword
        fields only when they differ from their defaults."""
        keywords = [f"threads_in_group={self.threads_in_group}"]
        for field in _MARKER_KEYWORD_FIELDS:
            value = getattr(self, field.name)
            if value != field.default:
                keywords.append(f"{field.name}={value!r}")
        return (
            f"{_HELION_LANE_REDUCE_MARKER}({self.input_name}, "
            f"{self.reduction_type!r}, {self.identity_expr}, {', '.join(keywords)})"
        )

    @classmethod
    def from_call(
        cls, call: ast.Call, *, result_var: str, wrap_template: str
    ) -> _LaneReduceMarker:
        """Decode a :meth:`to_call` text; malformed markers are rejected."""
        if len(call.args) != 3:
            raise exc.BackendUnsupported("cute", "invalid reduction marker")
        input_node, type_node, identity_node = call.args
        values: dict[str, object] = {}
        for keyword in call.keywords:
            name = keyword.arg
            if (
                name is None
                or name in values
                or not (name == "threads_in_group" or name in _MARKER_KEYWORD_NAMES)
            ):
                raise exc.BackendUnsupported("cute", "invalid reduction marker")
            values[name] = ast.literal_eval(keyword.value)
        threads_in_group = values.pop("threads_in_group", None)
        if type(threads_in_group) is not int:
            raise exc.BackendUnsupported("cute", "invalid reduction marker")
        reduction_type = ast.literal_eval(type_node)
        marker = cls(
            result_var=result_var,
            input_name=ast.unparse(input_node),
            reduction_type=reduction_type,
            identity_expr=ast.unparse(identity_node),
            threads_in_group=threads_in_group,
            wrap_template=wrap_template,
        )
        for field in _MARKER_KEYWORD_FIELDS:
            value = values.get(field.name, field.default)
            if field.name == "owner_lane":
                if value is not None and (not isinstance(value, str) or not value):
                    raise exc.BackendUnsupported("cute", "invalid reduction lane owner")
            elif type(value) is not type(field.default):
                label = field.name.replace("_", " ")
                raise exc.BackendUnsupported("cute", f"invalid {label} marker")
            setattr(marker, field.name, value)
        if marker.matmul_contribution and (
            marker.owner_lane is None or reduction_type != "sum"
        ):
            raise exc.BackendUnsupported("cute", "invalid matmul contribution marker")
        if marker.strided_restore and marker.owner_lane is None:
            raise exc.BackendUnsupported("cute", "invalid strided restore marker")
        if marker.broadcast_owner and marker.owner_lane is None:
            raise exc.BackendUnsupported("cute", "invalid broadcast owner marker")
        return marker


# The keyword fields of the marker call in their emitted order (every field
# after ``wrap_template`` except the required ``threads_in_group``).
_MARKER_KEYWORD_FIELDS = tuple(
    field
    for field in dataclasses.fields(_LaneReduceMarker)
    if field.name
    not in (
        "result_var",
        "input_name",
        "reduction_type",
        "identity_expr",
        "threads_in_group",
        "wrap_template",
    )
)
_MARKER_KEYWORD_NAMES = frozenset(field.name for field in _MARKER_KEYWORD_FIELDS)


def set_marker_owner(call: ast.Call, owner_lane: str) -> None:
    """Bind ``call`` (a marker) to the serial lane ``owner_lane``."""
    # The replication proof belonged to the original synthetic coordinate.
    # A rebound source lane is kept by its ordinary data dependencies.
    call.keywords = [kw for kw in call.keywords if kw.arg != "broadcast_owner"]
    for keyword in call.keywords:
        if keyword.arg == "owner_lane":
            keyword.value = ast.copy_location(
                ast.Constant(value=owner_lane), keyword.value
            )
            return
    call.keywords.append(ast.keyword(arg="owner_lane", value=ast.Constant(owner_lane)))


def _find_lane_reduce_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == _HELION_LANE_REDUCE_MARKER
        ):
            return sub
    return None


def has_lane_reduce_markers(body: Sequence[ast.AST]) -> bool:
    """Whether any statement of ``body`` may hold a ``_helion_lane_reduce``
    marker: a reference to the marker name anywhere in it.

    The lane-reduction passes rewrite, validate and restore only these
    markers (and never create one), so a body without any is left as is by
    every one of them; each still checks the exact call shape itself.
    """
    return any(
        isinstance(node, ast.Name) and node.id == _HELION_LANE_REDUCE_MARKER
        for statement in body
        for node in ast.walk(statement)
    )


def required_lane_reduce_owners(body: Sequence[ast.AST]) -> set[str]:
    """Lane owners referenced by broadcast_owner-marked reductions."""
    owners: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == _HELION_LANE_REDUCE_MARKER
            ):
                marker = _LaneReduceMarker.from_call(
                    node, result_var="", wrap_template=""
                )
                if marker.owner_lane is not None and marker.broadcast_owner:
                    owners.add(marker.owner_lane)
    return owners


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
    ``_helion_lane_reduce(IN, TYPE, ID, threads_in_group=T, ...)`` marker
    call, return a :class:`_LaneReduceMarker`; otherwise return ``None``.

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
    if call is None:
        return None
    # Build the wrap template by replacing the marker call with a sentinel.
    wrapped = _ReplaceLaneReduceCall().visit(ast.parse(ast.unparse(stmt.value)).body[0])
    assert isinstance(wrapped, ast.Expr)
    return _LaneReduceMarker.from_call(
        call, result_var=target.id, wrap_template=ast.unparse(wrapped.value)
    )


def validate_lane_reduce_owners(body: list[ast.AST]) -> None:
    """Reject markers inside a different serial lane before any rewriting.

    The compiler emitters attach the actual strategy's lane name. Older
    unowned markers remain supported by standalone AST helpers/tests; they are
    not emitted by production lowering. Serial non-lane loops are left to the
    existing interchange proof. Residual owned markers must not be restored to
    incomplete scalar inputs by the final safety net.
    """
    if not any(_find_lane_reduce_call(statement) for statement in body):
        # Without a marker there is no owner to validate; skip the per-node
        # marker parse of the whole body. (The codegen skips the whole
        # lane-reduction pass group without markers; this keeps direct
        # callers cheap too.)
        return

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


def _combine_expr(
    reduction_type: str, acc: str, val: str, dtype_ctor: str | None = None
) -> str:
    """Combine ``val`` into ``acc``; ``dtype_ctor`` is the accumulator's type
    constructor (``_dtype_ctor_from_identity``), when known."""
    if reduction_type == "sum":
        return f"({acc}) + ({val})"
    if reduction_type == "prod":
        return f"({acc}) * ({val})"
    if reduction_type in ("max", "min"):
        from ..math_templates import nan_extremum_expr

        return nan_extremum_expr(
            reduction_type,
            f"({acc})",
            f"({val})",
            float32=dtype_ctor == "cutlass.Float32",
        )
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
    return ReduceCall(
        scope=ReduceScope.WARP,
        op=reduction_type,
        input=acc,
        identity=identity_expr,
        lane_expr=lane_expr,
        pre=pre,
        group_span=group_span,
    ).render()


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
    reduce = ReduceCall(
        scope=ReduceScope.SHARED_TWO_STAGE,
        op=reduction_type,
        input=acc,
        identity=identity_expr,
        lane_expr=lane_var,
        lane_in_group=lane_in_group_var,
        lane_mod_pre=lane_mod_pre_var,
        pre=pre,
        group_span=group_span,
        group_count=group_count,
    )
    return [
        statement_from_string(f"{lane_var} = {lane_expr}"),
        statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}"),
        statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}"),
        statement_from_string(f"{result_acc} = {reduce.render()}"),
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
        from .... import exc

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
            from .... import exc

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
    if threads_in_group > 32:
        from .... import exc

        raise exc.BackendUnsupported(
            "cute",
            "lane reduction spanning multiple physical warps requires a proven "
            "CTA reduction group",
        )
    tg = f", threads_in_group={threads_in_group}"
    if reduction_type == "sum":
        return f"cute.arch.warp_reduction_sum({acc}{tg})"
    if reduction_type == "max":
        return f"cute.arch.warp_reduction_max({acc}, op=_cute_nan_max{tg})"
    if reduction_type == "min":
        return f"cute.arch.warp_reduction(({acc}), _cute_nan_min{tg})"
    if reduction_type == "prod":
        return f"cute.arch.warp_reduction(({acc}), lambda a, b: (a * b){tg})"
    raise NotImplementedError(f"lane warp reduce {reduction_type!r}")


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


def _lane_loop_extent(loop: ast.For) -> int:
    call = loop.iter
    assert isinstance(call, ast.Call)
    assert len(call.args) == 1
    return int(ast.literal_eval(call.args[0]))
