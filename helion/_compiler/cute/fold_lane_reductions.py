"""Fold a vectorized lane loop's per-lane reductions across its lanes.

A device loop whose tile axis is wider than ``threads x V`` gives every
thread several packets per trip, walked by a synthetic lane loop
(``PerThreadNDTileStrategy``).  A reduction over that axis is lowered inside
the lane loop as a per-thread fold over the packet's ``V`` elements
(``hoist_warp_reduce``) followed by one cross-thread combine, so the lane
loop runs the combine, and everything downstream of it, once per lane::

    for lane in cutlass.range_constexpr(R):        # after unroll_lane_loads
        vec = _lane_unroll_loads_0[lane]
        acc = -inf
        for v in cutlass.range_constexpr(V):
            acc = fmax(acc, f(vec[v]))
        amax = reduce(acc, 'max', ...)              # one lane's 2048 elements
        m_new = max(m, amax); alpha = exp(m - m_new)
        ...second fold over vec, reduce(sum)...
        s = s * alpha + partial; m = m_new            # updated R times a trip

The tile program reduces the whole tile once per trip.  The two-pass lane
marker that gives scalar lane loops that structure declines vectorized ones
(``_device_lane_loop_marker_expr``), so the per-lane form above is what the
strided reduce emits.  It computes the program only when the consumers form
an online fold (softmax's rescaled running sum); ``acc += sum(tile) ** 2``
squares per-lane partials.  It is also slow: a 16384-wide row at 128 threads
x V16 x R2 runs 8 cross-thread rounds (two block reductions each: shuffles,
shared-memory round trips, barriers) where 4 suffice.

This pass rewrites each such loop (recorded at emission in
``cute_state.lane_fold_reduce_sites``, keyed by ``fold_site_key``) into one
trace-time lane loop per reduction, each lane keeping its own accumulator
for instruction-level parallelism, a balanced tree over the per-lane
partials, then ONE cross-thread combine; the statements between reductions
that read no lane value run once::

    parts = []
    for lane in cutlass.range_constexpr(R):
        vec = _lane_unroll_loads_0[lane]
        acc = -inf
        for v in cutlass.range_constexpr(V): ...
        parts.append(acc)
    acc = fmax(parts[0], parts[1])
    amax = reduce(acc, 'max', ...)                  # the whole tile
    m_new = ...; alpha = ...
    parts_1 = []
    for lane in cutlass.range_constexpr(R):
        vec = _lane_unroll_loads_0[lane]
        ...second fold...
        parts_1.append(acc_1)
    acc_1 = parts_1[0] + parts_1[1]
    s_part = reduce(acc_1, 'sum', ...)
    s = s * alpha + s_part; m = m_new

The loop must already be a trace-time loop (``unroll_lane_loads`` unrolls
the recorded loops completely, ``forced_full``).  A statement of the body is
lane-dependent when it reads the lane variable or a name a lane-dependent
statement wrote; the accumulator initializations a lane-dependent statement
overwrites before the next reduction count as lane-dependent too (they
re-initialize per lane).  Each reduction must reduce such a per-lane fold:
its accumulator initialized to the op's identity inside the lane body
(``_identity_init``) and read by that fold and the reduce only; a value
carried across the lanes (initialized before the loop) is rejected, as
folding its per-lane snapshots would count the early lanes several times.
Every lane-dependent statement must be a pure register computation, load or
register-cache store; one that no fold reads is dropped when dead and
otherwise re-run once per lane in a trailing lane loop (a two-pass kernel's
row cache for its second device loop), unless an invariant it reads is
redefined later in the body.  A reduction nested in control flow, a reduce
that is not a recorded site, a side effect in the lane body, or a reduction
other than ``sum`` / ``max`` / ``min`` rejects the config
(``BackendUnsupported``) rather than keeping the per-lane form.

Sites are recorded on every path that emits a cross-thread reduce inside a
vectorized lane loop: the grouped strided helpers and the direct single-warp
``cute.arch.warp_reduction_*`` (``ReductionStrategy._record_lane_fold_site``).

Online pair.  When the body holds exactly a ``max`` reduction followed by a
``sum`` reduction whose summands are ``exp(f(x) - m)`` with ``f(x)`` the very
value the max folded and ``m`` derived from the max's result (an online
softmax step), the two per-lane folds are fused into ONE pass over the
packets: each lane folds its own max, exponentiates against its own
(``-inf``-guarded) frame and sums, so the packets die inside the lane and
the register file holds accumulators instead of the whole slice.  The
per-lane sums are brought to the global frame after the cross-thread max
(``s_l * exp(m_l - m)``, the online-softmax rescale) and summed.  The frame
chain is required to be non-decreasing in the max (``_is_monotone_in``), so
a lane's frame is ``>=`` every element of its lane and ``<=`` the global
frame: no summand overflows and the rescale factor of a non-empty lane is
``<= 1``.  An empty lane (every element ``-inf``) has an exact zero sum but
a guarded frame ``F(0)`` whose factor overflows once the row's frame is far
below zero; its term is selected to zero instead.  A summand whose minuend
is not the folded value, a factor beside the exponential, a frame that does
not derive from the max or derives from it non-monotonically, or an
invariant the sum fold needs that depends on the max (even through another
statement) keeps the two-fold form.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ... import exc
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _names_read

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping

# The grouped strided reduces: the op is a string literal in ``args[1]``,
# the identity in ``args[2]``.
_REDUCE_HELPERS = frozenset(
    {
        "_cute_grouped_reduce_shared_two_stage",
        "_cute_grouped_reduce_shared_tree",
        "_cute_grouped_reduce_warp",
        "_cute_grouped_reduce_cluster",
    }
)
# The direct single-warp reduces (``CuteBackend.reduction_expr``, emitted
# when the CTA is one warp on the reduced axis): the op is in the name, or
# for the generic ``cute.arch.warp_reduction(x, op_fn, ...)`` in ``args[1]``.
_DIRECT_REDUCES = {
    "cute.arch.warp_reduction_sum": "sum",
    "cute.arch.warp_reduction_max": "max",
    "cute.arch.warp_reduction_min": "min",
}
_DIRECT_GENERIC = "cute.arch.warp_reduction"
_DIRECT_OP_FUNCS = {"_cute_nan_max": "max", "_cute_nan_min": "min"}
_FOLD_OPS = frozenset({"sum", "max", "min"})
# The identity literal of each op (``hoist_warp_reduce`` writes the
# per-lane accumulator's initialization as ``<dtype>(<literal>)``).
_IDENTITY_LITERALS = {
    "sum": frozenset({"0", "0.0"}),
    "max": frozenset({"float('-inf')"}),
    "min": frozenset({"float('inf')"}),
}
_PARTS_PREFIX = "_fold_parts_"
_TREE_PREFIX = "_fold_tree_"
_BARRIER = "cute.arch.sync_threads"


def _reduce_helper(call: ast.Call) -> str | None:
    """The reduce helper or direct warp reduce ``call`` invokes, or None."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id if func.id in _REDUCE_HELPERS else None
    name = ast.unparse(func)
    if name in _DIRECT_REDUCES or name == _DIRECT_GENERIC:
        return name
    return None


def _canonical_identity(node: ast.expr) -> ast.expr:
    """``cutlass.Int64(cutlass.Int64(0))`` -> ``cutlass.Int64(0)``: the
    outermost dtype wrapper around the innermost literal.

    ``hoist_warp_reduce`` re-wraps a grouped helper's identity when it
    rewrites the input; the key must survive that.
    """
    if not (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func).startswith("cutlass.")
    ):
        return node
    return ast.Call(func=node.func, args=[_unwrap_casts(node.args[0])], keywords=[])


def fold_site_key(call: ast.Call | str) -> str:
    """A reduce call with its input replaced: the key of a fold site.

    ``hoist_warp_reduce`` rewrites the input of the call after it is
    emitted (the per-lane value becomes the ``_helion_vfold_acc_*``
    accumulator, a grouped helper's identity is re-wrapped in the
    accumulator dtype), so the key leaves the first argument out and keeps
    one dtype wrapper on the identity.
    """
    node = ast.parse(call if isinstance(call, str) else ast.unparse(call), mode="eval")
    expression = node.body
    if not isinstance(expression, ast.Call) or not expression.args:
        raise _reject(f"{ast.unparse(expression)} is not a reduce call")
    expression.args[0] = ast.Name(id="_", ctx=ast.Load())
    if _reduce_helper(expression) in _REDUCE_HELPERS and len(expression.args) >= 3:
        expression.args[2] = _canonical_identity(expression.args[2])
    return ast.unparse(ast.fix_missing_locations(expression))


def _reduce_calls(node: ast.AST) -> list[ast.Call]:
    """Every cross-thread reduce call (grouped helper or direct) in ``node``."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _reduce_helper(child) is not None
    ]


def _site_operation(call: ast.Call) -> tuple[str, ast.expr | None]:
    """The reduction op of a reduce call and, for a grouped helper, its
    identity expression (None for the direct warp reduces)."""
    helper = _reduce_helper(call)
    assert helper is not None
    if helper in _REDUCE_HELPERS:
        if len(call.args) < 3:
            raise _reject("reduce call without an identity")
        op_node = call.args[1]
        if not isinstance(op_node, ast.Constant) or not isinstance(op_node.value, str):
            raise _reject(f"{ast.unparse(call)} has no literal reduction op")
        return op_node.value, call.args[2]
    if helper in _DIRECT_REDUCES:
        return _DIRECT_REDUCES[helper], None
    op_fn = call.args[1] if len(call.args) > 1 else None
    if isinstance(op_fn, ast.Name) and op_fn.id in _DIRECT_OP_FUNCS:
        return _DIRECT_OP_FUNCS[op_fn.id], None
    raise _reject(f"{ast.unparse(call)} reduces with an op that has no lane fold")


def _identity_init(stmt: ast.stmt, accumulator: str) -> tuple[str, str] | None:
    """``accumulator = <dtype>(<literal>)`` -> (dtype, literal text), else None."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == accumulator
        and isinstance(stmt.value, ast.Call)
        and len(stmt.value.args) == 1
        and not stmt.value.keywords
        and ast.unparse(stmt.value.func).startswith("cutlass.")
    ):
        return None
    literal = stmt.value.args[0]
    if not (
        isinstance(literal, ast.Constant)
        or (
            isinstance(literal, ast.Call)
            and ast.unparse(literal.func) == "float"
            and len(literal.args) == 1
            and isinstance(literal.args[0], ast.Constant)
        )
    ):
        return None
    return ast.unparse(stmt.value.func), ast.unparse(literal)


def _constexpr_extent(loop: ast.For) -> int | None:
    iterator = loop.iter
    if (
        isinstance(iterator, ast.Call)
        and ast.unparse(iterator.func) == "cutlass.range_constexpr"
        and len(iterator.args) == 1
        and isinstance(iterator.args[0], ast.Constant)
        and isinstance(iterator.args[0].value, int)
    ):
        return iterator.args[0].value
    return None


def _rolled_extent(loop: ast.For) -> int | None:
    """``for x in range(N)`` -> N; None for any other iterator."""
    iterator = loop.iter
    if (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "range"
        and len(iterator.args) == 1
        and isinstance(iterator.args[0], ast.Constant)
        and isinstance(iterator.args[0].value, int)
    ):
        return iterator.args[0].value
    return None


def _unwrap_casts(node: ast.expr) -> ast.expr:
    """``cutlass.Float32(x)`` -> ``x`` (repeatedly)."""
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func).startswith("cutlass.")
    ):
        node = node.args[0]
    return node


def _exposed_reads(stmt: ast.AST) -> set[str]:
    """Names ``stmt`` reads before defining them (its own loop targets and
    earlier assignments do not count)."""
    defined: set[str] = set()
    exposed: set[str] = set()

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.For):
                exposed.update(set(ReadWrites.from_ast(node.iter).reads) - defined)
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.If, ast.While)):
                exposed.update(set(ReadWrites.from_ast(node.test).reads) - defined)
                visit(node.body)
                visit(node.orelse)
            else:
                rw = ReadWrites.from_ast(node)
                exposed.update(set(rw.reads) - defined)
                defined.update(rw.writes)

    visit([stmt])  # type: ignore[list-item]
    return exposed


def _reject(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"cross-lane reduction fold: {reason}")


_PURE_CALLS = frozenset(
    {
        # Packed fp32 pair arithmetic (``factor_affine_reductions``), the
        # direct exp2 and the per-thread register fragment of the sibling
        # V-loop merge (``merge_sibling_v_loops``): register computations the
        # relocatable-call proof of ``tile_strategy`` does not list.
        "cute.arch.add_packed_f32x2",
        "cute.arch.sub_packed_f32x2",
        "cute.arch.mul_packed_f32x2",
        "cute.arch.fma_packed_f32x2",
        "cute.arch.exp2",
        "cute.arch.rcp_approx",
        "cute.make_rmem_tensor",
    }
)
_LOOP_ITERATORS = frozenset({"range", "cutlass.range_constexpr"})


def _register_caches(statements: list[ast.stmt]) -> frozenset[str]:
    """Names bound to ``cute.make_rmem_tensor(...)`` anywhere in ``statements``:
    the per-thread register caches a subscript store may fill."""
    return frozenset(
        node.targets[0].id
        for stmt in statements
        for node in ast.walk(stmt)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
    )


def _pure_target(target: ast.expr, caches: frozenset[str]) -> bool:
    """A store target that touches registers only: a name, a tuple of names,
    or a subscript of a register cache."""
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_pure_target(elt, caches) for elt in target.elts)
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id in caches
    )


def _is_pure(stmt: ast.AST, caches: frozenset[str] = frozenset()) -> bool:
    """Register arithmetic, casts, packet reads and loads only, stored to
    names or to the register caches in ``caches``."""
    from ..tile_strategy import _is_proven_relocatable_call

    if not isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)):
        return False
    iterators = {
        id(node.iter)
        for node in ast.walk(stmt)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) in _LOOP_ITERATORS
        and node.orelse == []
    }
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            if id(node) in iterators or ast.unparse(node.func) in _PURE_CALLS:
                continue
            if not _is_proven_relocatable_call(node, allow_load=True):
                return False
        elif isinstance(node, ast.Assign):
            if not all(_pure_target(target, caches) for target in node.targets):
                return False
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            if not _pure_target(node.target, caches):
                return False
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await, ast.Delete)):
            return False
    return True


def _combine(op: str, dtype: str, a: str, b: str) -> str:
    if op == "sum":
        return f"({a}) + ({b})"
    if op == "max":
        if dtype == "cutlass.Float32":
            return f"cute.arch.fmax(({a}), ({b}), nan=True)"
        return f"cute.math.max(({a}), ({b}), propagate_nan=True)"
    if op == "min":
        if dtype == "cutlass.Float32":
            return f"cute.arch.fmin(({a}), ({b}), nan=True)"
        return f"cute.math.min(({a}), ({b}), propagate_nan=True)"
    raise _reject(f"reduction {op!r} has no cross-lane combine")


_EXP_FUNCS = frozenset({"cute.math.exp2", "cute.math.exp", "cute.arch.exp2"})
_ELEMENT = "ELEM"


def _clone(stmt: ast.stmt) -> ast.stmt:
    return statement_from_string(ast.unparse(stmt))


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=ast.Load()), node
            )
        return node


def _renamed(stmt: ast.stmt, mapping: Mapping[str, str]) -> ast.stmt:
    """A copy of ``stmt`` with its reads (not its targets) renamed."""
    copy = _clone(stmt)
    _Renamer(mapping).visit(copy)
    return ast.fix_missing_locations(copy)


def _local_definitions(
    loop: ast.For,
) -> tuple[dict[str, ast.expr], dict[str, ast.expr]]:
    """Names a V-loop body assigns once (name -> expression) and the register
    caches it fills by subscript (cache name -> element expression)."""
    counts: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    caches: dict[str, ast.expr] = {}
    for stmt in loop.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            counts[target.id] = counts.get(target.id, 0) + 1
            values[target.id] = stmt.value
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            caches[target.value.id] = stmt.value
    return {name: value for name, value in values.items() if counts[name] == 1}, caches


class _Fingerprint(ast.NodeTransformer):
    """The value of an expression as a function of the element: V-loop
    locals inlined, the element index replaced by a placeholder, register
    cache reads replaced by the fingerprint of what filled them (computed in
    the filling loop's own scope: the two V-loops reuse local names)."""

    def __init__(
        self,
        loop_var: str,
        locals_: Mapping[str, ast.expr],
        caches: Mapping[str, str],
        active: frozenset[str] = frozenset(),
    ) -> None:
        self.loop_var = loop_var
        self.locals = locals_
        self.caches = caches
        self.active = active

    def _inline(self, expression: ast.expr, name: str) -> ast.AST:
        nested = _Fingerprint(
            self.loop_var, self.locals, self.caches, self.active | {name}
        )
        return nested.visit(_expression_copy(expression))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.locals and node.id not in self.active:
            return self._inline(self.locals[node.id], node.id)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if self.loop_var in _names_read(node.slice):
            value = node.value
            if isinstance(value, ast.Name) and value.id in self.caches:
                return ast.Name(id=self.caches[value.id], ctx=ast.Load())
            return ast.Subscript(
                value=self.visit(value),
                slice=ast.Name(id=_ELEMENT, ctx=ast.Load()),
                ctx=ast.Load(),
            )
        return self.generic_visit(node)


def _expression_copy(node: ast.expr) -> ast.expr:
    copy = ast.parse(ast.unparse(node), mode="eval").body
    assert isinstance(copy, ast.expr)
    return copy


def _fingerprint(
    expression: ast.expr,
    loop_var: str,
    locals_: Mapping[str, ast.expr],
    caches: Mapping[str, str],
) -> str:
    """Outer fp32 casts do not count: the V-fold wraps its input in one
    (``_rewrite_cast_wrapper``) that the exponential's minuend lacks."""
    tree = _Fingerprint(loop_var, locals_, caches).visit(_expression_copy(expression))
    assert isinstance(tree, ast.expr)
    return ast.unparse(ast.fix_missing_locations(_strip_float_casts(tree)))


def _cache_fingerprints(
    loop: ast.For, locals_: Mapping[str, ast.expr], fills: Mapping[str, ast.expr]
) -> dict[str, str]:
    """Each register cache the V-loop fills, as the fingerprint of its
    element expression in that loop's scope."""
    assert isinstance(loop.target, ast.Name)
    return {
        name: _fingerprint(fill, loop.target.id, locals_, {})
        for name, fill in fills.items()
    }


def _strip_float_casts(node: ast.expr) -> ast.expr:
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) == "cutlass.Float32"
    ):
        node = node.args[0]
    return node


def _max_fold_input(loop: ast.For, accumulator: str) -> ast.expr | None:
    """``acc = cute.arch.fmax(acc, X, nan=True)`` in the V-loop -> ``X``."""
    for stmt in loop.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == accumulator
            and isinstance(stmt.value, ast.Call)
            and ast.unparse(stmt.value.func) in ("cute.arch.fmax", "cute.math.max")
            and len(stmt.value.args) == 2
            and isinstance(stmt.value.args[0], ast.Name)
            and stmt.value.args[0].id == accumulator
        ):
            return stmt.value.args[1]
    return None


def _sum_fold_terms(loop: ast.For) -> list[ast.expr] | None:
    """The summands of the V-loop's accumulations: ``t = t + T`` or the
    packed ``lo, hi = cute.arch.add_packed_f32x2((lo, hi), (T_lo, T_hi))``."""
    terms: list[ast.expr] = []
    for stmt in loop.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target, value = stmt.targets[0], stmt.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and isinstance(_strip_float_casts(value.left), ast.Name)
            and _strip_float_casts(value.left).id == target.id  # type: ignore[union-attr]
        ):
            terms.append(value.right)
        elif (
            isinstance(target, ast.Tuple)
            and isinstance(value, ast.Call)
            and ast.unparse(value.func) == "cute.arch.add_packed_f32x2"
            and len(value.args) == 2
            and isinstance(value.args[1], ast.Tuple)
        ):
            names = [ast.unparse(_strip_float_casts(e)) for e in value.args[0].elts]  # type: ignore[attr-defined]
            if names != [ast.unparse(t) for t in target.elts]:
                return None
            terms.extend(value.args[1].elts)
    return terms or None


@dataclasses.dataclass(frozen=True)
class _ExpTerm:
    """``exp(f - frame)`` in one of the lowered spellings."""

    func: str  # the exponential call
    keywords: str  # its keyword text (``, fastmath=True`` or empty)
    kind: str  # "fma": exp2(fma(Z, C, -S)); "sub_scaled": exp2((Z - V) * C); "sub": exp(Z - V)
    scale: str  # C for the scaled kinds
    frame: str  # S or V: the subtracted invariant
    minuend: ast.expr

    def factor(self, lane_frame: str, global_frame: str) -> str:
        """``exp(frame_l - frame)``: brings a lane's sum to the global frame."""
        if self.kind == "fma":
            argument = f"({lane_frame}) - ({global_frame})"
        elif self.kind == "sub_scaled":
            argument = f"(({lane_frame}) - ({global_frame})) * {self.scale}"
        else:
            argument = f"({lane_frame}) - ({global_frame})"
        return f"{self.func}({argument}{self.keywords})"


def _exp_term(node: ast.expr, locals_: Mapping[str, ast.expr]) -> _ExpTerm | None:
    def resolve(expression: ast.expr) -> ast.expr:
        """Strip casts and follow V-loop-local names to their definitions
        (``v_6 = v_1 - v_2`` is bound before the exponential reads it)."""
        seen: set[str] = set()
        expression = _strip_float_casts(expression)
        while (
            isinstance(expression, ast.Name)
            and expression.id in locals_
            and expression.id not in seen
        ):
            seen.add(expression.id)
            expression = _strip_float_casts(locals_[expression.id])
        return expression

    call = resolve(node)
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in _EXP_FUNCS
        and len(call.args) == 1
    ):
        return None
    func = ast.unparse(call.func)
    keywords = "".join(f", {ast.unparse(keyword)}" for keyword in call.keywords)
    argument = resolve(call.args[0])
    if (
        isinstance(argument, ast.Call)
        and ast.unparse(argument.func) == "cute.math.fma"
        and len(argument.args) == 3
        and isinstance(argument.args[1], ast.Constant)
        and isinstance(argument.args[2], ast.UnaryOp)
        and isinstance(argument.args[2].op, ast.USub)
        and isinstance(argument.args[2].operand, ast.Name)
        and func != "cute.math.exp"
    ):
        return _ExpTerm(
            func,
            keywords,
            "fma",
            ast.unparse(argument.args[1]),
            argument.args[2].operand.id,
            argument.args[0],
        )
    if (
        isinstance(argument, ast.BinOp)
        and isinstance(argument.op, ast.Mult)
        and isinstance(argument.right, ast.Constant)
        and func != "cute.math.exp"
    ):
        inner = resolve(argument.left)
        if (
            isinstance(inner, ast.BinOp)
            and isinstance(inner.op, ast.Sub)
            and isinstance(inner.right, ast.Name)
        ):
            return _ExpTerm(
                func,
                keywords,
                "sub_scaled",
                ast.unparse(argument.right),
                inner.right.id,
                inner.left,
            )
        return None
    if (
        isinstance(argument, ast.BinOp)
        and isinstance(argument.op, ast.Sub)
        and isinstance(argument.right, ast.Name)
    ):
        return _ExpTerm(func, keywords, "sub", "", argument.right.id, argument.left)
    return None


@dataclasses.dataclass
class _PairPlan:
    term: _ExpTerm
    chain: list[int]  # invariant statements from the max result to the frame
    independent: list[int]  # other invariants between the sites the sum fold needs


def _header_reads(stmt: ast.stmt) -> set[str]:
    """Names a compound statement's own header reads (loop iterator, branch
    test, context items)."""
    if isinstance(stmt, ast.For):
        return set(ReadWrites.from_ast(stmt.iter).reads)
    if isinstance(stmt, (ast.If, ast.While)):
        return set(ReadWrites.from_ast(stmt.test).reads)
    if isinstance(stmt, ast.With):
        return {
            name
            for item in stmt.items
            for name in ReadWrites.from_ast(item.context_expr).reads
        }
    return set()


class _LaneFolder:
    def __init__(
        self,
        sites_by_lane: Mapping[str, frozenset[str]],
        new_var: Callable[[str], str],
        caches: frozenset[str],
    ) -> None:
        self.sites_by_lane = sites_by_lane
        self.new_var = new_var
        # Names bound to ``cute.make_rmem_tensor`` anywhere in the kernel:
        # the per-thread register caches a lane body may fill (a two-pass
        # kernel's row cache for its second device loop is allocated outside
        # the lane body).
        self.caches = caches

    def visit_block(
        self, statements: list[ast.stmt], outer_reads: frozenset[str] = frozenset()
    ) -> list[ast.stmt]:
        """``outer_reads``: the exposed reads of every statement of the
        enclosing blocks other than the ancestors of this block (a lane
        value written here can be observed by any of them, after the
        enclosing loops or through their back edges)."""
        exposed: list[set[str]] | None = None

        def outside_reads(position: int) -> frozenset[str]:
            nonlocal exposed
            if exposed is None:
                exposed = [_exposed_reads(stmt) for stmt in statements]
            return outer_reads.union(
                *(
                    exposed[other]
                    for other in range(len(statements))
                    if other != position
                )
            )

        result: list[ast.stmt] = []
        for position, stmt in enumerate(statements):
            if isinstance(stmt, (ast.For, ast.If, ast.While, ast.With)):
                inner_reads = outside_reads(position) | _header_reads(stmt)
                stmt.body = self.visit_block(stmt.body, inner_reads)  # type: ignore[assignment]
                if not isinstance(stmt, ast.With):
                    stmt.orelse = self.visit_block(stmt.orelse, inner_reads)  # type: ignore[assignment]
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                keys = self.sites_by_lane.get(stmt.target.id)
                if keys and _reduce_calls(stmt) and _rolled_extent(stmt) != 1:
                    # A single-lane loop already reduces the whole tile.
                    result.extend(
                        self.fold_loop(stmt, keys, set(outside_reads(position)))
                    )
                    continue
            result.append(stmt)
        return result

    def fold_loop(
        self, loop: ast.For, keys: frozenset[str], outside_reads: set[str]
    ) -> list[ast.stmt]:
        lane = loop.target.id  # type: ignore[union-attr]
        extent = _constexpr_extent(loop)
        if extent is None:
            raise _reject(f"lane loop {lane} is not unrolled at trace time")
        lanes = extent
        body = list(loop.body)
        sites: dict[int, ast.Call] = {}
        for index, stmt in enumerate(body):
            calls = _reduce_calls(stmt)
            if not calls:
                continue
            if (
                len(calls) != 1
                or not isinstance(stmt, ast.Assign)
                or len(stmt.targets) != 1
                or not isinstance(stmt.targets[0], ast.Name)
            ):
                # Also a reduce the V-fold left per element inside the
                # ``range_constexpr(V)`` loop (an integer reduction).
                raise _reject(
                    f"reduction in {lane} is not a plain assignment of the lane body"
                )
            if fold_site_key(calls[0]) not in keys:
                # A reduce emitted by a path that did not record its site,
                # or whose call a later pass rewrote: the per-lane form would
                # be kept silently.
                raise _reject(
                    f"{ast.unparse(calls[0])} in {lane} is not a recorded fold site"
                )
            sites[index] = calls[0]
        if not sites:
            return [loop]

        # Lane dependence, in program order.  A statement's reads are its
        # exposed reads: a V-loop defines and reads its own temporaries, and
        # two V-loops of one body reuse those names.
        rw = [ReadWrites.from_ast(stmt) for stmt in body]
        reads = [_exposed_reads(stmt) for stmt in body]
        lane_dependent: set[str] = {lane}
        dependent = [False] * len(body)
        for index in range(len(body)):
            if index in sites:
                continue
            if reads[index] & lane_dependent:
                dependent[index] = True
                lane_dependent |= set(rw[index].writes)
        # An accumulator initialization a lane-dependent statement overwrites
        # before the next reduction re-initializes per lane.
        next_site = len(body)
        for index in range(len(body) - 1, -1, -1):
            if index in sites:
                next_site = index
                continue
            if dependent[index]:
                continue
            writes = set(rw[index].writes)
            if writes and any(
                dependent[later] and writes & set(rw[later].writes)
                for later in range(index + 1, next_site)
            ):
                dependent[index] = True
                lane_dependent |= writes

        def backward_slice(end: int, needs: set[str]) -> list[int]:
            chosen: list[int] = []
            for index in range(end - 1, -1, -1):
                if dependent[index] and set(rw[index].writes) & needs:
                    chosen.append(index)
                    needs |= reads[index]
            chosen.reverse()
            return chosen

        # Per-site accumulator, op and dtype.  The accumulator must be a
        # per-lane fold: initialized to the op's identity inside the lane
        # body (its backward slice holds exactly one such initialization),
        # read by that slice and the reduce only.  A value carried across
        # the lanes (initialized before the loop, updated per lane) already
        # accumulates across them; folding its per-lane snapshots would
        # count every lane but the last several times.
        accumulators: dict[int, str] = {}
        operations: dict[int, tuple[str, str]] = {}
        slices: dict[int, list[int]] = {}
        for index, call in sites.items():
            op, identity = _site_operation(call)
            if op not in _FOLD_OPS:
                raise _reject(f"reduction {op!r} has no cross-lane combine")
            source = _unwrap_casts(call.args[0])
            if not isinstance(source, ast.Name) or source.id not in lane_dependent:
                raise _reject(f"reduction in {lane} does not reduce a per-lane fold")
            accumulator = source.id
            other_reads = reads[index] - {accumulator}
            if other_reads & lane_dependent:
                raise _reject(
                    f"reduction in {lane} reads a lane value besides its input"
                )
            member = backward_slice(index, {accumulator})
            inits = [
                init
                for position in member
                if (init := _identity_init(body[position], accumulator)) is not None
            ]
            if len(inits) != 1:
                raise _reject(
                    f"the accumulator {accumulator} of the reduction in {lane} is "
                    "not initialized to the identity once per lane"
                )
            dtype, literal = inits[0]
            if literal not in _IDENTITY_LITERALS[op]:
                raise _reject(
                    f"{accumulator} starts from {literal}, not the identity of {op!r}"
                )
            if identity is not None and (
                not isinstance(identity, ast.Call)
                or ast.unparse(identity.func) != dtype
            ):
                raise _reject(
                    f"the identity of the reduction in {lane} is not a {dtype}"
                )
            if (
                any(
                    accumulator in reads[position]
                    for position in range(len(body))
                    if position != index and position not in member
                )
                or accumulator in outside_reads
            ):
                raise _reject(
                    f"{accumulator} is read outside the fold of its reduction in {lane}"
                )
            accumulators[index] = accumulator
            operations[index] = (op, dtype)
            slices[index] = member

        # Every lane-dependent statement is pure and feeds a reduction.
        caches = self.caches
        for index, stmt in enumerate(body):
            if not dependent[index]:
                continue
            if not _is_pure(stmt, caches):
                raise _reject(
                    f"{ast.unparse(stmt).splitlines()[0]} depends on {lane} "
                    "and is not a pure register computation"
                )
        for index, stmt in enumerate(body):
            if dependent[index] or index in sites:
                continue
            if isinstance(stmt, ast.Expr):
                call = stmt.value
                if not (
                    isinstance(call, ast.Call) and ast.unparse(call.func) == _BARRIER
                ):
                    raise _reject(f"{ast.unparse(stmt)} inside {lane} has effects")
            elif not isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                raise _reject(
                    f"{ast.unparse(stmt).splitlines()[0]} inside {lane} is not "
                    "a plain assignment"
                )
            elif any(
                isinstance(node, ast.Call)
                and node.func is not None
                and ast.unparse(node.func).endswith(".store")
                for node in ast.walk(stmt)
            ):
                raise _reject(f"{ast.unparse(stmt)} stores inside {lane}")

        def tree(parts: str, op: str, dtype: str, into: list[ast.stmt]) -> str:
            level = [f"{parts}[{index}]" for index in range(lanes)]
            while len(level) > 1:
                folded: list[str] = []
                for first in range(0, len(level) - 1, 2):
                    name = self.new_var(_TREE_PREFIX)
                    into.append(
                        statement_from_string(
                            f"{name} = {_combine(op, dtype, level[first], level[first + 1])}"
                        )
                    )
                    folded.append(name)
                if len(level) % 2:
                    folded.append(level[-1])
                level = folded
            return level[0]

        def lane_loop_of(statements: list[ast.stmt]) -> ast.For:
            loop = statement_from_string(
                f"for {lane} in cutlass.range_constexpr({lanes}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = statements
            return loop

        covered: set[int] = set()
        result: list[ast.stmt] = []
        ordered_sites = sorted(sites)
        pair = (
            _online_pair_plan(
                body,
                ordered_sites,
                accumulators,
                operations,
                dependent,
                reads,
                rw,
                backward_slice,
            )
            if len(ordered_sites) == 2
            else None
        )
        if pair is not None:
            first, second = ordered_sites
            acc_a, acc_b = accumulators[first], accumulators[second]
            slice_a, slice_b = slices[first], slices[second]
            covered.update(slice_a)
            covered.update(slice_b)
            for index in range(first):
                if not dependent[index]:
                    result.append(body[index])
            for index in pair.independent:
                result.append(body[index])
            parts_m = self.new_var(_PARTS_PREFIX)
            parts_v = self.new_var(_PARTS_PREFIX)
            parts_s = self.new_var(_PARTS_PREFIX)
            root = self.new_var("_pair_root")
            renames = {ast.unparse(body[first].targets[0]): root}  # type: ignore[attr-defined]
            for index in pair.chain:
                target = ast.unparse(body[index].targets[0])  # type: ignore[attr-defined]
                renames[target] = self.new_var(f"{target}_lane")
            ident = f"{operations[first][1]}(float('-inf'))"
            lane_body: list[ast.stmt] = [body[index] for index in slice_a]
            lane_body.append(
                statement_from_string(
                    f"{root} = {operations[first][1]}(0) if {acc_a} == {ident} else {acc_a}"
                )
            )
            for index in pair.chain:
                renamed = _renamed(body[index], renames)
                renamed.targets[0] = ast.Name(  # type: ignore[attr-defined]
                    id=renames[ast.unparse(body[index].targets[0])],  # type: ignore[attr-defined]
                    ctx=ast.Store(),
                )
                lane_body.append(renamed)
            for index in slice_b:
                if index not in slice_a:
                    lane_body.append(_renamed(body[index], renames))
            lane_body.extend(
                statement_from_string(text)
                for text in (
                    f"{parts_m}.append({acc_a})",
                    f"{parts_v}.append({renames[pair.term.frame]})",
                    f"{parts_s}.append({acc_b})",
                )
            )
            result.extend(
                (
                    *(
                        statement_from_string(f"{name} = []")
                        for name in (parts_m, parts_v, parts_s)
                    ),
                    lane_loop_of(lane_body),
                )
            )
            top = tree(parts_m, "max", operations[first][1], result)
            result.extend((statement_from_string(f"{acc_a} = {top}"), body[first]))
            for index in range(first + 1, second):
                if not dependent[index] and index not in pair.independent:
                    result.append(body[index])
            global_frame = self.new_var("_pair_frame")
            result.append(
                statement_from_string(
                    f"{global_frame} = {operations[first][1]}(0) "
                    f"if {pair.term.frame} == {ident} else {pair.term.frame}"
                )
            )
            # A lane whose max is -inf (every element -inf: a masked tail,
            # -inf padding) summed exact zeros against the guarded frame
            # F(0); its rescale factor exp(F(0) - F(m)) overflows once the
            # row's frame is far below 0 (0 * inf = NaN), so its term is
            # selected to zero instead.  Every other lane's frame is at
            # most the global frame (the chain is non-decreasing in the
            # max), so its factor is at most one.
            scaled = self.new_var(_PARTS_PREFIX)
            result.append(statement_from_string(f"{scaled} = []"))
            zero = f"{operations[second][1]}(0)"
            for index in range(lanes):
                factor = pair.term.factor(f"{parts_v}[{index}]", global_frame)
                result.append(
                    statement_from_string(
                        f"{scaled}.append({zero} if {parts_m}[{index}] == {ident} "
                        f"else {parts_s}[{index}] * {factor})"
                    )
                )
            total = tree(scaled, "sum", operations[second][1], result)
            result.extend((statement_from_string(f"{acc_b} = {total}"), body[second]))
            previous = second
        else:
            previous = -1
            for site_index in ordered_sites:
                accumulator = accumulators[site_index]
                op, dtype = operations[site_index]
                for index in range(previous + 1, site_index):
                    if not dependent[index]:
                        result.append(body[index])
                member = slices[site_index]
                covered.update(member)
                parts = self.new_var(_PARTS_PREFIX)
                result.extend(
                    (
                        statement_from_string(f"{parts} = []"),
                        lane_loop_of(
                            [
                                *(body[index] for index in member),
                                statement_from_string(f"{parts}.append({accumulator})"),
                            ]
                        ),
                    )
                )
                top = tree(parts, op, dtype, result)
                result.extend(
                    (statement_from_string(f"{accumulator} = {top}"), body[site_index])
                )
                previous = site_index
        for index in range(previous + 1, len(body)):
            if not dependent[index]:
                result.append(body[index])
        # A lane value no reduction folds: dead (the lane base the load phase
        # kept its own copy of) is dropped.  One read anywhere else (a
        # register cache the next device loop reads, a lane base read after
        # the loop) is recomputed once per lane in a trailing lane loop with
        # its slice: pure register work (checked above) whose lane-invariant
        # inputs are all defined by then, provided none of them is redefined
        # by a later statement of the body (the per-lane form read the
        # earlier value).
        all_reads: set[str] = set(outside_reads)
        for index in range(len(body)):
            all_reads |= reads[index]
        tail: set[int] = set()
        for index in range(len(body)):
            if (
                dependent[index]
                and index not in covered
                and set(rw[index].writes) & all_reads
            ):
                tail.update(backward_slice(index + 1, set(rw[index].writes)))
        if tail:
            for index in sorted(tail):
                for later in range(index + 1, len(body)):
                    redefined = (
                        set(rw[later].writes) & reads[index]
                        if not dependent[later]
                        else set()
                    )
                    if redefined:
                        raise _reject(
                            f"{ast.unparse(body[index]).splitlines()[0]} depends on "
                            f"{lane} and reads {min(redefined)}, which the lane "
                            "body redefines after it"
                        )
            result.append(lane_loop_of([body[index] for index in sorted(tail)]))
        for stmt in result:
            ast.fix_missing_locations(stmt)
        return result


def _online_pair_plan(
    body: list[ast.stmt],
    ordered_sites: list[int],
    accumulators: Mapping[int, str],
    operations: Mapping[int, tuple[str, str]],
    dependent: list[bool],
    reads: list[set[str]],
    rw: list[ReadWrites],
    backward_slice: Callable[[int, set[str]], list[int]],
) -> _PairPlan | None:
    """The online-pair shape of a max site followed by a sum site, or None.

    See the module docstring.  Proven here: the sum's summands are one
    exponential of ``f(x) - frame`` (one of the lowered spellings) with
    ``f(x)`` fingerprint-identical to the max fold's input, ``frame`` a
    lane-invariant name whose definition chain between the two sites reads
    the max site's result, and every statement of that chain a pure
    assignment; the other invariants between the sites that the sum fold
    reads do not depend on the max result (they are hoisted above the fused
    lane loop).
    """
    first, second = ordered_sites
    if operations[first][0] != "max" or operations[second][0] != "sum":
        return None
    if (
        operations[first][1] != "cutlass.Float32"
        or operations[second][1] != "cutlass.Float32"
    ):
        return None
    acc_a, acc_b = accumulators[first], accumulators[second]
    slice_a = backward_slice(first, {acc_a})
    slice_b = backward_slice(second, {acc_b})
    loops_a = [
        body[i]
        for i in slice_a
        if isinstance(body[i], ast.For) and acc_a in rw[i].writes
    ]
    loops_b = [body[i] for i in slice_b if isinstance(body[i], ast.For)]
    if len(loops_a) != 1 or not loops_b:
        return None
    loop_a = loops_a[0]
    loop_b = loops_b[-1]
    assert isinstance(loop_a, ast.For) and isinstance(loop_b, ast.For)
    if not (
        isinstance(loop_a.target, ast.Name) and isinstance(loop_b.target, ast.Name)
    ):
        return None
    locals_a, fills_a = _local_definitions(loop_a)
    locals_b, fills_b = _local_definitions(loop_b)
    max_input = _max_fold_input(loop_a, acc_a)
    terms = _sum_fold_terms(loop_b)
    if max_input is None or terms is None:
        return None
    caches_a = _cache_fingerprints(loop_a, locals_a, fills_a)
    wanted = _fingerprint(max_input, loop_a.target.id, locals_a, caches_a)
    parsed = [_exp_term(term, locals_b) for term in terms]
    if any(term is None for term in parsed):
        return None
    exp_terms = [term for term in parsed if term is not None]
    head = exp_terms[0]
    caches = {**caches_a, **_cache_fingerprints(loop_b, locals_b, fills_b)}
    for term in exp_terms:
        if (term.func, term.keywords, term.kind, term.scale, term.frame) != (
            head.func,
            head.keywords,
            head.kind,
            head.scale,
            head.frame,
        ):
            return None
        if _fingerprint(term.minuend, loop_b.target.id, locals_b, caches) != wanted:
            return None
    frame = head.frame
    # The frame's chain among the invariant statements between the sites.
    between = [i for i in range(first + 1, second) if not dependent[i]]
    defined_by = {}
    for i in between:
        stmt = body[i]
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return None
        defined_by[stmt.targets[0].id] = i
    result_name = ast.unparse(body[first].targets[0])  # type: ignore[attr-defined]
    if frame not in defined_by:
        return None
    chain: list[int] = []
    pending = [frame]
    while pending:
        name = pending.pop()
        i = defined_by.get(name)
        if i is None or i in chain:
            continue
        chain.append(i)
        pending.extend(n for n in reads[i] if n in defined_by)
    chain.sort()
    if not any(result_name in reads[i] for i in chain):
        return None
    # Chain statements depend on the max result only through names of the
    # chain or names defined before the first site, and each is pure and
    # non-decreasing in the max: a lane's frame is then at most the global
    # frame, so no lane sum overflows its own frame and every rescale factor
    # of a non-empty lane is at most one.
    chain_names = {ast.unparse(body[i].targets[0]) for i in chain}  # type: ignore[attr-defined]
    other_between = {ast.unparse(body[i].targets[0]) for i in between} - chain_names  # type: ignore[attr-defined]
    derived = frozenset(chain_names | {result_name})
    for i in chain:
        if reads[i] & other_between:
            return None
        if not _is_pure(body[i]):
            return None
        if not _is_monotone_in(body[i].value, derived):  # type: ignore[attr-defined]
            return None
    # Invariants between the sites the sum fold reads but the chain does not
    # define move above the fused loop, with every between-statement they
    # read (transitively); none of them may depend on the max result, even
    # through another between-statement.
    tainted = set(derived)
    for i in between:
        if reads[i] & tainted:
            tainted.add(ast.unparse(body[i].targets[0]))  # type: ignore[attr-defined]
    needed = set()
    for i in slice_b:
        needed |= reads[i]
    independent: set[int] = set()
    pending = [name for name in needed if name in defined_by]
    while pending:
        name = pending.pop()
        i = defined_by[name]
        if i in chain or i in independent:
            continue
        if name in tainted or not _is_pure(body[i]):
            return None
        independent.add(i)
        pending.extend(n for n in reads[i] if n in defined_by)
    # The lane-dependent statements of the sum fold may read the chain names
    # and the independent invariants, but not the max result itself.
    for i in slice_b:
        if result_name in reads[i]:
            return None
    return _PairPlan(head, chain, sorted(independent))


_MONOTONE_CALLS = frozenset(
    {"max", "min", "cute.math.max", "cute.math.min", "cute.arch.fmax", "cute.arch.fmin"}
)


def _positive_constant(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value > 0
    )


def _is_monotone_in(node: ast.expr, names: frozenset[str]) -> bool:
    """Whether ``node`` is non-decreasing in each of ``names``: casts, max /
    min, sums, differences with an independent subtrahend, products and
    fused multiply-adds by positive constants of such expressions.  An
    expression that reads none of ``names`` qualifies."""
    if not _names_read(node) & names:
        return True
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func.startswith("cutlass.") and len(node.args) == 1 and not node.keywords:
            return _is_monotone_in(node.args[0], names)
        if func in _MONOTONE_CALLS and len(node.args) == 2:
            return all(_is_monotone_in(arg, names) for arg in node.args)
        if func == "cute.math.fma" and len(node.args) == 3:
            a, b, c = node.args
            if _positive_constant(b):
                return _is_monotone_in(a, names) and _is_monotone_in(c, names)
            if _positive_constant(a):
                return _is_monotone_in(b, names) and _is_monotone_in(c, names)
        return False
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return _is_monotone_in(node.left, names) and _is_monotone_in(
                node.right, names
            )
        if isinstance(node.op, ast.Sub):
            return _is_monotone_in(node.left, names) and not (
                _names_read(node.right) & names
            )
        if isinstance(node.op, ast.Mult):
            if _positive_constant(node.right):
                return _is_monotone_in(node.left, names)
            if _positive_constant(node.left):
                return _is_monotone_in(node.right, names)
        return False
    return False


def lanes_with_sites(
    body: list[ast.stmt], sites: Mapping[str, Collection[str]]
) -> frozenset[str]:
    """The recorded lane variables (``sites``: lane variable -> the
    ``fold_site_key`` of each reduce recorded in its loop) whose lane loop
    still holds a cross-thread reduce.

    A site is recorded when the reduce is emitted inside the lane loop; an
    earlier pass may since have moved the combine out of the loop (a
    single-trip tile whose lane loop already accumulates across its lanes
    and reduces once after it).  Such loops are the forms this pass does not
    model: they must keep their code, so neither the forced unroll nor the
    fold applies to them.  A loop that still holds a reduce is folded, and
    the fold rejects a reduce that is not one of its recorded sites.
    """
    found: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id in sites
                and node.target.id not in found
                and _reduce_calls(node)
            ):
                found.add(node.target.id)
    return frozenset(found)


def fold_lane_reductions(
    body: list[ast.stmt],
    sites: Mapping[str, Collection[str]],
    new_var: Callable[[str], str],
) -> list[ast.stmt]:
    """Fold the recorded per-lane reductions (lane variable -> the
    ``fold_site_key`` of each reduce recorded in its loop) across their
    trace-time lane loops; see the module docstring."""
    if not sites:
        return body
    folder = _LaneFolder(
        {lane: frozenset(keys) for lane, keys in sites.items()},
        new_var,
        _register_caches(body),
    )
    return folder.visit_block(body)
