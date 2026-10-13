"""AST peephole that fuses redundant loads across two consecutive looped
reductions on the CuTe backend.

Generated reduction kernels (RMS norm, layernorm, softmax, ...) emit two
``for offset in range(...)`` loops that share the same outer iteration and
the same ``ptr.load()`` of the input tensor — once during the reduction
sweep, once again during the post-reduction consume/store sweep. The
second load is pure HBM bandwidth waste because the values already lived
in registers a few statements earlier.

This pass walks the kernel body, finds the pattern::

    for OFFSET in RANGE:
        ...
        NAME1 = (PTR).load() if MASK else CONST   # tracked load
        ...
    ... (post-reduction statements) ...
    for OFFSET in RANGE:                           # same offset, same range
        ...
        NAME2 = (PTR).load() if MASK else CONST   # identical load text
        ...

and rewrites it to cache the load in per-thread scalar registers between
the two loops::

    for OFFSET in RANGE:
        ...
        NAME1 = (PTR).load() if MASK else CONST
        _fuse_cache_<i>[(OFFSET - START) // STEP] = NAME1
        ...
    ... (post-reduction statements) ...
    for OFFSET in RANGE:
        ...
        NAME2 = _fuse_cache_<i>[(OFFSET - START) // STEP]
        ...

The cache is declared once above the first loop. The pass is conservative:
matches by exact unparsed text of the load expression, the mask, and the
two for-loops' ``target`` / ``iter`` / range bounds.
"""

from __future__ import annotations

import ast
from collections import deque
import copy
from itertools import islice
import re
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_extension import clone_ast
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterator

_PERSISTENT_BRANCH_VEC_LOAD = "_helion_persistent_branch_vec_load"
_PERSISTENT_BRANCH_VEC_STORE = "_helion_persistent_branch_vec_store"
_MAX_BINDING_KEY_NODES = 512

_LoopAliasNames = tuple[str | None, str | None, str | None, str | None]


def _loop_alias_names(loop: ast.For) -> _LoopAliasNames:
    """First assignments/loop target in the original ``ast.walk`` order."""
    prefixes = ("reduction_lane_base_", "reduction_vec_lane_", "lane_base_")
    names: list[str | None] = [None, None, None, None]
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    for index, prefix in enumerate(prefixes):
                        if names[index] is None and target.id.startswith(prefix):
                            names[index] = target.id
        elif (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("vec_lane_")
            and names[3] is None
        ):
            names[3] = node.target.id
        if all(name is not None for name in names):
            break
    return names[0], names[1], names[2], names[3]


def _scalar_bindings(node: ast.AST) -> set[str]:
    # ReadWrites deliberately omits for targets. Address keys need their
    # rebinding as well, including the value left after a completed loop.
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _own_statement_occurrences(
    body: list[ast.stmt], *, scalar_bindings: set[str] | None = None
) -> list[ast.stmt]:
    """Give repeated statements distinct lexical identities before analysis.

    Reduction splitting can reuse one assignment object in several sweeps.
    Identity-keyed memory positions would then retain only its last occurrence,
    and rewriting a consumer's children could also change its producer. Copy
    syntax fields per occurrence, retaining opaque compiler metadata by identity.
    Bodies that already have unique statements take the original path. The
    optional Store-name output shares this full walk; cloned bodies get the
    original fresh query so no pre-clone summary crosses the rewrite.
    """
    root = ast.Module(body=body, type_ignores=[])
    seen: set[int] = set()
    if scalar_bindings is not None:
        scalar_bindings.clear()
    for node in ast.walk(root):
        if (
            scalar_bindings is not None
            and isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
        ):
            scalar_bindings.add(node.id)
        if isinstance(node, ast.stmt):
            if id(node) in seen:
                break
            seen.add(id(node))
    else:
        return body

    def clone(node: ast.AST) -> ast.AST:
        result = node.copy() if isinstance(node, ExtendedAST) else copy.copy(node)
        assert isinstance(result, ast.AST)
        # ExtendedAST.copy() preserves its standard annotations; retain any
        # additional lowering markers as well, without copying their graphs.
        vars(result).update(vars(node))
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(result, field, clone(value))
            elif isinstance(value, list):
                setattr(
                    result,
                    field,
                    [
                        clone(item) if isinstance(item, ast.AST) else item
                        for item in value
                    ],
                )
        return result

    copied = clone(root)
    assert isinstance(copied, ast.Module)
    if scalar_bindings is not None:
        scalar_bindings.clear()
        scalar_bindings.update(_scalar_bindings(copied))
    return copied.body


def _unwrap_persistent_branch_vec_load(node: ast.AST) -> ast.AST:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _PERSISTENT_BRANCH_VEC_LOAD
        and len(node.args) == 6
    ):
        return node.args[5]
    return node


def _range_bounds(node: ast.AST) -> tuple[ast.expr, ast.expr, ast.expr] | None:
    """Extract ``(start, end, step)`` from a supported range call.

    Synthetic lanes can use either Python ``range`` or CuTe's compile-time
    ``cutlass.range_constexpr``.
    """
    if not isinstance(node, ast.Call):
        return None
    is_range = isinstance(node.func, ast.Name) and node.func.id == "range"
    is_constexpr_range = (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "range_constexpr"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
    )
    if not (is_range or is_constexpr_range):
        return None
    args = node.args
    if not args:
        return None
    if len(args) == 1:
        return (ast.Constant(value=0), args[0], ast.Constant(value=1))
    if len(args) == 2:
        return (args[0], args[1], ast.Constant(value=1))
    if len(args) == 3:
        return (args[0], args[1], args[2])
    return None


def _looks_like_tracked_load(node: ast.AST) -> ast.IfExp | None:
    """Return the IfExp if ``node`` matches the masked-load pattern emitted
    by ``_cute_scalar_load_expr``::

        (PTR).load() if MASK else CONST
    """
    if not isinstance(node, ast.IfExp):
        return None
    # body is the load expression
    body = node.body
    if not isinstance(body, ast.Call):
        return None
    func = body.func
    if not isinstance(func, ast.Attribute) or func.attr != "load":
        return None
    return node


def _looks_like_unmasked_load(node: ast.AST) -> ast.Call | None:
    """Return the Call if ``node`` is an unmasked scalar ``(PTR).load()``.

    The CuTe scalar load emitter drops the ``if mask else CONST`` wrapping
    when no mask is active (e.g. ``weight[:]`` loads in RMS-norm consume
    sweeps); we still want to fuse those across the two passes.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "load":
            return node
    return None


def _looks_like_vec_load(node: ast.AST) -> ast.Call | None:
    """Return the Call if ``node`` is a ``cute.arch.load(ptr, vec_type)``
    expression — the hoisted U16 vec load emitted by the LoopedReductionStrategy
    ``unroll`` mode — or a cache-hinted vector load helper of the same call
    shape (``_cute_load_l2_evict_last_8b(ptr, vec_type)``, see
    ``cache_policy_loads``).
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in _CUTE_CACHE_LOAD_HELPER_NAMES:
        return node
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != "load":
        return None
    val = func.value
    # Match either ``cute.arch.load`` or ``cute.arch.<whatever>.load`` — we
    # only care that ``arch`` is in the call chain because that's what the
    # cute helper uses.
    while isinstance(val, ast.Attribute):
        if val.attr == "arch":
            return node
        val = val.value
    return None


def _arch_scalar_dtype_mismatch(node: ast.AST, tensor_dtypes: dict[str, str]) -> bool:
    """True for a scalar ``cute.arch.load(ptr, DTYPE, ...)`` whose explicit
    DTYPE differs from the base tensor's element dtype.

    fp8/fp4 raw-byte loads (``cutlass.Uint8``) and byte-packed fp8 hoists
    (``cutlass.Uint32``/``Uint64``) load REINTERPRETED integers; caching
    them in the base tensor's dtype (what the unmasked path does) would
    numerically convert the value.  Cache-hinted fp32 loads pass
    (``cutlass.Float32`` == the tensor's dtype string).
    """
    if _looks_like_vec_load(node) is None or _vec_width(node) is not None:
        return False  # plain ``(ptr).load()``: implicit dtype == tensor dtype
    assert isinstance(node, ast.Call)
    if len(node.args) < 2:
        return True
    base = _unmasked_load_tensor_dtype(node, tensor_dtypes)
    return base is None or ast.unparse(node.args[1]) != base


def _load_kind(
    node: ast.AST, tensor_dtypes: dict[str, str] | None = None
) -> str | None:
    """Classify the load shape into ``"masked"`` / ``"unmasked"`` / ``"vec"``.

    Returns None when ``node`` doesn't look like a gmem load we can fuse.
    """
    node = _unwrap_persistent_branch_vec_load(node)
    if _looks_like_tracked_load(node) is not None:
        return "masked"
    # A cache-hinted SCALAR load also routes through ``cute.arch.load``
    # (``(ptr).load()`` has no hint kwargs), so "vec" additionally requires
    # a VectorType dtype argument; scalar arch loads fall through to
    # "unmasked" — but only when their explicit dtype matches the base
    # tensor's (see _arch_scalar_dtype_mismatch).
    if _looks_like_vec_load(node) is not None and _vec_width(node) is not None:
        return "vec"
    if _looks_like_unmasked_load(node) is not None:
        if tensor_dtypes is not None and _arch_scalar_dtype_mismatch(
            node, tensor_dtypes
        ):
            return None
        return "unmasked"
    return None


def _vec_width(node: ast.AST) -> int | None:
    """Extract the V from ``cute.arch.load(ptr, ir.VectorType.get([V], ...))``.

    Returns None when the expression isn't a recognised vec load.
    """
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    vec_arg = node.args[1]
    if (
        isinstance(vec_arg, ast.Call)
        and isinstance(vec_arg.func, ast.Attribute)
        and vec_arg.func.attr == "get"
        and len(vec_arg.args) >= 1
    ):
        shape_arg = vec_arg.args[0]
        if isinstance(shape_arg, ast.List) and len(shape_arg.elts) == 1:
            elt = shape_arg.elts[0]
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                return elt.value
    return None


def _dtype_from_default(node: ast.expr) -> str | None:
    """Extract dtype from the else branch of a masked load.

    For ``... if mask else cutlass.Float32(0)``, the else expression is
    ``cutlass.Float32(0)`` and the dtype is ``cutlass.Float32``.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
        return ast.unparse(node.func)
    return None


def _static_int(expr: ast.expr, constexpr_values: dict[str, int]) -> int | None:
    """Static value of a literal, a known constexpr name, or either wrapped in
    a one-argument call such as ``cutlass.Int32(...)``; otherwise None."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in constexpr_values:
        return constexpr_values[expr.id]
    if isinstance(expr, ast.Call) and len(expr.args) == 1:
        inner = expr.args[0]
        if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
            return inner.value
        if isinstance(inner, ast.Name) and inner.id in constexpr_values:
            return constexpr_values[inner.id]
    return None


def _wrap_dynamic_groups(
    body: list[ast.stmt],
    groups: list[tuple[ast.For, ast.For, str]],
    fused_spans: list[tuple[ast.For, ast.For]],
    originals: dict[int, ast.stmt],
) -> list[ast.stmt]:
    """Guard fused constexpr-trip groups with their trace-time fragment budgets.

    ``groups`` names the first and last rewritten sweep of each group and the
    condition under which every cache of the group fits its budget for the
    exact trip count.  The fused statements run when it holds; otherwise the
    cloned pre-fusion statements run and the later sweeps re-load from
    gmem/L2.  Only one branch is traced, so the fast path is unchanged.
    Overlapping groups share one branch under the conjunction of their
    budgets.

    ``fused_spans`` names the first and last rewritten sweep of every fused
    group in ``body``, static ones included.  A guarded range grows to cover
    each fused group it intersects: a group whose populate sweep is traced
    inside the guard while a consume sweep runs after it would otherwise read
    a cache only the fused branch fills.
    """
    ranges = sorted(
        (body.index(first), body.index(last), condition)
        for first, last, condition in groups
    )
    spans = [(body.index(first), body.index(last)) for first, last in fused_spans]
    while True:
        merged: list[tuple[int, int, str]] = []
        for lo, hi, condition in ranges:
            for span_lo, span_hi in spans:
                if span_lo <= hi and lo <= span_hi:
                    lo, hi = min(lo, span_lo), max(hi, span_hi)
            if merged and lo <= merged[-1][1]:
                prev_lo, prev_hi, prev_condition = merged[-1]
                merged[-1] = (
                    min(prev_lo, lo),
                    max(prev_hi, hi),
                    f"{prev_condition} and {condition}",
                )
            else:
                merged.append((lo, hi, condition))
        # Growing a range can reach further groups; iterate to a fixed point.
        if merged == ranges:
            break
        ranges = merged
    for lo, hi, condition in reversed(merged):
        fused = body[lo : hi + 1]
        fallback = [originals[id(stmt)] for stmt in fused if id(stmt) in originals]
        body[lo : hi + 1] = [
            create(
                ast.If,
                test=expr_from_string(f"cutlass.const_expr({condition})"),
                body=fused,
                orelse=fallback,
            )
        ]
    return body


def _trip_count_for(
    start: ast.expr,
    end: ast.expr,
    step: ast.expr,
    constexpr_values: dict[str, int],
    *,
    allow_dynamic_base: bool = False,
) -> int | None:
    """If start/end/step are constants (possibly wrapped in cutlass.Int32 or
    naming a known constexpr), return the static trip count. Otherwise None.
    """

    def _to_int(expr: ast.expr) -> int | None:
        return _static_int(expr, constexpr_values)

    def _peel_wrappers(expr: ast.expr) -> tuple[ast.expr, tuple[str, ...]]:
        # Peel ``cutlass.Int32(...)``-style 1-arg wrapper calls, recording
        # each callee so both bounds can be required to use the SAME chain
        # (a value-changing wrapper on one side must not match).
        callees: list[str] = []
        while isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords:
            callees.append(ast.unparse(expr.func))
            expr = expr.args[0]
        return expr, tuple(callees)

    s, e, t = _to_int(start), _to_int(end), _to_int(step)
    if t is None or t <= 0:
        return None
    if s is None or e is None:
        if not allow_dynamic_base:
            return None
        # Rank-offset ranges from the rolled cluster split:
        # ``range(W(BASE * C), W(BASE * C + SPAN), step)`` — the bounds are
        # dynamic (BASE is the CTA rank) but the SPAN is a constant shared
        # offset, so the trip count is still static.  Only the outer roll
        # loop may use this form: its cache index subtracts the start
        # expression, so a dynamic base still normalizes to slot 0..trip-1
        # (a lane loop's index uses the raw loop var and must stay 0-based).
        start_inner, start_wrappers = _peel_wrappers(start)
        end_inner, end_wrappers = _peel_wrappers(end)
        if start_wrappers != end_wrappers:
            return None
        if (
            isinstance(end_inner, ast.BinOp)
            and isinstance(end_inner.op, ast.Add)
            and isinstance(end_inner.right, ast.Constant)
            and isinstance(end_inner.right.value, int)
            and end_inner.right.value > 0
            and ast.unparse(end_inner.left) == ast.unparse(start_inner)
        ):
            return (end_inner.right.value + t - 1) // t
        return None
    if e <= s:
        return 0
    return (e - s + t - 1) // t


def _scalar_load_ptr_text(node: ast.AST) -> str | None:
    """Pointer text of a SCALAR gmem load in either emitted form:
    ``(PTR).load(...)`` or ``cute.arch.load(PTR, <scalar dtype>, ...)``.

    A cache-hinted scalar site routes through ``cute.arch.load`` while its
    unhinted twin in the other sweep stays ``(PTR).load()``; the two must
    compare equal for cross-sweep fusion, so both normalize to the pointer
    expression.  Returns None for vec loads and non-load expressions.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "load":
        return None
    if _looks_like_vec_load(node) is not None:
        if _vec_width(node) is not None:
            return None
        if node.args:
            return ast.unparse(node.args[0])
        return None
    return ast.unparse(func.value)


def _plain_vec_load_spelling(node: ast.AST) -> ast.AST:
    """A cache-hinted vector load helper call respelled as the plain
    ``cute.arch.load(ptr, vec_type)`` it stands for, so the hinted and
    unhinted twins of one load match across sweeps; other nodes unchanged."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _CUTE_CACHE_LOAD_HELPER_NAMES
    ):
        plain = cast("ast.Call", clone_ast(node))
        plain.func = cast("ast.expr", expr_from_string("cute.arch.load"))
        return plain
    return node


def _normalized_load_text(node: ast.AST) -> str:
    """Unparse with cache hints stripped and the two scalar load forms
    collapsed onto one spelling (see ``_scalar_load_ptr_text``); a hinted
    vector load helper collapses onto ``cute.arch.load``."""
    node = _plain_vec_load_spelling(_unwrap_persistent_branch_vec_load(node))
    ptr = _scalar_load_ptr_text(node)
    if ptr is not None:
        return f"__scalar_load__({ptr})"
    if isinstance(node, ast.IfExp):
        body_ptr = _scalar_load_ptr_text(node.body)
        if body_ptr is not None:
            return (
                f"(__scalar_load__({body_ptr}) if {ast.unparse(node.test)} "
                f"else {ast.unparse(node.orelse)})"
            )
    return re.sub(
        r",\s*(?:level1_eviction_priority|cop)='[a-z_]+'", "", ast.unparse(node)
    )


def _node_text(node: ast.AST) -> str:
    """Stable text key for matching AST nodes.

    Per-load-site cache hints are stripped from the key (and the hinted /
    unhinted scalar load spellings collapse): the same logical load in
    different sweeps carries a different (independently tunable) hint, and
    the hint must not defeat cross-sweep fusion — the surviving first-sweep
    load keeps its own hint.
    """
    return _normalized_load_text(node)


def _unmasked_load_tensor_dtype(
    node: ast.AST, tensor_dtypes: dict[str, str]
) -> str | None:
    """Dtype for caching an unmasked scalar ``(x.iterator + ...).load()``:
    the BASE TENSOR's own element dtype (looked up from the kernel's
    tensor arguments), so the cache roundtrip is bit-exact for every
    consumer regardless of how they cast the value.  Returns None when the
    base tensor cannot be identified (no fusion)."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "iterator"
            and isinstance(sub.value, ast.Name)
        ):
            return tensor_dtypes.get(sub.value.id)
    return None


def _is_store_call(node: ast.Call) -> bool:
    """Whether this is a generated memory write."""
    func = node.func
    is_store = (
        isinstance(func, ast.Attribute) and func.attr in ("store", "__setitem__")
    ) or (
        isinstance(func, ast.Name)
        and (
            func.id.startswith("_cute_store_")
            or func.id == _PERSISTENT_BRANCH_VEC_STORE
        )
    )
    is_atomic = (
        isinstance(func, ast.Attribute) and func.attr.startswith("atomic_")
    ) or (isinstance(func, ast.Name) and func.id.startswith("_cute_atomic_"))
    return is_store or is_atomic


def _contains_arch_attribute(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "arch"
        for child in _unique_ast_nodes(node)
    )


def _tensor_arg_roots(
    node: ast.AST,
    tensor_names: set[str],
) -> frozenset[str] | None:
    """Return kernel tensor arguments whose storage an expression addresses.

    ``None`` means the address cannot be resolved to named tensor arguments and
    must conservatively alias every tracked load.
    """
    roots: set[str] = set()
    unresolved_iterator = False
    for child in _unique_ast_nodes(node):
        if not isinstance(child, ast.Attribute) or child.attr != "iterator":
            continue
        if isinstance(child.value, ast.Name) and child.value.id in tensor_names:
            roots.add(child.value.id)
        else:
            unresolved_iterator = True
    if unresolved_iterator:
        return None
    if isinstance(node, ast.Name) and node.id in tensor_names:
        roots.add(node.id)
    return frozenset(roots) if roots else None


def _store_tensor_roots(
    node: ast.Call,
    tensor_names: set[str],
) -> frozenset[str] | None:
    """Resolve the destination tensor arguments of a generated write."""
    func = node.func
    pointer: ast.AST | None = None
    if (
        isinstance(func, ast.Name)
        and func.id == _PERSISTENT_BRANCH_VEC_STORE
        and len(node.args) == 6
    ):
        pointer = node.args[3]
    elif isinstance(func, ast.Name) and func.id.startswith(
        ("_cute_store_", "_cute_atomic_")
    ):
        pointer = node.args[0] if node.args else None
    elif isinstance(func, ast.Attribute):
        if func.attr.startswith("atomic_"):
            pointer = node.args[0] if node.args else None
        elif func.attr == "store":
            # ``cute.arch.store(ptr, value, ...)`` is a free function, while
            # ``ptr.store(value)`` carries its destination on ``func.value``.
            pointer = (
                node.args[0]
                if node.args and _contains_arch_attribute(func.value)
                else func.value
            )
        elif func.attr == "__setitem__":
            pointer = func.value
    return _tensor_arg_roots(pointer, tensor_names) if pointer is not None else None


def _tensor_roots_may_alias(
    left: frozenset[str] | None,
    right: frozenset[str] | None,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    if left is None or right is None:
        return True
    return any(
        left_name == right_name
        or frozenset((left_name, right_name)) not in proven_disjoint_tensor_pairs
        for left_name in left
        for right_name in right
    )


def _unique_ast_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Visit shared syntax once for occurrence-independent presence queries.

    Keep lexical visitors separate: repeated statements and memory accesses
    still have distinct execution positions even when their syntax is shared.
    """
    pending = deque([node])
    seen: set[int] = set()
    while pending:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(ast.iter_child_nodes(current))


class _LexicalMemoryOrder(ast.NodeVisitor):
    """Snapshot memory accesses while a fusion group's syntax is unchanged.

    Matching only reads the body. A new instance is built after each group's
    rewrites, so both lexical positions and per-loop store summaries stay valid.
    """

    def __init__(
        self,
        body: list[ast.stmt],
        tensor_names: set[str],
        proven_disjoint_tensor_pairs: set[frozenset[str]],
    ) -> None:
        super().__init__()
        self._position = 0
        self._tensor_names = tensor_names
        self._proven_disjoint_tensor_pairs = proven_disjoint_tensor_pairs
        self.assignment_positions: dict[int, int] = {}
        self.stores: list[tuple[int, frozenset[str] | None]] = []
        self._stores_in: dict[ast.AST, tuple[frozenset[str] | None, ...]] = {}
        for stmt in body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignment_positions[id(node)] = self._position
        self._position += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_store_call(node):
            self.stores.append(
                (
                    self._position,
                    _store_tensor_roots(node, self._tensor_names),
                )
            )
            self._position += 1
        self.generic_visit(node)

    def _may_alias(
        self,
        store_roots: frozenset[str] | None,
        load_roots: frozenset[str] | None,
    ) -> bool:
        return _tensor_roots_may_alias(
            store_roots,
            load_roots,
            self._proven_disjoint_tensor_pairs,
        )

    def has_aliasing_store_between(
        self,
        producer: ast.Assign,
        consumer: ast.Assign,
        load_roots: frozenset[str] | None,
    ) -> bool:
        producer_pos = self.assignment_positions.get(id(producer))
        consumer_pos = self.assignment_positions.get(id(consumer))
        if producer_pos is None or consumer_pos is None or consumer_pos <= producer_pos:
            return True
        return any(
            producer_pos < store_pos < consumer_pos
            and self._may_alias(store_roots, load_roots)
            for store_pos, store_roots in self.stores
        )

    def has_aliasing_store_in(
        self,
        node: ast.AST,
        load_roots: frozenset[str] | None,
    ) -> bool:
        if node not in self._stores_in:
            self._stores_in[node] = tuple(
                _store_tensor_roots(child, self._tensor_names)
                for child in _unique_ast_nodes(node)
                if isinstance(child, ast.Call) and _is_store_call(child)
            )
        return any(
            self._may_alias(store_roots, load_roots)
            for store_roots in self._stores_in[node]
        )


def _rewrite_vec_extract(
    node: ast.AST,
    hoist_var: str,
    cache: str,
    idx_expr: str,
    vec_w: int,
    *,
    use_smem: bool = False,
    cache_size: int = 1,
    tid_expr: str = "cutlass.Int32(cute.arch.thread_idx()[0])",
) -> None:
    """In-place rewrite of ``hoist_var[<vi>]`` -> cache read inside ``node``.

    Used after the vec hoist itself has been deleted from the consume
    sweep so the dependent extracts still resolve.

    Register-cache slot: ``cache[(idx_expr)*V + vi]`` (per-thread).
    SMEM-cache slot: per-thread contiguous layout —
        ``cache[tid * (cache_size * V) + (idx_expr) * V + vi]``
    matching the writer in ``_slot_expr``.
    """

    class _RewriteVecExtract(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == hoist_var:
                vi_text = ast.unparse(node.slice)
                inner = f"({idx_expr}) * {vec_w} + ({vi_text})"
                if use_smem:
                    slot = f"({tid_expr}) * {cache_size * vec_w} + ({inner})"
                else:
                    slot = inner
                new = ast.parse(f"{cache}[{slot}]").body[0]
                assert isinstance(new, ast.Expr)
                return new.value
            return node

    transformer = _RewriteVecExtract()
    transformer.visit(node)
    ast.fix_missing_locations(node)


def _binding_demand(body: list[ast.stmt], tensor_dtypes: dict[str, str]) -> set[str]:
    """Overapproximate names needed by any load key or induction-domain key.

    Union every assignment to a name: demand does not prove a reaching value.
    The existing freezer still versions every demanded definition and applies
    its original opacity and expanded-node limits. Cache rewrites only remove
    loads or replace extracts by cache slots over the original loop domains.
    """
    needed: set[str] = set()
    dependencies: dict[str, set[str]] = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.For):
            needed.update(ReadWrites.from_ast(node.iter).reads)
            needed.update(_scalar_bindings(node.target))
        elif (
            isinstance(node, (ast.Call, ast.IfExp))
            and _load_kind(node, tensor_dtypes) is not None
        ):
            needed.update(ReadWrites.from_ast(node).reads)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            dependencies.setdefault(node.targets[0].id, set()).update(
                ReadWrites.from_ast(node.value).reads
            )
    pending = list(needed)
    while pending:
        for name in dependencies.get(pending.pop(), ()):
            if name not in needed:
                needed.add(name)
                pending.append(name)
    return needed


def _canonical_load_text(
    node: ast.AST,
    lane_var_alias: dict[str, str],
    definitions: dict[str, ast.expr] | None = None,
) -> str:
    """Stringify a load node with lane-base / hoist variables canonicalised.

    The strategy creates fresh variable names for each sweep
    (``reduction_lane_base_1`` in the reduce sweep, ``reduction_lane_base_2``
    in the consume sweep, ``_unroll_vec_0`` vs ``_unroll_vec_1``...).  The
    fuser matches loads across the two sweeps, so it must compare their
    textual form modulo the rename — otherwise identical loads compare
    unequal and fusion bails.
    """
    if definitions:

        class _InlineDefinitions(ast.NodeTransformer):
            def __init__(self) -> None:
                super().__init__()
                self.expanding: set[str] = set()

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if not isinstance(node.ctx, ast.Load):
                    return node
                value = definitions.get(node.id)
                if value is None or node.id in self.expanding:
                    return node
                self.expanding.add(node.id)
                replacement = self.visit(
                    ast.parse(ast.unparse(value), mode="eval").body
                )
                self.expanding.remove(node.id)
                return ast.copy_location(replacement, node)

        node = _InlineDefinitions().visit(
            ast.parse(ast.unparse(node), mode="eval").body
        )

    # Per-load-site eviction hints differ between sweeps and must not
    # defeat matching (see _node_text).
    text = _normalized_load_text(node)
    for old, new in lane_var_alias.items():
        # Word-boundary replacement so suffixed vars don't pick up matches
        # for shorter prefixes.
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


class _CuteFuseTwoPassLoads:
    """See module docstring."""

    def __init__(
        self,
        constexpr_values: dict[str, int] | None = None,
        thread_block_dims: tuple[int, int, int] = (1, 1, 1),
        tensor_dtypes: dict[str, str] | None = None,
        reload_modes: dict[int, str] | None = None,
        proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
        proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
        dynamic_trip_counts: dict[str, tuple[int, str]] | None = None,
        preserve_scalar_invariants: bool = False,
    ) -> None:
        super().__init__()
        self._counter = 0
        self._computed_cache_elements = 0
        self._constexpr_values = constexpr_values or {}
        # Rolled reductions over a symbolic extent: offset variable (the
        # sweep loop's target) -> (size-hint trip count for the profitability
        # policy, name of the constexpr kernel parameter carrying the exact
        # runtime trip count).  The cache allocation is emitted in terms of
        # that parameter so the fragment is exact for every runtime extent.
        self._dynamic_trip_counts = dynamic_trip_counts or {}
        # Autotuner-selected reload mode per rolled or persistent reduction
        # block id ("auto" / "register" / "gmem").  Sweep loops are matched
        # back to their block id through their generated lane/offset variable.
        self._reload_modes = reload_modes or {}
        self._proven_disjoint_tensor_pairs = proven_disjoint_tensor_pairs or set()
        self._proven_tensor_stride_values = proven_tensor_stride_values or {}
        # Recursive matches append declarations here so allocations can be
        # emitted once at kernel scope, never inside a dynamic branch.
        self._root_declarations: list[ast.stmt] = []
        # Kernel tensor-arg name -> backend dtype string (e.g.
        # "cutlass.Float32"); resolves the cache dtype for unmasked
        # scalar loads.
        self._tensor_dtypes = tensor_dtypes or {}
        # Per-axis thread dims for the CUDA thread block.  Used by the
        # SMEM-backed cache path to (a) size the SMEM allocation by
        # total thread count and (b) emit a linear per-thread slot
        # index that is unique across all axes.
        dims = (
            max(1, int(thread_block_dims[0])),
            max(1, int(thread_block_dims[1])),
            max(1, int(thread_block_dims[2])),
        )
        self._thread_block_dims = dims
        self._thread_count = dims[0] * dims[1] * dims[2]
        self._readonly_tensors: set[str] = set()
        self._written_names: set[str] = set()
        self._binding_names: dict[tuple[ast.stmt, str], str] = {}
        self._reserved_names: set[str] = set()
        self._scalar_bypass_known_effects = False
        self._scalar_bypass_mutations: set[str] = set()
        self._preserve_scalar_invariants = preserve_scalar_invariants
        self._demanded_bindings: set[str] | None = None

    def _readonly_scalar_is_invariant(
        self,
        stmt: ast.Assign,
        kind: str,
        definitions: dict[str, ast.expr],
    ) -> bool:
        """Leave original invariant scalar loads for the existing LICM passes.

        A per-lane register cache would hide their invariance from LICM. This
        only declines caching: it neither moves a load nor changes its mask.
        Frozen bindings include the entire load, address, predicate and false
        value. Opaque induction/rebinding/size-limit tokens retain the old path.
        """
        from .hoist_lane_invariant_reductions import _is_allowed_pure_call

        if kind not in ("masked", "unmasked") or not self._scalar_bypass_known_effects:
            return False
        frozen = self._freeze_binding(stmt, stmt.value, definitions)
        roots = _tensor_arg_roots(frozen, set(self._tensor_dtypes))
        if roots is None or not roots <= self._readonly_tensors:
            return False
        reads = set(ReadWrites.from_ast(frozen).reads)
        if reads.intersection(self._binding_names.values()) or reads.intersection(
            self._scalar_bypass_mutations
        ):
            return False
        return all(
            _is_allowed_pure_call(node)
            for node in ast.walk(frozen)
            if isinstance(node, ast.Call)
        )

    def _canonical_aliases(self, local: dict[str, str]) -> dict[str, str]:
        """Resolve aliases proved by explicit copy assignments."""
        aliases = dict(local)

        def resolve(name: str) -> str:
            seen: set[str] = set()
            while name in aliases and name not in seen:
                seen.add(name)
                replacement = aliases[name]
                if replacement == name:
                    break
                name = replacement
            return name

        return {name: resolve(name) for name in aliases}

    def _pure_definitions_before(
        self,
        statements: list[ast.stmt],
        stop: int,
        inherited: dict[str, ast.expr] | None = None,
    ) -> dict[str, ast.expr]:
        """Freeze address bindings at their definitions, not their later uses.

        A memory-derived scalar is opaque unless its storage is read-only for
        the complete kernel. In particular, reusing the spelling ``slot`` after
        loading another index does not prove that two state addresses agree.
        """
        definitions = dict(inherited or {})
        self._extend_definitions(statements, 0, stop, definitions)
        return definitions

    def _extend_definitions(
        self,
        statements: list[ast.stmt],
        start: int,
        stop: int,
        definitions: dict[str, ast.expr],
    ) -> None:
        """Advance an owned binding map through an unchanged statement interval."""
        for stmt in statements[start:stop]:
            previous = dict(definitions)
            for name in _scalar_bindings(stmt):
                definitions[name] = self._opaque_binding(stmt, name)
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                target = stmt.targets[0].id
                if (
                    self._demanded_bindings is not None
                    and target not in self._demanded_bindings
                ):
                    # Keep the opaque write identity installed above. No load
                    # or domain key can depend on this scalar's expression.
                    continue
                has_memory_access = any(
                    isinstance(node, ast.Call)
                    and (
                        (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr in ("load", "store")
                        )
                        or (
                            isinstance(node.func, ast.Name)
                            and node.func.id.startswith(
                                ("_cute_load_", "_cute_store_", "_cute_atomic_")
                            )
                        )
                    )
                    for node in ast.walk(stmt.value)
                )
                roots = _tensor_arg_roots(stmt.value, set(self._tensor_dtypes))
                readonly_load = (
                    _load_kind(stmt.value, self._tensor_dtypes) is not None
                    and roots is not None
                    and roots <= self._readonly_tensors
                )
                if not has_memory_access or readonly_load:
                    definitions[target] = self._freeze_binding(
                        stmt, stmt.value, previous
                    )

    def _freeze_binding(
        self, stmt: ast.stmt, value: ast.expr, previous: dict[str, ast.expr]
    ) -> ast.expr:
        outer = self
        nodes = sum(1 for _ in islice(ast.walk(value), _MAX_BINDING_KEY_NODES + 1))
        if nodes > _MAX_BINDING_KEY_NODES:
            return self._opaque_binding(stmt, "expression")

        class FreezeBindings(ast.NodeTransformer):
            def __init__(self) -> None:
                self.nodes = nodes
                self.exceeded = False

            def visit_Name(self, node: ast.Name) -> ast.expr:
                if not isinstance(node.ctx, ast.Load):
                    return node
                if binding := previous.get(node.id):
                    # Earlier bindings are already frozen. Do not reinterpret
                    # them after a later reassignment of their source name.
                    self.nodes += sum(1 for _ in ast.walk(binding)) - 1
                    if self.nodes > _MAX_BINDING_KEY_NODES:
                        self.exceeded = True
                        return node
                    # Binding trees are frozen analysis values. Their consumers
                    # only read them or copy before rewriting. Share this
                    # immutable subtree; never attach it to emitted code.
                    return binding
                if node.id in outer._written_names:
                    return outer._opaque_binding(stmt, node.id)
                return node

        freezer = FreezeBindings()
        result = freezer.visit(ast.parse(ast.unparse(value), mode="eval").body)
        return self._opaque_binding(stmt, "expression") if freezer.exceeded else result

    def _opaque_binding(self, stmt: ast.stmt, name: str) -> ast.Name:
        key = (stmt, name)
        if key not in self._binding_names:
            fresh = f"_fuser_binding_{len(self._reserved_names)}"
            while fresh in self._reserved_names:
                fresh += "_"
            self._reserved_names.add(fresh)
            self._binding_names[key] = fresh
        return ast.Name(id=self._binding_names[key], ctx=ast.Load())

    def _container_definitions(
        self,
        loop: ast.For,
        container: list[ast.stmt],
        scope: dict[str, ast.expr],
        canonical_loop: ast.For,
        canonical_container: list[ast.stmt],
    ) -> dict[str, ast.expr] | None:
        """Enter only the corresponding, live induction domains of a pair.

        Loop-carried scalar bindings start opaque for each distinct loop;
        unconditional definitions before a load can still establish equality.
        Iteration tokens are immutable key names, so a captured induction value
        cannot be reinterpreted after its original name is rebound.
        """
        assert isinstance(loop.target, ast.Name)
        assert isinstance(canonical_loop.target, ast.Name)
        assigned = _scalar_bindings(ast.Module(body=loop.body, type_ignores=[]))
        if loop.target.id in assigned or assigned.intersection(
            ReadWrites.from_ast(loop.iter).reads
        ):
            # The cache index uses the original induction value and bounds.
            return None
        definitions = dict(scope)
        for name in assigned:
            definitions[name] = self._opaque_binding(loop, name)
        domain = _canonical_load_text(loop.iter, {}, scope)
        definitions[loop.target.id] = self._opaque_binding(
            canonical_loop, f"{canonical_loop.target.id}:{domain}"
        )
        if container is loop.body:
            return definitions if canonical_container is canonical_loop.body else None
        for index, statement in enumerate(loop.body):
            if isinstance(statement, ast.For) and statement.body is container:
                for canonical in canonical_loop.body:
                    if (
                        isinstance(canonical, ast.For)
                        and canonical.body is canonical_container
                        and _node_text(statement.iter) == _node_text(canonical.iter)
                    ):
                        before = self._pure_definitions_before(
                            loop.body, index, definitions
                        )
                        return self._container_definitions(
                            statement, container, before, canonical, canonical_container
                        )
        return None

    @staticmethod
    def _reduction_block_id(loop: ast.For) -> int | None:
        if not isinstance(loop.target, ast.Name):
            return None
        match = re.match(
            r"(?:roffset|synthetic_lane|reduction_lane)_(\d+)",
            loop.target.id,
        )
        return int(match.group(1)) if match is not None else None

    def _new_cache_name(self) -> str:
        name = f"_fuse_cache_{self._counter}"
        self._counter += 1
        return name

    def _consumer_store_preserves_snapshot(
        self,
        loop: ast.For,
        container: list[ast.stmt],
        load_index: int,
        load_stmt: ast.Assign,
        load_roots: frozenset[str] | None,
    ) -> bool:
        """Whether a consume sweep may read a pre-populated register cache.

        An in-place consume loop is not automatically a cache barrier. Once
        the producer sweep has populated every cache slot, a later exact
        per-lane read/modify/write may safely read that snapshot when each
        iteration's store is proved unable to affect another iteration's
        logical load. This is the same injectivity proof used when splitting
        persistent reductions in the first place.
        """
        from .persistent_branch_vec import _definition_snapshots
        from .persistent_branch_vec import _lane_accesses_are_iteration_independent
        from .persistent_branch_vec import _memory_load_calls

        if (
            load_roots is None
            or not isinstance(loop.target, ast.Name)
            or container is not loop.body
        ):
            return False
        bounds = _range_bounds(loop.iter)
        if bounds is None:
            return False
        lane_extent = _trip_count_for(*bounds, self._constexpr_values)
        if lane_extent is None:
            return False
        load_calls = _memory_load_calls(load_stmt.value)
        if len(load_calls) != 1:
            return False
        load_call = load_calls[0]

        snapshots = _definition_snapshots(container)
        if snapshots is None:
            return False
        definitely_written: set[str] = set()
        live_in: set[str] = set()
        may_writes: set[str] = set()
        for statement in container:
            effects = ReadWrites.from_ast(statement)
            live_in.update(set(effects.reads) - definitely_written - {loop.target.id})
            may_writes.update(effects.writes)
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                definitely_written.update(effects.writes)
        loop_carried_names = live_in & may_writes

        saw_aliasing_store = False
        tensor_names = set(self._tensor_dtypes)
        for statement_index, statement in enumerate(container):
            for store_call in (
                child
                for child in ast.walk(statement)
                if isinstance(child, ast.Call) and _is_store_call(child)
            ):
                store_roots = _store_tensor_roots(store_call, tensor_names)
                if not _tensor_roots_may_alias(
                    store_roots,
                    load_roots,
                    self._proven_disjoint_tensor_pairs,
                ):
                    continue
                saw_aliasing_store = True
                if statement_index <= load_index:
                    return False
                unstable_names = set(loop_carried_names)
                for crossed in container[load_index + 1 : statement_index + 1]:
                    unstable_names.update(ReadWrites.from_ast(crossed).writes)
                if not _lane_accesses_are_iteration_independent(
                    load_call,
                    store_call,
                    lane_var=loop.target.id,
                    lane_extent=lane_extent,
                    load_definitions=snapshots[load_index],
                    store_definitions=snapshots[statement_index],
                    proven_tensor_stride_values=self._proven_tensor_stride_values,
                    loop_carried_names=unstable_names,
                ):
                    return False
        return saw_aliasing_store

    def _resolve_load_container(
        self,
        outer_loop: ast.For,
        start: ast.expr,
        step: ast.expr,
        trip: int,
        *,
        fold_single_trip: bool = True,
    ) -> tuple[list[ast.stmt], str, int] | None:
        """Find the body list holding the actual gmem loads + the index
        expression used to address the cache for one iter of ``outer_loop``.

        Returns ``(container, cache_index_expr, cache_size)`` or None when
        the body shape is not recognised.

        Supported shapes:
          A) Flat body (``for offset ...: load...``) — original simple case.
          B) One nested for-loop (lane loop): ``for lane in range(K)`` —
             cache index is ``(offset - start) // step * K + lane`` and the
             container is the lane loop body.
          C) Two nested for-loops (lane + constexpr vec lane) — the load
             dispatcher hoists vec loads ABOVE the constexpr loop, so the
             container is the lane loop body (between base_stmt and
             vec_for); index is ``(offset - start) // step * K + lane``.
        """
        body = outer_loop.body
        assert isinstance(outer_loop.target, ast.Name)
        if trip == 1 and fold_single_trip:
            # Single-trip outer loop (e.g. a whole-row tile): the outer
            # index folds to a literal 0 so the cache slot expression stays
            # a compile-time constant — the fragment then lives in
            # registers instead of dynamically-indexed local memory.  A
            # size-hint trip count of a symbolic extent is not a proof of a
            # single trip, so it keeps the indexed form.
            cache_index_outer = "0"
        else:
            cache_index_outer = (
                f"({outer_loop.target.id} - ({_node_text(start)})) "
                f"// ({_node_text(step)})"
            )
        for_children = [s for s in body if isinstance(s, ast.For)]
        if not for_children:
            return body, cache_index_outer, trip
        if len(for_children) != 1:
            return None
        # Shape B/C: one nested lane loop
        lane_loop = for_children[0]
        if not isinstance(lane_loop.target, ast.Name):
            return None
        # Resolve lane extent from range(K)
        lane_range = _range_bounds(lane_loop.iter)
        if lane_range is None:
            return None
        lane_start, lane_end, lane_step = lane_range
        lane_trip = _trip_count_for(
            lane_start, lane_end, lane_step, self._constexpr_values
        )
        if lane_trip is None or lane_trip < 1 or lane_trip > 32:
            return None
        lane_var = lane_loop.target.id
        cache_index = f"({cache_index_outer}) * {lane_trip} + ({lane_var})"
        # Check if the lane body contains a NESTED for (the constexpr vec
        # loop in shape C).  If yes, the cache loads sit BEFORE that
        # constexpr loop (the vec-hoist pattern from the LoopedReduction
        # ``unroll`` mode).
        lane_inner_fors = [s for s in lane_loop.body if isinstance(s, ast.For)]
        if not lane_inner_fors:
            # Shape B: lane body itself is the load container.
            return lane_loop.body, cache_index, trip * lane_trip
        if len(lane_inner_fors) != 1:
            return None
        # Shape C: cache the hoisted vec loads (which sit BEFORE the
        # constexpr loop).  The constexpr V-loop itself is left alone
        # because it only reads ``hoist_var[vi].bitcast(...)`` — once the
        # hoist is replaced by a cache read, those inner extracts still
        # work unchanged.
        return lane_loop.body, cache_index, trip * lane_trip

    def _build_second_alias(
        self,
        first_loop: ast.For,
        second_loop: ast.For,
        *,
        summaries: dict[ast.For, _LoopAliasNames] | None = None,
    ) -> dict[str, str]:
        """Collect per-loop variable renames that need to be normalised so
        the two sweeps' load expressions compare equal.

        Currently handles:
          - LoopedReductionStrategy ``unroll`` mode:
            ``reduction_lane_base_<N>``, ``reduction_vec_lane_<N>``.
          - PerThreadNDTileStrategy lane / vec naming:
            ``lane_base_<N>``, ``_tile_unroll_vec_<N>_<sweep>``,
            ``vec_lane_<N>``.
        """
        # The caller owns this map only while the loop bodies are unchanged.
        # Direct callers get a fresh scope, including after an AST mutation.
        if summaries is None:
            summaries = {}
        for loop in (first_loop, second_loop):
            if loop not in summaries:
                summaries[loop] = _loop_alias_names(loop)
        return {
            second_var: first_var
            for first_var, second_var in zip(
                summaries[first_loop], summaries[second_loop], strict=True
            )
            if first_var and second_var and first_var != second_var
        }

    def _dtype_for_load_kind(self, node: ast.AST, kind: str) -> str | None:
        """Extract the storage dtype string for the fragment that backs the
        cached load.  For masked scalar loads, derive from the ``else
        CONST`` branch.  For unmasked / vec loads, derive from the pointer
        or vec-type expression.
        """
        node = _unwrap_persistent_branch_vec_load(node)
        if kind == "masked":
            assert isinstance(node, ast.IfExp)
            return _dtype_from_default(node.orelse)
        if kind == "unmasked":
            # ``(ptr_expr).load()`` — the load expression itself doesn't
            # carry a dtype, but we can look it up later via the assignment
            # target's downstream usage.  For now, fall back to a Float32
            # cache which the cute compiler will coerce.  (RMS-norm consume
            # sweeps in fp16/bf16 take the masked path; this fallback is
            # rarely exercised.)
            return None
        if kind == "vec":
            # ``cute.arch.load(ptr, ir.VectorType.get([V], <elem>.mlir_type))``
            # Pull the element type expression out so the fragment is the
            # right dtype.  The third arg to VectorType.get is the elem.
            assert isinstance(node, ast.Call)
            if len(node.args) >= 2:
                vec_arg = node.args[1]
                # ir.VectorType.get([V], <elem>.mlir_type)
                if (
                    isinstance(vec_arg, ast.Call)
                    and isinstance(vec_arg.func, ast.Attribute)
                    and vec_arg.func.attr == "get"
                    and len(vec_arg.args) >= 2
                ):
                    elem_arg = vec_arg.args[1]
                    if (
                        isinstance(elem_arg, ast.Attribute)
                        and elem_arg.attr == "mlir_type"
                    ):
                        return ast.unparse(elem_arg.value)
            return None
        return None

    def _try_fuse(
        self,
        body: list[ast.stmt],
        *,
        allow_smem: bool = True,
        inherited: dict[str, ast.expr] | None = None,
    ) -> list[ast.stmt] | None:
        # Find top-level ``for offset in range(...)`` loops with matching
        # target and range signature -- they don't need to be adjacent in
        # the body. Two-pass reduction kernels typically have
        # post-reduction statements between the two loops.
        loops = [stmt for stmt in body if isinstance(stmt, ast.For) and not stmt.orelse]
        if len(loops) < 2:
            return None

        # Group loops by (target, range) signature -- preserve order.
        groups_by_key: dict[tuple[str, str], list[ast.For]] = {}
        for loop in loops:
            if not isinstance(loop.target, ast.Name):
                continue
            key = (loop.target.id, _node_text(loop.iter))
            groups_by_key.setdefault(key, []).append(loop)
        # Also process every later suffix.  A value may first be loaded in
        # sweep two and reused in sweep three; after the full group caches
        # sweep-one loads, its [1:] suffix handles that later-origin value.
        groups = [
            group[start:]
            for group in groups_by_key.values()
            for start in range(len(group) - 1)
        ]

        any_fused = False
        new_body = list(body)
        # A group rolled over a constexpr trip count is fused under a
        # trace-time budget on its exact fragment size
        # (``_wrap_dynamic_groups``); the pre-fusion statements are cloned
        # first so that branch can fall back to them.
        fallback_originals: dict[int, ast.stmt] = {}
        if any(
            isinstance(loop.target, ast.Name)
            and loop.target.id in self._dynamic_trip_counts
            for loop in loops
        ):
            fallback_originals = {
                id(stmt): cast("ast.stmt", clone_ast(stmt)) for stmt in body
            }
        dynamic_groups: list[tuple[ast.For, ast.For, str]] = []
        fused_spans: list[tuple[ast.For, ast.For]] = []
        scope_prefixes: dict[int, dict[str, ast.expr]] = {0: dict(inherited or {})}
        loop_alias_summaries: dict[ast.For, _LoopAliasNames] = {}

        def scope_before(stop: int) -> dict[str, ast.expr]:
            # Suffix groups revisit the same enclosing scopes. Reuse frozen
            # bindings only while every statement in the prefix is unchanged.
            start = max(index for index in scope_prefixes if index <= stop)
            definitions = dict(scope_prefixes[start])
            if start != stop:
                self._extend_definitions(new_body, start, stop, definitions)
                scope_prefixes[stop] = dict(definitions)
            return definitions

        for group in groups:
            if len(group) < 2:
                continue
            first_loop = group[0]
            range_args = _range_bounds(first_loop.iter)
            if range_args is None:
                continue
            start, end, step = range_args
            trip = _trip_count_for(
                start, end, step, self._constexpr_values, allow_dynamic_base=True
            )
            # A rolled reduction over a symbolic extent has no static trip
            # count; its strategy exposes the exact count as a constexpr
            # kernel parameter and the size-hint count for the policy
            # below.  The fragment is then allocated as ``TRIPS * slots``.
            trip_expr: str | None = None
            if trip is None:
                assert isinstance(first_loop.target, ast.Name)
                dynamic = self._dynamic_trip_counts.get(first_loop.target.id)
                if dynamic is None or _static_int(start, self._constexpr_values) != 0:
                    continue
                trip, trip_expr = dynamic
            # Require a static, bounded trip count.  Single-trip loops
            # (whole-row tiles) are the most profitable case: the cache
            # index folds to a constant and the fragment stays in
            # registers.  The ``cache_size`` cap is enforced below (and
            # is SMEM-aware), so we allow large trip counts here — the
            # SMEM backing path covers caches that wouldn't fit in a
            # per-thread register fragment.
            if trip is None or trip < 1 or trip > 2048:
                continue

            first_ctx = self._resolve_load_container(
                first_loop, start, step, trip, fold_single_trip=trip_expr is None
            )
            if first_ctx is None:
                continue
            first_container, first_cache_index, first_cache_size = first_ctx
            cache_size = first_cache_size
            first_loop_index = new_body.index(first_loop)
            first_scope_definitions = scope_before(first_loop_index)
            first_scope_definitions = self._container_definitions(
                first_loop,
                first_container,
                first_scope_definitions,
                first_loop,
                first_container,
            )
            if first_scope_definitions is None:
                continue

            # Gather ALL subsequent sweeps in the group that share the
            # first sweep's container shape.  Multi-pass kernels (e.g.
            # 3-pass softmax: max sweep, sum sweep, normalize sweep)
            # re-load the same values in every later sweep, so each one
            # should read the cache instead.
            sweeps: list[
                tuple[
                    int,
                    ast.For,
                    list[ast.stmt],
                    str,
                    dict[str, str],
                    dict[str, ast.expr],
                ]
            ] = []
            for loop_k in group[1:]:
                ctx_k = self._resolve_load_container(
                    loop_k, start, step, trip, fold_single_trip=trip_expr is None
                )
                if ctx_k is None or ctx_k[2] != cache_size:
                    continue
                body_idx = new_body.index(loop_k)
                alias_k = self._canonical_aliases(
                    self._build_second_alias(
                        first_loop, loop_k, summaries=loop_alias_summaries
                    )
                )
                scope_definitions = scope_before(body_idx)
                scope_definitions = self._container_definitions(
                    loop_k, ctx_k[0], scope_definitions, first_loop, first_container
                )
                if scope_definitions is None:
                    continue
                sweeps.append(
                    (
                        body_idx,
                        loop_k,
                        ctx_k[0],
                        ctx_k[1],
                        alias_k,
                        scope_definitions,
                    )
                )
            if not sweeps:
                continue
            # Default policy: register-backed cache only when
            # ``cache_size <= 64``; otherwise skip fusion.
            #
            # Why skip beyond 64 instead of switching to SMEM:
            # empirically (see task P10), the SMEM-backed cache eats
            # 25+ KB of per-CTA SMEM for the autotuner-picked
            # warp-reduction softmax config (cache_size=99, V=4,
            # threads=32), which crashes occupancy (~14% on B200).  In
            # parallel, the SMEM-write + barrier + SMEM-read overhead
            # outweighs the second-pass gmem read that the kernel's
            # natural L1 reuse mostly serves already.  The autotuner
            # accordingly steers to fusion-friendly configs (wider
            # block + more threads, cache_size=13) when fusion is
            # available, which still leaves the warp-reduction shape
            # ~13% faster overall.
            #
            # Env-var escape hatches for experimentation:
            #   HELION_FUSER_MODE=disabled — never fuse.
            #   HELION_FUSER_MODE=register — force register fragment up
            #     to cache_size=1024 (will spill / drop occupancy for
            #     large caches).
            #   HELION_FUSER_MODE=smem — force SMEM-backed cache up to
            #     cache_size=1024.  Requires ``thread_count`` to be
            #     plumbed correctly from the dispatch layer.
            import os

            # Effective per-thread cache footprint in ELEMENTS: for vec
            # loads each cache slot fans out into V scalar lanes, so the
            # raw slot count understates the register cost by V.
            _vec_m = re.search(r"VectorType\.get\(\[(\d+)\]", ast.unparse(first_loop))
            cache_elems = cache_size * (int(_vec_m.group(1)) if _vec_m else 1)
            # Both rolled and persistent reduction sweeps have a reload-mode
            # knob; tile loops retain the automatic policy.
            block_id = self._reduction_block_id(first_loop)
            is_reduction_sweep = block_id is not None
            _fuser_mode = os.environ.get("HELION_FUSER_MODE")
            if _fuser_mode is None:
                _fuser_mode = (
                    self._reload_modes.get(block_id, "auto")
                    if block_id is not None
                    else "auto"
                )
                # The config knob spells "never fuse" as ``"gmem"``
                # (re-load later sweeps from gmem/L2).
                if _fuser_mode == "gmem":
                    _fuser_mode = "disabled"
            if _fuser_mode in ("disabled", "computed"):
                continue
            if _fuser_mode == "register":
                use_smem = False
                fragment_cap = 1024
                if cache_elems > fragment_cap:
                    continue
            elif _fuser_mode == "smem":
                # The per-thread SMEM slot stride is baked into every slot
                # expression; a constexpr trip count has no static stride.
                if not allow_smem or trip_expr is not None:
                    continue
                use_smem = True
                fragment_cap = 1024
                if cache_elems > fragment_cap:
                    continue
            else:  # auto
                # Reduction sweeps cap on the true element footprint (the
                # vec path multiplies slots by V); tile-loop sweeps keep
                # the historical slot-count cap their tunings were
                # calibrated against.
                fragment_cap = 64
                if (cache_elems if is_reduction_sweep else cache_size) > fragment_cap:
                    continue
                use_smem = False
            cache_index = first_cache_index

            # Collect tracked loads from the first container.  Vec loads
            # (the ``unroll`` mode's U16 vec hoist) cache via a *scalar*
            # fragment of ``cache_size * V`` slots — one slot per
            # extracted lane — because the CUTLASS DSL's
            # ``cute.make_rmem_tensor(N, dtype)`` does not currently accept a
            # vec element type.  Scalar (masked / unmasked) loads cache as
            # the existing fast path.
            tracked: dict[str, tuple[int, str, str, str | None, int | None]] = {}
            tracked_statements: dict[str, ast.Assign] = {}
            tracked_tensor_roots: dict[str, frozenset[str] | None] = {}
            scalar_bypass_keys: set[str] = set()
            first_keys: dict[int, str] = {}
            first_definitions = dict(first_scope_definitions)
            first_analyzed = 0
            for j, s in enumerate(first_container):
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    kind = _load_kind(s.value, self._tensor_dtypes)
                    if kind is None:
                        continue
                    dtype = self._dtype_for_load_kind(s.value, kind)
                    if dtype is None and kind == "unmasked":
                        # An unmasked scalar load (mask elided because the
                        # extent divides the block) carries no dtype in its
                        # text; cache in the base tensor's OWN dtype so the
                        # roundtrip is exact for every consumer.
                        dtype = _unmasked_load_tensor_dtype(
                            s.value, self._tensor_dtypes
                        )
                    if dtype is None:
                        continue
                    v_width = _vec_width(s.value) if kind == "vec" else 1
                    if kind == "vec" and v_width is None:
                        continue
                    self._extend_definitions(
                        first_container, first_analyzed, j, first_definitions
                    )
                    first_analyzed = j
                    key = _canonical_load_text(s.value, {}, first_definitions)
                    if self._readonly_scalar_is_invariant(s, kind, first_definitions):
                        scalar_bypass_keys.add(key)
                    else:
                        scalar_bypass_keys.discard(key)
                    tracked[key] = (
                        j,
                        s.targets[0].id,
                        kind,
                        dtype,
                        v_width,
                    )
                    tracked_statements[key] = s
                    first_keys[id(s)] = key
                    tracked_tensor_roots[key] = _tensor_arg_roots(
                        s.value, set(self._tensor_dtypes)
                    )
            if not tracked:
                continue

            # Match loads in each subsequent sweep's container.
            # ``auto`` is a profitability policy, not an instruction to keep
            # reduction values live as far as correctness permits.  A disjoint
            # write is still a useful phase boundary: crossing it can extend
            # several per-lane fragments through unrelated reductions and put
            # a high-thread-count CTA on a much longer dependency chain.  Let
            # explicit register/smem choices opt into that tradeoff.  Preserve
            # the historical tile-loop policy, whose much shorter live ranges
            # are not selected through this reduction-reload knob.
            disjoint_pairs = (
                self._proven_disjoint_tensor_pairs
                if not is_reduction_sweep or _fuser_mode in ("register", "smem")
                else set()
            )
            memory_order = _LexicalMemoryOrder(
                new_body,
                set(self._tensor_dtypes),
                disjoint_pairs,
            )
            proven_memory_order = _LexicalMemoryOrder(
                new_body,
                set(self._tensor_dtypes),
                self._proven_disjoint_tensor_pairs,
            )
            per_sweep_matches: list[list[tuple[int, str, str]]] = []
            matched_keys: set[str] = set()
            for (
                _body_idx,
                loop_k,
                container_k,
                _cache_index_k,
                alias_k,
                scope_definitions,
            ) in sweeps:
                matches: list[tuple[int, str, str]] = []
                definitions_at_load = dict(scope_definitions)
                analyzed = 0
                for j, s in enumerate(container_k):
                    if (
                        isinstance(s, ast.Assign)
                        and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)
                    ):
                        kind = _load_kind(s.value, self._tensor_dtypes)
                        if kind is None:
                            continue
                        # Canonicalise this sweep's load text against the
                        # first sweep's variable names before keying.
                        self._extend_definitions(
                            container_k, analyzed, j, definitions_at_load
                        )
                        analyzed = j
                        key = _canonical_load_text(
                            s.value, alias_k, definitions_at_load
                        )
                        if (
                            key in scalar_bypass_keys
                            and not self._readonly_scalar_is_invariant(
                                s, kind, definitions_at_load
                            )
                        ):
                            scalar_bypass_keys.discard(key)
                        producer = tracked_statements.get(key)
                        load_roots = tracked_tensor_roots.get(key)
                        consumer_has_aliasing_store = (
                            memory_order.has_aliasing_store_in(loop_k, load_roots)
                        )
                        snapshot_safe = (
                            _fuser_mode == "register"
                            and consumer_has_aliasing_store
                            and self._consumer_store_preserves_snapshot(
                                loop_k,
                                container_k,
                                j,
                                s,
                                load_roots,
                            )
                        )
                        crosses_phase_boundary = (
                            memory_order.has_aliasing_store_between(
                                producer,
                                s,
                                load_roots,
                            )
                            if producer is not None
                            else True
                        )
                        # An explicit register policy may retain an exact
                        # in-place snapshot through the short writeback
                        # epilogue. Keep ``auto`` conservative: this tradeoff
                        # saves bandwidth but extends the fragment live range,
                        # which can regress small launches through pressure.
                        if snapshot_safe and crosses_phase_boundary:
                            crosses_phase_boundary = (
                                producer is None
                                or proven_memory_order.has_aliasing_store_between(
                                    producer,
                                    s,
                                    load_roots,
                                )
                            )
                        if (
                            producer is not None
                            and not memory_order.has_aliasing_store_in(
                                first_loop, load_roots
                            )
                            and (not consumer_has_aliasing_store or snapshot_safe)
                            and not crosses_phase_boundary
                        ):
                            matches.append((j, s.targets[0].id, key))
                            matched_keys.add(key)
                per_sweep_matches.append(matches)
            # Every matched occurrence retained the same frozen invariant key.
            # Keep its original loads in every sweep; later LICM independently
            # rechecks reaching definitions, aliasing, effects and profitability.
            matched_keys.difference_update(scalar_bypass_keys)
            per_sweep_matches = [
                [match for match in matches if match[2] not in scalar_bypass_keys]
                for matches in per_sweep_matches
            ]
            if not matched_keys:
                continue

            # Build cache declarations.  For vec loads, allocate
            # ``cache_size * V`` scalar slots; for scalar loads use the
            # original ``cache_size`` count.
            #
            # Backing:
            #   - Register fragment for ``cache_size <= 64``: lowest
            #     latency, no sync needed, fits in registers.
            #   - SMEM tensor for larger caches: allocated once at the
            #     top, indexed per-thread.  Sync inserted between the
            #     sweeps so the consume reads see populated slots.
            cache_names: dict[str, tuple[str, int]] = {}
            cache_decls: list[ast.stmt] = []
            # Trace-time conditions under which each constexpr-trip fragment
            # fits the budget the size hint was admitted with.
            fragment_caps: list[str] = []
            # Build the linear per-thread index expression covering all
            # populated thread-block axes (axis 0 = warp lanes, axis 1
            # = additional thread rows, axis 2 = z).  For a 1-D thread
            # block this is just thread_idx[0].  For a 2-D block (used
            # by per-row layernorm-style kernels) this picks up
            # thread_idx[1] * dim0 so each row gets its own SMEM
            # region.
            dim0, dim1, dim2 = self._thread_block_dims
            tid_terms = ["cutlass.Int32(cute.arch.thread_idx()[0])"]
            if dim1 > 1:
                tid_terms.append(f"cutlass.Int32(cute.arch.thread_idx()[1]) * {dim0}")
            if dim2 > 1:
                tid_terms.append(
                    f"cutlass.Int32(cute.arch.thread_idx()[2]) * {dim0 * dim1}"
                )
            tid_expr = " + ".join(tid_terms)
            for key, (_j, _name, _kind, dtype, vec_w) in tracked.items():
                if key not in matched_keys:
                    continue
                if dtype is None or vec_w is None:
                    continue
                cache = self._new_cache_name()
                cache_names[key] = (cache, vec_w)
                cache_total_per_thread = cache_size * vec_w
                if use_smem:
                    cache_total = cache_total_per_thread * self._thread_count
                    smem_ptr = f"{cache}_ptr"
                    cache_decls.extend(
                        [
                            statement_from_string(
                                f"{smem_ptr} = cute.arch.alloc_smem({dtype}, {cache_total})"
                            ),
                            statement_from_string(
                                f"{cache} = cute.make_tensor({smem_ptr}, ({cache_total},))"
                            ),
                        ]
                    )
                else:
                    cache_total_expr = str(cache_total_per_thread)
                    if trip_expr is not None:
                        # ``cache_size`` is ``trip * lanes``; scale the exact
                        # constexpr trip count by the same per-trip slots.
                        cache_total_expr = f"{trip_expr} * {cache_size // trip * vec_w}"
                    declaration = (
                        f"{cache} = cute.make_rmem_tensor({cache_total_expr}, {dtype})"
                    )
                    if trip_expr is not None:
                        # The size-hint fragment (``cache_total_per_thread``
                        # elements of ``dtype`` for this thread count) is what
                        # the policy admitted and any autotuning measured.  The
                        # exact fragment is bounded by it at trace time, so a
                        # kernel bound at a short extent and reused at a long
                        # one never grows the dynamically indexed per-thread
                        # array past the measured footprint: longer extents
                        # take the pre-fusion sweeps instead.  The group's
                        # sweeps are guarded by the same condition.
                        fragment_caps.append(
                            f"{cache_total_expr} <= {cache_total_per_thread}"
                        )
                        declaration = (
                            f"if cutlass.const_expr({fragment_caps[-1]}):\n"
                            f"    {declaration}"
                        )
                    cache_decls.append(statement_from_string(declaration))
            if not cache_names:
                continue

            # Helper: build the cache slot expression for a given outer
            # index expression and vec lane.
            #
            # SMEM cache layout: per-thread contiguous region —
            #   slot = tid * (cache_size * vec_w) + iter * vec_w + vi
            # This keeps each thread's slots clustered in SMEM (so the
            # compiler can issue vector stores/loads), and avoids bank
            # conflicts as long as the per-thread stride is not a
            # multiple of 32 * 4 bytes.
            #
            # For register fragments we keep the original per-thread
            # layout: slot = iter_idx * vec_w + vi.
            #
            # Capture loop-local ``cache_size`` / ``tid_expr`` /
            # ``use_smem`` via default args to satisfy ruff B023.
            def _slot_expr(
                idx_text: str,
                vec_w_: int,
                vi_text: str,
                _cache_size: int = cache_size,
                _tid_expr: str = tid_expr,
                _use_smem: bool = use_smem,
            ) -> str:
                if vec_w_ == 1:
                    inner = f"({idx_text})"
                else:
                    inner = f"({idx_text}) * {vec_w_} + ({vi_text})"
                if _use_smem:
                    return f"({_tid_expr}) * {_cache_size * vec_w_} + ({inner})"
                return inner

            # Rewrite the first container: append cache writes after each
            # tracked load assignment.  Vec loads write V scalars per
            # cache slot; scalar loads write a single scalar.
            new_first_body: list[ast.stmt] = []
            for s in first_container:
                new_first_body.append(s)
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    if _load_kind(s.value, self._tensor_dtypes) is None:
                        continue
                    key = first_keys.get(id(s))
                    if key is None:
                        continue
                    entry = cache_names.get(key)
                    if entry is None:
                        continue
                    cache, vec_w = entry
                    name = s.targets[0].id
                    if vec_w == 1:
                        slot = _slot_expr(cache_index, 1, "0")
                        new_first_body.append(
                            statement_from_string(f"{cache}[{slot}] = {name}")
                        )
                    else:
                        for v in range(vec_w):
                            slot = _slot_expr(cache_index, vec_w, str(v))
                            new_first_body.append(
                                statement_from_string(f"{cache}[{slot}] = {name}[{v}]")
                            )
            first_container[:] = new_first_body

            # Rewrite each subsequent sweep's container: replace each
            # matched load.  Scalar loads become a single cache read.
            # Vec loads are eliminated entirely (the consume sweep's
            # hoist disappears) and any downstream ``hoist_var[vi]``
            # extracts inside the nested constexpr V-loop are rewritten
            # to read from the cache at the appropriate slot expression.
            smem_barrier_positions: list[int] = []
            rewrote_vec_extracts = False
            for (
                body_idx,
                _loop_k,
                container_k,
                cache_index_k,
                _alias_k,
                _scope_definitions,
            ), matches in zip(sweeps, per_sweep_matches, strict=True):
                if not matches:
                    continue
                vec_extract_rewrites: list[
                    tuple[str, str, str, int, bool, int, str]
                ] = []
                new_sweep_body: list[ast.stmt] = []
                matched_by_index = {j: key for j, _name, key in matches}
                for statement_index, s in enumerate(container_k):
                    if (
                        isinstance(s, ast.Assign)
                        and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)
                    ):
                        kind = _load_kind(s.value, self._tensor_dtypes)
                        if kind is not None:
                            key = matched_by_index.get(statement_index)
                            if key is None:
                                new_sweep_body.append(s)
                                continue
                            entry = cache_names.get(key)
                            if entry is not None:
                                cache, vec_w = entry
                                name = s.targets[0].id
                                if vec_w == 1:
                                    slot = _slot_expr(cache_index_k, 1, "0")
                                    new_sweep_body.append(
                                        statement_from_string(
                                            f"{name} = {cache}[{slot}]"
                                        )
                                    )
                                else:
                                    # Drop the hoist entirely; remember
                                    # that ``name[vi]`` needs to be
                                    # rewritten to a cache read in
                                    # subsequent statements (especially
                                    # inside the constexpr V-loop).
                                    vec_extract_rewrites.append(
                                        (
                                            name,
                                            cache,
                                            cache_index_k,
                                            vec_w,
                                            use_smem,
                                            cache_size,
                                            tid_expr,
                                        )
                                    )
                                continue
                    new_sweep_body.append(s)
                rewrote_vec_extracts |= bool(vec_extract_rewrites)
                # Apply the vec extract rewrites recursively.
                for (
                    hoist_var,
                    cache,
                    idx_expr,
                    vec_w,
                    use_smem_,
                    cache_size_,
                    tid_expr_,
                ) in vec_extract_rewrites:
                    for stmt in new_sweep_body:
                        _rewrite_vec_extract(
                            stmt,
                            hoist_var,
                            cache,
                            idx_expr,
                            vec_w,
                            use_smem=use_smem_,
                            cache_size=cache_size_,
                            tid_expr=tid_expr_,
                        )
                container_k[:] = new_sweep_body
                smem_barrier_positions.append(body_idx)

            # SMEM-backed cache requires a CTA-wide barrier after the
            # populate sweep and before each consume sweep.  Insert in
            # descending position order so earlier insertions don't
            # shift later positions.
            if use_smem:
                for pos in sorted(smem_barrier_positions, reverse=True):
                    new_body.insert(
                        pos,
                        statement_from_string("cute.arch.sync_threads()"),
                    )

            rewritten = [
                loop_k
                for (_body_idx, loop_k, *_rest), matches in zip(
                    sweeps, per_sweep_matches, strict=True
                )
                if matches
            ]
            last_rewritten = rewritten[-1] if rewritten else first_loop
            fused_spans.append((first_loop, last_rewritten))
            if fragment_caps:
                dynamic_groups.append(
                    (first_loop, last_rewritten, " and ".join(fragment_caps))
                )

            # Declarations are inserted by ``transform`` at kernel scope.
            # This remains valid when the matching sweeps live below a
            # dynamic branch, where CuTe forbids local-memory allocation.
            self._root_declarations.extend(cache_decls)
            any_fused = True
            # Scalar cache rewrites replace statements only inside the first
            # and later loops. Barriers are inserted at later consumer indices,
            # so the strict prefix before the first loop is unchanged. Vector
            # extract rewrites mutate descendants in place: retain the original
            # full invalidation for potentially shared expression subtrees.
            scope_prefixes = (
                {0: dict(inherited or {})}
                if rewrote_vec_extracts
                else {
                    index: definitions
                    for index, definitions in scope_prefixes.items()
                    if index <= first_loop_index
                }
            )
            loop_alias_summaries.clear()

        if not any_fused:
            return None
        if dynamic_groups:
            new_body = _wrap_dynamic_groups(
                new_body, dynamic_groups, fused_spans, fallback_originals
            )
        return new_body

    def _transform_body(
        self,
        body: list[ast.stmt],
        *,
        under_dynamic_branch: bool,
        inherited: dict[str, ast.expr] | None = None,
        live_after: set[str] | None = None,
    ) -> list[ast.stmt]:
        """Transform children with the frozen bindings visible at their entry.

        Only a live for-body receives its induction token. Completed loops and
        loop-carried scalars retain the existing opaque versioning; an if arm
        inherits its entry without exporting bindings into its sibling arm.
        """
        entry = dict(inherited or {})
        transformed = list(body)
        definitions_before = dict(entry)
        analyzed = 0
        for index, stmt in enumerate(transformed):
            child_under_branch = under_dynamic_branch or isinstance(stmt, ast.If)
            children = [
                (field, child)
                for field in ("body", "orelse", "finalbody")
                if isinstance(child := getattr(stmt, field, None), list)
                and all(isinstance(child_stmt, ast.stmt) for child_stmt in child)
            ]
            if not children:
                continue
            # Advance after the preceding child transformations, so later
            # siblings inherit the bindings of the transformed statements.
            self._extend_definitions(transformed, analyzed, index, definitions_before)
            analyzed = index
            before = dict(definitions_before)
            for field, child in children:
                scope: dict[str, ast.expr] = {}
                child_live_after: set[str] = set()
                if "computed" in self._reload_modes.values():
                    from .computed_sweep_cache import live_before

                    child_live_after = live_before(
                        transformed[index + 1 :], live_after or set()
                    )
                    if isinstance(stmt, ast.For) and field == "body":
                        child_live_after.update(live_before(child, set()))
                if (
                    isinstance(stmt, ast.For)
                    and isinstance(stmt.target, ast.Name)
                    and not _scalar_bindings(stmt.iter)
                ):
                    if field == "body":
                        scope = (
                            self._container_definitions(
                                stmt, child, before, stmt, child
                            )
                            or {}
                        )
                    else:
                        scope = self._pure_definitions_before([stmt], 1, before)
                elif isinstance(stmt, ast.If) and not _scalar_bindings(stmt.test):
                    scope = before
                setattr(
                    stmt,
                    field,
                    self._transform_body(
                        child,
                        under_dynamic_branch=child_under_branch,
                        inherited=scope,
                        live_after=child_live_after,
                    ),
                )
        if "computed" in self._reload_modes.values():
            from .computed_sweep_cache import cache_computed_sweeps

            cache_computed_sweeps(self, transformed, entry, live_after or set())
        fused = self._try_fuse(
            transformed,
            allow_smem=not under_dynamic_branch,
            inherited=entry,
        )
        return transformed if fused is None else fused

    def transform(self, body: list[ast.stmt]) -> list[ast.stmt]:
        from .hoist_lane_invariant_reductions import _is_allowed_pure_call
        from .hoist_lane_invariant_reductions import _mutation_roots
        from .hoist_lane_invariant_reductions import _rmem_declaration

        self._written_names = set()
        body = _own_statement_occurrences(body, scalar_bindings=self._written_names)
        root = ast.Module(body=body, type_ignores=[])
        names = ReadWrites.from_ast(root)
        self._reserved_names = set(names.reads) | self._written_names
        # Computed caching creates new value expressions. Keep its original
        # eager binding path; demand covers only redundant-load replacement.
        self._demanded_bindings = (
            None
            if "computed" in self._reload_modes.values()
            else _binding_demand(body, self._tensor_dtypes)
        )
        memory = _LexicalMemoryOrder(
            body, set(self._tensor_dtypes), self._proven_disjoint_tensor_pairs
        )
        self._readonly_tensors = {
            name
            for name in self._tensor_dtypes
            if not memory.has_aliasing_store_in(root, frozenset((name,)))
        }
        if self._preserve_scalar_invariants:
            # The readonly facts cover all stores in the kernel, not just this pair
            # of sweeps. Unknown effects and non-register subscript/attribute writes
            # keep the existing cache route. No additional motion is authorized.
            self._scalar_bypass_mutations = _mutation_roots(root)
            local_registers = {
                stmt.targets[0].id
                for stmt in ast.walk(root)
                if isinstance(stmt, ast.Assign)
                and _rmem_declaration(stmt) is not None
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            }
            allocation_calls = {
                id(stmt.value)
                for stmt in ast.walk(root)
                if isinstance(stmt, ast.Assign) and _rmem_declaration(stmt) is not None
            }
            self._scalar_bypass_known_effects = (
                self._preserve_scalar_invariants
                and self._scalar_bypass_mutations <= local_registers
                and all(
                    _is_allowed_pure_call(node)
                    or _is_store_call(node)
                    # Existing fuser markers retain their original scalar fallback;
                    # recognizing their effects here never authorizes moving one.
                    or _unwrap_persistent_branch_vec_load(node) is not node
                    or id(node) in allocation_calls
                    for node in ast.walk(root)
                    if isinstance(node, ast.Call)
                )
            )
        transformed = self._transform_body(body, under_dynamic_branch=False)
        return [*self._root_declarations, *transformed]


def fuse_two_pass_loads(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None = None,
    *,
    thread_block_dims: tuple[int, int, int] = (1, 1, 1),
    tensor_dtypes: dict[str, str] | None = None,
    reload_modes: dict[int, str] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
    dynamic_trip_counts: dict[str, tuple[int, str]] | None = None,
    preserve_scalar_invariants: bool = False,
) -> list[ast.stmt]:
    """Apply two-pass load fusion to a list of statements (the device kernel
    body). Returns the (possibly modified) body.

    ``constexpr_values`` maps constexpr name -> static integer value so
    the pass can resolve ``range(..., step=cutlass.Int32(NAME))`` trip
    counts when NAME is inlined as a kernel-level constexpr.

    ``thread_block_dims`` is the launch-time thread-block shape
    ``(x, y, z)``.  Used to (a) decide between register-backed cache
    (small caches, occupancy-friendly) and SMEM-backed cache (large
    caches), and (b) build a per-thread linear slot index for the SMEM
    path so kernels with 2-D / 3-D thread blocks (layernorm,
    block-pointer softmax) don't have rows clobber each other's slots.

    ``proven_disjoint_tensor_pairs`` contains only argument-name pairs whose
    allocations cannot overlap.  Distinct argument names are otherwise
    conservatively treated as aliases across every store or atomic write.

    ``proven_tensor_stride_values`` supplies cache-keyed exact strides for the
    injectivity proof used by an in-place consume sweep. Without that proof,
    a store to the loaded tensor remains a hard cache barrier.

    ``dynamic_trip_counts`` maps the offset variable (sweep loop target) of
    a rolled reduction whose extent is symbolic to ``(size_hint_trips,
    constexpr_name)``: the profitability policy uses the hint, the emitted
    fragment is allocated as ``constexpr_name * slots_per_trip`` so it is
    exact at trace time, and the fused group is guarded by a trace-time check
    that the exact fragment does not exceed the size-hint fragment the policy
    admitted, falling back to the pre-fusion sweeps otherwise.  Every fused
    group that intersects such a guarded range is guarded with it.

    ``preserve_scalar_invariants`` leaves proven readonly scalar loads at their
    original sites instead of making a lane-indexed cache. The ordinary device
    pipeline enables this before its independently checked scalar LICM passes.
    It does not authorize any load motion or change the original expression.

    Safe to call on any kernel body — only rewrites when a strict pattern
    match succeeds.
    """
    transformer = _CuteFuseTwoPassLoads(
        constexpr_values=constexpr_values,
        thread_block_dims=thread_block_dims,
        tensor_dtypes=tensor_dtypes,
        reload_modes=reload_modes,
        proven_disjoint_tensor_pairs=proven_disjoint_tensor_pairs,
        proven_tensor_stride_values=proven_tensor_stride_values,
        dynamic_trip_counts=dynamic_trip_counts,
        preserve_scalar_invariants=preserve_scalar_invariants,
    )
    return transformer.transform(body)
