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
two for-loops' ``target`` / ``iter`` / range bounds.  A load that reads the
same element in every iteration (a scalar such as a dropout seed) takes a
single slot.

Later sweeps also read the first sweep's expensive pure values from
fragments (``sweep_values``): the results of pure inline asm, such as the
Philox chain of a word0 ``hl.rand`` draw, and with the explicit ``register``
policy transcendentals.  The computation and the loads only those values
needed are then removed from the later sweeps.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
import functools
import re
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import clone_ast
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _canonical_name
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import is_name_assign
from ._ast_pass_utils import live_in
from ._ast_pass_utils import loop_live_end
from ._ast_pass_utils import names_read
from ._ast_pass_utils import tensor_roots_may_alias
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .packet_access import call_write_roots
from .packet_access import is_store_call as _is_store_call  # noqa: F401
from .packet_access import match_packet_access
from .sweep_values import MIN_SAVED_COST
from .sweep_values import Cut
from .sweep_values import SweepValues
from .sweep_values import bound_names
from .sweep_values import eliminate_dead
from .sweep_values import heavy_cost
from .sweep_values import is_pure_expression
from .sweep_values import number_sweep
from .sweep_values import plan_cut

if TYPE_CHECKING:
    from collections.abc import Callable

_PERSISTENT_BRANCH_VEC_LOAD = "_helion_persistent_branch_vec_load"


def _pure_scalar_definition(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    has_memory_access = any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and (
                node.func.attr in ("load", "store")
                # ``cute.arch.atomic_add(ptr, v)`` and ``ptr.atomic_add(v)``.
                or node.func.attr.startswith("atomic_")
            )
            # A helper of the generated code may read or write memory besides
            # its ``_cute_load_``, ``_cute_store_`` and ``_cute_atomic_`` ones
            # (``_cute_flex_load_vec``, the rank-1 and resident loads): only
            # the integer modulo is known to compute from its operands alone
            # (the asm helper below).
            or isinstance(node.func, ast.Name)
            and node.func.id.startswith("_cute_")
            and node.func.id not in ("_cute_python_mod", "_cute_inline_asm_elementwise")
            # Asm not declared pure (a clock read, an atomic) may give each
            # run another value.
            or isinstance(node.func, ast.Name)
            and node.func.id == "_cute_inline_asm_elementwise"
            and not any(
                keyword.arg == "is_pure"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
        )
        # An element of a named fragment: a store through another name
        # bound to the same fragment (``p = offs; offs[0] = 8``) changes it
        # without a write of the name read here.
        or isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        for node in ast.walk(stmt.value)
    )
    return None if has_memory_access else (stmt.targets[0].id, stmt.value)


def _define(
    definitions: dict[str, ast.expr],
    depends: dict[str, frozenset[str]],
    writes: frozenset[str],
    definition: tuple[str, ast.expr] | None,
    definition_depends: frozenset[str],
) -> None:
    """Advance ``definitions`` (and ``depends``, the names whose writes end
    each: its own and those it reads) past a statement that writes
    ``writes`` and makes ``definition``.  A write ends the definition of the
    written name and of every name defined from it: ``c = k * 2`` no longer
    spells ``c`` once ``k`` changes.  The names are compared as
    ``_definition_effects`` gives them, through the final rename's groups."""
    for name in [name for name in definitions if not depends[name].isdisjoint(writes)]:
        del definitions[name]
        del depends[name]
    if definition is not None:
        name, value = definition
        definitions[name] = value
        depends[name] = definition_depends


def _definition_effects(
    stmt: ast.stmt, canonical: Callable[[str], str]
) -> tuple[frozenset[str], tuple[str, ast.expr] | None, frozenset[str]]:
    """What ``stmt`` writes, the pure definition it makes, and the names whose
    writes end that definition (its own and those it reads), all through
    ``canonical``.  A loop writes its target (``for base in range(2)``), and
    a write of a name the final rename merges into ``idx`` (``idx_next =
    ...``) is a write of ``idx``; a definition that reads its own name (``k =
    k + 1``, or ``idx_next = idx + 1``) spells a value no name holds any
    more, so it defines nothing."""
    writes = frozenset(map(canonical, bound_names(stmt)))
    definition = _pure_scalar_definition(stmt)
    if definition is None:
        return writes, None, frozenset()
    target = canonical(definition[0])
    reads = frozenset(map(canonical, names_read(definition[1])))
    if target in reads:
        return writes, None, frozenset()
    return writes, definition, reads | {target}


def _definition_names(
    name: str, value: ast.expr, canonical: Callable[[str], str]
) -> set[str]:
    """The names whose writes end the definition ``name = value``, through
    ``canonical``: its own and those ``value`` reads (``_define``)."""
    return {canonical(each) for each in {name} | _value_names(value)}


def _value_names(value: ast.expr) -> set[str]:
    """The names the expression ``value`` reads.  Every name in a value is a
    read, so their ``ctx`` is not consulted: a name built without one has no
    such attribute before Python 3.13."""
    return {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}


@dataclass
class _PrefixState:
    statements: list[ast.stmt]
    stop: int
    writes: set[str]
    definitions: dict[str, ast.expr]
    # The names whose writes end each of ``definitions`` (``_define``).
    depends: dict[str, frozenset[str]]


class _AnalysisCache:
    """Read-only facts for one unchanged AST epoch of a sibling fusion scope.

    Every successful rewrite clears all facts before mutating the AST. Neither
    mutable expressions nor statement-list prefixes are cached across scopes,
    passes, or configurations. Capacity limits only decline caching.
    """

    _CAPACITY = 512

    def __init__(self, canonical: Callable[[str], str]) -> None:
        # Definitions compare names through the final rename's groups.
        self._canonical = canonical
        self._effects: dict[
            ast.stmt,
            tuple[frozenset[str], tuple[str, ast.expr] | None, frozenset[str]],
        ] = {}
        self._scopes: dict[int, _PrefixState] = {}
        self._prefixes: OrderedDict[
            tuple[int, int], tuple[frozenset[str], dict[str, ast.expr]]
        ] = OrderedDict()
        self._aliases: dict[ast.For, dict[str, str]] = {}

    def clear(self) -> None:
        self._effects.clear()
        self._scopes.clear()
        self._prefixes.clear()
        self._aliases.clear()

    def pure_definitions_before(
        self,
        statements: list[ast.stmt],
        stop: int,
        inherited: dict[str, ast.expr] | None,
    ) -> dict[str, ast.expr]:
        key = (id(statements), stop)
        prefix = self._prefixes.get(key)
        if prefix is None:
            state = self._scopes.get(id(statements))
            if state is None or stop < state.stop:
                state = _PrefixState(statements, 0, set(), {}, {})
                if len(self._scopes) < self._CAPACITY or id(statements) in self._scopes:
                    self._scopes[id(statements)] = state
            for stmt in statements[state.stop : stop]:
                effects = self._effects.get(stmt)
                if effects is None:
                    effects = _definition_effects(stmt, self._canonical)
                    if len(self._effects) < self._CAPACITY:
                        self._effects[stmt] = effects
                writes, definition, definition_depends = effects
                state.writes.update(writes)
                _define(
                    state.definitions,
                    state.depends,
                    writes,
                    definition,
                    definition_depends,
                )
            state.stop = stop
            prefix = (frozenset(state.writes), dict(state.definitions))
            # Retain the list as well as its id so a temporary container cannot
            # be collected and its id reused while a prefix is cached.
            if self._scopes.get(id(statements)) is state:
                self._prefixes[key] = prefix
                if len(self._prefixes) > self._CAPACITY:
                    self._prefixes.popitem(last=False)
        else:
            self._prefixes.move_to_end(key)
        writes, local = prefix
        # An inherited definition ends where the prefix writes its name or a
        # name it reads (``_define``).
        definitions = {
            name: value
            for name, value in (inherited or {}).items()
            if writes.isdisjoint(_definition_names(name, value, self._canonical))
        }
        definitions.update(local)
        return definitions

    def loop_aliases(self, loop: ast.For) -> dict[str, str]:
        aliases = self._aliases.get(loop)
        if aliases is None:
            aliases = {}
            # The first vector-lane loop target, in ast.walk's order.
            for node in ast.walk(loop):
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id.startswith("vec_lane_")
                ):
                    aliases.setdefault("vec_lane_", node.target.id)
            if len(self._aliases) < self._CAPACITY:
                self._aliases[loop] = aliases
        return aliases


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
    """Return the Call if ``node`` is a ``cute.arch.load(...)`` -- the hoisted
    U16 vec load emitted by the LoopedReductionStrategy ``unroll`` mode, or a
    cache-hinted scalar load -- or a cache-hinted vector load helper of the
    same call shape (``_cute_load_l2_evict_last_8b(ptr, vec_type)``, see
    ``cache_policy_loads``).  ``_vec_width`` tells the packets apart.
    """
    access = match_packet_access(node)
    if access is not None and access.kind in ("arch_load", "packet", "hinted_packet"):
        return access.call
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
    """Extract the V from ``cute.arch.load(ptr, ir.VectorType.get([V], ...))``
    or its L2-hinted helper form.

    Returns None when the expression isn't a recognised vec load.
    """
    access = match_packet_access(node)
    if access is not None and access.kind in ("packet", "hinted_packet"):
        return access.width
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


def _contains_arch_attribute(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "arch"
        for child in ast.walk(node)
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
    for child in ast.walk(node):
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
    """Resolve the destination tensor arguments of a generated write
    (``call_write_roots`` over ``_tensor_arg_roots``)."""
    return call_write_roots(
        node, lambda pointer: _tensor_arg_roots(pointer, tensor_names)
    )


class _LexicalMemoryOrder(ast.NodeVisitor):
    """Track assignments and writes in generated source execution order: the
    ``stores`` are every call that may write memory, a call whose effects are
    unknown included."""

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
        for stmt in body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignment_positions[id(node)] = self._position
        self._position += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Every call that may write memory orders the sweeps: a generated
        # write, and a call whose effects are unknown (impure inline asm),
        # which may write any tensor (``call_write_roots``).
        roots = _store_tensor_roots(node, self._tensor_names)
        if roots != frozenset():
            self.stores.append((self._position, roots))
            self._position += 1
        self.generic_visit(node)

    def _may_alias(
        self,
        store_roots: frozenset[str] | None,
        load_roots: frozenset[str] | None,
    ) -> bool:
        return tensor_roots_may_alias(
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
        """Whether ``node`` holds a write that may reach ``load_roots`` (a
        call whose effects are unknown reaches every tensor)."""
        return any(
            isinstance(child, ast.Call)
            and (roots := _store_tensor_roots(child, self._tensor_names)) != frozenset()
            and self._may_alias(roots, load_roots)
            for child in ast.walk(node)
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


def _canonical_load_text(
    node: ast.AST,
    lane_var_alias: dict[str, str],
    definitions: dict[str, ast.expr] | None = None,
) -> str:
    """Stringify a load node with ``definitions`` inlined and the later
    sweep's lane variable spelled as the first sweep's (``lane_var_alias``).

    Each sweep binds names of its own (``reduction_lane_base_1`` in the
    reduce sweep, ``reduction_lane_base_2`` in the consume sweep), which the
    inlined definitions spell out, so loads of the same element compare
    equal; only the ``vec_lane_<N>`` loop variable is renamed.
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


def _nested_lane_loop(loop: ast.For) -> ast.For | None:
    """The lane loop of the sweep loop ``loop``: its one nested ``for``, in
    the shapes ``_resolve_load_container`` accepts."""
    nested = [statement for statement in loop.body if isinstance(statement, ast.For)]
    return nested[0] if len(nested) == 1 else None


def _jumps(body: list[ast.stmt]) -> bool:
    """Whether ``body`` holds a ``break`` or ``continue``.  The backward
    liveness the sweep-value cut deletes by (``live_in``) lets both fall
    through, so a statement only a jump's path reads would look dead."""
    return any(
        isinstance(node, (ast.Break, ast.Continue))
        for statement in body
        for node in ast.walk(statement)
    )


def _exits_early(loop: ast.For) -> bool:
    """Whether a ``return`` in ``loop``, or a ``break`` or ``continue`` of
    ``loop`` or of its lane loop, may end an iteration early: the statements
    after it then do not run in every iteration, and a later sweep would
    read slots this sweep never wrote.  A ``break`` or ``continue`` of
    another loop nested in it ends only that loop."""
    lane = _nested_lane_loop(loop)
    own = {loop} if lane is None else {loop, lane}

    def exits(node: ast.AST, innermost: ast.AST) -> bool:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, (ast.Break, ast.Continue)):
            return innermost in own
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # A loop's ``else`` runs after it, in the enclosing loop.
            return any(exits(child, node) for child in node.body) or any(
                exits(child, innermost) for child in node.orelse
            )
        return any(exits(child, innermost) for child in ast.iter_child_nodes(node))

    return any(exits(statement, loop) for statement in loop.body)


def _rebinds_index(loop: ast.For, canonical: Callable[[str], str]) -> bool:
    """Whether the body of the sweep loop ``loop`` binds its index or its
    lane loop's other than as the loop target, comparing names through
    ``canonical``: a cache slot is the one the indices name, so a rebound
    index names another iteration's slot, and the final rename makes a write
    of ``lane_next`` (merged into an index) a write of that index."""
    lane = _nested_lane_loop(loop)
    loops = [loop] if lane is None else [loop, lane]
    indices = {
        canonical(each.target.id) for each in loops if isinstance(each.target, ast.Name)
    }
    targets = {id(each.target) for each in loops}
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and canonical(node.id) in indices
        and id(node) not in targets
        for statement in loop.body
        for node in ast.walk(statement)
    )


def _subscript_reads(node: ast.AST) -> set[str]:
    """The names ``node`` reads elements of (``offs`` in ``offs[0]``)."""
    return {
        each.value.id
        for each in ast.walk(node)
        if isinstance(each, ast.Subscript) and isinstance(each.value, ast.Name)
    }


def _stores_elements(*nodes: ast.AST) -> bool:
    """Whether ``nodes`` store into an element, through a subscript or by a
    call that stores into a fragment in place (``fragment_writes``, the
    fragment grouped reduce).  Another name may hold the same fragment
    (``q = offs``; ``q[0] = 8``), so such a store may change an element
    whatever name reads it."""
    from .packet_access import fragment_writes

    return any(
        isinstance(each, ast.Subscript)
        and isinstance(each.ctx, (ast.Store, ast.Del))
        or isinstance(each, ast.Call)
        and fragment_writes(each)
        for node in nodes
        for each in ast.walk(node)
    )


def _key_names(key: str) -> set[str]:
    """The names a load key (``_canonical_load_text``) reads."""
    return names_read(ast.parse(key, mode="eval"))


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
        binding_dtypes: dict[str, str] | None = None,
        rename_groups: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._counter = 0
        self._constexpr_values = constexpr_values or {}
        # Names the final rename merges: the checks and liveness compare
        # canonical names.
        self._renames = rename_groups or {}
        # The DSL type of names their binding statement does not show.
        self._binding_dtypes = binding_dtypes or {}
        # Element type of each fragment filled so far, by name: a later
        # sweep's ``cache[slot]`` read is that sweep-invariant value.
        self._cache_dtypes: dict[str, str] = {}
        # Whether the kernel body holds a ``break`` or ``continue``
        # (``_jumps``): it then carries no values.
        self._jumps = False
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
        *,
        analysis: _AnalysisCache | None = None,
    ) -> dict[str, ast.expr]:
        """Collect dominating, side-effect-free scalar definitions that still
        spell their names' values (``_define``), comparing names through the
        final rename's groups as ``analysis`` does."""
        if analysis is not None:
            return analysis.pure_definitions_before(statements, stop, inherited)
        definitions = dict(inherited or {})
        depends = {
            name: frozenset(_definition_names(name, value, self._canonical))
            for name, value in definitions.items()
        }
        for stmt in statements[:stop]:
            _define(definitions, depends, *_definition_effects(stmt, self._canonical))
        return definitions

    def _iteration_definitions(
        self, definitions: dict[str, ast.expr], loop: ast.For
    ) -> dict[str, ast.expr]:
        """The ``definitions`` holding where ``loop`` starts that hold in
        every iteration.  A definition of a name the loop writes holds in
        the first iteration only, and one that reads such a name means that
        name's value before the loop, which the name no longer spells."""
        written = self._bound(loop)
        return {
            name: value
            for name, value in definitions.items()
            if not _definition_names(name, value, self._canonical) & written
        }

    def _container_definitions(
        self,
        loop: ast.For,
        scope: dict[str, ast.expr],
        *,
        analysis: _AnalysisCache,
    ) -> dict[str, ast.expr]:
        """The definitions holding where each iteration of the load container
        of ``loop`` starts (``_resolve_load_container``): ``scope``, those
        holding in every iteration of ``loop``, and for a nested lane loop
        also those of ``loop``'s statements before it that hold in each of
        its iterations."""
        lane = _nested_lane_loop(loop)
        if lane is None:
            return scope
        return self._iteration_definitions(
            self._pure_definitions_before(
                loop.body, loop.body.index(lane), scope, analysis=analysis
            ),
            lane,
        )

    def _sweep_definitions(
        self, body: list[ast.stmt], index: int, analysis: _AnalysisCache
    ) -> dict[str, ast.expr]:
        """The definitions holding where each iteration of the load container
        of the sweep loop ``body[index]`` starts (``_container_definitions``)."""
        loop = cast("ast.For", body[index])
        return self._container_definitions(
            loop,
            self._iteration_definitions(
                self._pure_definitions_before(body, index, analysis=analysis), loop
            ),
            analysis=analysis,
        )

    def _unstable_names(
        self,
        first_loop: ast.For,
        loop: ast.For,
        between: list[ast.stmt],
        alias: dict[str, str],
    ) -> set[str]:
        """Names a load key of the later sweep ``loop`` may not read to match
        a first-sweep load, spelled as the first sweep spells them (``alias``):
        a name either loop writes or a statement ``between`` them rebinds,
        other than the indices both sweeps loop over, which hold equal values
        at equal iterations; in the later sweep the first sweep's own lane
        index holds its last value.  A key inlines the definitions that hold,
        so a name either loop writes stays in it only where the text does not
        show its value: a value carried from the iteration before, or a loaded
        index (``t = y[lane]`` in one sweep and ``t = z[lane]`` in the
        other)."""
        first_indices = {
            self._canonical(cast("ast.Name", nested.target).id)
            for nested in self._sweep_loops(first_loop)
        }
        later_indices = {
            self._canonical(alias.get(index, index))
            for index in (
                cast("ast.Name", nested.target).id for nested in self._sweep_loops(loop)
            )
        }
        indices = first_indices & later_indices
        later = {self._canonical(alias.get(name, name)) for name in bound_names(loop)}
        return (self._bound(first_loop, *between) | later) - indices

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

    def _canonical(self, name: str) -> str:
        return _canonical_name(name, self._renames)

    def _reads(self, node: ast.AST) -> set[str]:
        return {self._canonical(name) for name in ReadWrites.from_ast(node).reads}

    def _bound(self, *nodes: ast.AST) -> set[str]:
        """The canonical names ``nodes`` may bind."""
        return {self._canonical(name) for node in nodes for name in bound_names(node)}

    def _key_reads(self, key: str) -> set[str]:
        """The canonical names a load key (``_canonical_load_text``) reads."""
        return {self._canonical(name) for name in _key_names(key)}

    def _assigned(self, node: ast.AST) -> str | None:
        name = assigned_name(node)
        return None if name is None else self._canonical(name)

    def _removable(self, statement: ast.stmt) -> bool:
        """An assignment that may be dropped once nothing reads it: a pure
        value or a plain load."""
        if not isinstance(statement, ast.Assign):
            return False
        if is_pure_expression(statement.value):
            return True
        return _load_kind(statement.value, self._tensor_dtypes) is not None and all(
            _is_proven_relocatable_call(node, allow_load=True)
            or call_path_str(node.func) == _PERSISTENT_BRANCH_VEC_LOAD
            for node in ast.walk(statement.value)
            if isinstance(node, ast.Call)
        )

    def _cache_declarations(
        self,
        cache: str,
        dtype: str,
        *,
        slots: int,
        use_smem: bool,
        trip: int,
        trip_expr: str | None,
        fragment_caps: list[str],
    ) -> list[ast.stmt]:
        """Allocate ``cache``: ``slots`` elements per thread, in shared memory
        or in a register fragment sized by the exact constexpr trip count
        (``fragment_caps`` collects its trace-time budget)."""
        self._cache_dtypes[cache] = dtype
        if use_smem:
            cache_total = slots * self._thread_count
            smem_ptr = f"{cache}_ptr"
            return [
                statement_from_string(
                    f"{smem_ptr} = cute.arch.alloc_smem({dtype}, {cache_total})"
                ),
                statement_from_string(
                    f"{cache} = cute.make_tensor({smem_ptr}, ({cache_total},))"
                ),
            ]
        cache_total_expr = str(slots)
        if trip_expr is not None:
            # ``slots`` is ``trip * per-trip slots``; scale the exact
            # constexpr trip count by the same per-trip slots.
            cache_total_expr = f"{trip_expr} * {slots // trip}"
        declaration = f"{cache} = cute.make_rmem_tensor({cache_total_expr}, {dtype})"
        if trip_expr is not None:
            # The size-hint fragment (``slots`` elements of ``dtype`` for this
            # thread count) is what the policy admitted and any autotuning
            # measured.  The exact fragment is bounded by it at trace time, so
            # a kernel bound at a short extent and reused at a long one never
            # grows the dynamically indexed per-thread array past the measured
            # footprint: longer extents take the pre-fusion sweeps instead.
            # The group's sweeps are guarded by the same condition.
            fragment_caps.append(f"{cache_total_expr} <= {slots}")
            declaration = (
                f"if cutlass.const_expr({fragment_caps[-1]}):\n    {declaration}"
            )
        return [statement_from_string(declaration)]

    @staticmethod
    def _sweep_loops(loop: ast.For) -> tuple[ast.For, ...]:
        """The sweep loop and, for a nested lane loop (the shapes
        ``_resolve_load_container`` accepts), that lane loop."""
        lane = _nested_lane_loop(loop)
        return (loop,) if lane is None else (loop, lane)

    @staticmethod
    def _has_heavy_operation(container: list[ast.stmt], transcendentals: bool) -> bool:
        return any(
            isinstance(node, ast.Call)
            and heavy_cost(node, transcendentals=transcendentals)
            for statement in container
            for node in ast.walk(statement)
        )

    def _lane_invariant_keys(
        self,
        first_loop: ast.For,
        tracked: dict[str, tuple[int, str, str, str | None, int | None]],
    ) -> set[str]:
        """Tracked scalar loads that read the same element in every
        iteration: their key (address with the definitions inlined that hold
        in every iteration) names neither a loop index nor anything the loop
        writes, compared through the final rename's groups."""
        written = self._bound(first_loop)
        return {
            key
            for key, (_j, _name, kind, _dtype, _vec_w) in tracked.items()
            if kind != "vec" and not self._key_reads(key) & written
        }

    def _live_after_container(
        self,
        body: list[ast.stmt],
        loops: tuple[ast.For, ...],
        live_out: Callable[[], frozenset[str]],
    ) -> set[str]:
        """Canonical names live after the loop whose body is a sweep's load
        container (the lane loop of a nested sweep)."""
        after = live_in(
            body[body.index(loops[0]) + 1 :],
            live_out(),
            reads=self._reads,
            assigned=self._assigned,
        )
        if len(loops) == 1:
            return after
        outer, lane = loops
        end = loop_live_end(outer, after, reads=self._reads, assigned=self._assigned)
        return live_in(
            outer.body[outer.body.index(lane) + 1 :],
            end,
            reads=self._reads,
            assigned=self._assigned,
        )

    def _plan_sweep_values(
        self,
        body: list[ast.stmt],
        first_loop: ast.For,
        later_loops: list[ast.For],
        *,
        first_loads: dict[int, str],
        later_loads: list[dict[int, str]],
        scope_loads: dict[int, str],
        load_dtypes: dict[str, str],
        invariant_keys: set[str],
        transcendentals: bool,
        live_out: Callable[[], frozenset[str]],
    ) -> tuple[SweepValues, dict[int, ast.stmt], list[Cut | None]] | None:
        """Number the group's sweeps and choose, per later sweep, the values
        it reads from the first sweep instead of recomputing them.

        ``*_loads`` give the load statements proven to read a first-sweep
        load's value (its key), in the first sweep, each later sweep and the
        statements between the loops.  Returns the numbering, the first
        sweep's statement computing each number, and each later sweep's cut,
        or None when no sweep saves enough to carry anything.
        """
        values = SweepValues(
            transcendentals=transcendentals,
            cache_dtypes=self._cache_dtypes,
            binding_dtypes=self._binding_dtypes,
            canonical=self._canonical,
        )

        def numbers(loads: dict[int, str]) -> dict[int, int]:
            return {
                statement_id: values.load(
                    key, load_dtypes.get(key), variant=key not in invariant_keys
                )
                for statement_id, key in loads.items()
            }

        # One forward walk over the enclosing statements: every sweep sees the
        # same numbers for the definitions it shares with the others.
        sweep_ids = {id(first_loop), *(id(loop) for loop in later_loops)}
        last = max(body.index(loop) for loop in (first_loop, *later_loops))
        env: dict[str, int] = {}
        snapshots: dict[int, dict[str, int]] = {}
        scope = numbers(scope_loads)
        for statement in body[: last + 1]:
            if id(statement) in sweep_ids:
                snapshots[id(statement)] = dict(env)
            values.run([statement], env, loads=scope, scope=None)

        def number(loop: ast.For, loads: dict[int, str]) -> dict[int, int]:
            loops = self._sweep_loops(loop)
            prefix = loop.body[: loop.body.index(loops[1])] if len(loops) > 1 else []
            return number_sweep(
                values,
                loops,
                prefix,
                loops[-1].body,
                snapshots[id(loop)],
                loads=numbers(loads),
            )

        first_loops = self._sweep_loops(first_loop)
        first_numbers = number(first_loop, first_loads)
        first: dict[int, ast.stmt] = {}
        for statement in first_loops[-1].body:
            value = first_numbers.get(id(statement))
            if value is not None:
                first.setdefault(value, statement)
        cuts: list[Cut | None] = []
        for loop, loads in zip(later_loops, later_loads, strict=True):
            loops = self._sweep_loops(loop)
            # The loop indices compare equal across sweeps: ``_try_fuse``
            # groups only sweeps that run the first sweep's indices, with as
            # many loops.
            if len(loops) != len(first_loops):
                cuts.append(None)
                continue
            cut = plan_cut(
                values,
                loops[-1].body,
                number(loop, loads),
                first,
                live_after=self._live_after_container(body, loops, live_out),
                loop_targets={self._canonical(cast("ast.Name", loops[-1].target).id)},
                reads=self._reads,
                assigned=self._assigned,
                removable=self._removable,
            )
            cuts.append(cut if cut.cached and cut.saved >= MIN_SAVED_COST else None)
        if not any(cuts):
            return None
        return values, first, cuts

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
        from ._ast_pass_utils import rename_names
        from .persistent_branch_vec import _definition_snapshots
        from .persistent_branch_vec import _lane_accesses_are_iteration_independent
        from .persistent_branch_vec import _memory_load_calls

        if load_roots is None or container is not loop.body:
            return False
        if self._renames:
            # Prove on the loop as the final rename spells it, where a name
            # stands for its whole group: ``idx_next = idx + 1`` writes ``idx``.
            loop = rename_names(cast("ast.For", clone_ast(loop)), self._renames)
            container = loop.body
            load_stmt = cast("ast.Assign", container[load_index])
        if not isinstance(loop.target, ast.Name):
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
            for write_call in (
                child for child in ast.walk(statement) if isinstance(child, ast.Call)
            ):
                write_roots = _store_tensor_roots(write_call, tensor_names)
                if write_roots == frozenset() or not tensor_roots_may_alias(
                    write_roots,
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
                    write_call,
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
        # The cache index takes the lane variable for the lane's slot, which
        # it is only when the loop counts from 0 in steps of 1 under a name of
        # its own: a lane loop reusing the sweep's target hides the sweep
        # index the slot reads too.
        if (
            _static_int(lane_start, self._constexpr_values) != 0
            or _static_int(lane_step, self._constexpr_values) != 1
            or lane_loop.target.id == outer_loop.target.id
        ):
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
        analysis: _AnalysisCache | None = None,
    ) -> dict[str, str]:
        """The rename that spells the vector-lane loop variable of the later
        sweep ``second_loop`` (``vec_lane_<N>``, a ``cutlass.range_constexpr``
        loop target of the PerThreadNDTileStrategy) as the first sweep
        spells it, so that the two sweeps' load keys compare equal.  Names
        the sweeps assign are not renamed: a key spells such a name by its
        definition where that holds in each iteration, and otherwise the
        name is written in a sweep, which no key may read
        (``_unstable_names``).  Nothing is renamed either when the later
        sweep reads, compared through the final rename's groups, the first
        sweep's variable, or its own under another name: the first is that
        variable's last value, or a value a statement between the sweeps
        gives it, which the rename would turn into the later sweep's own
        index, and the second is that index under a spelling the rename
        does not change."""
        analysis = analysis or _AnalysisCache(self._canonical)
        first = analysis.loop_aliases(first_loop).get("vec_lane_")
        second = analysis.loop_aliases(second_loop).get("vec_lane_")
        if first is None or second is None or first == second:
            return {}
        variables = {self._canonical(first), self._canonical(second)}
        if any(
            name != second and self._canonical(name) in variables
            for name in names_read(second_loop)
        ):
            return {}
        return {second: first}

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
        live_out: Callable[[], frozenset[str]],
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
        analysis = _AnalysisCache(self._canonical)
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
            if (
                first_ctx is None
                or _exits_early(first_loop)
                or _rebinds_index(first_loop, self._canonical)
            ):
                continue
            first_container, first_cache_index, first_cache_size = first_ctx
            cache_size = first_cache_size
            first_loop_index = new_body.index(first_loop)
            first_scope_definitions = self._sweep_definitions(
                new_body, first_loop_index, analysis
            )

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
                if (
                    ctx_k is None
                    or ctx_k[2] != cache_size
                    or _rebinds_index(loop_k, self._canonical)
                ):
                    continue
                body_idx = new_body.index(loop_k)
                # The lane loops' ranges must hold the values of the names they
                # read from the first sweep through the later one, as the sweep
                # loops' ranges below: the cache holds the lanes their static
                # trip counts name (a constexpr ``K`` rebound between the
                # sweeps runs another lane count over the slots).
                if {
                    name
                    for lane in map(_nested_lane_loop, (first_loop, loop_k))
                    if lane is not None
                    for name in names_read(lane.iter)
                } & {
                    name
                    for statement in new_body[first_loop_index : body_idx + 1]
                    for name in ReadWrites.from_ast(statement).writes
                }:
                    continue
                # A range bound read from a fragment element may change
                # through another name for the fragment.
                if _subscript_reads(loop_k.iter) and _stores_elements(
                    *new_body[first_loop_index : body_idx + 1]
                ):
                    continue
                # Equal range text runs the first sweep's indices only while
                # the names it reads keep their values, through the later
                # sweep too: its cache index reads them in every iteration.
                if {
                    self._canonical(name) for name in names_read(loop_k.iter)
                } & self._bound(*new_body[first_loop_index : body_idx + 1]):
                    continue
                alias_k = self._canonical_aliases(
                    self._build_second_alias(first_loop, loop_k, analysis=analysis)
                )
                scope_definitions = self._sweep_definitions(
                    new_body, body_idx, analysis
                )
                # A key inlines these definitions, and one may read the first
                # sweep's lane (``saved = vec_lane_1`` between the sweeps, or
                # ``saved = idx`` with ``idx`` merged into it), its last
                # value: renaming the later sweep's lane to it would take
                # that value for the later lane.  Without the rename the
                # first sweep's lane is a name it writes (``_unstable_names``).
                if any(
                    self._canonical(first) in self._reads(value)
                    for first in alias_k.values()
                    for value in scope_definitions.values()
                ):
                    alias_k = {}
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
            if _fuser_mode == "register_prefetch":
                _fuser_mode = "register"
            if _fuser_mode == "disabled":
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
            first_keys: dict[int, str] = {}
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
                    first_definitions = self._pure_definitions_before(
                        first_container, j, first_scope_definitions, analysis=analysis
                    )
                    key = _canonical_load_text(s.value, {}, first_definitions)
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
            # Later reduction sweeps may also read the first sweep's expensive
            # pure values: its pure inline asm (the Philox chain of a word0
            # ``hl.rand`` draw) whenever the policy fuses into registers, and
            # its transcendentals when it explicitly accepts the registers.
            # Tile loops, which no reload knob governs, carry loads only: the
            # autotuner could not turn a costly carry off.
            transcendentals = _fuser_mode == "register"
            carry_values = (
                is_reduction_sweep
                and not use_smem
                and not self._jumps
                and self._has_heavy_operation(first_container, transcendentals)
            )
            if not tracked and not carry_values:
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
                body_idx,
                loop_k,
                container_k,
                _cache_index_k,
                alias_k,
                scope_definitions,
            ) in sweeps:
                matches: list[tuple[int, str, str]] = []
                unstable = self._unstable_names(
                    first_loop,
                    loop_k,
                    new_body[first_loop_index + 1 : body_idx],
                    alias_k,
                )
                # A store into an element, through any name, may change the
                # element a key reads from a fragment (``q = offs`` and
                # ``q[0] = 8`` change ``offs[0]``).
                stores_elements = _stores_elements(
                    *new_body[first_loop_index : body_idx + 1]
                )
                for j, s in enumerate(container_k):
                    if (
                        isinstance(s, ast.Assign)
                        and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)
                    ):
                        kind = _load_kind(s.value, self._tensor_dtypes)
                        if kind is None:
                            continue
                        # A fused vector load is deleted, and the reads of its
                        # name become cache reads, by spelling: under a name
                        # the final rename merges with another, the reads of
                        # the other spelling would be left behind.
                        if kind == "vec" and any(
                            s.targets[0].id in pair
                            for pair in self._renames.items()
                            if pair[0] != pair[1]
                        ):
                            continue
                        # Canonicalise this sweep's load text against the
                        # first sweep's variable names before keying.
                        definitions_at_load = self._pure_definitions_before(
                            container_k, j, scope_definitions, analysis=analysis
                        )
                        key = _canonical_load_text(
                            s.value, alias_k, definitions_at_load
                        )
                        if self._key_reads(key) & unstable:
                            continue
                        if stores_elements and _subscript_reads(
                            ast.parse(key, mode="eval")
                        ):
                            continue
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

            # A load that reads the same element in every iteration needs one
            # slot, not one per lane.
            invariant_keys = self._lane_invariant_keys(first_loop, tracked)
            # Each later sweep's cut: the statements that read a carried pure
            # value instead of recomputing it, and those that became dead.
            cuts: list[Cut | None] = [None] * len(sweeps)
            values: SweepValues | None = None
            first_values: dict[int, ast.stmt] = {}
            if carry_values:
                # Loads between the sweeps (a later sweep's lane-invariant
                # prefix) that read a lane-invariant first-sweep load.
                scope_loads: dict[int, str] = {}
                for index in range(first_loop_index + 1, sweeps[-1][0]):
                    statement = new_body[index]
                    if not (
                        is_name_assign(statement)
                        and _load_kind(statement.value, self._tensor_dtypes)
                    ):
                        continue
                    key = _canonical_load_text(
                        statement.value,
                        {},
                        self._pure_definitions_before(
                            new_body, index, analysis=analysis
                        ),
                    )
                    if key not in invariant_keys:
                        continue
                    # The same element, and unchanged: the names the key
                    # reads keep their values, no store into an element in
                    # the first sweep or after it (through any name) may
                    # change a fragment element the key reads, as for a later
                    # sweep's load, and no store may write the element.
                    load_roots = tracked_tensor_roots.get(key)
                    if not (
                        self._key_reads(key)
                        & self._bound(*new_body[first_loop_index + 1 : index])
                        or _stores_elements(*new_body[first_loop_index:index])
                        and _subscript_reads(ast.parse(key, mode="eval"))
                        or memory_order.has_aliasing_store_in(first_loop, load_roots)
                        or memory_order.has_aliasing_store_between(
                            tracked_statements[key], statement, load_roots
                        )
                    ):
                        scope_loads[id(statement)] = key
                plan = self._plan_sweep_values(
                    new_body,
                    first_loop,
                    [loop_k for _, loop_k, *_rest in sweeps],
                    first_loads={
                        id(statement): key
                        for key, statement in tracked_statements.items()
                    },
                    later_loads=[
                        {id(sweep[2][j]): key for j, _name, key in matches}
                        for sweep, matches in zip(
                            sweeps, per_sweep_matches, strict=True
                        )
                    ],
                    scope_loads=scope_loads,
                    load_dtypes={
                        key: dtype
                        for key, (_j, _name, _kind, dtype, _w) in tracked.items()
                        if dtype is not None
                    },
                    invariant_keys=invariant_keys,
                    transcendentals=transcendentals,
                    live_out=live_out,
                )
                if plan is not None:
                    values, first_values, cuts = plan
                    # Loads only the cut computation read are dead now.
                    per_sweep_matches = [
                        [
                            match
                            for match in matches
                            if cut is None or id(sweep[2][match[0]]) not in cut.dead
                        ]
                        for sweep, matches, cut in zip(
                            sweeps, per_sweep_matches, cuts, strict=True
                        )
                    ]
                    matched_keys = {
                        key
                        for matches in per_sweep_matches
                        for _j, _name, key in matches
                    }
            if not matched_keys and values is None:
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
            #   - A lane-invariant value: one register slot.
            cache_names: dict[str, tuple[str, int]] = {}
            # Carried pure values: number -> fragment.
            value_caches: dict[int, str] = {}
            single_slot: set[str] = set()
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
                invariant = key in invariant_keys
                if invariant:
                    single_slot.add(cache)
                cache_decls.extend(
                    self._cache_declarations(
                        cache,
                        dtype,
                        slots=1 if invariant else cache_size * vec_w,
                        use_smem=use_smem and not invariant,
                        trip=trip,
                        trip_expr=None if invariant else trip_expr,
                        fragment_caps=fragment_caps,
                    )
                )
            if values is not None:
                for cut in cuts:
                    for number in cut.cached.values() if cut is not None else ():
                        if number in value_caches:
                            continue
                        cache = value_caches[number] = self._new_cache_name()
                        value = values.values[number]
                        assert value.dtype is not None
                        if not value.variant:
                            single_slot.add(cache)
                        cache_decls.extend(
                            self._cache_declarations(
                                cache,
                                value.dtype,
                                slots=cache_size if value.variant else 1,
                                use_smem=False,
                                trip=trip,
                                trip_expr=trip_expr if value.variant else None,
                                fragment_caps=fragment_caps,
                            )
                        )
            if not cache_names and not value_caches:
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

            def _scalar_slot(
                cache: str, idx_text: str, _single_slot: set[str] = single_slot
            ) -> str:
                return "0" if cache in _single_slot else _slot_expr(idx_text, 1, "0")

            # The first sweep's statement computing each carried value.
            value_writes = {
                id(first_values[number]): cache
                for number, cache in value_caches.items()
            }

            # Rewrite the first container: append cache writes after each
            # tracked load assignment and each carried value.  Vec loads
            # write V scalars per cache slot; scalar loads write a single
            # scalar.
            new_first_body: list[ast.stmt] = []
            analysis.clear()
            for s in first_container:
                new_first_body.append(s)
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    if (cache := value_writes.get(id(s))) is not None:
                        slot = _scalar_slot(cache, cache_index)
                        new_first_body.append(
                            statement_from_string(
                                f"{cache}[{slot}] = {s.targets[0].id}"
                            )
                        )
                        continue
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
                        slot = _scalar_slot(cache, cache_index)
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
            for (
                body_idx,
                _loop_k,
                container_k,
                cache_index_k,
                _alias_k,
                _scope_definitions,
            ), matches, cut in zip(sweeps, per_sweep_matches, cuts, strict=True):
                if not matches and cut is None:
                    continue
                vec_extract_rewrites: list[
                    tuple[str, str, str, int, bool, int, str]
                ] = []
                new_sweep_body: list[ast.stmt] = []
                matched_by_index = {j: key for j, _name, key in matches}
                for statement_index, s in enumerate(container_k):
                    if cut is not None and id(s) in cut.dead:
                        continue
                    if cut is not None and id(s) in cut.cached:
                        cache = value_caches[cut.cached[id(s)]]
                        slot = _scalar_slot(cache, cache_index_k)
                        # The constructor is a no-op on the element type; it
                        # keeps the value's type as provable as the
                        # expression it replaces (the exact-division pass
                        # needs an fp32 dividend).
                        target = assigned_name(s)
                        assert target is not None
                        new_sweep_body.append(
                            statement_from_string(
                                f"{target} = {self._cache_dtypes[cache]}({cache}[{slot}])"
                            )
                        )
                        continue
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
                                    slot = _scalar_slot(cache, cache_index_k)
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
            # populate sweep and before each consume sweep (a single slot
            # stays in registers).  Insert in descending position order so
            # earlier insertions don't shift later positions.
            if use_smem and any(key not in invariant_keys for key in cache_names):
                for pos in sorted(smem_barrier_positions, reverse=True):
                    new_body.insert(
                        pos,
                        statement_from_string("cute.arch.sync_threads()"),
                    )

            rewritten = [
                loop_k
                for (_body_idx, loop_k, *_rest), matches, cut in zip(
                    sweeps, per_sweep_matches, cuts, strict=True
                )
                if matches or cut is not None
            ]
            last_rewritten = rewritten[-1] if rewritten else first_loop
            if values is not None and not fallback_originals:
                # Drop what only the cut computation needed between the
                # sweeps, such as a store sweep's lane-invariant prefix of
                # the Philox chain.  (A constexpr-trip group's fallback
                # replays these statements, so they stay there.)
                new_body = eliminate_dead(
                    new_body,
                    range(
                        new_body.index(first_loop) + 1, new_body.index(last_rewritten)
                    ),
                    live_out(),
                    removable=self._removable,
                    reads=self._reads,
                    assigned=self._assigned,
                )
                analysis.clear()
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

        if not any_fused:
            return None
        if dynamic_groups:
            new_body = _wrap_dynamic_groups(
                new_body, dynamic_groups, fused_spans, fallback_originals
            )
        return new_body

    def _child_live_out(
        self,
        body: list[ast.stmt],
        index: int,
        live_out: Callable[[], frozenset[str]],
    ) -> Callable[[], frozenset[str]]:
        """Names live at the end of a statement list nested in
        ``body[index]``, computed on first use."""

        @functools.cache
        def compute() -> frozenset[str]:
            statement = body[index]
            after = live_in(
                body[index + 1 :],
                live_out(),
                reads=self._reads,
                assigned=self._assigned,
            )
            if isinstance(statement, (ast.If, ast.With)):
                # Each nested list runs once, right before the statements
                # after ``statement``.
                return frozenset(after)
            if isinstance(statement, ast.For) and isinstance(
                statement.target, ast.Name
            ):
                # Later iterations, then the ``else`` (unless a ``break``
                # skips it), run after the body.
                after |= live_in(
                    statement.orelse, after, reads=self._reads, assigned=self._assigned
                )
                after = loop_live_end(
                    statement, after, reads=self._reads, assigned=self._assigned
                )
            else:
                # Another loop's later iterations and ``else``, or a ``try``'s
                # handlers, ``else`` and ``finally``, may run after the nested
                # list: what the statement reads stays live.
                after |= self._reads(statement)
            return frozenset(after)

        return compute

    def _transform_body(
        self,
        body: list[ast.stmt],
        *,
        under_dynamic_branch: bool,
        live_out: Callable[[], frozenset[str]],
    ) -> list[ast.stmt]:
        """Recursively transform statement-list fields before siblings.

        ``live_out`` gives the names live after ``body``: the carried-value
        cut removes only computation nothing reads afterwards.
        """
        transformed = list(body)
        # The first call sees the whole kernel body.
        self._jumps = self._jumps or _jumps(body)
        for index, stmt in enumerate(transformed):
            child_under_branch = under_dynamic_branch or isinstance(stmt, ast.If)
            for field in ("body", "orelse", "finalbody"):
                child = getattr(stmt, field, None)
                if isinstance(child, list) and all(
                    isinstance(child_stmt, ast.stmt) for child_stmt in child
                ):
                    setattr(
                        stmt,
                        field,
                        self._transform_body(
                            child,
                            under_dynamic_branch=child_under_branch,
                            live_out=self._child_live_out(transformed, index, live_out),
                        ),
                    )
        fused = self._try_fuse(
            transformed,
            allow_smem=not under_dynamic_branch,
            live_out=live_out,
        )
        return transformed if fused is None else fused

    def transform(self, body: list[ast.stmt]) -> list[ast.stmt]:
        transformed = self._transform_body(
            body, under_dynamic_branch=False, live_out=frozenset
        )
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
    binding_dtypes: dict[str, str] | None = None,
    rename_groups: dict[str, str] | None = None,
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

    Later sweeps also read the first sweep's expensive pure values from
    fragments instead of recomputing them (``sweep_values``): results of pure
    inline asm such as a word0 ``hl.rand`` draw, and with the explicit
    ``register`` policy transcendentals.  ``rename_groups`` maps each name to
    the name the final rename gives it, so the computation the carried values
    replace is removed only where nothing reads it, and ``binding_dtypes``
    gives the DSL type of names their binding statement does not show (the
    uniform cutoffs of a dropout mask), so a value computed from them has
    one.

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
        binding_dtypes=binding_dtypes,
        rename_groups=rename_groups,
    )
    return transformer.transform(body)
