"""Run the complete chunks of a looped sweep in a loop of their own.

Role: device-body passes (``cute_proven_bounds``): ``peel_partial_chunks``
before ``simplify_proven_loop_bounds``, ``merge_unproven_peels`` after it.

A rolled reduction or full-slice sweep walks ``range(0, n, chunk)``.  When the
chunk does not divide ``n`` the last chunk is partial, and every chunk pays for
it: each lane compares its index against ``n``, selects a safe address for its
masked loads and guards its stores (3.35 of the 24.6 SASS instructions per
element of a cross-entropy backward's sweep over a 30522-wide row).  Split
into ``range(0, m, chunk)`` and ``range(m, n, chunk)`` with ``m = n - n %
chunk``, the first loop covers complete chunks only, whose predicates
``simplify_proven_loop_bounds`` proves and removes; the second runs the partial
chunk once, predicates kept.  The two loops run the same iterations in the same
order.  Where the proof leaves the complete chunks' copy as it was, or as it
leaves the partial chunk's, ``merge_unproven_peels`` puts the loop back
together: the split would only have doubled it.

A loop qualifies with a literal end that its body compares against (a bounds
predicate the proof may remove), a static start and step, no control
transfer, and no call that may take shared memory of its own
(``smem_footprint.allocates_smem``), which every copy would allocate again.
As a conservative choice, an mbarrier's initialization and the reduces across
threads that allocate nothing (the warp and cluster grouped-reduce
dispatchers, the DSL's warp reduction) keep the loop whole too, though they
claim nothing per call site.  A loop holding a peeled loop stays whole, so
nesting never multiplies the copies, and a loop the caller names
(``keep_whole``, at any depth) stays whole with every loop nested in it.  The
copies keep their bounds' integer type (an ``Int64`` induction variable stays
``Int64``).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..ast_extension import _STATEMENT_LIST_FIELDS
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from .reduce_call import GROUPED_REDUCE_HELPERS
from .smem_footprint import allocates_smem

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    KeepWhole = Callable[[ast.stmt | None, ast.For], bool]

_INT_CASTS = frozenset({"cutlass.Int32", "cutlass.Int64"})
# Set on both copies of a peeled loop: ``(original loop, copy)``.
_PEELED = "_helion_peeled_chunks"


def _int_cast_arg(node: ast.expr) -> ast.expr | None:
    """``x`` of ``cutlass.Int32(x)`` or ``cutlass.Int64(x)``, else ``None``."""
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) in _INT_CASTS
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _static_int(node: ast.expr, constexpr_values: Mapping[str, int]) -> int | None:
    while (arg := _int_cast_arg(node)) is not None:
        node = arg
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return constexpr_values.get(node.id)
    return None


def _keeps_loop_whole(call: ast.Call) -> bool:
    """Whether ``call`` keeps its loop whole: shared memory it may take
    (``allocates_smem``) every copy would take again.  An mbarrier's
    initialization and the reduces across threads that allocate nothing (the
    warp and cluster grouped-reduce dispatchers, the DSL's warp reduction)
    keep it whole too, conservatively: they claim nothing per call site."""
    if allocates_smem(call):
        return True
    path = call_path_str(call.func)
    return path is not None and (
        path == "cute.arch.mbarrier_init"
        or path in GROUPED_REDUCE_HELPERS
        or path.startswith("cute.arch.warp_reduction")
    )


def _peelable(
    loop: ast.For, constexpr_values: Mapping[str, int]
) -> tuple[int, int, int] | None:
    """``(start, end, full_end)`` of a loop to split, or ``None``."""
    call = loop.iter
    if (
        loop.orelse
        or not isinstance(loop.target, ast.Name)
        or not isinstance(call, ast.Call)
        or call_path_str(call.func) != "range"
        or len(call.args) != 3
        or call.keywords
    ):
        return None
    start, step = (_static_int(call.args[i], constexpr_values) for i in (0, 2))
    # The body compares against a literal end (a name for it never matches).
    end = _static_int(call.args[1], {})
    if (
        start is None
        or end is None
        or step is None
        or step <= 0
        or end - start <= step
        or (end - start) % step == 0
    ):
        return None
    compares_end = False
    for node in ast.walk(ast.Module(body=loop.body, type_ignores=[])):
        if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
            return None
        if isinstance(node, ast.Call) and _keeps_loop_whole(node):
            return None
        if isinstance(node, ast.Compare) and any(
            isinstance(right, ast.Constant) and right.value == end
            for right in node.comparators
        ):
            compares_end = True
    if not compares_end:
        return None
    return start, end, end - (end - start) % step


def _rebound(bound: ast.expr, value: int) -> ast.expr:
    """``bound`` holding ``value`` instead, its integer casts kept."""
    arg = _int_cast_arg(bound)
    if arg is not None:
        cast = deep_clone(bound)
        assert isinstance(cast, ast.Call)
        cast.args = [_rebound(arg, value)]
        return cast
    return ast.copy_location(ast.Constant(value=value), bound)


def _with_bounds(loop: ast.For, start: int, end: int) -> ast.For:
    copy = deep_clone(loop)
    assert isinstance(copy.iter, ast.Call)
    begin, stop, step = copy.iter.args
    copy.iter.args = [_rebound(begin, start), _rebound(stop, end), step]
    setattr(copy, _PEELED, loop)
    return copy


def peel_partial_chunks(
    body: list[ast.stmt],
    constexpr_values: Mapping[str, int],
    *,
    keep_whole: KeepWhole | None = None,
) -> list[ast.stmt]:
    """Split every qualifying loop of ``body``, nested ones included;
    ``keep_whole(previous, loop)`` names the loops, at any depth, to leave
    whole with the loops nested in them, given the statement before each."""
    return _peel(body, constexpr_values, keep_whole)[0]


def _peel(
    body: list[ast.stmt],
    constexpr_values: Mapping[str, int],
    keep_whole: KeepWhole | None,
) -> tuple[list[ast.stmt], bool]:
    """The statements with their loops split, and whether any was."""
    result: list[ast.stmt] = []
    peeled = False
    previous: ast.stmt | None = None
    for statement in body:
        if (
            keep_whole is not None
            and isinstance(statement, ast.For)
            and keep_whole(previous, statement)
        ):
            # The caller matches the nest whole: its inner loops stay too.
            result.append(statement)
            previous = statement
            continue
        inner = _peel_blocks(statement, constexpr_values, keep_whole)
        split = (
            None
            if inner or not isinstance(statement, ast.For)
            else _peelable(statement, constexpr_values)
        )
        previous = statement
        peeled = peeled or inner
        if split is None:
            result.append(statement)
            continue
        assert isinstance(statement, ast.For)
        start, end, full_end = split
        result.extend(
            (
                _with_bounds(statement, start, full_end),
                _with_bounds(statement, full_end, end),
            )
        )
        peeled = True
    return result, peeled


def _peel_blocks(
    node: ast.AST, constexpr_values: Mapping[str, int], keep_whole: KeepWhole | None
) -> bool:
    """Peel the statement lists nested in ``node``; whether a loop was."""
    peeled = False

    def peel(statements: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal peeled
        statements, inner = _peel(statements, constexpr_values, keep_whole)
        peeled = peeled or inner
        return statements

    _map_blocks(node, peel)
    return peeled


def _map_blocks(
    node: ast.AST, rewrite: Callable[[list[ast.stmt]], list[ast.stmt]]
) -> None:
    """Replace each statement list nested in ``node`` (an ``except``
    handler's and a ``match`` case's too) with ``rewrite`` of it."""
    for field in _STATEMENT_LIST_FIELDS:
        children = getattr(node, field, None)
        if not isinstance(children, list):
            continue
        if field in ("handlers", "cases"):
            for child in children:
                _map_blocks(child, rewrite)
            continue
        setattr(node, field, rewrite(children))


def merge_unproven_peels(body: list[ast.stmt]) -> list[ast.stmt]:
    """Put back together each loop ``peel_partial_chunks`` split whose
    complete chunks' copy ``simplify_proven_loop_bounds`` left as it left the
    partial chunk's, or as it was.  In the first case the original loop
    takes the copies' (identical) bodies, so any predicate the proof removed
    from both stays removed; in the second it keeps its own, unproven body
    (the copies are clones), dropping what the proof removed from the
    partial chunk alone."""
    result: list[ast.stmt] = []
    for statement in body:
        _map_blocks(statement, merge_unproven_peels)
        original = getattr(statement, _PEELED, None)
        full = result[-1] if result else None
        if (
            original is not None
            and isinstance(full, ast.For)
            and isinstance(statement, ast.For)
            and getattr(full, _PEELED, None) is original
        ):
            # ``statement`` runs the partial chunk, ``full`` the complete ones.
            delattr(full, _PEELED)
            delattr(statement, _PEELED)
            proven = _dump(full.body)
            if proven == _dump(statement.body):
                original.body = full.body
                result[-1] = original
                continue
            if proven == _dump(original.body):
                result[-1] = original
                continue
        result.append(statement)
    return result


def _dump(body: list[ast.stmt]) -> str:
    return ast.dump(ast.Module(body=body, type_ignores=[]))
