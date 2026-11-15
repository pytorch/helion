"""Match the online-softmax ``(max, sum)`` cluster-reduce pair of a device body.

After the lane-reduce collapse a cluster-split online-softmax row reduces its
maximum and its exponential sum with two ``_cute_grouped_reduce_cluster``
calls, each directly inside a top-level ``for``.  Three passes rewrite that
pair (``cluster_online_pair`` fuses the two exchanges into one packed one,
``rolled_cluster_online_pair`` and ``register_row_reduction`` do the same for
rolled traversals) and start from the same facts, gathered here; the
per-thread ``online_chunk_reduction`` pass matches warp / shared-memory
reduce sites of its own but shares the value and literal helpers:

* :class:`_Site` / :func:`_find_sites` -- every cluster-reduce site, in order,
  or ``None`` when any site does not have the exact shape the rewrites
  understand;
* :class:`_ElementValues` / :func:`_reduced_values` -- the canonical
  per-element values site A max-reduces, with the register caches written
  alongside its fold;
* :func:`_prove` -- the rolled pair's frame and purity proof (sweep B folds
  ``exp2((x - m) * C)`` or ``exp(x - m)`` (:func:`_exponent`) of the same
  elements under the same mask, the maximum escapes only into those
  subtractions);
* :func:`_wrap_sites` -- the one-trip owners that give top-level rolled
  finalizers the shape the proofs expect (erased again by the passes).

:class:`OnlinePairMatcher` iterates the adjacent site pairs an
:class:`OnlinePairGate` admits (``candidates``) and, for the rolled passes,
the ones whose proof holds (``proved``).  Acceptance is parameterized, not
unified: the explicit pass accepts any group span and cluster size above one,
the rolled passes need both sites inside owners, a group span of at least 64
and a cluster of at most 16 CTAs.  Nothing here mutates the body; every pass
re-matches after its own rewrite.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
from itertools import pairwise
import math
from typing import TYPE_CHECKING
from typing import Callable
from typing import TypeGuard
from typing import cast

from ...ast_read_writes import ReadWrites
from .._ast_pass_utils import _is_proven_relocatable_assignment
from .._ast_pass_utils import is_name_assign
from .._ast_pass_utils import reparse_expr

if TYPE_CHECKING:
    from collections.abc import Iterator

_CLUSTER_REDUCE = "_cute_grouped_reduce_cluster"
_BLOCK_REDUCE = "_cute_grouped_reduce_block"
_PAIR_REDUCE = "_cute_grouped_reduce_cluster_online_pair"


def _stmt(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


def _assigned_name(stmt: ast.Assign) -> str:
    """Name of a target already proved by ``is_name_assign``."""
    return cast("ast.Name", stmt.targets[0]).id


def _loads(node: ast.AST) -> list[str]:
    return [
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    ]


def _int_kw(call: ast.Call, name: str) -> int | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, int):
                return value
    return None


def _float32(expr: ast.expr) -> TypeGuard[ast.Call]:
    return (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) == "cutlass.Float32"
        and len(expr.args) == 1
        and not expr.keywords
    )


def _literal(expr: ast.expr) -> float | None:
    while _float32(expr):
        assert isinstance(expr, ast.Call)
        expr = expr.args[0]
    if isinstance(expr, ast.Constant) and type(expr.value) in (int, float):
        return float(cast("int | float", expr.value))
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        value = _literal(expr.operand)
        return -value if value is not None else None
    if (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) == "float"
        and len(expr.args) == 1
        and isinstance(expr.args[0], ast.Constant)
        and expr.args[0].value in ("inf", "-inf")
    ):
        return float(expr.args[0].value)
    return None


class _Site:
    def __init__(
        self,
        top_idx: int,
        stmt_idx: int,
        for_node: ast.For,
        assign: ast.Assign,
        call: ast.Call,
    ) -> None:
        self.top_idx = top_idx
        self.stmt_idx = stmt_idx
        self.for_node = for_node
        self.assign = assign
        self.call = call
        self.target: str = assign.targets[0].id  # type: ignore[attr-defined]
        self.op: str = call.args[1].value  # type: ignore[attr-defined]
        self.buf_name: str = call.args[4].id  # type: ignore[attr-defined]
        self.mbar_name: str = call.args[5].id  # type: ignore[attr-defined]
        self.group_span = _int_kw(call, "group_span")
        self.cluster_n = _int_kw(call, "cluster_n")


def _find_sites(body: list[ast.stmt]) -> list[_Site] | None:
    """All ``_cute_grouped_reduce_cluster`` sites, in order.  Returns None
    when any site does not have the exact shape the rewrite understands
    (a single-Name-target Assign directly inside a top-level For)."""
    sites: list[_Site] = []
    seen = 0
    for i, top in enumerate(body):
        for node in ast.walk(top):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == _CLUSTER_REDUCE
            ):
                seen += 1
        if not isinstance(top, ast.For):
            continue
        for j, inner in enumerate(top.body):
            if not (is_name_assign(inner) and isinstance(inner.value, ast.Call)):
                continue
            call = inner.value
            if not (
                isinstance(call.func, ast.Name) and call.func.id == _CLUSTER_REDUCE
            ):
                continue
            if (
                len(call.args) != 6
                or not isinstance(call.args[1], ast.Constant)
                or not isinstance(call.args[4], ast.Name)
                or not isinstance(call.args[5], ast.Name)
            ):
                return None
            sites.append(_Site(i, j, top, inner, call))
    if seen != len(sites):
        return None
    return sites


class _Inliner(ast.NodeTransformer):
    """Substitute vec-body local single-assignment names into an expression
    (bounded depth, load context only)."""

    def __init__(self, defs: dict[str, ast.expr], depth: int = 0) -> None:
        self.defs = defs
        self.depth = depth

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.defs and self.depth < 10:
            replacement = reparse_expr(self.defs[node.id])
            return _Inliner(self.defs, self.depth + 1).visit(replacement)
        return node


def _is_cute_math_call(call: ast.AST, name: str) -> TypeGuard[ast.Call]:
    return (
        isinstance(call, ast.Call)
        and len(call.args) == 1
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == name
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "math"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "cute"
    )


def _is_exp2(call: ast.AST) -> TypeGuard[ast.Call]:
    return _is_cute_math_call(call, "exp2")


def _exponent(call: ast.AST) -> tuple[ast.expr, float | None] | None:
    """The exponent of an online-softmax exponential and its base-2 scale.

    ``exp2(shift * C)`` (``C`` a positive finite float literal, the
    ``fast_math`` and ``torch.exp2`` spelling) gives ``(shift, C)``; the
    natural ``exp(shift)`` (``torch.exp`` without ``fast_math``) gives
    ``(shift, None)``.  Keywords are left to the caller.
    """
    if _is_cute_math_call(call, "exp"):
        return call.args[0], None
    if not _is_exp2(call):
        return None
    argument = call.args[0]
    if (
        isinstance(argument, ast.BinOp)
        and isinstance(argument.op, ast.Mult)
        and isinstance(argument.right, ast.Constant)
        and type(argument.right.value) is float
        and math.isfinite(argument.right.value)
        and argument.right.value > 0
    ):
        return argument.left, argument.right.value
    return None


def _exponential_src(difference: str, scale: float | None, *, ftz: bool) -> str:
    """``exp(difference)`` in the spelling of an exponent of ``scale``
    (:func:`_exponent`): ``exp2((difference) * scale)``, ``ftz`` choosing its
    flush-to-zero form, or the natural ``exp(difference)`` for ``None``,
    whose exact form has no denormal fixup to drop."""
    if scale is None:
        return f"cute.math.exp({difference})"
    return f"cute.math.exp2(({difference}) * {scale!r}, fastmath={ftz!r})"


def _strip_fp32_casts(node: ast.expr) -> ast.expr:
    """Drop ``cutlass.Float32(v)`` around a value that already is Float32."""

    class Strip(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if not (
                ast.unparse(node.func) == "cutlass.Float32"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(inner := node.args[0], ast.Call)
            ):
                return node
            if ast.unparse(inner.func) == "cutlass.Float32" or (
                isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "bitcast"
                and [ast.unparse(arg) for arg in inner.args] == ["cutlass.Float32"]
            ):
                return inner
            return node

    result = Strip().visit(node)
    assert isinstance(result, ast.expr)
    return result


def _domain(expr: ast.expr, identity: float) -> tuple[ast.expr | None, ast.expr]:
    """Separate a reduction's neutral mask without dropping its FP32 cast."""
    expr = _strip_fp32_casts(expr)
    if _float32(expr):
        assert isinstance(expr, ast.Call)
        inner = expr.args[0]
        if isinstance(inner, ast.IfExp):
            expr = ast.IfExp(
                test=inner.test,
                body=ast.Call(func=expr.func, args=[inner.body], keywords=[]),
                orelse=ast.Call(func=expr.func, args=[inner.orelse], keywords=[]),
            )
    if isinstance(expr, ast.IfExp) and _literal(expr.orelse) == identity:
        return expr.test, expr.body
    return None, expr


def _under_domain(expr: ast.expr, domain: ast.expr | None) -> str:
    condition = ast.unparse(domain) if domain is not None else None

    class Assume(ast.NodeTransformer):
        def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
            if condition is not None and ast.unparse(node.test) == condition:
                return cast("ast.expr", self.visit(node.body))
            return cast("ast.expr", self.generic_visit(node))

    result = Assume().visit(ast.parse(ast.unparse(expr), mode="eval").body)
    assert isinstance(result, ast.expr)
    return ast.unparse(_strip_fp32_casts(result))


def _loop_chain(
    node: ast.AST, parents: dict[int, ast.AST], root: ast.For
) -> list[tuple[ast.For, ast.stmt]] | None:
    """The ``for`` loops from ``root`` down to ``node`` (outermost first),
    each with the statement of its body that holds ``node``; None if
    ``node`` sits under other control flow or in a loop's ``else``."""
    chain: list[tuple[ast.For, ast.stmt]] = []
    child: ast.AST = node
    while child is not root:
        parent = parents.get(id(child))
        if parent is None:
            return None
        if isinstance(child, ast.stmt):
            if not (
                isinstance(parent, ast.For) and any(s is child for s in parent.body)
            ):
                return None
            chain.append((parent, child))
        child = parent
    chain.reverse()
    return chain


def _same_loop_ranges(
    first: list[tuple[ast.For, ast.stmt]], second: list[tuple[ast.For, ast.stmt]]
) -> bool:
    return len(first) == len(second) and all(
        ast.unparse(a.iter) == ast.unparse(b.iter)
        for (a, _), (b, _) in zip(first, second, strict=True)
    )


def _stable_outer_definitions(body: list[ast.stmt], before: int) -> dict[str, ast.expr]:
    """Pure scalar aliases whose dependencies have one unchanged SSA value.

    Expand only definitions preceding the first reduction. Every candidate and
    dependency is either never assigned in this body or was already proved as
    a single-assignment scalar. A later write, including a loop-carried write,
    therefore prevents substitution into a subsequent load site's address.
    """
    module = ast.Module(body=body, type_ignores=[])
    writes = Counter(
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    stable = set(_loads(module)) - writes.keys()
    definitions: dict[str, ast.expr] = {}
    for statement in body[:before]:
        if not is_name_assign(statement):
            continue
        name = _assigned_name(statement)
        if (
            writes[name] != 1
            or not set(_loads(statement.value)) <= stable
            or not _is_proven_relocatable_assignment(statement, allow_load=False)
        ):
            continue
        # Canonicalize this acyclic prefix once, so an expression's nesting
        # depth does not change how far its outer aliases are substituted.
        definitions[name] = _Inliner(definitions).visit(reparse_expr(statement.value))
        stable.add(name)
    return definitions


class _ElementValues:
    """Canonical per-element values of expressions inside ``for`` nests
    whose innermost loop is ``range_constexpr(N)``: local definitions are
    inlined, the innermost index becomes each constant ``0..N-1``, the
    other loop indices become positional placeholders, and register-cache
    reads become the values written into them."""

    def __init__(self, outer_definitions: dict[str, ast.expr] | None = None) -> None:
        # (cache name, canonical index) -> canonical stored value
        self.cache_values: dict[tuple[str, str], ast.expr] = {}
        self.outer_definitions = outer_definitions or {}

    @staticmethod
    def _definitions(
        chain: list[tuple[ast.For, ast.stmt]],
    ) -> dict[str, ast.expr] | None:
        definitions: dict[str, ast.expr] = {}
        for loop, holder in chain:
            seen: set[str] = set()
            for statement in loop.body:
                if statement is holder:
                    break
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        if node.id in seen:
                            return None  # rebound: inlining could skip a write
                        seen.add(node.id)
                if is_name_assign(statement):
                    name = statement.targets[0].id  # type: ignore[attr-defined]
                    if name not in _loads(statement.value):  # type: ignore[attr-defined]
                        definitions[name] = statement.value  # type: ignore[attr-defined]
        return definitions

    @staticmethod
    def _rename(node: ast.expr, names: dict[str, ast.expr]) -> ast.expr:
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                replacement = names.get(node.id)
                if replacement is None or not isinstance(node.ctx, ast.Load):
                    return node
                return reparse_expr(replacement)

        result = Rename().visit(reparse_expr(node))
        assert isinstance(result, ast.expr)
        return result

    def _canonical(
        self,
        node: ast.expr,
        chain: list[tuple[ast.For, ast.stmt]],
        constant: dict[str, int],
    ) -> ast.expr | None:
        definitions = self._definitions(chain)
        if definitions is None:
            return None
        expr = _Inliner(self.outer_definitions | definitions).visit(reparse_expr(node))
        names: dict[str, ast.expr] = {
            loop.target.id: ast.Name(id=f"_loop{level}_", ctx=ast.Load())  # type: ignore[attr-defined]
            for level, (loop, _) in enumerate(chain)
            if isinstance(loop.target, ast.Name)
        }
        names.update(
            {name: ast.Constant(value=value) for name, value in constant.items()}
        )
        return _strip_fp32_casts(self._rename(expr, names))

    def record_cache_store(
        self, store: ast.Assign, chain: list[tuple[ast.For, ast.stmt]]
    ) -> bool:
        target = store.targets[0]
        assert isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
        index = self._canonical(target.slice, chain, {})
        value = self._canonical(store.value, chain, {})
        if index is None or value is None:
            return False
        key = (target.value.id, ast.unparse(index))
        if key in self.cache_values:
            return False
        self.cache_values[key] = value
        return True

    def values(
        self,
        node: ast.expr,
        chain: list[tuple[ast.For, ast.stmt]],
        caches: set[str],
    ) -> list[str] | None:
        """``node``'s canonical value at each innermost-loop position."""
        if not chain:
            return None
        inner = chain[-1][0]
        iterator = inner.iter
        if not (
            isinstance(inner.target, ast.Name)
            and isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Attribute)
            and iterator.func.attr == "range_constexpr"
            and len(iterator.args) == 1
            and isinstance(iterator.args[0], ast.Constant)
            and isinstance(iterator.args[0].value, int)
        ):
            return None
        owner = self
        results: list[str] = []
        for position in range(iterator.args[0].value):
            expr = self._canonical(node, chain, {inner.target.id: position})
            if expr is None:
                return None
            missing = False

            class ReadCache(ast.NodeTransformer):
                def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                    nonlocal missing
                    self.generic_visit(node)
                    if not (
                        isinstance(node.value, ast.Name) and node.value.id in caches
                    ):
                        return node
                    stored = owner.cache_values.get(
                        (node.value.id, ast.unparse(node.slice))
                    )
                    if stored is None:
                        missing = True
                        return node
                    return reparse_expr(stored)

            expr = ReadCache().visit(expr)
            if missing:
                return None
            results.append(ast.unparse(_strip_fp32_casts(expr)))
        return results


def _reduced_values(
    body: list[ast.stmt],
    site_a: _Site,
    canon: Callable[[str], str],
    caches: set[str],
) -> tuple[_ElementValues, list[tuple[ast.For, ast.stmt]], list[str]] | None:
    """The values site A max-reduces, per position of its fold's
    ``range_constexpr`` loop, with the register caches written alongside."""
    acc = site_a.call.args[0]
    if not isinstance(acc, ast.Name):
        return None
    acc_name = canon(acc.id)
    root = site_a.for_node
    parents = {
        id(child): node
        for node in ast.walk(root)
        for child in ast.iter_child_nodes(node)
    }
    folds: list[tuple[ast.Assign, ast.expr]] = []
    for statement in root.body[: site_a.stmt_idx]:
        for node in ast.walk(statement):
            if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                return None
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and canon(t.id) == acc_name
                    for t in node.targets
                )
            ):
                continue
            value = node.value
            if not is_name_assign(node):
                return None
            if ast.unparse(value) == "cutlass.Float32(float('-inf'))":
                continue
            if not (isinstance(value, ast.Call) and len(value.args) == 2):
                return None
            # NaN-propagating maxima only, as the packed exchange folds.
            call = (
                ast.unparse(value.func),
                tuple((kw.arg, ast.unparse(kw.value)) for kw in value.keywords),
            )
            if call not in {
                ("cute.arch.fmax", (("nan", "True"),)),
                ("cute.math.max", (("propagate_nan", "True"),)),
                ("_cute_nan_max", ()),
            }:
                return None
            first, second = value.args
            if isinstance(first, ast.Name) and canon(first.id) == acc_name:
                element = second
            elif isinstance(second, ast.Name) and canon(second.id) == acc_name:
                element = first
            else:
                return None
            if acc_name in {canon(name) for name in _loads(element)}:
                return None
            folds.append((node, element))
    if len(folds) != 1:
        return None
    fold, element = folds[0]
    chain = _loop_chain(fold, parents, root)
    if chain is None:
        return None
    elements = _ElementValues(_stable_outer_definitions(body, site_a.top_idx))
    # Register caches written beside the fold (same loops), and nowhere else.
    for statement in body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in caches
            ):
                continue
            store_chain = _loop_chain(node, parents, root)
            if (
                store_chain is None
                or len(store_chain) >= len(chain)
                or any(
                    a is not b
                    for (a, _), (b, _) in zip(
                        store_chain, chain[: len(store_chain)], strict=True
                    )
                )
                or not elements.record_cache_store(node, store_chain)
            ):
                return None
    values = elements.values(element, chain, caches)
    if values is None:
        return None
    return elements, chain, values


def _wrap_sites(body: list[ast.stmt]) -> tuple[list[ast.stmt], set[int]]:
    """Give top-level rolled finalizers the one-trip owner used by pair proofs.

    The temporary owners are erased before code is emitted. Only a roll,
    pure lane-index assignments, its finalizer, and the immediate FP32
    result cast enter an owner; unrelated statements keep their positions.
    """
    result: list[ast.stmt] = []
    owners: set[int] = set()
    index = 0
    while index < len(body):
        statement = body[index]
        if not isinstance(statement, ast.For):
            result.append(statement)
            index += 1
            continue
        end = index + 1
        while end < len(body) and is_name_assign(body[end]):
            candidate = body[end]
            assert isinstance(candidate, ast.Assign)
            if (
                isinstance(candidate.value, ast.Call)
                and ast.unparse(candidate.value.func) == _CLUSTER_REDUCE
            ):
                break
            if not _is_proven_relocatable_assignment(candidate, allow_load=False):
                break
            end += 1
        if end + 1 >= len(body) or not is_name_assign(body[end]):
            result.append(statement)
            index += 1
            continue
        finalize = body[end]
        result_cast = body[end + 1]
        assert isinstance(finalize, ast.Assign)
        call = finalize.value
        if not (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == _CLUSTER_REDUCE
            and len(call.args) == 6
            and is_name_assign(result_cast)
            and _float32(result_cast.value)
            and ast.unparse(result_cast.value.args[0])
            == ast.unparse(finalize.targets[0])
        ):
            result.append(statement)
            index += 1
            continue
        wrapper = _stmt(
            "for _rolled_pair_owner in cutlass.range_constexpr(1):\n    pass"
        )
        assert isinstance(wrapper, ast.For)
        wrapper.body = body[index : end + 2]
        owners.add(id(wrapper))
        result.append(wrapper)
        index = end + 2
    return result, owners


def _pure_assignments(statements: list[ast.stmt]) -> list[ast.Assign] | None:
    result: list[ast.Assign] = []
    for statement in statements:
        if isinstance(statement, ast.For):
            nested = _pure_assignments(statement.body)
            if nested is None:
                return None
            result.extend(nested)
        elif is_name_assign(statement) and _is_proven_relocatable_assignment(
            statement, allow_load=True
        ):
            assert isinstance(statement, ast.Assign)
            result.append(statement)
        elif not isinstance(statement, ast.Pass):
            return None
    return result


def _prove(
    body: list[ast.stmt], site_a: _Site, site_b: _Site
) -> tuple[str, float | None, list[ast.Name]] | None:
    tail = site_a.for_node.body[site_a.stmt_idx + 1 :]
    if len(tail) != 1 or not is_name_assign(tail[0]):
        return None
    maximum = _assigned_name(tail[0])
    region = [
        *body[site_a.top_idx + 1 : site_b.top_idx],
        *site_b.for_node.body[: site_b.stmt_idx],
    ]
    assignments = _pure_assignments(region)
    acc = site_b.call.args[0]
    if assignments is None or not isinstance(acc, ast.Name):
        return None
    folds = []
    for node in assignments:
        if _assigned_name(node) != acc.id:
            continue
        value = node.value
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and isinstance(value.left, ast.Name)
            and value.left.id == acc.id
        ):
            folds.append((node, value.right))
        elif _literal(value) != 0.0:
            return None
    if len(folds) != 1:
        return None
    caches = {
        _assigned_name(top)
        for top in body
        if is_name_assign(top)
        and isinstance(top.value, ast.Call)
        and ast.unparse(top.value.func) == "cute.make_rmem_tensor"
    }
    reduced = _reduced_values(body, site_a, lambda name: name, caches)
    if reduced is None:
        return None
    elements, max_chain, max_values = reduced
    fold, term = folds[0]
    parents = {
        id(child): parent
        for parent in ast.walk(site_b.for_node)
        for child in ast.iter_child_nodes(parent)
    }
    sum_chain = _loop_chain(fold, parents, site_b.for_node)
    if sum_chain is None or not _same_loop_ranges(max_chain, sum_chain):
        return None
    terms = elements.values(term, sum_chain, caches)
    if terms is None or len(terms) != len(max_values):
        return None
    scales: set[float | None] = set()
    for max_value, sum_term in zip(max_values, terms, strict=True):
        max_domain, value = _domain(ast.parse(max_value, mode="eval").body, -math.inf)
        sum_domain, exponential = _domain(ast.parse(sum_term, mode="eval").body, 0.0)
        if (ast.unparse(max_domain) if max_domain is not None else None) != (
            ast.unparse(sum_domain) if sum_domain is not None else None
        ):
            return None
        while _float32(exponential):
            assert isinstance(exponential, ast.Call)
            exponential = exponential.args[0]
        exponent = _exponent(exponential)
        if exponent is None:
            return None
        assert isinstance(exponential, ast.Call)
        if any(
            keyword.arg != "fastmath"
            or not isinstance(keyword.value, ast.Constant)
            or type(keyword.value.value) is not bool
            for keyword in exponential.keywords
        ):
            return None
        shift, scale = exponent
        while _float32(shift):
            assert isinstance(shift, ast.Call)
            shift = shift.args[0]
        if not (
            isinstance(shift, ast.BinOp)
            and isinstance(shift.op, ast.Sub)
            and isinstance(shift.right, ast.Name)
            and shift.right.id == maximum
            and _under_domain(shift.left, max_domain)
            == _under_domain(value, max_domain)
        ):
            return None
        scales.add(scale)
    if len(scales) != 1:
        return None
    dependent = {maximum}
    changed = True
    while changed:
        changed = False
        for node in assignments:
            if set(ReadWrites.from_ast(node.value).reads) & dependent:
                target = _assigned_name(node)
                if target not in dependent:
                    dependent.add(target)
                    changed = True
    escaped = dependent - {maximum, acc.id}
    for statement in [
        *site_b.for_node.body[site_b.stmt_idx + 1 :],
        *body[site_b.top_idx + 1 :],
    ]:
        if set(ReadWrites.from_ast(statement).reads) & escaped:
            return None
    subtractions = [
        node.right
        for node in ast.walk(ast.Module(body=region, type_ignores=[]))
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.right, ast.Name)
        and node.right.id == maximum
    ]
    reads = [
        node
        for node in ast.walk(ast.Module(body=region, type_ignores=[]))
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == maximum
    ]
    if not subtractions or {id(node) for node in reads} != {
        id(node) for node in subtractions
    }:
        return None
    return maximum, next(iter(scales)), subtractions


@dataclasses.dataclass(frozen=True)
class OnlinePairGate:
    """Which adjacent ``(max, sum)`` site pairs a pass may rewrite.

    Both sites must reduce over the same group span and cluster, the cluster
    must have more than one CTA, and site A must precede site B.  The rolled
    passes additionally require both sites inside the one-trip ``owners``
    made by :func:`_wrap_sites`, a group span of at least
    ``group_span_floor`` and a cluster of at most ``cluster_n_ceiling`` CTAs.
    """

    owners: frozenset[int] | None = None
    group_span_floor: int = 0
    cluster_n_ceiling: int | None = None

    def admits(self, site_a: _Site, site_b: _Site) -> bool:
        return (
            site_a.op == "max"
            and site_b.op == "sum"
            and site_a.cluster_n is not None
            and site_a.cluster_n > 1
            and site_a.cluster_n == site_b.cluster_n
            and (
                self.cluster_n_ceiling is None
                or site_a.cluster_n <= self.cluster_n_ceiling
            )
            and site_a.group_span is not None
            and site_a.group_span == site_b.group_span
            and site_a.group_span >= self.group_span_floor
            and site_a.top_idx < site_b.top_idx
            and (
                self.owners is None
                or (
                    id(site_a.for_node) in self.owners
                    and id(site_b.for_node) in self.owners
                )
            )
        )


def rolled_pair_gate(owners: set[int]) -> OnlinePairGate:
    """The gate of the rolled passes (warp-level pair, register rows)."""
    return OnlinePairGate(
        owners=frozenset(owners), group_span_floor=64, cluster_n_ceiling=16
    )


@dataclasses.dataclass(frozen=True)
class OnlinePairCandidate:
    """An admitted adjacent site pair; ``index`` is site A's position."""

    index: int
    site_a: _Site
    site_b: _Site


@dataclasses.dataclass(frozen=True)
class OnlinePairMatch(OnlinePairCandidate):
    """A candidate whose rolled proof holds: ``maximum`` is the name sweep B
    subtracts (at ``subtractions``), ``scale`` the exponent's constant
    (``None`` for the natural ``exp``, see :func:`_exponent`)."""

    maximum: str
    scale: float | None
    subtractions: list[ast.Name]


class OnlinePairMatcher:
    """The cluster-reduce site pairs of ``body`` admitted by ``gate``."""

    def __init__(
        self, body: list[ast.stmt], gate: OnlinePairGate | None = None
    ) -> None:
        self.body = body
        self.gate = gate if gate is not None else OnlinePairGate()
        self.sites = _find_sites(body)

    def candidates(self) -> Iterator[OnlinePairCandidate]:
        """Admitted adjacent pairs in body order; nothing when any site has an
        unexpected shape or there are fewer than two sites."""
        if not self.sites or len(self.sites) < 2:
            return
        for index, (site_a, site_b) in enumerate(pairwise(self.sites)):
            if self.gate.admits(site_a, site_b):
                yield OnlinePairCandidate(index, site_a, site_b)

    def proved(self) -> Iterator[OnlinePairMatch]:
        """Candidates whose frame and purity proof (:func:`_prove`) holds."""
        for candidate in self.candidates():
            proof = _prove(self.body, candidate.site_a, candidate.site_b)
            if proof is not None:
                maximum, scale, subtractions = proof
                yield OnlinePairMatch(
                    candidate.index,
                    candidate.site_a,
                    candidate.site_b,
                    maximum,
                    scale,
                    subtractions,
                )
