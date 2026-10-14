# pyrefly: ignore-errors
"""Fuse the two DSM cluster exchanges of an online-softmax reduction pair
into ONE packed ``(max, sum)`` exchange.

After the lane-reduce collapse, a cluster-split online-softmax row does:

    sweep A:  acc0 = fold(max, x)      ; m  = _cute_grouped_reduce_cluster(acc0, 'max', ...)
    sweep B:  acc1 = fold(+, exp2(f(x) - m*C)) ; s = _cute_grouped_reduce_cluster(acc1, 'sum', ...)
    sweep C:  out  = exp2(f(x) - m*C) * g(s)

i.e. TWO cluster-wide mbarrier round-trips per row, and sweep B stalls on
the cross-cluster max before it can start.  Quack's hand kernel does the
standard online-softmax combine instead: each CTA reduces its slice
locally, exchanges the ``(local_max, local_sum)`` pair ONCE, and folds
with the rescale ``s += s_r * exp2((m_r - m) * C)``.  On a B200 this is
worth +5..9% at cluster_n 2..16 (one fewer cluster round-trip, and a
``cluster_n``-slot fold instead of ``warps * cluster_n``).

This pass rewrites the emitted AST into that shape when it can prove the
pattern:

  * site A becomes a CTA-local block reduce (``_cute_grouped_reduce_block``),
    so ``m`` holds the CTA-slice max and sweep B starts without waiting on
    the cluster,
  * sweep B's ``exp2`` values are cached in a new register fragment (they
    are exact partial results of sweep C: ``out_i = e_i * exp2(m_local*C -
    m_global*C) * g(s)``, softmax being shift-invariant),
  * site B becomes ``_cute_grouped_reduce_cluster_online_pair`` which
    block-reduces the local sum, exchanges the packed pair once, folds
    with the rescale, and returns the GLOBAL ``(max, sum)``; the max
    variable is reassigned so every later read sees the global value,
  * sweep C's ``exp2`` recompute is replaced by the cached value times the
    (CTA-uniform) rescale factor.

The cached and rescaled exponentials round each output differently, so
that form needs the ``fast_math`` setting.  Without it the rewrite keeps
every exponential exact: only sweep B moves into the frame of the CTA's own
maximum (``_try_rewrite_pair_exact``), and sweep C recomputes its ``exp2``
from the global maximum.

Every rewrite condition fails closed: if any structural check does not
match, the kernel keeps the two-exchange form.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
from typing import Callable

_CLUSTER_REDUCE = "_cute_grouped_reduce_cluster"
_BLOCK_REDUCE = "_cute_grouped_reduce_block"
_PAIR_REDUCE = "_cute_grouped_reduce_cluster_online_pair"

# Module aliases / builtins that appear as Name loads in emitted code but
# are not kernel-local values.
_IGNORED_NAMES = frozenset(
    {
        "cutlass",
        "cute",
        "ir",
        "math",
        "mlir_math",
        "operator",
        "torch",
        "hl",
        "helion",
        "float",
        "int",
        "bool",
        "min",
        "max",
        "range",
    }
)


def _local_loads(node: ast.AST) -> list[str]:
    return [
        n
        for n in _loads(node)
        if n not in _IGNORED_NAMES and not n.startswith("_cute_")
    ]


def _stmt(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


def _reparse_expr(node: ast.expr) -> ast.expr:
    """Plain-ast copy of an expression (``copy.deepcopy`` fails on the
    ExtendedAST nodes the emitter produces)."""
    return ast.parse(ast.unparse(node), mode="eval").body


def _is_name_assign(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    )


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
            if not (_is_name_assign(inner) and isinstance(inner.value, ast.Call)):
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
            replacement = _reparse_expr(self.defs[node.id])
            return _Inliner(self.defs, self.depth + 1).visit(replacement)
        return node


def _inline_locals(expr: ast.expr, block: list[ast.stmt], upto: ast.stmt) -> ast.expr:
    defs: dict[str, ast.expr] = {}
    for stmt in block:
        if stmt is upto:
            break
        if _is_name_assign(stmt):
            name = stmt.targets[0].id  # type: ignore[attr-defined]
            if name not in _loads(stmt.value):
                defs[name] = stmt.value
    inlined = _Inliner(defs).visit(_reparse_expr(expr))
    ast.fix_missing_locations(inlined)
    return inlined


def _canonical_src(expr: ast.expr, rename: dict[str, str]) -> str:
    node = _reparse_expr(expr)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in rename:
            sub.id = rename[sub.id]
    return ast.unparse(node)


def _is_exp2(call: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and len(call.args) == 1
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "exp2"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "math"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "cute"
    )


def _is_pure(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if isinstance(
            node,
            (
                ast.Name,
                ast.Constant,
                ast.Load,
                ast.BinOp,
                ast.UnaryOp,
                ast.Subscript,
                ast.Attribute,
                ast.Tuple,
                ast.operator,
                ast.unaryop,
                ast.expr_context,
            ),
        ):
            continue
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "bitcast":
                continue
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in ("cutlass", "cute"):
                    continue
            if isinstance(func, ast.Name):
                continue
            return False
        if isinstance(node, ast.keyword):
            continue
        return False
    return True


def _prune_dead_assigns(block: list[ast.stmt], external_readers: list[ast.AST]) -> None:
    """Remove pure single-Name assignments in ``block`` whose target is
    never loaded again (in the block or by any of ``external_readers``)."""
    changed = True
    while changed:
        changed = False
        for idx in range(len(block) - 1, -1, -1):
            stmt = block[idx]
            if not (_is_name_assign(stmt) and _is_pure(stmt.value)):
                continue
            name = stmt.targets[0].id  # type: ignore[attr-defined]
            read = False
            for other in block:
                node = other.value if other is stmt else other
                if name in _loads(node):
                    read = True
                    break
            if not read:
                for reader in external_readers:
                    if name in _loads(reader):
                        read = True
                        break
            if not read:
                del block[idx]
                changed = True


@dataclasses.dataclass
class _VecExpMatch:
    vec_for: ast.For
    exp_stmt: ast.stmt
    exp_call: ast.Call
    scaled_name: str
    slot: ast.expr
    canonical: str


def _scaled_exp_subtrahend(
    argument: ast.expr, scaled_values: dict[str, float]
) -> str | None:
    if isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Sub):
        subtrahend = argument.right
    elif (
        isinstance(argument, ast.Call)
        and ast.unparse(argument.func) == "cute.math.fma"
        and len(argument.args) == 3
        and not argument.keywords
        and isinstance(argument.args[1], ast.Constant)
        and type(argument.args[1].value) is float
        and math.isfinite(argument.args[1].value)
        and argument.args[1].value > 0
        and isinstance(argument.args[2], ast.UnaryOp)
        and isinstance(argument.args[2].op, ast.USub)
    ):
        # The distributed-scale contraction makes the existing FP32
        # ``value * C - scaled_max`` explicit. Recognize its subtractive
        # addend without expanding the FMA back into rounded operations.
        subtrahend = argument.args[2].operand
        if not (
            isinstance(subtrahend, ast.Name)
            and scaled_values.get(subtrahend.id) == argument.args[1].value
        ):
            return None
    else:
        return None
    if isinstance(subtrahend, ast.Name) and subtrahend.id in scaled_values:
        return subtrahend.id
    return None


def _is_fp32_cache_read(value: ast.expr, cache_names: set[str]) -> bool:
    if not (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cutlass.Float32"
        and len(value.args) == 1
        and not value.keywords
    ):
        return False
    value = value.args[0]
    # Register caches can contain FP32 values, half values, or their raw
    # Uint16 bits. Admit only those reads and typed conversions, not a
    # second scaled-max dependency or an opaque/effectful value recipe.
    while isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
        if ast.unparse(value.func) in {
            "cutlass.Float32",
            "cutlass.Float16",
            "cutlass.BFloat16",
            "cutlass.Uint16",
        }:
            value = value.args[0]
        elif (
            isinstance(value.func, ast.Attribute)
            and value.func.attr == "bitcast"
            and ast.unparse(value.args[0])
            in {"cutlass.Float32", "cutlass.Float16", "cutlass.BFloat16"}
        ):
            value = value.func.value
        else:
            return False
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id in cache_names
        and all(
            isinstance(
                node,
                (
                    ast.Name,
                    ast.Load,
                    ast.Constant,
                    ast.BinOp,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.UnaryOp,
                    ast.USub,
                    ast.UAdd,
                ),
            )
            and (not isinstance(node, ast.Constant) or type(node.value) is int)
            for node in ast.walk(value.slice)
        )
    )


def _find_vec_exp(
    root: ast.stmt,
    stop_at: ast.stmt | None,
    scaled_values: dict[str, float],
    cache_names: set[str],
) -> _VecExpMatch | None:
    """Find the (unique) constexpr vec loop under ``root`` (searching
    statements before ``stop_at`` only) whose body computes
    ``cute.math.exp2(<expr> - <scaled>)`` or its explicit FP32 FMA form,
    reading exactly one fuse-cache slot; return its canonical form. The
    canonical form retains the operation and rounding choice, so a fused
    sum expression never matches an unfused output expression."""
    matches: list[_VecExpMatch] = []
    for node in ast.walk(root):
        if node is stop_at:
            continue
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Attribute)
            and it.func.attr == "range_constexpr"
        ):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        vec_var = node.target.id
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if not _is_exp2(sub):
                    continue
                arg = sub.args[0]
                scaled_name = _scaled_exp_subtrahend(arg, scaled_values)
                if scaled_name is None:
                    continue
                inlined = _inline_locals(arg, node.body, stmt)
                if isinstance(inlined, ast.Call):
                    # Keep the new admission limited to the emitter's
                    # FP32 contraction, independently of the raw cache dtype.
                    if not (
                        _is_fp32_cache_read(inlined.args[0], cache_names)
                        and sum(name == scaled_name for name in _loads(inlined)) == 1
                    ):
                        continue
                cache_slots = [
                    s
                    for s in ast.walk(inlined)
                    if isinstance(s, ast.Subscript)
                    and isinstance(s.value, ast.Name)
                    and s.value.id in cache_names
                ]
                if len(cache_slots) != 1:
                    continue
                canonical = _canonical_src(
                    inlined, {vec_var: "_V_", scaled_name: "_S_"}
                )
                matches.append(
                    _VecExpMatch(
                        node,
                        stmt,
                        sub,
                        scaled_name,
                        copy.deepcopy(cache_slots[0].slice),
                        canonical,
                    )
                )
    # Multiple textual exp2 sites are fine when they are the SAME
    # computation (the emitter duplicates the exp2 into the accumulator
    # update); they must agree on the vec loop, canonical form, and scaled
    # var.  Prefer the standalone ``name = exp2(...)`` match.
    if not matches:
        return None
    first = matches[0]
    for m in matches:
        if (
            m.vec_for is not first.vec_for
            or m.canonical != first.canonical
            or m.scaled_name != first.scaled_name
        ):
            return None
    for m in matches:
        if _is_name_assign(m.exp_stmt) and m.exp_stmt.value is m.exp_call:
            return m
    return first


def fuse_cluster_online_pair(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None,
    rename_groups: dict[str, str] | None = None,
    fast_math: bool = False,
) -> list[ast.stmt]:
    renames = rename_groups or {}

    def canon(name: str) -> str:
        return renames.get(name, name)

    sites = _find_sites(body)
    if not sites or len(sites) < 2:
        return body

    # Top-level register-fragment caches (fuse_two_pass_loads output).
    caches: dict[str, tuple[int, int, str]] = {}
    for i, top in enumerate(body):
        if not (_is_name_assign(top) and isinstance(top.value, ast.Call)):
            continue
        call = top.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "make_rmem_tensor"
            and len(call.args) == 2
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, int)
        ):
            caches[top.targets[0].id] = (  # type: ignore[attr-defined]
                call.args[0].value,
                i,
                ast.unparse(call.args[1]),
            )

    for pair_idx in range(len(sites) - 1):
        site_a, site_b = sites[pair_idx], sites[pair_idx + 1]
        if (
            site_a.op != "max"
            or site_b.op != "sum"
            or site_a.cluster_n is None
            or site_a.cluster_n <= 1
            or site_a.cluster_n != site_b.cluster_n
            or site_a.group_span is None
            or site_a.group_span != site_b.group_span
            or site_a.top_idx >= site_b.top_idx
        ):
            continue
        if (
            fast_math
            and _try_rewrite_pair(
                body, site_a, site_b, caches, constexpr_values, canon, pair_idx
            )
        ) or _try_rewrite_pair_exact(
            body,
            site_a,
            site_b,
            set(caches),
            constexpr_values,
            canon,
            pair_idx,
            fast_math,
        ):
            # Sites list is stale after a rewrite; one online pair per
            # kernel is the supported shape (softmax/logsumexp).
            break
    return body


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


def _strip_fp32_casts(node: ast.expr) -> ast.expr:
    """Drop ``cutlass.Float32(v)`` around a value that already is Float32."""

    class Strip(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call) -> ast.AST:
            self.generic_visit(call)
            if not (
                ast.unparse(call.func) == "cutlass.Float32"
                and len(call.args) == 1
                and not call.keywords
                and isinstance(inner := call.args[0], ast.Call)
            ):
                return call
            if ast.unparse(inner.func) == "cutlass.Float32" or (
                isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "bitcast"
                and [ast.unparse(arg) for arg in inner.args] == ["cutlass.Float32"]
            ):
                return inner
            return call

    result = Strip().visit(node)
    assert isinstance(result, ast.expr)
    return result


class _ElementValues:
    """Canonical per-element values of expressions inside ``for`` nests
    whose innermost loop is ``range_constexpr(N)``: local definitions are
    inlined, the innermost index becomes each constant ``0..N-1``, the
    other loop indices become positional placeholders, and register-cache
    reads become the values written into them."""

    def __init__(self) -> None:
        # (cache name, canonical index) -> canonical stored value
        self.cache_values: dict[tuple[str, str], ast.expr] = {}

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
                if _is_name_assign(statement):
                    name = statement.targets[0].id  # type: ignore[attr-defined]
                    if name not in _loads(statement.value):  # type: ignore[attr-defined]
                        definitions[name] = statement.value  # type: ignore[attr-defined]
        return definitions

    @staticmethod
    def _rename(node: ast.expr, names: dict[str, ast.expr]) -> ast.expr:
        class Rename(ast.NodeTransformer):
            def visit_Name(self, name: ast.Name) -> ast.AST:
                replacement = names.get(name.id)
                if replacement is None or not isinstance(name.ctx, ast.Load):
                    return name
                return _reparse_expr(replacement)

        result = Rename().visit(_reparse_expr(node))
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
        expr = _Inliner(definitions).visit(_reparse_expr(node))
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
                def visit_Subscript(self, read: ast.Subscript) -> ast.AST:
                    nonlocal missing
                    self.generic_visit(read)
                    if not (
                        isinstance(read.value, ast.Name) and read.value.id in caches
                    ):
                        return read
                    stored = owner.cache_values.get(
                        (read.value.id, ast.unparse(read.slice))
                    )
                    if stored is None:
                        missing = True
                        return read
                    return _reparse_expr(stored)

            expr = ReadCache().visit(expr)
            if missing:
                return None
            results.append(ast.unparse(_strip_fp32_casts(expr)))
        return results


def _same_loop_ranges(
    first: list[tuple[ast.For, ast.stmt]], second: list[tuple[ast.For, ast.stmt]]
) -> bool:
    return len(first) == len(second) and all(
        ast.unparse(a.iter) == ast.unparse(b.iter)
        for (a, _), (b, _) in zip(first, second, strict=True)
    )


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
            if not _is_name_assign(node):
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
    elements = _ElementValues()
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


def _try_rewrite_pair_exact(
    body: list[ast.stmt],
    site_a: _Site,
    site_b: _Site,
    caches: set[str],
    constexpr_values: dict[str, int] | None,
    canon: Callable[[str], str],
    k: int,
    fast_math: bool,
) -> bool:
    """Pair the exchanges while every exponential stays as written.

    ``_try_rewrite_pair`` caches sweep B's exponentials and rescales them in
    sweep C, which rounds each output differently; it needs fast_math.  This one
    only moves sweep B's ``exp2((x - mi) * C)`` into the frame of the CTA's
    own maximum: the packed exchange folds the CTA sums with ``sum +=
    s_r * exp2((m_r - m) * C)``, the reassociation the kernel's online
    recurrence already applies between tiles, and sweep C still computes
    ``exp2((x - mi) * C)`` from the global maximum.

    Every read of ``mi`` (or a copy of it) between the sweeps must be the
    subtrahend of such an exponent, and site B must reduce exactly their
    sum.  Each exponent's minuend must be the very element site A
    max-reduced (``x - max(x)``): only then is it at most the CTA maximum,
    so the CTA frame cannot overflow, and an all--inf slice contributes
    nothing.
    """
    mi_name = _max_variable(body, site_a, site_b, constexpr_values, canon)
    if mi_name is None:
        return False
    acc = site_b.call.args[0]
    if not isinstance(acc, ast.Name):
        return False
    acc_name = canon(acc.id)
    region = [
        *body[site_a.top_idx + 1 : site_b.top_idx],
        *site_b.for_node.body[: site_b.stmt_idx],
    ]
    if any(
        isinstance(node, (ast.AugAssign, ast.AnnAssign))
        for statement in region
        for node in ast.walk(statement)
    ):
        return False
    assigns = [
        node
        for statement in region
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
    ]
    if any(not _is_name_assign(node) for node in assigns):
        return False

    def target(node: ast.Assign) -> str:
        return canon(node.targets[0].id)  # type: ignore[attr-defined]

    # Names holding ``mi``'s value: ``mi`` and its copies.
    frames = {mi_name}
    changed = True
    while changed:
        changed = False
        for node in assigns:
            value = node.value
            if (
                isinstance(value, ast.Name)
                and canon(value.id) in frames
                and target(node) not in frames
            ):
                frames.add(target(node))
                changed = True
    for node in assigns:
        value = node.value
        if target(node) in frames and not (
            isinstance(value, ast.Name) and canon(value.id) in frames
        ):
            return False

    def frame_subtraction(node: ast.AST) -> ast.BinOp | None:
        """``a - frame`` with no frame value in ``a``."""
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.right, ast.Name)
            and canon(node.right.id) in frames
            and not any(canon(name) in frames for name in _loads(node.left))
        ):
            return node
        return None

    shifted = {
        target(node): node.value for node in assigns if frame_subtraction(node.value)
    }
    if any(
        sum(target(node) == name for node in assigns) != 1 or name == acc_name
        for name in shifted
    ):
        return False

    # ``exp2(Float32(shift) * C)`` with ``shift`` a frame subtraction.
    exponents: dict[int, float] = {}
    allowed: set[int] = set()
    subtractions: list[ast.BinOp] = [
        value for value in shifted.values() if isinstance(value, ast.BinOp)
    ]
    for statement in region:
        for node in ast.walk(statement):
            if not (_is_exp2(node) and not node.keywords):  # type: ignore[union-attr]
                continue
            argument = node.args[0]  # type: ignore[union-attr]
            if not (
                isinstance(argument, ast.BinOp)
                and isinstance(argument.op, ast.Mult)
                and isinstance(argument.right, ast.Constant)
                and type(argument.right.value) is float
                and math.isfinite(argument.right.value)
                and argument.right.value > 0
            ):
                continue
            shift = argument.left
            if (
                isinstance(shift, ast.Call)
                and ast.unparse(shift.func) == "cutlass.Float32"
                and len(shift.args) == 1
                and not shift.keywords
            ):
                shift = shift.args[0]
            if isinstance(shift, ast.Name) and canon(shift.id) in shifted:
                allowed.add(id(shift))
            elif (subtraction := frame_subtraction(shift)) is not None:
                allowed.add(id(subtraction.right))
                subtractions.append(subtraction)
            else:
                continue
            exponents[id(node)] = argument.right.value
    scales = set(exponents.values())
    if len(scales) != 1:
        return False
    (scale,) = scales
    subtrahends = [
        value.right
        for value in shifted.values()
        if isinstance(value, ast.BinOp) and isinstance(value.right, ast.Name)
    ]
    allowed.update(id(node) for node in subtrahends)
    for node in assigns:
        if isinstance(node.value, ast.Name) and target(node) in frames:
            allowed.add(id(node.value))

    # Each exponential is named only to be accumulated (or not at all).
    named = {
        target(node)
        for node in assigns
        if id(node.value) in exponents and target(node) != acc_name
    }
    consumed = {
        id(node.value)
        for node in assigns
        if id(node.value) in exponents and target(node) in named
    }

    def accumulated(value: ast.expr) -> bool:
        """``acc + term`` with ``term`` an exponential or its name."""
        if not (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and isinstance(value.left, ast.Name)
            and canon(value.left.id) == acc_name
        ):
            return False
        term = value.right
        if (
            isinstance(term, ast.Call)
            and ast.unparse(term.func) == "cutlass.Float32"
            and len(term.args) == 1
            and not term.keywords
        ):
            term = term.args[0]
        if id(term) in exponents:
            consumed.add(id(term))
            return True
        if isinstance(term, ast.Name) and canon(term.id) in named:
            allowed.add(id(term))
            return True
        return False

    updates = 0
    for node in assigns:
        if target(node) != acc_name:
            continue
        value = node.value
        if accumulated(value):
            updates += 1
            allowed.add(id(value.left))  # type: ignore[union-attr]
        elif ast.unparse(value) not in {"cutlass.Float32(0)", "cutlass.Float32(0.0)"}:
            return False
    if updates == 0 or set(exponents) - consumed:
        return False
    # Every read of a frame value, shift, exponential name or the
    # accumulator is one of the uses above.
    tracked = frames | set(shifted) | named | {acc_name}
    for statement in region:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and canon(node.id) in tracked
                and id(node) not in allowed
            ):
                return False

    # Each minuend is the element site A max-reduced, at the same loop
    # positions.
    reduced = _reduced_values(body, site_a, canon, caches)
    if reduced is None:
        return False
    elements, reduced_chain, reduced_values = reduced
    parents = {
        id(child): node
        for node in ast.walk(site_b.for_node)
        for child in ast.iter_child_nodes(node)
    }
    for subtraction in subtractions:
        chain = _loop_chain(subtraction, parents, site_b.for_node)
        if (
            chain is None
            or not _same_loop_ranges(chain, reduced_chain)
            or elements.values(subtraction.left, chain, caches) != reduced_values
        ):
            return False

    # The region's frame-dependent values stay in it: after site B only the
    # global ``mi`` (and site B's own reduce input) is read.
    local_values = (frames - {mi_name}) | set(shifted) | named
    for statement in [
        *site_b.for_node.body[site_b.stmt_idx :],
        *body[site_b.top_idx + 1 :],
    ]:
        for node in ast.walk(statement):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            name = canon(node.id)
            if name in local_values or (name == acc_name and node is not acc):
                return False

    # --- everything matched; apply the rewrite. -------------------------
    negative_inf = _unique_name(body, canon, f"_pair_negative_inf_{k}")
    frame = _unique_name(body, canon, f"_pair_frame_{k}")
    _localize_site_a(site_a)
    # An all--inf CTA slice has the zero frame, so exp2((-inf - -inf) * C)
    # does not poison its sum; the packed exchange keeps the true local
    # maximum, so a globally all--inf row still produces NaN.
    after_a = body.index(site_a.for_node) + 1
    body[after_a:after_a] = [
        _stmt(f"{negative_inf} = {mi_name} == cutlass.Float32(float('-inf'))"),
        _stmt(f"{frame} = cutlass.Float32(0) if {negative_inf} else {mi_name}"),
    ]
    for statement in region:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.right, ast.Name)
                and id(node.right) in allowed
                and frame_subtraction(node) is not None
            ):
                node.right = ast.Name(id=frame, ctx=ast.Load())
    _pair_site_b(site_b, mi_name, scale, fast_math, k)
    _rebuffer_pair(body, site_a, site_b)
    for statement in body:
        ast.fix_missing_locations(statement)
    return True


def _max_variable(
    body: list[ast.stmt],
    site_a: _Site,
    site_b: _Site,
    constexpr_values: dict[str, int] | None,
    canon: Callable[[str], str],
) -> str | None:
    """The (canonical) max variable ``mi`` that site A's result feeds, when
    both sites run once and site A's result reaches only ``mi``."""
    from .hoist_warp_reduce import _static_trip_count

    if _static_trip_count(site_a.for_node, constexpr_values) != 1:
        return None
    if _static_trip_count(site_b.for_node, constexpr_values) != 1:
        return None

    # --- max carrier chain: everything after site A in sweep A must be a
    # linear cast/accumulate chain ending at the max variable ``mi``.
    tail = site_a.for_node.body[site_a.stmt_idx + 1 :]
    if not tail or not all(_is_name_assign(s) for s in tail):
        return None
    chain_targets = [s.targets[0].id for s in tail]  # type: ignore[attr-defined]
    mi_name = canon(chain_targets[-1])
    allowed = {canon(a) for a in {site_a.target, mi_name} | set(chain_targets)}
    for stmt in tail:
        if any(canon(n) not in allowed for n in _local_loads(stmt.value)):
            return None
    intermediates = {site_a.target} | {t for t in chain_targets if canon(t) != mi_name}

    # ``mi`` must start at -inf (single trip => mi ends as a cast of the
    # site-A result).
    init_found = False
    for top in body[: site_a.top_idx]:
        if _is_name_assign(top) and canon(top.targets[0].id) == mi_name:  # type: ignore[attr-defined]
            init_found = "-inf" in ast.unparse(top.value)
    if not init_found:
        return None

    # No escape of the local-valued intermediates past sweep A.
    for top in body[site_a.top_idx + 1 :]:
        for name in _loads(top):
            if canon(name) in {canon(x) for x in intermediates}:
                return None
    return mi_name


def _unique_name(body: list[ast.stmt], canon: Callable[[str], str], name: str) -> str:
    used_names = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    used_names.update(canon(used) for used in tuple(used_names))
    while name in used_names:
        name += "_"
    return name


def _localize_site_a(site_a: _Site) -> None:
    """Site A becomes a CTA-local block reduce (drop the buf/mbar args)."""
    block_args = ", ".join(ast.unparse(a) for a in site_a.call.args[:4])
    site_a.assign.value = _stmt(
        f"_x = {_BLOCK_REDUCE}({block_args}, group_span={site_a.group_span})"
    ).value
    ast.fix_missing_locations(site_a.assign)


def _pair_site_b(
    site_b: _Site, mi_name: str, scale: float, fast_math: bool, k: int
) -> None:
    """Site B exchanges the ``(local_max, local_sum)`` pair once and every
    later read of ``mi`` sees the global maximum."""
    gmax = f"_pair_gmax_{k}"
    pair_call_src = (
        f"{gmax}, {site_b.target} = {_PAIR_REDUCE}("
        f"{ast.unparse(site_b.call.args[0])}, {mi_name}, "
        f"{ast.unparse(site_b.call.args[3])}, {site_b.buf_name}, "
        f"{site_b.mbar_name}, group_span={site_b.group_span}, "
        f"cluster_n={site_b.cluster_n}, scale={scale!r}, "
        f"fastmath={fast_math})"
    )
    site_b.for_node.body[site_b.stmt_idx : site_b.stmt_idx + 1] = [
        _stmt(pair_call_src),
        _stmt(f"{mi_name} = {gmax}"),
    ]


def _try_rewrite_pair(
    body: list[ast.stmt],
    site_a: _Site,
    site_b: _Site,
    caches: dict[str, tuple[int, int]],
    constexpr_values: dict[str, int] | None,
    canon: Callable[[str], str],
    k: int,
) -> bool:
    mi_name = _max_variable(body, site_a, site_b, constexpr_values, canon)
    if mi_name is None:
        return False

    # --- reads of ``mi`` between the sweeps: only ``scaled = mi * C``
    # assigns (or dead assigns).  Collect the scaled candidates.
    scaled_before: dict[str, float] = {}
    region: list[tuple[ast.stmt, bool]] = [
        (stmt, False) for stmt in body[site_a.top_idx + 1 : site_b.top_idx]
    ]
    region.extend((stmt, True) for stmt in site_b.for_node.body[: site_b.stmt_idx])

    def _read_anywhere(name: str) -> bool:
        target_canon = canon(name)
        for top in body:
            for other in ast.walk(top):
                if (
                    isinstance(other, ast.Name)
                    and isinstance(other.ctx, ast.Load)
                    and canon(other.id) == target_canon
                ):
                    return True
        return False

    for stmt, _in_sweep_b in region:
        for sub_stmt in ast.walk(stmt):
            if not isinstance(sub_stmt, ast.Assign):
                continue
            if not any(canon(n) == mi_name for n in _loads(sub_stmt.value)):
                continue
            value = sub_stmt.value
            if (
                _is_name_assign(sub_stmt)
                and isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Mult)
                and isinstance(value.left, ast.Name)
                and canon(value.left.id) == mi_name
                and isinstance(value.right, ast.Constant)
                and isinstance(value.right.value, float)
            ):
                scaled_before[sub_stmt.targets[0].id] = value.right.value  # type: ignore[attr-defined]
                continue
            # Dead alias copies (e.g. ``mi_copy = mi``) are harmless.
            if _is_name_assign(sub_stmt) and not _read_anywhere(
                sub_stmt.targets[0].id  # type: ignore[attr-defined]
            ):
                continue
            return False
        # Statements that read mi outside any Assign (calls, stores)?
        assign_reads = set()
        for sub_stmt in ast.walk(stmt):
            if isinstance(sub_stmt, ast.Assign):
                assign_reads.update(id(n) for n in ast.walk(sub_stmt.value))
                for tgt in sub_stmt.targets:
                    assign_reads.update(id(n) for n in ast.walk(tgt))
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and canon(node.id) == mi_name
                and id(node) not in assign_reads
            ):
                return False
    if not scaled_before:
        return False

    # --- sweep B: the vec-loop exp2 keyed by one of the scaled vars.
    match_b = _find_vec_exp(site_b.for_node, site_b.assign, scaled_before, set(caches))
    if match_b is None:
        return False
    scaled0 = match_b.scaled_name
    scale_const = scaled_before[scaled0]
    # No OTHER scaled candidate may feed anything (they'd keep local-max
    # values alive with unknown consumers).
    for name in scaled_before:
        if name != scaled0 and _read_anywhere(name):
            return False

    # The accumulator update in the same vec body must add exactly this
    # exp2 to the site-B input.
    exp_src = ast.unparse(match_b.exp_call)
    acc_name = None
    if isinstance(site_b.call.args[0], ast.Name):
        acc_name = site_b.call.args[0].id
    if acc_name is None:
        return False
    acc_stmt = None
    for stmt in match_b.vec_for.body:
        if (
            _is_name_assign(stmt)
            and canon(stmt.targets[0].id) == canon(acc_name)  # type: ignore[attr-defined]
            and any(
                _is_exp2(node) and ast.unparse(node) == exp_src
                for node in ast.walk(stmt.value)
            )
        ):
            acc_stmt = stmt
            break
    if acc_stmt is None:
        return False

    # --- sweep C: a later top-level loop recomputing the same exp2 with a
    # different (post-exchange) scaled variable.
    scaled_after: dict[str, tuple[float, int]] = {}
    for i in range(site_b.top_idx + 1, len(body)):
        top = body[i]
        if (
            _is_name_assign(top)
            and isinstance(top.value, ast.BinOp)
            and isinstance(top.value.op, ast.Mult)
            and isinstance(top.value.left, ast.Name)
            and canon(top.value.left.id) == mi_name
            and isinstance(top.value.right, ast.Constant)
            and top.value.right.value == scale_const
        ):
            scaled_after[top.targets[0].id] = (top.value.right.value, i)  # type: ignore[attr-defined]
    if not scaled_after:
        return False
    match_c = None
    sweep_c_top_idx = None
    for i in range(site_b.top_idx + 1, len(body)):
        top = body[i]
        if not isinstance(top, ast.For):
            continue
        found = _find_vec_exp(
            top,
            None,
            {name: value for name, (value, _) in scaled_after.items()},
            set(caches),
        )
        if found is not None:
            if match_c is not None:
                return False
            match_c = found
            sweep_c_top_idx = i
    if match_c is None or match_c.canonical != match_b.canonical:
        return False
    scaled1 = match_c.scaled_name
    if canon(scaled1) == canon(scaled0):
        return False
    # The sweep-C exp2 must be the whole RHS of a Name assignment so the
    # cached value can substitute for it.
    if not (
        _is_name_assign(match_c.exp_stmt) and match_c.exp_stmt.value is match_c.exp_call
    ):
        return False

    # --- everything matched; apply the rewrite. -------------------------
    cache_size = None
    cache_alloc_idx = None
    inlined_b = _inline_locals(
        match_b.exp_call.args[0], match_b.vec_for.body, match_b.exp_stmt
    )
    for sub in ast.walk(inlined_b):
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id in caches
        ):
            cache_size, cache_alloc_idx = caches[sub.value.id][:2]
    if cache_size is None:
        return False

    exp_cache = f"_pair_exp_cache_{k}"
    rescale = f"_pair_rescale_{k}"
    negative_inf = _unique_name(body, canon, f"_pair_negative_inf_{k}")
    scale_assignments = [
        node
        for statement, _in_sweep_b in region
        for node in ast.walk(statement)
        if _is_name_assign(node) and node.targets[0].id == scaled0
    ]
    if len(scale_assignments) != 1:
        return False

    # 1) site A -> CTA-local block reduce (drop buf/mbar args).
    _localize_site_a(site_a)

    # An all--inf CTA slice contributes zero when another CTA has a finite
    # maximum. Normalize that slice in the zero frame so exp(-inf - -inf)
    # does not poison the local sum. Keep the true local maximum in the
    # packed exchange: a globally all--inf row must still produce NaN, and
    # NaN input elements continue to poison their local exponential/sum.
    scale_assignments[0].value = _stmt(
        f"_x = (cutlass.Float32(0) if {negative_inf} else {mi_name}) * {scale_const!r}"
    ).value

    # 2) sweep B: name the exp2 (reuse an existing ``v = exp2(...)`` if the
    # emitter already produced one), cache it, and reference it in the
    # accumulator update.
    vec_body = match_b.vec_for.body
    exp_assign: ast.stmt | None = None
    for stmt in vec_body:
        if stmt is acc_stmt:
            break
        if (
            _is_name_assign(stmt)
            and _is_exp2(stmt.value)
            and ast.unparse(stmt.value) == exp_src
        ):
            exp_assign = stmt
    if exp_assign is not None:
        exp_name = exp_assign.targets[0].id  # type: ignore[attr-defined]
    else:
        exp_name = f"_pair_e_{k}"
        exp_assign = _stmt(f"{exp_name} = {exp_src}")
        vec_body.insert(vec_body.index(acc_stmt), exp_assign)
    store = _stmt(f"{exp_cache}[{ast.unparse(match_b.slot)}] = {exp_name}")
    vec_body.insert(vec_body.index(exp_assign) + 1, store)

    class _SwapExp(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if _is_exp2(node) and ast.unparse(node) == exp_src:
                return ast.Name(id=exp_name, ctx=ast.Load())
            return node

    acc_stmt.value = _SwapExp().visit(acc_stmt.value)
    ast.fix_missing_locations(acc_stmt)

    # 3) site B -> single packed pair exchange + global-max reassignment.
    _pair_site_b(site_b, mi_name, scale_const, fast_math=True, k=k)

    # 4) sweep C: the cached exp replaces the recompute.  When the exp
    # feeds exactly one multiply by a scalar that is only read inside this
    # sweep (the hoisted ``1/denom``), fold the CTA-uniform rescale into
    # that scalar instead of paying a per-element multiply.
    exp_target = match_c.exp_stmt.targets[0].id  # type: ignore[attr-defined]
    scaled1_idx = scaled_after[scaled1][1]
    fold_scalar = None
    exp_reads = sum(
        n == exp_target
        for stmt in body[sweep_c_top_idx].body  # type: ignore[attr-defined]
        for n in _loads(stmt)
    )
    if exp_reads == 1:
        for stmt in match_c.vec_for.body:
            if not (_is_name_assign(stmt) and isinstance(stmt.value, ast.BinOp)):
                continue
            value = stmt.value
            if not isinstance(value.op, ast.Mult):
                continue
            names = [o.id for o in (value.left, value.right) if isinstance(o, ast.Name)]
            if exp_target in names and len(names) == 2:
                other = next(n for n in names if n != exp_target)
                read_elsewhere = any(
                    other in _loads(top)
                    for j, top in enumerate(body)
                    if j != sweep_c_top_idx
                )
                defined_before = any(
                    _is_name_assign(top) and top.targets[0].id == other  # type: ignore[attr-defined]
                    for top in body[: scaled1_idx + 1]
                )
                if not read_elsewhere and defined_before:
                    fold_scalar = other
                break
    if fold_scalar is not None:
        match_c.exp_stmt.value = _stmt(
            f"_x = {exp_cache}[{ast.unparse(match_c.slot)}]"
        ).value
    else:
        match_c.exp_stmt.value = _stmt(
            f"_x = {exp_cache}[{ast.unparse(match_c.slot)}] * {rescale}"
        ).value
    ast.fix_missing_locations(match_c.exp_stmt)
    _prune_dead_assigns(
        match_c.vec_for.body,
        [body[sweep_c_top_idx], *body[sweep_c_top_idx + 1 :]],
    )

    # 5) top-level declarations: the f32 exp cache next to the load cache,
    # and the rescale factor after the post-exchange scaled var.
    # The empty slice's cached exponentials are zero. Select a zero rescale
    # directly; evaluating exp(0 - a very negative global maximum) could
    # overflow and turn a valid 0*scale into NaN. A globally invalid row
    # still has a NaN denominator from the packed combine.
    rescale_stmts = [
        _stmt(
            f"{rescale} = cutlass.Float32(0) if {negative_inf} "
            f"else cute.math.exp2({scaled0} - {scaled1})"
        )
    ]
    if fold_scalar is not None:
        rescale_stmts.append(_stmt(f"{fold_scalar} = {fold_scalar} * {rescale}"))
    body[scaled1_idx + 1 : scaled1_idx + 1] = rescale_stmts
    body.insert(
        cache_alloc_idx + 1,
        _stmt(f"{exp_cache} = cute.make_rmem_tensor({cache_size}, cutlass.Float32)"),
    )
    body.insert(
        body.index(site_a.for_node) + 1,
        _stmt(f"{negative_inf} = {mi_name} == cutlass.Float32(float('-inf'))"),
    )

    _rebuffer_pair(body, site_a, site_b)
    return True


def _rebuffer_pair(body: list[ast.stmt], site_a: _Site, site_b: _Site) -> None:
    """Site B's receive buffer becomes ``cluster_n`` Int64 pair slots, and
    site A's buffer/mbarrier (now unused) are removed."""
    for top in body:
        if _is_name_assign(top) and top.targets[0].id == site_b.buf_name:  # type: ignore[attr-defined]
            top.value = _stmt(
                f"_x = cute.arch.alloc_smem(cutlass.Int64, {site_b.cluster_n})"
            ).value
            ast.fix_missing_locations(top)
    removable: list[int] = []
    for i, top in enumerate(body):
        if (
            _is_name_assign(top)
            and top.targets[0].id
            in (  # type: ignore[attr-defined]
                site_a.buf_name,
                site_a.mbar_name,
            )
            or isinstance(top, ast.If)
            and site_a.mbar_name in _loads(top)
        ):
            removable.append(i)
    kept = [top for i, top in enumerate(body) if i not in removable]
    leftover_reads = [
        name
        for top in kept
        for name in _loads(top)
        if name in (site_a.buf_name, site_a.mbar_name)
    ]
    if not leftover_reads:
        for i in reversed(removable):
            del body[i]
