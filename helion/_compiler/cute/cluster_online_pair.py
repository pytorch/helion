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

Every rewrite condition fails closed: if any structural check does not
match, the kernel keeps the two-exchange form.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
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


def _find_vec_exp(
    root: ast.stmt,
    stop_at: ast.stmt | None,
    scaled_names: set[str],
    cache_names: set[str],
) -> _VecExpMatch | None:
    """Find the (unique) constexpr vec loop under ``root`` (searching
    statements before ``stop_at`` only) whose body computes
    ``cute.math.exp2(<expr> - <scaled>)`` reading exactly one fuse-cache
    slot; return its canonical form."""
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
                if not (
                    isinstance(arg, ast.BinOp)
                    and isinstance(arg.op, ast.Sub)
                    and isinstance(arg.right, ast.Name)
                    and arg.right.id in scaled_names
                ):
                    continue
                inlined = _inline_locals(arg, node.body, stmt)
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
                    inlined, {vec_var: "_V_", arg.right.id: "_S_"}
                )
                matches.append(
                    _VecExpMatch(
                        node,
                        stmt,
                        sub,
                        arg.right.id,
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
        if _try_rewrite_pair(
            body, site_a, site_b, caches, constexpr_values, canon, pair_idx, fast_math
        ):
            # Sites list is stale after a rewrite; one online pair per
            # kernel is the supported shape (softmax/logsumexp).
            break
    return body


def _try_rewrite_pair(
    body: list[ast.stmt],
    site_a: _Site,
    site_b: _Site,
    caches: dict[str, tuple[int, int]],
    constexpr_values: dict[str, int] | None,
    canon: Callable[[str], str],
    k: int,
    fast_math: bool,
) -> bool:
    from .hoist_warp_reduce import _static_trip_count

    if _static_trip_count(site_a.for_node, constexpr_values) != 1:
        return False
    if _static_trip_count(site_b.for_node, constexpr_values) != 1:
        return False

    # --- max carrier chain: everything after site A in sweep A must be a
    # linear cast/accumulate chain ending at the max variable ``mi``.
    tail = site_a.for_node.body[site_a.stmt_idx + 1 :]
    if not tail or not all(_is_name_assign(s) for s in tail):
        return False
    chain_targets = [s.targets[0].id for s in tail]  # type: ignore[attr-defined]
    mi_name = canon(chain_targets[-1])
    allowed = {canon(a) for a in {site_a.target, mi_name} | set(chain_targets)}
    for stmt in tail:
        if any(canon(n) not in allowed for n in _local_loads(stmt.value)):
            return False
    intermediates = {site_a.target} | {t for t in chain_targets if canon(t) != mi_name}

    # ``mi`` must start at -inf (single trip => mi ends as a cast of the
    # site-A result).
    init_found = False
    for top in body[: site_a.top_idx]:
        if _is_name_assign(top) and canon(top.targets[0].id) == mi_name:  # type: ignore[attr-defined]
            init_found = "-inf" in ast.unparse(top.value)
    if not init_found:
        return False

    # No escape of the local-valued intermediates past sweep A.
    for top in body[site_a.top_idx + 1 :]:
        for name in _loads(top):
            if canon(name) in {canon(x) for x in intermediates}:
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
    match_b = _find_vec_exp(
        site_b.for_node, site_b.assign, set(scaled_before), set(caches)
    )
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
        found = _find_vec_exp(top, None, set(scaled_after), set(caches))
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
    gmax = f"_pair_gmax_{k}"

    # 1) site A -> CTA-local block reduce (drop buf/mbar args).
    block_args = ", ".join(ast.unparse(a) for a in site_a.call.args[:4])
    site_a.assign.value = _stmt(
        f"_x = {_BLOCK_REDUCE}({block_args}, group_span={site_a.group_span})"
    ).value
    ast.fix_missing_locations(site_a.assign)

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
    pair_call_src = (
        f"{gmax}, {site_b.target} = {_PAIR_REDUCE}("
        f"{ast.unparse(site_b.call.args[0])}, {mi_name}, "
        f"{ast.unparse(site_b.call.args[3])}, {site_b.buf_name}, "
        f"{site_b.mbar_name}, group_span={site_b.group_span}, "
        f"cluster_n={site_b.cluster_n}, scale={scale_const!r}, "
        f"fastmath={fast_math})"
    )
    site_b.for_node.body[site_b.stmt_idx : site_b.stmt_idx + 1] = [
        _stmt(pair_call_src),
        _stmt(f"{mi_name} = {gmax}"),
    ]

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
    rescale_stmts = [_stmt(f"{rescale} = cute.math.exp2({scaled0} - {scaled1})")]
    if fold_scalar is not None:
        rescale_stmts.append(_stmt(f"{fold_scalar} = {fold_scalar} * {rescale}"))
    body[scaled1_idx + 1 : scaled1_idx + 1] = rescale_stmts
    body.insert(
        cache_alloc_idx + 1,
        _stmt(f"{exp_cache} = cute.make_rmem_tensor({cache_size}, cutlass.Float32)"),
    )

    # 6) preamble: site B's receive buffer becomes ``cluster_n`` Int64 pair
    # slots; site A's buffer/mbarrier (now unused) are removed.
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
    return True
