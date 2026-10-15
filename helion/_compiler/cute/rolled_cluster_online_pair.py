"""Warp-local online-softmax pairs for a proved rolled cluster reduction."""

from __future__ import annotations

import ast
from itertools import pairwise
import math
from typing import TypeGuard
from typing import cast

from ..ast_read_writes import ReadWrites
from . import cluster_online_pair as pair

_WARP_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair"


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


def _domain(expr: ast.expr, identity: float) -> tuple[ast.expr | None, ast.expr]:
    """Separate a reduction's neutral mask without dropping its FP32 cast."""
    expr = pair._strip_fp32_casts(expr)
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
    return ast.unparse(pair._strip_fp32_casts(result))


def _wrap_sites(body: list[ast.stmt]) -> tuple[list[ast.stmt], set[int]]:
    """Give top-level rolled finalizers the one-trip owner used by pair proofs.

    The temporary owners are erased before code is emitted. Only a roll,
    pure lane-index assignments, its finalizer, and the immediate FP32
    result cast enter an owner; unrelated statements keep their positions.
    """
    from ..tile_strategy import _is_proven_relocatable_assignment

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
        while end < len(body) and pair._is_name_assign(body[end]):
            candidate = body[end]
            assert isinstance(candidate, ast.Assign)
            if (
                isinstance(candidate.value, ast.Call)
                and ast.unparse(candidate.value.func) == pair._CLUSTER_REDUCE
            ):
                break
            if not _is_proven_relocatable_assignment(candidate, allow_load=False):
                break
            end += 1
        if end + 1 >= len(body) or not pair._is_name_assign(body[end]):
            result.append(statement)
            index += 1
            continue
        finalize = body[end]
        result_cast = body[end + 1]
        assert isinstance(finalize, ast.Assign)
        call = finalize.value
        if not (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == pair._CLUSTER_REDUCE
            and len(call.args) == 6
            and pair._is_name_assign(result_cast)
            and _float32(result_cast.value)
            and ast.unparse(result_cast.value.args[0])
            == ast.unparse(finalize.targets[0])
        ):
            result.append(statement)
            index += 1
            continue
        wrapper = pair._stmt(
            "for _rolled_pair_owner in cutlass.range_constexpr(1):\n    pass"
        )
        assert isinstance(wrapper, ast.For)
        wrapper.body = body[index : end + 2]
        owners.add(id(wrapper))
        result.append(wrapper)
        index = end + 2
    return result, owners


def _pure_assignments(statements: list[ast.stmt]) -> list[ast.Assign] | None:
    from ..tile_strategy import _is_proven_relocatable_assignment

    result: list[ast.Assign] = []
    for statement in statements:
        if isinstance(statement, ast.For):
            nested = _pure_assignments(statement.body)
            if nested is None:
                return None
            result.extend(nested)
        elif pair._is_name_assign(statement) and _is_proven_relocatable_assignment(
            statement, allow_load=True
        ):
            assert isinstance(statement, ast.Assign)
            result.append(statement)
        elif not isinstance(statement, ast.Pass):
            return None
    return result


def _prove(
    body: list[ast.stmt], site_a: pair._Site, site_b: pair._Site
) -> tuple[str, float, list[ast.Name]] | None:
    tail = site_a.for_node.body[site_a.stmt_idx + 1 :]
    if len(tail) != 1 or not pair._is_name_assign(tail[0]):
        return None
    maximum = pair._assigned_name(tail[0])
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
        if pair._assigned_name(node) != acc.id:
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
        pair._assigned_name(top)
        for top in body
        if pair._is_name_assign(top)
        and isinstance(top.value, ast.Call)
        and ast.unparse(top.value.func) == "cute.make_rmem_tensor"
    }
    reduced = pair._reduced_values(body, site_a, lambda name: name, caches)
    if reduced is None:
        return None
    elements, max_chain, max_values = reduced
    fold, term = folds[0]
    parents = {
        id(child): parent
        for parent in ast.walk(site_b.for_node)
        for child in ast.iter_child_nodes(parent)
    }
    sum_chain = pair._loop_chain(fold, parents, site_b.for_node)
    if sum_chain is None or not pair._same_loop_ranges(max_chain, sum_chain):
        return None
    terms = elements.values(term, sum_chain, caches)
    if terms is None or len(terms) != len(max_values):
        return None
    scales: set[float] = set()
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
        if not pair._is_exp2(exponential):
            return None
        assert isinstance(exponential, ast.Call)
        if any(
            keyword.arg != "fastmath"
            or not isinstance(keyword.value, ast.Constant)
            or type(keyword.value.value) is not bool
            for keyword in exponential.keywords
        ):
            return None
        argument = exponential.args[0]
        if not (
            isinstance(argument, ast.BinOp)
            and isinstance(argument.op, ast.Mult)
            and isinstance(argument.right, ast.Constant)
            and type(argument.right.value) is float
            and math.isfinite(argument.right.value)
            and argument.right.value > 0
        ):
            return None
        shift = argument.left
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
        scales.add(argument.right.value)
    if len(scales) != 1:
        return None
    dependent = {maximum}
    changed = True
    while changed:
        changed = False
        for node in assignments:
            if set(ReadWrites.from_ast(node.value).reads) & dependent:
                target = pair._assigned_name(node)
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


def fuse_rolled_cluster_online_pair(body: list[ast.stmt]) -> list[ast.stmt]:
    working, owners = _wrap_sites(body)
    sites = pair._find_sites(working)
    if sites is not None:
        for site_a, site_b in pairwise(sites):
            if not (
                id(site_a.for_node) in owners
                and id(site_b.for_node) in owners
                and site_a.op == "max"
                and site_b.op == "sum"
                and site_a.group_span == site_b.group_span
                and site_a.group_span is not None
                and site_a.group_span >= 64
                and site_a.cluster_n == site_b.cluster_n
                and site_a.cluster_n is not None
                and 1 < site_a.cluster_n <= 16
            ):
                continue
            proof = _prove(working, site_a, site_b)
            if proof is None:
                continue
            maximum, scale, subtractions = proof
            frame = pair._unique_name(working, lambda name: name, "_rolled_pair_frame")
            after_a = working.index(site_a.for_node) + 1
            working.insert(
                after_a,
                pair._stmt(
                    f"{frame} = cutlass.Float32(0) if {maximum} == "
                    f"cutlass.Float32(float('-inf')) else {maximum}"
                ),
            )
            for node in subtractions:
                node.id = frame
            call = site_a.call
            site_a.assign.value = cast(
                "ast.Assign",
                pair._stmt(
                    f"_x = _cute_grouped_reduce_warp("
                    f"{', '.join(ast.unparse(argument) for argument in call.args[:4])}, "
                    "pre=1, group_span=32)"
                ),
            ).value
            pair._pair_site_b(site_b, maximum, scale, False, 0)
            new_call = cast("ast.Assign", site_b.for_node.body[site_b.stmt_idx]).value
            assert isinstance(new_call, ast.Call)
            new_call.func = ast.Name(id=_WARP_PAIR, ctx=ast.Load())
            pair._rebuffer_pair(working, site_a, site_b)
            for statement in working:
                if (
                    pair._is_name_assign(statement)
                    and pair._assigned_name(statement) == site_b.buf_name
                ):
                    statement.value = cast(
                        "ast.Assign",
                        pair._stmt(
                            f"_x = cute.arch.alloc_smem(cutlass.Int64, "
                            f"{cast('int', site_b.group_span) // 32 * cast('int', site_b.cluster_n)})"
                        ),
                    ).value
            break
    result: list[ast.stmt] = []
    for statement in working:
        result.extend(
            cast("ast.For", statement).body if id(statement) in owners else [statement]
        )
    return result
