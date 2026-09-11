"""Pack four independent cluster sums into one transaction-counted exchange."""

from __future__ import annotations

import ast
from typing import cast


def _assign(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id, stmt.value
    return None


def _sum_call(stmt: ast.stmt) -> ast.Call | None:
    assignment = _assign(stmt)
    if assignment is None:
        return None
    call = assignment[1]
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cute_grouped_reduce_cluster"
        and len(call.args) == 6
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "sum"
        and all(isinstance(call.args[i], ast.Name) for i in (0, 3, 4, 5))
    ):
        return call
    return None


def fuse_cluster_sums(body: list[ast.stmt]) -> list[ast.stmt]:
    """Fuse straight-line, independent FP32 sums with matching launch layouts.

    Only lane-index assignments and result casts may separate the four sites.
    The single-use shared buffers and barriers must have canonical allocation
    and initialization statements in this same scope.
    """
    body = list(body)
    for start, stmt in enumerate(body):
        first = _sum_call(stmt)
        if first is None:
            continue
        sites = [start]
        assignment = _assign(stmt)
        assert assignment is not None
        results = {assignment[0]}
        for index in range(start + 1, len(body)):
            assignment = _assign(body[index])
            if assignment is None:
                break
            name, value = assignment
            if _sum_call(body[index]) is not None:
                sites.append(index)
                results.add(name)
                if len(sites) == 4:
                    break
            elif not (
                ast.unparse(value) == "cutlass.Int32(cute.arch.thread_idx()[0])"
                or (
                    isinstance(value, ast.Call)
                    and ast.unparse(value.func) == "cutlass.Float32"
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id in results
                    and not value.keywords
                )
            ):
                break
        if len(sites) != 4 or len(results) != 4:
            continue
        calls = [cast("ast.Call", _sum_call(body[i])) for i in sites]
        keywords = {kw.arg: kw.value for kw in first.keywords}
        if set(keywords) != {"group_span", "cluster_n"} or not all(
            isinstance(v, ast.Constant) and isinstance(v.value, int)
            for v in keywords.values()
        ):
            continue
        span = cast("int", cast("ast.Constant", keywords["group_span"]).value)
        cluster = cast("int", cast("ast.Constant", keywords["cluster_n"]).value)
        if span % 32 or not 32 <= span <= 1024 or not 2 <= cluster <= 32:
            continue
        slots = span // 32 * cluster
        definitions = [
            {
                item[0]: item[1]
                for node in body[:site]
                if (item := _assign(node)) is not None
            }
            for site in sites
        ]
        writes = {
            item[0]
            for node in body[start : sites[-1] + 1]
            if (item := _assign(node)) is not None
        }
        if any(
            ast.unparse(call.args[2]) != "cutlass.Float32(0)"
            or [(k.arg, ast.unparse(k.value)) for k in call.keywords]
            != [(k.arg, ast.unparse(k.value)) for k in first.keywords]
            or cast("ast.Name", call.args[0]).id in writes
            or ast.unparse(defs.get(cast("ast.Name", call.args[3]).id, call.args[3]))
            != "cutlass.Int32(cute.arch.thread_idx()[0])"
            for call, defs in zip(calls, definitions, strict=True)
        ):
            continue
        buffers = [cast("ast.Name", call.args[4]).id for call in calls]
        barriers = [cast("ast.Name", call.args[5]).id for call in calls]
        if len(set(buffers + barriers)) != 8:
            continue
        allocations: dict[str, int] = {}
        initializers: dict[str, int] = {}
        for index, node in enumerate(body[:start]):
            if assignment := _assign(node):
                name, value = assignment
                expected = (
                    f"cute.arch.alloc_smem(cutlass.Float32, {slots})"
                    if name in buffers
                    else "cute.arch.alloc_smem(cutlass.Int64, 1)"
                )
                if name in buffers + barriers and ast.unparse(value) == expected:
                    allocations[name] = index
            if isinstance(node, ast.If):
                for name in barriers:
                    if (
                        len(node.body) == 1
                        and ast.unparse(node.body[0])
                        == f"cute.arch.mbarrier_init({name}, 1)"
                        and all(isinstance(s, ast.Pass) for s in node.orelse)
                        and ast.unparse(node.test)
                        == "cutlass.Int32(cute.arch.thread_idx()[0]) == 0"
                    ):
                        initializers[name] = index
        if set(allocations) != set(buffers + barriers) or set(initializers) != set(
            barriers
        ):
            continue
        loads = dict.fromkeys(buffers + barriers, 0)
        stores = dict.fromkeys(loads, 0)
        for node in body:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id in loads:
                        loads[sub.id] += 1
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    if sub.id in stores:
                        stores[sub.id] += 1
        if any(count != 1 for count in stores.values()):
            continue
        if any(loads[name] != 1 for name in buffers) or any(
            loads[name] != 2 for name in barriers
        ):
            continue
        targets = ", ".join(
            cast("tuple[str, ast.expr]", _assign(body[i]))[0] for i in sites
        )
        inputs = ", ".join(ast.unparse(call.args[0]) for call in calls)
        replacement = ast.parse(
            f"{targets} = _cute_grouped_reduce_cluster_sum4("
            f"{inputs}, {ast.unparse(first.args[3])}, {buffers[0]}, {barriers[0]}, "
            f"group_span={span}, cluster_n={cluster})"
        ).body[0]
        body[start] = replacement
        body[allocations[buffers[0]]] = ast.parse(
            f"{buffers[0]} = cute.arch.alloc_smem(cutlass.Float32, {slots * 4}, alignment=16)"
        ).body[0]
        remove = set(sites[1:])
        remove.update(allocations[name] for name in buffers[1:] + barriers[1:])
        remove.update(initializers[name] for name in barriers[1:])
        return fuse_cluster_sums([s for i, s in enumerate(body) if i not in remove])
    return body
