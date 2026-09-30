"""Overlap packed cluster synchronization with independent global loads."""

from __future__ import annotations

import ast

from ._ast_pass_utils import _names_read
from .cluster_sum import _assign
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _trip_count_for

_COLLECTIVE = "_cute_grouped_reduce_cluster_sum4"
_ADDRESS_CALLS = {
    "cutlass.Int32",
    "cutlass.Int64",
    "cute.arch.thread_idx",
    "cute.arch.block_idx",
    "cute.arch.block_idx_in_cluster",
    "ir.VectorType.get",
}
_PURE_CALLS = _ADDRESS_CALLS | {
    "cutlass.Float32",
    "cutlass.Uint16",
    "cutlass.Uint32",
    "cutlass.BFloat16",
    "cutlass.Float16",
    "cutlass.range_constexpr",
    "range",
    "cute.math.rsqrt",
    "cute.math.fma",
    "cute.math.max",
    "cute.math.min",
    "cute.arch.fmax",
    "cute.arch.fmin",
    "max",
    "min",
}


def _call_name(expr: ast.AST) -> str:
    return ast.unparse(expr.func) if isinstance(expr, ast.Call) else ""


def _pure(expr: ast.AST, *, loads: bool = False) -> bool:
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in _PURE_CALLS or (loads and name == "cute.arch.load"):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "bitcast"
            and isinstance(node.func.value, ast.Call)
            and _call_name(node.func.value) in _PURE_CALLS
        ):
            continue
        return False
    return True


def _local_work(stmt: ast.stmt, registers: set[str]) -> bool:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        return _pure(stmt.value, loads=True) and (
            isinstance(target, ast.Name)
            or (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in registers
                and _pure(target.slice)
            )
        )
    if isinstance(stmt, ast.For):
        return (
            isinstance(stmt.target, ast.Name)
            and _range_bounds(stmt.iter) is not None
            and _pure(stmt.iter)
            and not stmt.orelse
            and all(_local_work(s, registers) for s in stmt.body)
        )
    return False


def _flatten_once(body: list[ast.stmt], constants: dict[str, int]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for stmt in body:
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and not stmt.orelse
            and (bounds := _range_bounds(stmt.iter)) is not None
            and _trip_count_for(*bounds, constants, allow_dynamic_base=True) == 1
            and not any(
                isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(stmt)
            )
        ):
            result.extend(
                ast.parse(f"{stmt.target.id} = {ast.unparse(bounds[0])}").body
            )
            result.extend(_flatten_once(stmt.body, constants))
        else:
            result.append(stmt)
    return result


def _collective_index(body: list[ast.stmt]) -> int | None:
    return next(
        (
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign) and _call_name(stmt.value) == _COLLECTIVE
        ),
        None,
    )


def _stage_epilogue_load(body: list[ast.stmt], collective: int) -> list[ast.stmt]:
    collective_outputs = {
        node.id
        for node in ast.walk(body[collective])
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    written = {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    for index in range(collective + 1, len(body)):
        assignment = _assign(body[index])
        if assignment is None or not _pure(assignment[1], loads=True):
            break
        value = assignment[1]
        if _call_name(value) != "cute.arch.load":
            continue
        assert isinstance(value, ast.Call)
        if not value.args:
            continue
        roots = [
            node.value.id
            for node in ast.walk(value.args[0])
            if isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
        ]
        # Kernel parameters are global tensors; locally constructed pointers
        # or tensors can refer to the collective's shared receive buffer.
        if len(roots) != 1 or roots[0] in written:
            continue
        definitions = {
            item[0]: (i, item[1])
            for i, node in enumerate(body[:index])
            if (item := _assign(node)) is not None
        }
        required: set[int] = set()
        visiting: set[str] = set()

        def collect(
            expr: ast.AST,
            definitions: dict[str, tuple[int, ast.expr]] = definitions,
            visiting: set[str] = visiting,
            required: set[int] = required,
        ) -> bool:
            for node in ast.walk(expr):
                if isinstance(node, ast.Name) and node.id in collective_outputs:
                    return False
                if (
                    isinstance(node, ast.Call)
                    and _call_name(node) not in _ADDRESS_CALLS
                ):
                    return False
                if isinstance(node, ast.Name) and node.id in definitions:
                    defining_index, definition = definitions[node.id]
                    if defining_index >= collective:
                        if node.id in visiting:
                            return False
                        # Moving a definition must not change an earlier use.
                        if any(
                            node.id in _names_read(s)
                            for s in body[collective:defining_index]
                        ):
                            return False
                        visiting.add(node.id)
                        if not collect(definition):
                            return False
                        visiting.remove(node.id)
                        required.add(defining_index)
            return True

        if value.keywords or not all(collect(arg) for arg in value.args):
            continue
        if assignment[0] in set().union(
            *(_names_read(s) for s in body[collective:index])
        ):
            continue
        required.add(index)
        return (
            body[:collective]
            + [body[i] for i in sorted(required)]
            + [
                node
                for i, node in enumerate(body[collective:], collective)
                if i not in required
            ]
        )
    return body


def prefetch_cluster_epilogue(
    body: list[ast.stmt], constants: dict[str, int]
) -> list[ast.stmt]:
    """Stage a read-only epilogue load and overlap the initial cluster barrier.

    Only the straight-line packed-sum path is supported. Unknown calls, global
    writes, shared accesses, and control flow prevent synchronization motion.
    """
    if _collective_index(body) is None:
        return body
    body = _flatten_once(body, constants)
    collective = _collective_index(body)
    assert collective is not None
    body = _stage_epilogue_load(body, collective)
    collective = _collective_index(body)
    assert collective is not None
    registers = {
        item[0]
        for stmt in body[:collective]
        if (item := _assign(stmt)) is not None
        and _call_name(item[1]) == "cute.make_rmem_tensor"
    }
    shared = {
        item[0]
        for stmt in body[:collective]
        if (item := _assign(stmt)) is not None
        and _call_name(item[1]) == "cute.arch.alloc_smem"
    }
    # A shared pointer can be wrapped in a tensor or copied to a local alias.
    changed = True
    while changed:
        before = len(shared)
        for stmt in body[:collective]:
            if (item := _assign(stmt)) is not None and _names_read(item[1]) & shared:
                shared.add(item[0])
        changed = len(shared) != before
    wait = next(
        (
            i
            for i, stmt in enumerate(body[:collective])
            if ast.unparse(stmt) == "cute.arch.cluster_wait()"
        ),
        None,
    )
    if wait is None or not all(
        _local_work(s, registers) and not (_names_read(s) & shared)
        for s in body[wait + 1 : collective]
    ):
        return body
    sync = body.pop(wait)
    body.insert(collective - 1, sync)
    arrive = wait - 1
    fence = wait - 2
    if (
        fence < 0
        or ast.unparse(body[arrive]) != "cute.arch.cluster_arrive_relaxed()"
        or ast.unparse(body[fence]) != "cute.arch.mbarrier_init_fence()"
    ):
        return body
    initializers = []
    for i in range(fence - 1, -1, -1):
        stmt = body[i]
        if not (
            isinstance(stmt, ast.If)
            and ast.unparse(stmt.test)
            == "cutlass.Int32(cute.arch.thread_idx()[0]) == 0"
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Expr)
            and _call_name(stmt.body[0].value) == "cute.arch.mbarrier_init"
            and isinstance(stmt.body[0].value, ast.Call)
            and len(stmt.body[0].value.args) == 2
            and isinstance(stmt.body[0].value.args[0], ast.Name)
            and stmt.body[0].value.args[0].id in shared
            and ast.unparse(stmt.body[0].value.args[1]) == "1"
            and not stmt.body[0].value.keywords
            and all(isinstance(s, ast.Pass) for s in stmt.orelse)
        ):
            break
        initializers.append(i)
    if not initializers:
        return body
    destination = next(
        (i for i in range(arrive + 1, collective - 1) if isinstance(body[i], ast.For)),
        None,
    )
    if destination is None or not any(
        isinstance(stmt, ast.Assign) and _call_name(stmt.value) == "cute.arch.load"
        for stmt in body[arrive + 1 : destination]
    ):
        return body
    moved = set(initializers) | {fence, arrive}
    # Loads and local arithmetic do not access any barrier being initialized.
    return (
        [stmt for i, stmt in enumerate(body[:destination]) if i not in moved]
        + [body[i] for i in sorted(moved)]
        + body[destination:]
    )
