"""Carry same-element collective seeds through a shared TMEM allocation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ._ast_pass_utils import _bound_names
from ._ast_pass_utils import _fresh_prefix
from .collective_epilogue import _RemoveFullThreadBounds
from .scalar_recipe import _clone
from .scalar_recipe import _read_names

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .collective_tcgen05 import CollectiveTcgen05Plan


@dataclass(frozen=True)
class NativeCollectiveLifetime:
    plan: CollectiveTcgen05Plan
    m_offset: str
    n_offset: str
    local_m: str
    local_n: str
    thread_dimensions: tuple[int, int, int]
    before_seed: tuple[ast.stmt, ...]
    seed_nodes: tuple[ast.stmt, ...]
    seed_statements: tuple[ast.Assign, ...]
    seed_value: ast.expr
    finish: tuple[ast.stmt, ...]
    initialized: ast.expr


def prune_pure_scalar_statements(
    statements: Sequence[ast.stmt], live: set[str]
) -> tuple[list[ast.stmt], set[str]]:
    """Backward liveness inside an independently proven pure scalar region.

    The caller proves all assignments and predicates pure and no values escape
    the region except ``live``. This handles overwritten definitions, unlike
    name-wide dead-assignment elimination. Do not use on arbitrary device ASTs.
    """
    result: list[ast.stmt] = []
    live = set(live)
    for statement in reversed(statements):
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Assign):
            assert len(statement.targets) == 1
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            if target.id not in live:
                continue
            live.remove(target.id)
            live.update(_read_names(statement.value))
            result.append(statement)
            continue
        assert isinstance(statement, ast.If)
        left, left_live = prune_pure_scalar_statements(statement.body, live)
        right, right_live = prune_pure_scalar_statements(statement.orelse, live)
        if not left and not right:
            continue
        live = left_live | right_live | _read_names(statement.test)
        result.append(ast.If(statement.test, left or [ast.Pass()], right))
    return list(reversed(result)), live


def _same_cell(index: ast.expr, offset: str, coordinate: str) -> bool:
    if isinstance(index, ast.Name) and index.id == coordinate:
        return True
    # Scalar collective reads are emitted as (tile_offset + local) - tile_offset.
    # Require this exact integer coordinate construction, not algebraic guesses
    # about arbitrary expressions, casts, aliases or shifted cells.
    return (
        isinstance(index, ast.BinOp)
        and isinstance(index.op, ast.Sub)
        and isinstance(index.right, ast.Name)
        and index.right.id == offset
        and isinstance(index.left, ast.BinOp)
        and isinstance(index.left.op, ast.Add)
        and isinstance(index.left.left, ast.Name)
        and index.left.left.id == offset
        and isinstance(index.left.right, ast.Name)
        and index.left.right.id == coordinate
    )


def _seed_recipe(
    source: NativeCollectiveLifetime,
    target: NativeCollectiveLifetime,
    registers: str,
    element: str,
) -> tuple[list[ast.Assign], ast.expr] | None:
    from .collective_matmul import _uses_thread_coordinates

    shared = f"{source.plan.prefix}_c"

    class ReplaceCell(ast.NodeTransformer):
        seen = False
        invalid = False

        def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
            if isinstance(node.value, ast.Name) and node.value.id == shared:
                if not (
                    isinstance(node.ctx, ast.Load)
                    and isinstance(node.slice, ast.Tuple)
                    and len(node.slice.elts) == 2
                    and _same_cell(node.slice.elts[0], target.m_offset, target.local_m)
                    and _same_cell(node.slice.elts[1], target.n_offset, target.local_n)
                ):
                    self.invalid = True
                    return node
                self.seen = True
                return ast.Subscript(
                    ast.Name(registers, ast.Load()),
                    ast.Name(element, ast.Load()),
                    ast.Load(),
                )
            return cast("ast.expr", self.generic_visit(node))

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == shared:
                self.invalid = True
            return node

    replace = ReplaceCell()
    bounds = _RemoveFullThreadBounds(target.thread_dimensions)
    statements = [
        cast("ast.Assign", bounds.visit(replace.visit(_clone(statement))))
        for statement in target.seed_statements
    ]
    value = cast("ast.expr", bounds.visit(replace.visit(_clone(target.seed_value))))
    if (
        not replace.seen
        or replace.invalid
        or _uses_thread_coordinates([*statements, value])
    ):
        return None
    return statements, value


def _statement_lists(body: list[ast.stmt]) -> list[list[ast.stmt]]:
    result = [body]
    for statement in body:
        if isinstance(statement, (ast.For, ast.If)):
            result.extend(_statement_lists(statement.body))
            result.extend(_statement_lists(statement.orelse))
    return result


def _position(body: list[ast.stmt], nodes: Sequence[ast.stmt]) -> int | None:
    if not nodes:
        return None
    for index, statement in enumerate(body):
        if (
            statement is nodes[0]
            and index + len(nodes) <= len(body)
            and all(
                item is expected
                for item, expected in zip(
                    body[index : index + len(nodes)], nodes, strict=True
                )
            )
        ):
            return index
    return None


def _reads_outside(
    body: Sequence[ast.stmt], names: set[str], ignored: set[int]
) -> bool:
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if id(node) in ignored:
            continue
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in names
        ):
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def bridge_collective_tmem_seeds(
    body: list[ast.stmt],
    lifetimes: Sequence[NativeCollectiveLifetime],
    fresh_name: Callable[[str], str],
) -> int:
    """Replace an adjacent native seed upload with an exact register recipe.

    Only the next collective in the same statement list may reuse a TMEM
    value. Its setup must be the entire interval after the source's completed
    drain. No control-flow motion or intermediate TMEM writer is admitted.
    Shared drains remain whenever any other consumer still reads their values.
    """
    changes = 0
    for target in lifetimes:
        if not target.seed_nodes:
            continue
        for source in lifetimes:
            if source is target or not (
                source.plan.resource is target.plan.resource
                and source.plan.bm == target.plan.bm
                and source.plan.bn == target.plan.bn
                and source.m_offset == target.m_offset
                and source.n_offset == target.n_offset
            ):
                continue
            location = next(
                (
                    (statements, start, end)
                    for statements in _statement_lists(body)
                    if (start := _position(statements, source.finish)) is not None
                    and (end := _position(statements, target.seed_nodes)) is not None
                    and start + len(source.finish) <= end
                    and all(
                        id(statement) in {id(item) for item in target.before_seed}
                        for statement in statements[start + len(source.finish) : end]
                    )
                ),
                None,
            )
            if location is None:
                continue
            prefix = _fresh_prefix(
                "collective_tmem_seed",
                _bound_names(ast.Module(body=body, type_ignores=[])),
                fresh_name,
            )
            registers, element = f"{prefix}_values", f"{prefix}_i"
            recipe = _seed_recipe(source, target, registers, element)
            if recipe is None:
                continue
            statements, start, end = location
            setup, value = recipe
            dependent = {target.local_m, target.local_n, registers, element}
            invariant: list[ast.Assign] = []
            varying: list[ast.Assign] = []
            for statement in setup:
                assert len(statement.targets) == 1
                name = statement.targets[0]
                assert isinstance(name, ast.Name)
                if _read_names(statement.value) & dependent:
                    dependent.add(name.id)
                    varying.append(statement)
                else:
                    invariant.append(statement)
            bm, bn = source.plan.bm, source.plan.bn
            operation, repetitions = (
                ("St16x128bOp", bn // 4) if bm == 64 else ("St32x32bOp", bn)
            )
            replacement = ast.parse(f"""
{prefix}_mn = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({source.plan.prefix}_acc)
{prefix}_load_atom = cutlass.utils.blackwell_helpers.get_tmem_load_op(({bm}, {bn}, {source.plan.bk}), cutlass.utils.layout.LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32, ({bm}, {bn}), False)
{prefix}_load_copy = cute.nvgpu.tcgen05.make_tmem_copy({prefix}_load_atom, {prefix}_mn)
{prefix}_load_thread = {prefix}_load_copy.get_slice({target.plan.tid})
{prefix}_source = {prefix}_load_thread.partition_S({prefix}_mn)
{prefix}_coords = {prefix}_load_thread.partition_D(cute.make_identity_tensor(({bm}, {bn})))
{registers} = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)
{registers}.fill(0.0)
if {prefix}_initialized:
    cute.copy({prefix}_load_copy, {prefix}_source, {registers})
cute.arch.fence_view_async_tmem_load()
cute.arch.sync_threads()
""").body
            replacement.extend(invariant)
            loop = cast(
                "ast.For",
                ast.parse(f"""
for {element} in cutlass.range_constexpr(cute.size({registers})):
    {target.local_m} = {prefix}_coords[{element}][0]
    {target.local_n} = {prefix}_coords[{element}][1]
""").body[0],
            )
            loop.body.extend(varying)
            loop.body.extend(
                ast.parse(
                    f"{registers}[{element}] = cutlass.Float32({ast.unparse(value)})"
                ).body
            )
            replacement.append(loop)
            replacement.extend(
                ast.parse(f"""
{prefix}_destination_mn = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({target.plan.prefix}_acc)
{prefix}_store_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.{operation}(cute.nvgpu.tcgen05.Repetition({repetitions})), cutlass.Float32)
{prefix}_store_copy = cute.nvgpu.tcgen05.make_tmem_copy({prefix}_store_atom, {prefix}_destination_mn)
{prefix}_store_thread = {prefix}_store_copy.get_slice({target.plan.tid})
{prefix}_destination = {prefix}_store_thread.partition_D({prefix}_destination_mn)
cute.copy({prefix}_store_copy, {registers}, {prefix}_destination)
cute.arch.fence_view_async_tmem_store()
cute.arch.sync_threads()
""").body
            )
            capture = ast.Assign(
                [ast.Name(f"{prefix}_initialized", ast.Store())],
                _clone(source.initialized),
            )
            removable_names = {f"{source.plan.prefix}_c", f"{source.plan.prefix}_cp"}
            storage = [
                node
                for node in source.before_seed
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in removable_names
            ]
            removable_names.update(
                node.id
                for statement in source.finish
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            ignored = {
                id(node) for node in (*source.finish, *target.seed_nodes, *storage)
            }
            remove_drain = not _reads_outside(body, removable_names, ignored)
            statements[end : end + len(target.seed_nodes)] = replacement
            if remove_drain:
                statements[start : start + len(source.finish)] = [capture]
                for block in _statement_lists(body):
                    block[:] = [node for node in block if node not in storage]
            else:
                statements.insert(start + len(source.finish), capture)
            changes += 1
            break
    return changes
