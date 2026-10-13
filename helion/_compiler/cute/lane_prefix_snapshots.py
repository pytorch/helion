"""Freeze a straight-line load prefix for the existing lane-unroll pass.

Only the address slice is cloned. Computation, masks, stores and their order
remain in the original loop. The old lane-access proof checks every moved
load against every write before any replacement is installed.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ..tile_strategy import _is_proven_relocatable_call
from ..tile_strategy import _lane_split_reorders_aliasing_memory
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .hoist_lane_invariant_reductions import _rmem_declaration
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _definition_snapshots
from .persistent_branch_vec import _freeze_definition
from .persistent_branch_vec import _generated_access_pointer
from .persistent_branch_vec import _lane_dependent_alias_expansion
from .persistent_branch_vec import _memory_load_calls

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, original: ast.expr, replacement: ast.expr) -> None:
        self.original = original
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST | list[ast.stmt] | None:
        if node is self.original:
            return self.replacement
        return super().visit(node)


@dataclasses.dataclass
class _Definition:
    index: int
    stmt: ast.Assign
    dependencies: dict[str, _Definition]
    is_load: bool


def _same(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def _known_loop_effects(loop: ast.For, register_names: frozenset[str]) -> bool:
    """Unknown mutation/synchronization or early exit pins the prefix."""
    lists = {
        stmt.targets[0].id
        for stmt in ast.walk(loop)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.List)
        and not stmt.value.elts
    }
    bindings = [
        node.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    # Sibling-loop fusion may allocate a private register cache inside the
    # row. Keep its constructor and all accesses in the compute phase, while
    # allowing unrelated leading global loads to cross this local allocation.
    # Rebinding, aliases written through other names, and unknown uses still
    # fail the ordinary effect checks below.
    local_registers = {
        declaration[0]: stmt.value
        for stmt in ast.walk(loop)
        if isinstance(stmt, ast.Assign)
        and (declaration := _rmem_declaration(stmt)) is not None
        and bindings.count(declaration[0]) == 1
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.args[0], ast.Constant)
        and type(stmt.value.args[0].value) is int
    }
    allocation_calls = {id(call) for call in local_registers.values()}
    local_storage = {name for name in lists if bindings.count(name) == 1}
    local_storage.update(local_registers)
    local_storage.update(name for name in register_names if name not in bindings)
    for node in ast.walk(loop):
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in local_storage
            ):
                return False
        if isinstance(
            node,
            (
                ast.Break,
                ast.Continue,
                ast.Return,
                ast.Raise,
                ast.Yield,
                ast.YieldFrom,
                ast.Try,
                ast.With,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.NamedExpr,
                ast.Delete,
            ),
        ):
            return False
        if not isinstance(node, ast.Call):
            continue
        if _is_store_call(node):
            if _generated_access_pointer(node) is None:
                return False
            continue
        if id(node) in allocation_calls:
            continue
        if _is_proven_relocatable_call(node, allow_load=True):
            continue
        name = ast.unparse(node.func)
        if name in {"range", "cutlass.range_constexpr", "cute.arch.warp_reduction_sum"}:
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in lists
        ):
            continue
        return False
    return True


def _independent(
    loop: ast.For,
    loads: list[_Definition],
    disjoint: set[frozenset[str]],
    strides: dict[tuple[str, int], int],
    extent: int,
) -> bool:
    lane = loop.target.id  # type: ignore[union-attr]
    tensor_names = {
        node.value.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }
    writes: list[tuple[int, ast.Call, tuple[ast.expr, ...]]] = []

    def visit(stmt: ast.stmt, index: int, guards: tuple[ast.expr, ...]) -> None:
        if isinstance(stmt, ast.If):
            for child in stmt.body:
                visit(child, index, (*guards, stmt.test))
            for child in stmt.orelse:
                visit(child, index, guards)
        elif isinstance(stmt, ast.For):
            for child in [*stmt.body, *stmt.orelse]:
                visit(child, index, guards)
        else:
            writes.extend(
                (index, call, guards)
                for call in ast.walk(stmt)
                if isinstance(call, ast.Call) and _is_store_call(call)
            )

    for index, stmt in enumerate(loop.body):
        visit(stmt, index, ())

    def may_alias(left: frozenset[str] | None, right: frozenset[str] | None) -> bool:
        return (
            left is None
            or right is None
            or any(
                a == b or frozenset((a, b)) not in disjoint for a in left for b in right
            )
        )

    required = {
        name
        for item in loads
        for call in _memory_load_calls(item.stmt)
        for name in ReadWrites.from_ast(call).reads
    }
    required.update(
        name
        for _, call, guards in writes
        for expression in (call, *guards)
        for name in ReadWrites.from_ast(expression).reads
    )
    definitions = _definition_snapshots(loop.body, required_names=required)
    if definitions is None:
        return False
    all_writes = _binding_write_roots(loop)
    for item in loads:
        for call in _memory_load_calls(item.stmt):
            access = _generated_access_pointer(call)
            if access is None:
                return False
            pointer, _width = access
            expanded_pointer = _lane_dependent_alias_expansion(
                pointer, definitions[item.index], lane
            )

            _ReplaceNode(pointer, expanded_pointer).visit(call)
            pointer = expanded_pointer
            conditionals = [
                node for node in ast.walk(pointer) if isinstance(node, ast.IfExp)
            ]
            if not conditionals:
                continue
            # A masked load still reads its fallback address. Under an exact
            # lane-invariant store guard, the false path has no aliased writes.
            # Specialization is proof-only: emitted pointer/mask stays original.
            if len(conditionals) != 1:
                return False
            choice = conditionals[0]
            condition = _freeze_definition(choice.test, definitions[item.index])
            if (
                condition is None
                or set(ReadWrites.from_ast(condition).reads) & all_writes
            ):
                return False
            predicate_roots = _tensor_arg_roots(condition, tensor_names)
            for write_index, store, guards in writes:
                store_roots = _store_tensor_roots(store, tensor_names)
                if _memory_load_calls(condition) and may_alias(
                    predicate_roots, store_roots
                ):
                    return False
                if not may_alias(_tensor_arg_roots(pointer, tensor_names), store_roots):
                    continue
                if write_index <= item.index or not any(
                    frozen is not None and _same(condition, frozen)
                    for guard in guards
                    for frozen in [_freeze_definition(guard, definitions[write_index])]
                ):
                    return False

            # Replace only this conditional in the disposable analysis AST.
            _ReplaceNode(choice, choice.body).visit(call)
    return not _lane_split_reorders_aliasing_memory(
        loop.body,
        [],
        disjoint,
        lane,
        extent,
        strides,
        additional_inputs=[(item.index + 1, item.stmt.targets[0].id) for item in loads],  # type: ignore[union-attr]
    )


def snapshot_load_prefix(
    loop: ast.For,
    extent: int,
    new_var: Callable[[str], str],
    *,
    disjoint: set[frozenset[str]],
    strides: dict[tuple[str, int], int],
    rename_groups: Mapping[str, str],
    register_names: frozenset[str] = frozenset(),
) -> list[tuple[ast.Assign, ast.Assign, bool]]:
    """Original statement, fresh load-phase statement, and load flag.

    Every dependency binds to the exact preceding definition, even if its
    spelling is overwritten later. A live-in that is written by this loop,
    unknown effect, unsupported compound prefix, or failed alias proof declines.
    """
    from .unroll_lane_loads import _candidate_target

    analysis = statement_from_string(ast.unparse(loop))
    assert isinstance(analysis, ast.For)
    ast_rename(analysis, dict(rename_groups))
    if not _known_loop_effects(
        analysis, frozenset(rename_groups.get(name, name) for name in register_names)
    ):
        return []
    lane = analysis.target.id  # type: ignore[union-attr]
    writes = _binding_write_roots(analysis)
    current: dict[str, _Definition] = {}
    selected: dict[int, _Definition] = {}
    loads: list[_Definition] = []

    def pull(definition: _Definition) -> None:
        if definition.index in selected:
            return
        for dependency in definition.dependencies.values():
            pull(dependency)
        selected[definition.index] = definition

    for index, stmt in enumerate(analysis.body):
        target = _candidate_target(stmt)
        if target is None or target == lane:
            break
        assert isinstance(stmt, ast.Assign)
        reads = ReadWrites.from_ast(stmt.value).reads
        if any(
            name in writes and name != lane and name not in current for name in reads
        ):
            break
        definition = _Definition(
            index,
            stmt,
            {name: current[name] for name in reads if name in current},
            bool(_memory_load_calls(stmt.value)),
        )
        current[target] = definition
        if definition.is_load:
            pull(definition)
            loads.append(definition)
    if not loads or not _independent(analysis, loads, disjoint, strides, extent):
        return []
    fresh = {index: new_var("_lane_unroll_prefix") for index in sorted(selected)}
    result: list[tuple[ast.Assign, ast.Assign, bool]] = []
    for index, definition in sorted(selected.items()):
        original = loop.body[index]
        assert isinstance(original, ast.Assign)
        # Clone original RHS: the disposable alias-proof AST may have selected
        # an address branch. Its proof simplification must never be emitted.
        clone = statement_from_string(ast.unparse(original))
        assert isinstance(clone, ast.Assign)
        mapping = {
            node.id: fresh[
                definition.dependencies[rename_groups.get(node.id, node.id)].index
            ]
            for node in ast.walk(clone.value)
            if isinstance(node, ast.Name)
            and rename_groups.get(node.id, node.id) in definition.dependencies
        }
        ast_rename(clone.value, mapping)
        clone.targets[0] = ast.Name(id=fresh[index], ctx=ast.Store())
        result.append((original, clone, definition.is_load))
    return result
