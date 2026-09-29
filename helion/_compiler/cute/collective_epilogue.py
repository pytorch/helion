"""Scalar replay and thread bounds shared by cooperative epilogues."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ..ast_read_writes import ReadWrites
from .contiguous_copy import _path
from .scalar_recipe import _GLOBALS
from .scalar_recipe import ScalarRecipe
from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence


class _RemoveFullThreadBounds(ast.NodeTransformer):
    """Remove only tautological comparisons proved by the original CTA shape."""

    def __init__(self, dimensions: tuple[int, int, int]) -> None:
        self.dimensions = dimensions

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            return node
        left, right = node.left, node.comparators[0]
        if (
            isinstance(left, ast.Call)
            and ast.unparse(left.func) in {"cutlass.Int32", "cutlass.Int64"}
            and len(left.args) == 1
            and not left.keywords
        ):
            left = left.args[0]
        if (
            isinstance(left, ast.Subscript)
            and isinstance(left.value, ast.Call)
            and ast.unparse(left.value.func) == "cute.arch.thread_idx"
            and not left.value.args
            and not left.value.keywords
            and isinstance(left.slice, ast.Constant)
            and type(left.slice.value) is int
            and 0 <= left.slice.value < 3
            and isinstance(right, ast.Constant)
            and type(right.value) is int
            and self.dimensions[left.slice.value] <= right.value
        ):
            return ast.copy_location(ast.Constant(True), node)
        return node


@dataclasses.dataclass
class _ScalarEpilogue:
    statements: Sequence[ast.stmt]
    recipes: dict[int, ScalarRecipe]
    value: ScalarRecipe
    local_names: set[str]

    @property
    def boundary_names(self) -> frozenset[str]:
        names = self.value.boundary_names.union(
            *(recipe.boundary_names for recipe in self.recipes.values())
        )
        return names.difference(self.local_names)

    def emit(
        self,
        replacements: Mapping[str, ast.expr],
        fresh_name: Callable[[str], str],
    ) -> tuple[list[ast.stmt], ast.expr]:
        substitutions = dict(replacements)
        substitutions.update(
            (name, ast.Name(fresh_name(f"{name}_epilogue"), ast.Load()))
            for name in sorted(self.local_names)
        )

        def emit_statements(statements: Sequence[ast.stmt]) -> list[ast.stmt]:
            result: list[ast.stmt] = []
            for statement in statements:
                if isinstance(statement, ast.Pass):
                    result.append(_clone(statement))
                    continue
                if isinstance(statement, ast.Assign):
                    setup, value = self.recipes[id(statement.value)].emit(
                        substitutions, fresh_name
                    )
                    target = statement.targets[0]
                    assert isinstance(target, ast.Name)
                    renamed = _clone(substitutions[target.id])
                    assert isinstance(renamed, ast.Name)
                    renamed.ctx = ast.Store()
                    result.extend([*setup, ast.Assign(targets=[renamed], value=value)])
                    continue
                assert isinstance(statement, ast.If)
                setup, test = self.recipes[id(statement.test)].emit(
                    substitutions, fresh_name
                )
                result.extend(
                    [
                        *setup,
                        ast.If(
                            test=test,
                            body=emit_statements(statement.body),
                            orelse=emit_statements(statement.orelse),
                        ),
                    ]
                )
            return result

        statements = emit_statements(self.statements)
        setup, value = self.value.emit(substitutions, fresh_name)
        return [*statements, *setup], value


def _scalar_epilogue(
    statements: Sequence[ast.stmt],
    value: ast.expr,
    *,
    m_index: str,
    n_index: str,
    dominating: Sequence[ast.stmt],
    boundaries: set[str],
    fresh_name: Callable[[str], str],
) -> _ScalarEpilogue | None:
    from .collective_matmul import _uses_thread_coordinates

    # The collective's ownership proof supplies logical M/N coordinates. Only
    # the original leading row-coordinate definition may be discarded.
    first = statements[0] if statements else None
    if (
        not isinstance(first, ast.Assign)
        or len(first.targets) != 1
        or not isinstance(first.targets[0], ast.Name)
        or first.targets[0].id != m_index
        or build_recipe(first.value, [], _read_names(first.value)) is None
    ):
        return None
    statements = statements[1:]
    local_names = set(ReadWrites.from_list(list(statements)).writes)
    if local_names & (_GLOBALS | boundaries | {m_index, n_index}):
        return None
    recipes: dict[int, ScalarRecipe] = {}

    def expression(expr: ast.expr, available: set[str]) -> ScalarRecipe | None:
        if (_read_names(expr) & local_names) - available:
            return None
        recipe = build_recipe(expr, dominating, boundaries | local_names)
        if recipe is None:
            return None
        # A dominating alias may have captured an older definition of a name
        # that this epilogue overwrites. Its capture cannot use the replayed
        # local binding, even though direct local reads should use that value.
        # Leave such epilogues on the scalar path until their full definition
        # versions can be carried across the replay boundary.
        if any(
            _read_names(dominating[index]) & (local_names | {m_index})
            for index in recipe.source_statement_indices
        ):
            return None
        setup, result = recipe.emit({}, fresh_name)
        if _uses_thread_coordinates([*setup, result]):
            return None
        recipes[id(expr)] = recipe
        return recipe

    def validate(body: Sequence[ast.stmt], available: set[str]) -> set[str] | None:
        available = set(available)
        for statement in body:
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Assign):
                if (
                    len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                    or expression(statement.value, available) is None
                ):
                    return None
                available.add(statement.targets[0].id)
                continue
            if not isinstance(statement, ast.If):
                return None
            if expression(statement.test, available) is None:
                return None
            left = validate(statement.body, available)
            right = validate(statement.orelse, available)
            if left is None or right is None:
                return None
            available = left & right
        return available

    available = validate(statements, set())
    if available is None or (result := expression(value, available)) is None:
        return None
    return _ScalarEpilogue(statements, recipes, result, local_names)


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


def _load_pointer_roots(
    statements: Sequence[ast.AST],
    *,
    tensor_names: frozenset[str],
    integer_names: set[str],
) -> set[str] | None:
    """Require an explicit tensor base and integral offsets for every load.

    Counting iterator syntax can miss another base behind a pointer alias or
    conditional. Decline those bases rather than infer their identity. Track
    only integer type facts in statement order; aliases never expand into an
    expression tree, and a rebind cannot retain an earlier integer fact.
    """
    roots: set[str] = set()

    def integer_type(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "cutlass"
            and value.attr
            in {
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Uint8",
                "Uint16",
                "Uint32",
                "Uint64",
            }
        )

    def integer(value: ast.expr, known: set[str]) -> bool:
        pending: list[ast.AST] = [value]
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Constant) and type(node.value) is int:
                continue
            if isinstance(node, ast.Name) and node.id in known:
                continue
            if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
                if integer_type(node.func) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"to", "bitcast"}
                    and integer_type(node.args[0])
                ):
                    continue
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub, ast.Invert)
            ):
                pending.append(node.operand)
                continue
            if isinstance(node, ast.BinOp) and isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ):
                pending.extend((node.left, node.right))
                continue
            if isinstance(node, ast.IfExp):
                pending.extend((node.body, node.orelse))
                continue
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value >= 0
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in {"shape", "stride"}
            ):
                owner = node.value.value
                if isinstance(owner, ast.Attribute) and owner.attr == "layout":
                    owner = owner.value
                if isinstance(owner, ast.Name) and owner.id in tensor_names:
                    continue
            if (
                isinstance(node, ast.Call)
                and _path(node.func) == "cute.crd2idx"
                and len(node.args) == 2
                and not node.keywords
                and isinstance(node.args[0], ast.Tuple)
                and isinstance(node.args[1], ast.Attribute)
                and node.args[1].attr == "layout"
                and isinstance(node.args[1].value, ast.Name)
                and node.args[1].value.id in tensor_names
            ):
                pending.extend(node.args[0].elts)
                continue
            return False
        return True

    def expression(value: ast.AST, known: set[str]) -> bool:
        for node in ast.walk(value):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in tensor_names
            ):
                roots.add(node.value.id)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
            ):
                continue
            pointer = node.func.value
            if _path(pointer) == "cute.arch":
                if len(node.args) != 2 or node.keywords:
                    return False
                pointer = node.args[0]
            elif node.args or node.keywords:
                return False
            while isinstance(pointer, ast.BinOp) and isinstance(
                pointer.op, (ast.Add, ast.Sub)
            ):
                if not integer(pointer.right, known):
                    return False
                pointer = pointer.left
            if not (
                isinstance(pointer, ast.Attribute)
                and pointer.attr == "iterator"
                and isinstance(pointer.value, ast.Name)
                and pointer.value.id in tensor_names
            ):
                return False
            roots.add(pointer.value.id)
        return True

    def visit(body: Sequence[ast.AST], known: set[str]) -> bool:
        for statement in body:
            if isinstance(statement, ast.Assign):
                if (
                    len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)
                    or not expression(statement.value, known)
                ):
                    return False
                is_integer = integer(statement.value, known)
                known.discard(statement.targets[0].id)
                if is_integer:
                    known.add(statement.targets[0].id)
            elif isinstance(statement, ast.If):
                if not expression(statement.test, known):
                    return False
                left, right = set(known), set(known)
                if not visit(statement.body, left) or not visit(
                    statement.orelse, right
                ):
                    return False
                known.clear()
                known.update(left & right)
            elif isinstance(statement, ast.Pass):
                continue
            elif not isinstance(statement, (ast.expr, ast.Expr)) or not expression(
                statement, known
            ):
                return False
        return True

    return roots if visit(statements, set(integer_names)) else None
