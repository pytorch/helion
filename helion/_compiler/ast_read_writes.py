from __future__ import annotations

import ast
import collections
import typing
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    _A = TypeVar("_A", bound=ast.AST)


# TODO(oulgen): This visitor is extremely primitive, does not consider alpha renaming or scopes
class _ReadWriteVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.rw = ReadWrites(collections.Counter(), collections.Counter())

    def _update(self, name: str, ctx: ast.expr_context) -> None:
        if isinstance(ctx, ast.Load):
            self.rw.reads[name] += 1
        elif isinstance(ctx, ast.Store):
            self.rw.writes[name] += 1

    def visit_Name(self, node: ast.Name) -> None:
        self._update(node.id, node.ctx)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            self._update(node.value.id, node.ctx)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # Skip target
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)


class ReadWrites(typing.NamedTuple):
    reads: dict[str, int]
    writes: dict[str, int]

    def __iter__(self) -> typing.Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return iter({**self.reads, **self.writes})

    @staticmethod
    def from_list(body: list[ast.AST] | list[ast.stmt]) -> ReadWrites:
        visitor = _ReadWriteVisitor()
        for node in body:
            visitor.visit(node)
        return visitor.rw

    @staticmethod
    def from_ast(node: ast.AST) -> ReadWrites:
        """
        Analyze an Abstract Syntax Tree (AST) node to determine the variables
        that are read and written within it.

        This function traverses the given AST node and collects information
        about variable reads and writes using the `_ReadWriteVisitor` class.

        Args:
            node: The root AST node to analyze.

        Returns:
            A `ReadWrites` object containing dictionaries of read and
            written variable names.
        """
        visitor = _ReadWriteVisitor()
        visitor.visit(node)
        return visitor.rw


class _RenameVisitor(ast.NodeVisitor):
    def __init__(self, renames: dict[str, str]) -> None:
        super().__init__()
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> None:
        node.id = self.renames.get(node.id, node.id)


def ast_rename(node: _A, renames: dict[str, str]) -> _A:
    """
    Rename variables in an Abstract Syntax Tree (AST) node, in-place.

    This function traverses the given AST node and renames variables
    based on the provided mapping of old names to new names.

    Args:
        node: The root AST node to rename variables in.
        renames: A dictionary mapping old variable names to new variable names.

    Returns:
        The modified AST node with variables renamed.
    """
    visitor = _RenameVisitor(renames)
    visitor.visit(node)
    return node


class _DeleteAssignments(ast.NodeTransformer):
    def __init__(self, to_remove: set[str]) -> None:
        super().__init__()
        self.to_remove = to_remove

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        """
        Visit an assignment node and remove it if the target variable is in the to_remove set.

        Args:
            node: The assignment node to visit.

        Returns:
            The modified assignment node, or None if it should be removed.
        """
        if len(node.targets) == 1:
            (target,) = node.targets
            if isinstance(target, ast.Name) and target.id in self.to_remove:
                return None
        return node


def ast_delete_assignments(body: list[ast.AST], to_remove: set[str]) -> list[ast.AST]:
    new_body = []
    transformer = _DeleteAssignments(to_remove)
    for node in body:
        new_node = transformer.visit(node)
        if new_node is not None:
            new_body.append(new_node)
    return new_body


class _NotPureException(Exception):
    pass


class _PureExpressionVisitor(ast.NodeVisitor):
    """
    AST visitor that determines if an expression is guaranteed to be pure.
    """

    def generic_visit(self, node: ast.AST) -> None:
        # Anything without a specific visitor is not pure
        raise _NotPureException

    def visit_Constant(self, node: ast.Constant) -> None:
        pass

    def visit_Num(self, node: ast.Num) -> None:
        pass

    def visit_Str(self, node: ast.Str) -> None:
        pass

    def visit_Bytes(self, node: ast.Bytes) -> None:
        pass

    def visit_NameConstant(self, node: ast.NameConstant) -> None:
        pass

    def visit_Ellipsis(self, node: ast.Ellipsis) -> None:
        pass

    def visit_Name(self, node: ast.Name) -> None:
        pass

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Set(self, node: ast.Set) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if key is not None:  # Handle dict unpacking
                self.visit(key)
        for value in node.values:
            self.visit(value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.visit(node.operand)

    def visit_Starred(self, node: ast.Starred) -> None:
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        # Math methods are all pure, so allow them
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            raise _NotPureException

        # Recurse into children except for func
        for arg in node.args:
            self.visit(arg)

        for keyword in node.keywords:
            self.visit(keyword.value)


def definitely_does_not_have_side_effects(expr: ast.expr) -> bool:
    try:
        _PureExpressionVisitor().visit(expr)
        return True
    except _NotPureException:
        return False


class _DeletePureExpressions(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if definitely_does_not_have_side_effects(node.value):
            return None
        return node


def dead_assignment_elimination(
    body: list[ast.AST],
    dce_vars: list[str],
    num_iterations: int = 8,
    input_rw: ReadWrites | None = None,
) -> None:
    """
    Eliminates dead assignments from body
    """

    # num_iterations and input_rw are not compatible with each other
    assert num_iterations == 1 or input_rw is None
    for _ in range(num_iterations):
        rw = input_rw if input_rw is not None else ReadWrites.from_list(body)
        to_remove = set()
        for name in dce_vars:
            if name in rw.writes and name not in rw.reads:
                to_remove.add(name)
        if not to_remove:
            break
        body[:] = ast_delete_assignments(body, to_remove)


def is_string_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def dead_expression_elimination(body: list[ast.AST]) -> None:
    """
    Eliminates dead expressions from body
    """
    new_body = []
    for node in body:
        if is_string_expr(node):
            # triple quoted comments and strings are indistinguishable
            # do not eliminate them
            new_body.append(node)
            continue
        new_node = _DeletePureExpressions().visit(node)
        if new_node is not None:
            new_body.append(new_node)
    body[:] = new_body
