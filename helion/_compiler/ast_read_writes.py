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

    def __iter__(self) -> typing.Iterator[str]:
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

        :param node: The root AST node to analyze.
        :return: A `ReadWrites` object containing dictionaries of read and
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

    :param node: The root AST node to rename variables in.
    :param renames: A dictionary mapping old variable names to new variable names.
    :return: The modified AST node with variables renamed.
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

        :param node: The assignment node to visit.
        :return: The modified assignment node, or None if it should be removed.
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
