"""Shared AST helpers for the CuTe source-to-source rewrite passes.

These small utilities are used by several of the post-codegen AST passes
(``hoist_loop_invariant_recip``, ``hoist_warp_reduce``,
``merge_sibling_v_loops``, ``pipeline_inner_loads``).  They previously
lived as byte-identical copies in each pass module.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection


class _NameRefCollector(ast.NodeVisitor):
    """Collect all ``ast.Name`` ids that appear as Load contexts in ``node``."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _names_read(node: ast.AST) -> set[str]:
    collector = _NameRefCollector()
    collector.visit(node)
    return collector.names


def _assignment_lhs_name(stmt: ast.stmt) -> str | None:
    """If ``stmt`` is ``LHS = RHS`` with LHS a single ``ast.Name``, return LHS.id."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _bound_names(tree: ast.AST) -> set[str]:
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".", 1)[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
    return names


def _fresh_prefix(
    hint: str, occupied: Collection[str], fresh_name: Callable[[str], str]
) -> str:
    """Reserve a template prefix whose derived identifiers cannot capture names.

    ``fresh_name`` reserves a new identifier on each call. Templates derive
    locals by appending underscores and suffixes, so reserving only the prefix
    does not protect an existing argument or local such as ``prefix_value``.
    The caller supplies all original AST names and external boundaries, and
    registers the emitted identifiers before later passes allocate more names.
    """
    while True:
        prefix = fresh_name(hint)
        if not any(
            name == prefix or name.startswith(f"{prefix}_") for name in occupied
        ):
            return prefix
