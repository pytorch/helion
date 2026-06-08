"""Shared AST helpers for the CuTe source-to-source rewrite passes.

These small utilities are used by several of the post-codegen AST passes
(``hoist_loop_invariant_recip``, ``hoist_warp_reduce``,
``merge_sibling_v_loops``, ``pipeline_inner_loads``).  They previously
lived as byte-identical copies in each pass module.
"""

from __future__ import annotations

import ast


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
