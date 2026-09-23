"""Preserve logical coordinates when a whole region owns a flat physical CTA."""

from __future__ import annotations

import ast
from typing import cast

from .proven_loop_bounds import _call_path


def _coordinate_kind(node: ast.AST) -> str | None:
    path = _call_path(node)
    if path is not None and path in {
        ("cute", "arch", "thread_idx"),
        ("cute", "arch", "block_dim"),
    }:
        return path[-1]
    return None


class _Project(ast.NodeTransformer):
    def __init__(self, logical_dims: tuple[int, int, int]) -> None:
        self.logical_dims = logical_dims

    def coordinate(self, kind: str, axis: int) -> ast.expr:
        size = self.logical_dims[axis]
        if kind == "block_dim":
            return ast.parse(f"cutlass.Int32({size})", mode="eval").body
        if size == 1:
            return ast.parse("cutlass.Int32(0)", mode="eval").body
        stride = 1
        for previous in self.logical_dims[:axis]:
            stride *= previous
        expression = "cute.arch.thread_idx()[0]"
        if stride != 1:
            expression = f"({expression} // {stride})"
        if any(dim != 1 for dim in self.logical_dims[axis + 1 :]):
            expression = f"({expression} % {size})"
        return ast.parse(expression, mode="eval").body

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if (
            isinstance(node.value, ast.Call)
            and (kind := _coordinate_kind(node.value.func)) is not None
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
            and 0 <= node.slice.value < 3
        ):
            return ast.copy_location(self.coordinate(kind, node.slice.value), node)
        return cast("ast.expr", self.generic_visit(node))

    def visit_Call(self, node: ast.Call) -> ast.expr:
        kind = _coordinate_kind(node.func)
        if kind is not None:
            return ast.copy_location(
                ast.Tuple(
                    elts=[self.coordinate(kind, axis) for axis in range(3)],
                    ctx=ast.Load(),
                ),
                node,
            )
        return cast("ast.expr", self.generic_visit(node))


def flatten_thread_coordinates(
    body: list[ast.stmt], logical_dims: tuple[int, int, int]
) -> bool:
    """Project an owned body to (product(logical_dims), 1, 1) in place.

    CUDA's physical linear thread index and warp/lane membership are unchanged.
    All explicit coordinate and block-dimension reads retain their old values.
    Function aliases and noncanonical calls decline before any AST mutation.
    The caller must also use the new dimensions in its launcher and analyses.
    """
    module = ast.Module(body=body, type_ignores=[])
    parents = {
        id(child): parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(module):
        if _coordinate_kind(node) is None:
            continue
        parent = parents[id(node)]
        if (
            not isinstance(parent, ast.Call)
            or parent.func is not node
            or parent.args
            or parent.keywords
        ):
            return False
    _Project(logical_dims).visit(module)
    return True


def update_launch_block(
    host_statements: list[ast.AST],
    kernel_name: str,
    block_dims: tuple[int, int, int],
) -> int:
    """Refresh only marked calls after a late device lowering fixes its CTA."""
    changed = 0
    for statement in host_statements:
        for node in ast.walk(statement):
            if vars(node).get("_is_kernel_call") is not True:
                continue
            assert isinstance(node, (ast.Expr, ast.Assign))
            call = node.value
            assert isinstance(call, ast.Call)
            assert isinstance(call.func, ast.Name) and call.func.id == "_launcher"
            target = call.args[0]
            assert isinstance(target, ast.Name)
            if target.id != kernel_name:
                continue
            keyword = next(value for value in call.keywords if value.arg == "block")
            keyword.value = ast.copy_location(
                ast.Tuple(
                    elts=[ast.Constant(value) for value in block_dims], ctx=ast.Load()
                ),
                keyword.value,
            )
            changed += 1
    return changed
