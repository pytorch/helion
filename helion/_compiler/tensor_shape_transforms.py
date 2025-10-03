from __future__ import annotations

import ast
import typing
from typing import TYPE_CHECKING

from .ast_extension import expr_from_string

if TYPE_CHECKING:
    from .host_function import HostFunction


class PadSpecializedDeviceTensorShapesToPowerOfTwo(ast.NodeTransformer):
    """Pad any hl.specialize'd variable in device tensor shape to next power of 2."""

    def __init__(self) -> None:
        self.in_device_context = False
        self.specialized_vars: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        # Track variables assigned from hl.specialize() calls
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "hl"
            and node.value.func.attr == "specialize"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.specialized_vars.add(target.id)
        return typing.cast("ast.Assign", self.generic_visit(node))

    def visit_For(self, node: ast.For) -> ast.For:
        # Track when we enter hl.tile or hl.grid loops (device context)
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "hl"
            and node.iter.func.attr in ("tile", "grid")
        ):
            old_context = self.in_device_context
            self.in_device_context = True

            try:
                return typing.cast("ast.For", self.generic_visit(node))
            finally:
                self.in_device_context = old_context
        return typing.cast("ast.For", self.generic_visit(node))

    def visit_Call(self, node: ast.Call) -> ast.Call:
        # Only transform tensor creation calls in device context
        if not (
            self.in_device_context
            and isinstance(node.func, ast.Attribute)
            and node.args
        ):
            return typing.cast("ast.Call", self.generic_visit(node))

        # Check for torch.zeros/empty/ones/full or tensor.new_zeros/new_empty/new_ones/new_full
        if isinstance(node.args[0], ast.Name):
            # Only wrap specialized variables
            if node.args[0].id in self.specialized_vars:
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr in ("zeros", "empty", "ones", "full")
                ) or node.func.attr in (
                    "new_zeros",
                    "new_empty",
                    "new_ones",
                    "new_full",
                ):
                    node.args[0] = typing.cast(
                        "ast.expr",
                        expr_from_string(
                            "helion.next_power_of_2({arg})", arg=node.args[0]
                        ),
                    )

        # Check for hl.zeros/empty/ones/full where first arg is a list
        elif isinstance(node.args[0], ast.List):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hl"
                and node.func.attr in ("zeros", "empty", "ones", "full")
            ):
                node.args[0].elts = typing.cast(
                    "list[ast.expr]",
                    [
                        expr_from_string("helion.next_power_of_2({arg})", arg=elt)
                        if isinstance(elt, ast.Name) and elt.id in self.specialized_vars
                        else elt
                        for elt in node.args[0].elts
                    ],
                )

        return typing.cast("ast.Call", self.generic_visit(node))


def pad_specialized_device_tensor_shapes_to_power_of_2(
    host_function: HostFunction,
) -> None:
    """Pad any hl.specialize'd variable in device tensor shape to next power of 2."""
    transformer = PadSpecializedDeviceTensorShapesToPowerOfTwo()
    host_function.body = [transformer.visit(stmt) for stmt in host_function.body]
