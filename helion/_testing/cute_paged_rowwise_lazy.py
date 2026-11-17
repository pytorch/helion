"""Rowwise lazy recurrence derived from the shared folded attention source."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import helion
from helion._testing.cute_paged_base2 import _base2_kernel
from helion._testing.cute_paged_postscale import _postscale_kernel
from helion._testing.cute_paged_reciprocal import _reciprocal_kernel
from helion._testing.cute_source import transform_kernel

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


class _LazyRows(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = dict.fromkeys(("specialize", "maximum", "value", "reverse"), 0)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        assert [arg.arg for arg in node.args.args[-2:]] == [
            "reference_scores",
            "reverse_pages",
        ]
        assert len(node.args.defaults) == 2
        node.args.args = node.args.args[:-2]
        node.args.defaults = []
        window = next(arg for arg in node.args.args if arg.arg == "window_left")
        window.annotation = ast.parse("hl.constexpr", mode="eval").body
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        if isinstance(node.test, ast.Name) and node.test.id == "reference_scores":
            # Specialize the shared source's existing reference-score branch.
            result = []
            for statement in node.body:
                visited = self.visit(statement)
                if visited is not None:
                    result.extend(visited if isinstance(visited, list) else [visited])
            return result
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        if isinstance(node.test, ast.Name) and node.test.id == "reverse_pages":
            self.changed["reverse"] += 1
            return self.visit(node.body)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt] | None:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name = node.targets[0].id
        if name == "window_left":
            assert isinstance(node.value, ast.Call)
            assert ast.unparse(node.value.func) == "hl.specialize"
            self.changed["specialize"] += 1
            return None
        if name == "value":
            self.changed["value"] += 1
            node.value = ast.BinOp(node.value, ast.Add(), ast.Constant(0.0))
        if name == "next_maximum":
            self.changed["maximum"] += 1
            assert ast.unparse(node.value) == (
                "torch.maximum(maximum, torch.amax(scores, dim=1) + reference)"
            )
            return ast.parse(
                'invalid_maximum = ~((maximum >= 0.0) & (maximum < float("inf")))\n'
                "tile_maximum = torch.amax(scores, dim=1)\n"
                "needs_rescale = (tile_maximum > 2.0) | invalid_maximum | torch.isnan(tile_maximum)\n"
                "proposed_maximum = torch.maximum(maximum, tile_maximum + reference)\n"
                "next_maximum = torch.where(needs_rescale, proposed_maximum, maximum)\n"
            ).body
        return node


def _rowwise_kernel() -> Kernel:
    kernel = _postscale_kernel(
        _reciprocal_kernel(_base2_kernel("mapped"), False), -6, -22
    )
    transform = _LazyRows()
    result = transform_kernel(
        kernel,
        transform,
        name="paged_query_first_direct_attention",
        filename="<helion_rowwise_lazy_attention>",
    )
    assert all(count == 1 for count in transform.changed.values())
    return helion.kernel(result.fn, backend="cute", static_shapes=True, fast_math=True)


paged_query_first_direct_attention = _rowwise_kernel()
