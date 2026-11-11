"""Explicit base-two source variants of the shared reference-score fixture."""

from __future__ import annotations

import ast
import functools
import math
from typing import TYPE_CHECKING

from .cute_folded_attention import folded_paged_attention
from .cute_paged_attention import paged_online_attention
from .cute_paged_reference import reference_attention_fixture
from .cute_source import transform_kernel

if TYPE_CHECKING:
    import torch

    from helion.runtime.kernel import Kernel


class _Base2Source(ast.NodeTransformer):
    def __init__(self) -> None:
        self.exponentials = 0
        self.scales = 0
        self.lse = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "exp"
        ):
            node.func.attr = "exp2"
            self.exponentials += 1
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.left, ast.Name)
            and node.left.id == "dim"
            and isinstance(node.right, ast.UnaryOp)
            and isinstance(node.right.op, ast.USub)
            and isinstance(node.right.operand, ast.Constant)
            and node.right.operand.value == 0.5
        ):
            self.scales += 1
            return ast.BinOp(node, ast.Mult(), ast.Constant(math.log2(math.e)))
        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Name)
            and node.left.id == "maximum"
            and isinstance(node.right, ast.Call)
            and isinstance(node.right.func, ast.Attribute)
            and node.right.func.attr == "log"
        ):
            self.lse += 1
            node.left = ast.BinOp(node.left, ast.Mult(), ast.Constant(math.log(2)))
        return node


@functools.cache
def _base2_kernel(form: str) -> Kernel[tuple[torch.Tensor, torch.Tensor | None]]:
    assert form in ("mapped", "per_key", "reduced")
    kernel = folded_paged_attention if form == "mapped" else paged_online_attention
    transform = _Base2Source()
    result = transform_kernel(
        kernel,
        transform,
        name=kernel.fn.__name__ + "_base2",
        filename=f"<helion_paged_base2_{form}>",
    )
    assert (transform.exponentials, transform.scales, transform.lse) == (3, 1, 1)
    return result


def base2_attention_fixture(
    form: str, query_len: int, return_lse: bool, device: torch.device | str = "cpu"
) -> tuple[Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]]:
    _kernel, args = reference_attention_fixture(form, query_len, return_lse, device)
    return _base2_kernel(form), args
