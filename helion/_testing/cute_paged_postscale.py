"""Source-selected post-BF16 scaling of the shared quantized attention fixtures."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING

from .cute_paged_reciprocal import reciprocal_attention_fixture
from .cute_source import transform_kernel

if TYPE_CHECKING:
    import torch

    from helion.runtime.kernel import Kernel


class _PostScale(ast.NodeTransformer):
    def __init__(self, key: int, value: int) -> None:
        self.exponents = {"key": key, "value": value}
        self.changed = dict.fromkeys(("key", "value", "scores", "output"), 0)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name = node.targets[0].id
        if name in self.exponents:
            node.value = ast.BinOp(
                node.value, ast.Mult(), ast.Constant(2.0 ** self.exponents[name])
            )
            self.changed[name] += 1
        # Match the initial score expression, not later mask/reference assignments.
        if name == "scores" and isinstance(node.value, ast.BinOp):
            if (
                isinstance(node.value.left, ast.Call)
                and isinstance(node.value.left.func, ast.Attribute)
                and node.value.left.func.attr == "dot"
            ):
                node.value.right = ast.BinOp(
                    node.value.right,
                    ast.Mult(),
                    ast.Constant(2.0 ** -self.exponents["key"]),
                )
                self.changed["scores"] += 1
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Div)
            and isinstance(node.value.left, ast.Constant)
            and node.value.left.value == 1.0
        ):
            node.value = ast.BinOp(
                node.value, ast.Mult(), ast.Constant(2.0 ** -self.exponents["value"])
            )
            self.changed["output"] += 1
        return node


@functools.cache
def _postscale_kernel(kernel: Kernel, key: int, value: int) -> Kernel:
    transform = _PostScale(key, value)
    result = transform_kernel(
        kernel,
        transform,
        name=kernel.fn.__name__ + "_postscale",
        filename=f"<helion_paged_postscale_{kernel.fn.__name__}_{key}_{value}>",
    )
    assert all(count == 1 for count in transform.changed.values())
    return result


def postscale_attention_fixture(
    form: str,
    query_len: int,
    return_lse: bool,
    device: torch.device | str = "cpu",
    *,
    key: int = -6,
    value: int = -6,
    reference_scores: bool = False,
    base2_scores: bool = False,
    reverse_pages: bool = False,
) -> tuple[Kernel, tuple[object, ...]]:
    kernel, args = reciprocal_attention_fixture(
        form,
        query_len,
        return_lse,
        device,
        reference_scores=reference_scores,
        base2_scores=base2_scores,
        reverse_pages=reverse_pages,
    )
    return _postscale_kernel(kernel, key, value), args


POSTSCALE_ATTENTION_CASES = [
    ("reduced", 8, -6, -6, False, False, False, "attention", 1, False, True),
    ("per_key", 1, -22, 0, False, False, False, "attention", 1, True, False),
    ("mapped", 4, 0, -18, False, False, True, "attention_scoped", 2, False, True),
    ("per_key", 5, -6, -3, True, False, False, "attention_scoped", 2, True, False),
    ("mapped", 5, -6, -22, True, True, True, "attention_paired", 1, True, True),
    ("per_key", 4, -14, -7, True, True, False, "attention_paired", 1, False, False),
]
