"""A source-visible positive-zero operation after the final BF16 V rounding."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING

from .cute_paged_postscale import postscale_attention_fixture
from .cute_source import transform_kernel

if TYPE_CHECKING:
    import torch

    import helion


POSITIVE_ZERO_ATTENTION_CASES = [
    ("reduced", "attention", 1, 4, -14, False),
    ("per_key", "attention", 1, 1, 0, True),
    ("per_key", "attention_scoped", 2, 4, -22, False),
    ("mapped", "attention_paired", 1, 5, -22, True),
]


class _PositiveZeroValue(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "value"
        ):
            # p=0 also covers the admitted decode +0 form without a postscale.
            if (
                isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Mult)
                and isinstance(node.value.right, ast.Constant)
                and node.value.right.value == 1.0
            ):
                node.value = node.value.left
            node.value = ast.BinOp(node.value, ast.Add(), ast.Constant(0.0))
            self.changed += 1
        return node


@functools.cache
def positive_zero_attention_fixture(
    query_len: int = 5,
    post_scale: int = -22,
    device: torch.device | str = "cpu",
    *,
    form: str = "mapped",
    reference_scores: bool = True,
) -> tuple[helion.Kernel, tuple[object, ...]]:
    kernel, args = postscale_attention_fixture(
        form,
        query_len,
        True,
        device,
        key=-6,
        value=post_scale,
        reference_scores=reference_scores,
        base2_scores=reference_scores,
        reverse_pages=reference_scores,
    )
    transform = _PositiveZeroValue()
    result = transform_kernel(
        kernel,
        transform,
        name="positive_zero_attention",
        filename=f"<helion_positive_zero_attention_{form}_{reference_scores}_{query_len}_{post_scale}>",
    )
    assert transform.changed == 1
    return result, args
