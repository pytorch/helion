"""Explicit row-reciprocal output variants of the shared attention sources."""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING

from .cute_optional_lse import optional_lse_fixture
from .cute_paged_base2 import base2_attention_fixture
from .cute_source import transform_kernel

if TYPE_CHECKING:
    import torch

    from helion.runtime.kernel import Kernel


class _ReciprocalOutput(ast.NodeTransformer):
    def __init__(self, intrinsic: bool) -> None:
        self.intrinsic = intrinsic
        self.replacements = 0

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "accumulator"
        ):
            assert isinstance(node.right, ast.Subscript)
            assert isinstance(node.right.value, ast.Name)
            assert node.right.value.id == "safe_denominator"
            denominator = node.right.value
            node.right.value = (
                ast.Call(
                    ast.Attribute(
                        ast.Name("torch", ast.Load()), "reciprocal", ast.Load()
                    ),
                    [denominator],
                    [],
                )
                if self.intrinsic
                else ast.BinOp(ast.Constant(1.0), ast.Div(), denominator)
            )
            node.op = ast.Mult()
            self.replacements += 1
        return node


@functools.cache
def _reciprocal_kernel(
    kernel: Kernel[tuple[torch.Tensor, torch.Tensor | None]], intrinsic: bool
) -> Kernel[tuple[torch.Tensor, torch.Tensor | None]]:
    transform = _ReciprocalOutput(intrinsic)
    result = transform_kernel(
        kernel,
        transform,
        name=kernel.fn.__name__ + "_reciprocal",
        filename=f"<helion_paged_reciprocal_{kernel.fn.__name__}_{intrinsic}>",
    )
    assert transform.replacements == 1
    return result


def reciprocal_attention_fixture(
    form: str,
    query_len: int,
    return_lse: bool,
    device: torch.device | str = "cpu",
    *,
    reference_scores: bool = False,
    base2_scores: bool = False,
    reverse_pages: bool = False,
    intrinsic: bool = False,
) -> tuple[Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]]:
    if base2_scores:
        assert reference_scores
        kernel, args = base2_attention_fixture(form, query_len, return_lse, device)
    else:
        kernel, args = optional_lse_fixture(
            form, return_lse, device, query_len=query_len
        )
        args = (*args, reference_scores)
    return _reciprocal_kernel(kernel, intrinsic), (*args, reverse_pages)


# Eight source/schedule representatives cover each topology, panel, output
# mapping and score form without a Cartesian compile grid.
RECIPROCAL_ATTENTION_CASES = [
    ("reduced", 8, False, False, False, "attention", 1, False, True, False),
    ("per_key", 1, False, False, False, "attention", 1, False, False, False),
    ("mapped", 4, False, False, False, "attention_scoped", 2, True, True, True),
    ("per_key", 5, True, False, False, "attention_scoped", 2, False, False, False),
    ("mapped", 1, True, True, True, "attention", 1, True, True, False),
    ("per_key", 4, True, True, True, "attention_paired", 1, False, False, False),
    ("mapped", 5, True, False, False, "attention_paired", 1, True, True, False),
    ("mapped", 1, False, False, False, "attention_paired", 1, False, True, False),
]
