"""An authored native integer decoder in the shared paged-attention source."""

from __future__ import annotations

import ast

import torch

from .cute_mma_dag_grouped import grouped_matmul
from .cute_native_s2f6 import integer_s2f6_decode
from .cute_optional_lse import optional_lse_fixture
from .cute_source import transform_kernel
import helion
import helion.language as hl


class _NativeDecode(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("key", "value")
        ):
            return self.generic_visit(node)
        rounded = node.value
        assert isinstance(rounded, ast.Call)
        assert isinstance(rounded.func, ast.Attribute) and rounded.func.attr == "to"
        product = rounded.func.value
        assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
        cast, exp = product.left, product.right
        assert isinstance(cast, ast.Call) and isinstance(cast.func, ast.Attribute)
        assert isinstance(exp, ast.Call) and ast.unparse(exp.func) == "torch.exp2"
        bias = exp.args[0]
        assert isinstance(bias, ast.BinOp) and isinstance(bias.op, ast.Sub)
        scale = bias.left
        assert isinstance(scale, ast.Call) and isinstance(scale.func, ast.Attribute)
        node.value = ast.Call(
            ast.Name(integer_s2f6_decode.__name__, ast.Load()),
            [cast.func.value, scale.func.value],
            [],
        )
        self.changed += 1
        return node


def native_paged_operand_fixture() -> tuple[
    helion.Kernel[tuple[torch.Tensor, torch.Tensor | None]], tuple[object, ...]
]:
    kernel, args = optional_lse_fixture("per_key", True)
    assert kernel.fn.__globals__["hl"] is hl
    assert kernel.fn.__globals__["torch"] is torch
    transform = _NativeDecode()
    rebuilt = transform_kernel(
        kernel,
        transform,
        name="native_paged_operand",
        filename="<helion_native_paged_operand>",
        namespace=globals(),
    )
    assert transform.changed == 2
    assert rebuilt.fn.__globals__ is globals()
    return rebuilt, args


def native_grouped_operand_fixture() -> tuple[
    helion.Kernel[torch.Tensor], tuple[object, ...]
]:
    """Use the existing grouped-GEMM source to check ordinary default fallback."""
    transform = _NativeDecode()
    kernel = transform_kernel(
        helion.kernel(grouped_matmul, backend="cute", static_shapes=True),
        transform,
        name="native_grouped_operand",
        filename="<helion_native_grouped_operand>",
        namespace=globals(),
    )
    assert transform.changed == 1
    args = (
        torch.empty((128, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((128, 2), dtype=torch.uint8),
        32,
        False,
        False,
    )
    return kernel, args
