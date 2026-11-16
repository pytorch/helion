"""Literal half arithmetic must round before a widening consumer."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", autotune_effort="none")
def _product_then_widen(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    for tile in hl.tile(x.shape):
        out[tile] = (x[tile] * 3).to(torch.float32)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_literal_product_keeps_its_result_dtype(dtype: torch.dtype) -> None:
    with cpu_render_target():
        bound = cpu_bind(_product_then_widen, (torch.empty(128, dtype=dtype),))
        source = bound.to_code(helion.Config(block_sizes=[128]))
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    products = [
        name
        for name, value in assignments.items()
        if isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Mult)
        and isinstance(value.right, ast.Name)
        and isinstance(constant := assignments.get(value.right.id), ast.Constant)
        and constant.value == 3
    ]
    assert len(products) == 1
    product = products[0]
    casts = {
        ast.unparse(value.func): name
        for name, value in assignments.items()
        if isinstance(value, ast.Call)
        and value.args
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == product
    }
    if dtype != torch.float32:
        cast = "cutlass.Float16" if dtype == torch.float16 else "cutlass.BFloat16"
        assert cast in casts, source
        rounded = casts[cast]
        assert any(
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "cutlass.Float32"
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == rounded
            for value in assignments.values()
        )
    else:
        assert not {"cutlass.Float16", "cutlass.BFloat16"}.intersection(casts)
