from __future__ import annotations

import ast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .test_cute_chained_tcgen05_guards import _code
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrow_batch_offset(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, overflow: hl.constexpr
) -> torch.Tensor:
    batch, rows, state = a.shape
    columns = v.shape[-1]
    state = hl.specialize(state)
    out = torch.empty_like(v)
    for block, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        q = hl.arange(128)
        # This narrowing is explicit user semantics, not a pointer-width cast.
        bi = torch.scalar_tensor(block.begin, dtype=torch.int32)
        if overflow:
            bi = ((bi * 1073741824) // 1073741824) % 3
        first = hl.dot(a[bi, row, :], b[bi, q, :].T)
        result = hl.dot(first.to(a.dtype), v[bi, q, col])
        out[block.begin, row, col] = result.to(v.dtype)
    return out


def test_explicit_narrow_index_promotes_before_wide_stride_product() -> None:
    # No physical allocation: each index fits Int32, but index * stride does
    # not. At bi=131072, the A/B offset is 2**31 elements, not -2**31.
    with FakeTensorMode():
        values = tuple(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in (
                (1048577, 128, 128),
                (1048577, 128, 128),
                (1048577, 128, 64),
            )
        )
        code = _code((*values, False), None, _narrow_batch_offset)
    pointers = [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.endswith("_async_pointer")
            for target in node.targets
        )
    ]
    assert len(pointers) == 3
    for pointer in pointers:
        text = ast.unparse(pointer)
        assert "cutlass.Int64(cutlass.Int32(" in text
        for product in ast.walk(pointer):
            if isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult):
                for operand in (product.left, product.right):
                    assert not (
                        isinstance(operand, ast.Call)
                        and ast.unparse(operand.func) == "cutlass.Int32"
                    ), "Physical stride multiplication must not execute in Int32"


def test_explicit_wrapping_index_arithmetic_uses_scalar_fallback() -> None:
    values = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((4, 128, 128), (4, 128, 128), (4, 128, 64))
    )
    code = _code((*values, True), None, _narrow_batch_offset)
    # All selectors are valid, but cancellation in unbounded integer algebra
    # would incorrectly use [0, 1, 2, 0] instead of the wrapped [0, 1, 1, 2].
    selector = ((torch.arange(4, dtype=torch.int32) * 1073741824) // 1073741824) % 3
    assert selector.tolist() == [0, 1, 1, 2]
    assert "OperandSource.TMEM" in code
    assert "_async_pointer" not in code
    assert "1073741824" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wrapping_batch_selector_runtime_exact() -> None:
    function = None
    selected = torch.tensor([0, 1, 1, 2], device=DEVICE)
    for seed in range(5):
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        values = tuple(
            torch.randint(-2, 3, shape, device=DEVICE, generator=generator).to(
                torch.bfloat16
            )
            / 16
            for shape in ((4, 128, 128), (4, 128, 128), (4, 128, 64))
        )
        saved = tuple(value.clone() for value in values)
        if function is None:
            function = _narrow_batch_offset._bind_isolated(
                (*values, True)
            ).compile_config(
                helion.Config(
                    block_sizes=[128, 64],
                    num_warps=4,
                    cute_chained_mma_schedule="tcgen05_tmem",
                )
            )
        a, b, v = (value.index_select(0, selected).double() for value in values)
        first = (a @ b.transpose(-1, -2)).to(torch.bfloat16).double()
        expected = (first @ v).to(torch.bfloat16)
        actual = function(*values, True)
        repeated = function(*values, True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        assert actual.data_ptr() != repeated.data_ptr()
        for value, before in zip(values, saved, strict=True):
            torch.testing.assert_close(value, before, atol=0, rtol=0)
