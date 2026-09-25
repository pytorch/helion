from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .test_cute_chained_tcgen05_guards import _code
import helion
from helion._compiler.cute.chained_matmul import _copy_code
from helion._compiler.cute.chained_matmul import _copy_index
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
def test_copy_index_preserves_uniform_integer_cast(dtype: str) -> None:
    row, origin = sympy.symbols("row origin", integer=True)
    value = _copy_index(
        f"cutlass.{dtype}(origin * 128)", {}, {"row": row, "origin": origin}
    )
    assert sympy.diff(value, row) == 0
    env = SimpleNamespace(
        index_dtype=torch.int64,
        backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Int64"),
    )
    with patch(
        "helion._compiler.cute.chained_matmul.CompileEnvironment.current",
        return_value=env,
    ):
        expected = f"cutlass.{dtype}((128 * cutlass.Int64(origin)))"
        if dtype == "Int32":
            expected = f"cutlass.Int64({expected})"
        assert _copy_code(value) == expected


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
@pytest.mark.parametrize(
    "expression",
    ["row + index", "index - 1", "index * 0", "index // 2", "index % 2", "-index"],
)
def test_copy_index_rejects_post_cast_arithmetic(dtype: str, expression: str) -> None:
    with pytest.raises(_UnsupportedChain, match="arithmetic after fixed-width"):
        _copy_index(
            expression,
            {"index": f"cutlass.{dtype}(origin)"},
            {name: sympy.Symbol(name, integer=True) for name in ("row", "origin")},
        )


@pytest.mark.parametrize("dtype", ["Int32", "Int64"])
def test_copy_index_varying_cast_is_not_proven_affine(dtype: str) -> None:
    row = sympy.Symbol("row", integer=True)
    value = _copy_index(f"cutlass.{dtype}(row)", {}, {"row": row})
    assert not isinstance(sympy.diff(value, row), sympy.Integer)


@pytest.mark.parametrize(
    "source", ["cutlass.Float32(row)", "other.Int32(row)", "cutlass.Int32(row, 2)"]
)
def test_copy_index_rejects_non_integer_or_unknown_calls(source: str) -> None:
    with pytest.raises(_UnsupportedChain):
        _copy_index(source, {}, {"row": sympy.Symbol("row", integer=True)})


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chunked_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, length, state = a.shape
    dim = v.shape[-1]
    chunks = length // 128
    state = hl.specialize(state)
    out = torch.empty_like(v)
    for block, row, col in hl.tile(
        [batch * chunks, 128, dim], block_size=[1, None, None]
    ):
        bi = block.begin // chunks
        ci = block.begin % chunks
        q = hl.arange(128)
        qr = ci * 128 + row.index
        qk = ci * 128 + q
        first = hl.dot(a[bi, qr, :], b[bi, qk, :].T)
        result = hl.dot(first.to(a.dtype), v[bi, qk, col])
        out[bi, qr, col] = (result + v[bi, qr, col].float()).to(v.dtype)
    return out


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
    )


def test_chunked_chain_codegen_keeps_scalar_post_cast_indexing() -> None:
    values = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, 384, 128), (2, 384, 128), (2, 384, 64))
    )
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        code = _chunked_chain._bind_isolated(values).to_code(_config())
    assert "OperandSource.TMEM" in code
    assert "_async_pointer" not in code
    assert "chain_store_copy" not in code
    assert "chain_epi_input_0_values" not in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chunked_chain_correctness() -> None:
    torch.manual_seed(73)
    values = tuple(
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) * 0.1
        for shape in ((2, 384, 128), (2, 384, 128), (2, 384, 64))
    )
    saved = tuple(value.clone() for value in values)
    a, b, v = (value.reshape(2, 3, 128, -1) for value in values)
    expected = (
        (
            (a.float() @ b.float().transpose(-1, -2)).to(a.dtype).float() @ v.float()
            + v.float()
        )
        .to(v.dtype)
        .reshape(values[-1].shape)
    )
    function = _chunked_chain._bind_isolated(values).compile_config(_config())
    actual = function(*values)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
    torch.testing.assert_close(actual, function(*values), atol=0, rtol=0)
    for value, before in zip(values, saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)


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
