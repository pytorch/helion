from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy
import torch

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
