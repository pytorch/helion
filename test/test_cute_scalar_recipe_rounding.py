from __future__ import annotations

import ast
import operator
import struct
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.scalar_recipe_rounding import preserve_fp32_multiply_rounding
from helion._testing import skipUnlessBackends
import helion.language as hl

CUDA_DEVICE = "cuda"


def _rewrite(source: str) -> str:
    tree = ast.parse(source)
    statements, result = preserve_fp32_multiply_rounding(
        cast("list[ast.Assign]", tree.body), ast.Name(id="result", ctx=ast.Load())
    )
    module = ast.Module(body=[*statements], type_ignores=[])
    # The original recipe is still available to other consumers/proofs.
    assert ast.unparse(tree) == ast.unparse(ast.parse(source))
    assert isinstance(result, ast.Name) and result.id == "result"
    return ast.unparse(ast.fix_missing_locations(module))


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def test_separate_products_keep_cancellation_before_bf16_rounding() -> None:
    source = _rewrite(
        "left = cutlass.Float32(a)\n"
        "right = cutlass.Float32(b)\n"
        "scale = 1.44269504\n"
        "left_scaled = left * scale\n"
        "right_scaled = operator.mul(right, scale)\n"
        "difference = left_scaled - right_scaled\n"
        "decay = cute.math.exp2(difference)\n"
        "weighted = decay * -1.25\n"
        "result = cutlass.BFloat16(weighted * 0.953125)\n"
    )
    assert source.count("mul.rn.f32") == 4

    def multiply(args, *, asm, constraints, dtype, is_pure):
        assert asm == "mul.rn.f32 $0, $1, $2;"
        assert constraints == "=f,f,f"
        assert dtype is _f32 and is_pure
        return _f32(args[0] * args[1])

    namespace = {
        "a": -12.125,
        "b": -12.125,
        "cutlass": SimpleNamespace(
            Float32=_f32,
            BFloat16=lambda value: float(torch.tensor(value, dtype=torch.bfloat16)),
        ),
        "cute": SimpleNamespace(math=SimpleNamespace(exp2=lambda value: 2.0**value)),
        "operator": operator,
        "_cute_inline_asm_elementwise": multiply,
    }
    exec(compile(source, "<rounding-test>", "exec"), namespace)
    assert namespace["difference"] == 0.0
    assert namespace["result"] == -1.1875
    # A newly contracted multiply/subtract changes a halfway BF16 rounding.
    scale = _f32(1.44269504)
    fused_difference = _f32(-12.125 * scale - _f32(-12.125 * scale))
    assert fused_difference != 0.0
    assert (
        float(
            torch.tensor(
                _f32(_f32(2.0**fused_difference * -1.25) * 0.953125),
                dtype=torch.bfloat16,
            )
        )
        == -1.1953125
    )


@pytest.mark.parametrize(
    "source",
    [
        "x = cutlass.Int32(a)\nresult = x * 1.44269504",
        "x = cutlass.Float16(a)\nresult = x * 1.44269504",
        "x = cutlass.BFloat16(a)\nresult = x * 1.44269504",
        "x = cutlass.Float64(a)\nresult = x * 1.44269504",
        "x = cutlass.Float32(a)\ny = cutlass.Float64(b)\nresult = x * y",
        "x = cutlass.Float32(a)\nresult = x * unknown",
        "x = cutlass.Float32(a)\nresult = x * 1099511627777",
        "x = cutlass.Float32(a)\nx = unknown\nresult = x * 1.44269504",
        "x = 1.44269504\nresult = x * 2.0",
        "x = cutlass.Float32(a)\nresult = cute.math.fma(x, 1.44269504, x)",
    ],
)
def test_unproven_other_dtypes_and_explicit_fma_stay_unchanged(source: str) -> None:
    assert _rewrite(source) == ast.unparse(ast.parse(source))


def test_explicit_fma_survives_with_a_separately_rounded_product_input() -> None:
    source = _rewrite(
        "x = cutlass.Float32(a)\nresult = cute.math.fma(x * 1.44269504, x, x)"
    )
    assert source.count("mul.rn.f32") == 1
    assert "cute.math.fma(" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_operand_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            left = a[tile_m, tile_k].float()
            row = a[tile_m, 0].float()
            scaled = left * 1.44269504
            decay = torch.exp2(scaled - row[:, None] * 1.44269504)
            operand = ((decay * -1.25) * 0.953125).to(a.dtype)
            acc = hl.dot(operand, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_collective_rounding_boundary_respects_fast_math(
    fast_math: bool, dtype: torch.dtype
) -> None:
    args = (torch.empty((64, 32), dtype=dtype), torch.empty((32, 32), dtype=dtype))
    kernel = helion.kernel(
        _rounded_operand_dot.fn,
        backend="cute",
        static_shapes=True,
        fast_math=fast_math,
        autotune_effort="none",
    )
    config = helion.Config(
        block_sizes=[64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        source = _cpu_bind(kernel, args).to_code(config)
    assert "cute.gemm(" in source
    assert ("mul.rn.f32" in source) is not fast_math
    if not fast_math:
        assert "from helion._compiler.cute.inline_asm_helpers import" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("copy", ["scalar", "async_cached"])
@skipUnlessBackends(["cute"])
def test_cancelled_exponent_preserves_halfway_operand_rounding(
    dtype: torch.dtype, copy: str
) -> None:
    a = torch.full((64, 32), -12.125, dtype=dtype, device=CUDA_DEVICE)
    b = torch.eye(32, dtype=dtype, device=CUDA_DEVICE)
    bound = _rounded_operand_dot._bind_isolated((a, b))
    bound.set_config(
        helion.Config(
            block_sizes=[64, 32, 32],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1],
            cute_collective_mma=True,
            cute_collective_copy=copy,
        )
    )
    expected = torch.full_like(a, -1.25 * 0.953125)
    torch.testing.assert_close(bound(a, b), expected, atol=0, rtol=0)
