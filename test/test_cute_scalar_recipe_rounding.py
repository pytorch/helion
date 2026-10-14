from __future__ import annotations

import ast
import operator
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

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


def _render(
    kernel: Kernel, args: tuple[torch.Tensor, ...], *, fast_math: bool = False
) -> str:
    bound_kernel = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=fast_math,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(bound_kernel, args)
        return bound.to_code(bound.config_spec.autotune_reference_config())


def _half(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float16)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_pointwise_products(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(a)
    product = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = a[tile].float()
        second = b[tile].float()
        # Both products are rounded before the subtraction.
        out[tile] = (first * second).to(a.dtype) - (second * 1.5).to(a.dtype)
        # A stored product is consumed as rounded; nothing can contract it.
        product[tile] = (first * 0.75).to(a.dtype)
    return out, product


@pytest.mark.parametrize("fast_math", [False, True])
@skipUnlessBackends(["cute"])
def test_explicitly_rounded_pointwise_products_cannot_contract(
    fast_math: bool,
) -> None:
    """The half FMA ptxas forms from an unrounded ``mul.f16`` + ``sub.f16``
    would drop the product's explicit rounding (``examples/rope.py``)."""
    source = _render(
        _rounded_pointwise_products, (_half(64), _half(64)), fast_math=fast_math
    )
    assert source.count("mul.rn.f32") == (0 if fast_math else 2)
    assert "* 0.75" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_stored_and_added(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    stored = torch.empty_like(a)
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        rounded = (a[tile].float() * b[tile].float()).to(a.dtype)
        stored[tile] = rounded
        out[tile] = rounded - c[tile]
    return stored, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_through_view(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(c)
    for tile in hl.tile(a.size(0)):
        product = a[tile].float() * b[tile].float()
        out[tile, :] = product[:, None].to(a.dtype) - c[tile, :]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_square_negation_and_inplace(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = a[tile].float()
        second = b[tile].float()
        scaled = a[tile].float()
        scaled.mul_(second)
        out[tile] = (
            (first**2).to(a.dtype)
            + (-(first * second)).to(a.dtype)
            + scaled.to(a.dtype)
            - c[tile]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp32_product_before_subtraction(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = a[tile].float()
        # ``sigmoid`` is a genuine FP32 operand: LLVM cannot narrow the product
        # (``examples/swiglu.py``), so the cast already rounds it.
        out[tile] = (first * torch.sigmoid(first)).to(a.dtype) - c[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_with_constant_tensor(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        # A scalar-filled tensor is a splat constant LLVM narrows like a scalar.
        half = hl.full([tile], 0.5, dtype=torch.float32)
        out[tile] = (a[tile].float() * half).to(a.dtype) - c[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_negated_by_multiply(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        rounded = (a[tile].float() * b[tile].float()).to(a.dtype)
        # ``* -1.0`` folds to a negation; the addition can still fuse the product.
        out[tile] = rounded * -1.0 + c[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_dot_operand(
    a: torch.Tensor, b: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            operand = (a[tile_m, tile_k].float() * s[tile_k][None, :].float()).to(
                a.dtype
            )
            acc = hl.dot(operand, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_transposed_dot_operand(
    a: torch.Tensor, b: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    k, m = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            operand = (a[tile_k, tile_m].float() * s[tile_k][:, None].float()).to(
                a.dtype
            )
            acc = hl.dot(operand.T, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rounded_product_selected_dot_operand(
    a: torch.Tensor, b: torch.Tensor, s: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            operand = (a[tile_m, tile_k].float() * s[tile_k][None, :].float()).to(
                a.dtype
            )
            operand = torch.where(
                keep[tile_m, tile_k], operand, torch.zeros_like(operand)
            )
            acc = hl.dot(operand, b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@pytest.mark.parametrize(
    ("kernel", "args", "rounded_products"),
    [
        pytest.param(
            _rounded_product_stored_and_added,
            (_half(64), _half(64), _half(64)),
            1,
            id="store_and_subtract",
        ),
        pytest.param(
            _rounded_product_through_view,
            (_half(64), _half(64), _half(64, 4)),
            1,
            id="view_before_cast",
        ),
        pytest.param(
            _rounded_square_negation_and_inplace,
            (_half(64), _half(64), _half(64)),
            3,
            id="square_negation_inplace",
        ),
        pytest.param(
            _fp32_product_before_subtraction,
            (_half(64), _half(64), _half(64)),
            0,
            id="fp32_operand",
        ),
        pytest.param(
            _rounded_product_with_constant_tensor,
            (_half(64), _half(64), _half(64)),
            1,
            id="constant_tensor_operand",
        ),
        pytest.param(
            _rounded_product_negated_by_multiply,
            (_half(64), _half(64), _half(64)),
            1,
            id="multiply_by_minus_one",
        ),
        pytest.param(
            _rounded_product_dot_operand,
            (_half(64, 32), _half(32, 32), _half(32)),
            0,
            id="dot_operand",
        ),
        pytest.param(
            _rounded_product_transposed_dot_operand,
            (_half(32, 64), _half(32, 32), _half(32)),
            0,
            id="transposed_dot_operand",
        ),
        pytest.param(
            _rounded_product_selected_dot_operand,
            (
                _half(64, 32),
                _half(32, 32),
                _half(32),
                torch.empty((64, 32), dtype=torch.bool),
            ),
            0,
            id="where_before_dot_operand",
        ),
    ],
)
@skipUnlessBackends(["cute"])
def test_only_fusable_rounded_products_use_mul_rn(
    kernel: Kernel, args: tuple[torch.Tensor, ...], rounded_products: int
) -> None:
    """Views, ``where``, negations and multiplies by -1.0 are followed to the
    real producer and consumer; a store or a matmul operand cannot fuse the
    rounded product, a product with a genuine FP32 operand is never narrowed,
    and a scalar-filled tensor operand is a narrowable constant."""
    assert _render(kernel, args).count("mul.rn.f32") == rounded_products


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
