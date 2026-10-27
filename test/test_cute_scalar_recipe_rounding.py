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
from helion._compiler.cute.scalar_recipe_rounding import (
    NARROW_MULTIPLY_ROUNDING_META_KEY,
)
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
    kernel: Kernel,
    args: tuple[torch.Tensor, ...],
    *,
    fast_math: bool = False,
    capability: tuple[int, int] | None = (10, 0),
    config: helion.Config | None = None,
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
        bound = _cpu_bind(bound_kernel, args, capability=capability)
        return bound.to_code(
            bound.config_spec.autotune_reference_config() if config is None else config
        )


def _half(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float16)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16)


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
    would drop the product's explicit rounding (``examples/rope.py``).  The
    product of two half upcasts is exact in FP32, so it becomes an explicitly
    rounded ``mul.rn.f16`` of the halves (``narrow_exact_products``); the
    product with a Python scalar keeps the FP32 form."""
    source = _render(
        _rounded_pointwise_products, (_half(64), _half(64)), fast_math=fast_math
    )
    assert source.count("mul.rn.f32") == (0 if fast_math else 1)
    assert source.count("mul.rn.f16") == (0 if fast_math else 1)
    if not fast_math:
        assert (
            "_cute_inline_asm_elementwise((cutlass.Float16(load), "
            "cutlass.Float16(load_1)), asm='mul.rn.f16 $0, $1, $2;', "
            "constraints='=h,h,h', dtype=cutlass.Float16, is_pure=True)"
        ) in source
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
def _rounded_self_product(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = a[tile].float()
        # ``strip_unused_inputs`` hands the lowering ``mul(first, None)``.
        out[tile] = (first * first).to(a.dtype) - c[tile]
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
    ("kernel", "args", "rounded_products", "narrow_products"),
    [
        pytest.param(
            _rounded_product_stored_and_added,
            (_half(64), _half(64), _half(64)),
            0,
            1,
            id="store_and_subtract",
        ),
        pytest.param(
            _rounded_product_through_view,
            (_half(64), _half(64), _half(64, 4)),
            1,
            0,
            id="view_before_cast",
        ),
        pytest.param(
            _rounded_square_negation_and_inplace,
            (_half(64), _half(64), _half(64)),
            1,
            2,
            id="square_negation_inplace",
        ),
        pytest.param(
            _rounded_self_product,
            (_half(64), _half(64), _half(64)),
            0,
            1,
            id="self_product",
        ),
        pytest.param(
            _fp32_product_before_subtraction,
            (_half(64), _half(64), _half(64)),
            0,
            0,
            id="fp32_operand",
        ),
        pytest.param(
            _rounded_product_with_constant_tensor,
            (_half(64), _half(64), _half(64)),
            1,
            0,
            id="constant_tensor_operand",
        ),
        pytest.param(
            _rounded_product_negated_by_multiply,
            (_half(64), _half(64), _half(64)),
            0,
            1,
            id="multiply_by_minus_one",
        ),
        pytest.param(
            _rounded_product_dot_operand,
            (_half(64, 32), _half(32, 32), _half(32)),
            0,
            0,
            id="dot_operand",
        ),
        pytest.param(
            _rounded_product_transposed_dot_operand,
            (_half(32, 64), _half(32, 32), _half(32)),
            0,
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
            0,
            id="where_before_dot_operand",
        ),
    ],
)
@skipUnlessBackends(["cute"])
def test_only_fusable_rounded_products_use_mul_rn(
    kernel: Kernel,
    args: tuple[torch.Tensor, ...],
    rounded_products: int,
    narrow_products: int,
) -> None:
    """Views, ``where``, negations and multiplies by -1.0 are followed to the
    real producer and consumer; a store or a matmul operand cannot fuse the
    rounded product, a product with a genuine FP32 operand is never narrowed,
    and a scalar-filled tensor operand is a narrowable constant.  A flagged
    product of two half upcasts whose only uses are casts back to half is
    the exact ``mul.rn.f16`` instead (``narrow_exact_products``), the
    in-place ``scaled.mul_(second)`` (it lowers as the functional product)
    and the self-products ``first ** 2`` and ``first * first`` included; a
    view or a negation between the product and its cast and a
    constant-tensor operand keep the FP32 form."""
    source = _render(kernel, args)
    assert source.count("mul.rn.f32") == rounded_products
    assert source.count("mul.rn.f16") == narrow_products
    if kernel in (_rounded_self_product, _rounded_square_negation_and_inplace):
        assert (
            "_cute_inline_asm_elementwise((cutlass.Float16(load), "
            "cutlass.Float16(load)), asm='mul.rn.f16 $0, $1, $2;'"
        ) in source


@skipUnlessBackends(["cute"])
def test_narrowed_product_node_starts_from_the_cast_value_only() -> None:
    """The 16-bit multiply ``narrow_exact_products`` adds takes the cast's
    value and source position and nothing else: the cast's lowering-derived
    keys (``orig_node``, ``output_index``) and anything cached on the convert
    (``masked_value``) describe the erased node."""
    import helion._compiler.inductor_lowering as inductor_lowering

    recorded: list[dict[str, object]] = []
    prepare = inductor_lowering.prepare_added_node_lowering

    def spy(graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        recorded.append(dict(node.meta))
        prepare(graph, node)

    with patch.object(inductor_lowering, "prepare_added_node_lowering", spy):
        _render(_rounded_product_stored_and_added, (_half(64), _half(64), _half(64)))
    assert len(recorded) == 1
    (meta,) = recorded
    assert set(meta) <= {
        "val",
        "location",
        "stack_trace",
        NARROW_MULTIPLY_ROUNDING_META_KEY,
    }
    assert meta[NARROW_MULTIPLY_ROUNDING_META_KEY] is True
    assert cast("torch.Tensor", meta["val"]).dtype is torch.float16
    assert "location" in meta


@pytest.mark.parametrize(
    ("dtype", "capability", "narrowed"),
    [
        pytest.param(torch.bfloat16, (8, 0), False, id="bf16_sm80"),
        pytest.param(torch.bfloat16, (8, 9), False, id="bf16_sm89"),
        pytest.param(torch.bfloat16, None, False, id="bf16_unknown_target"),
        pytest.param(torch.bfloat16, (9, 0), True, id="bf16_sm90"),
        pytest.param(torch.bfloat16, (10, 0), True, id="bf16_sm100"),
        pytest.param(torch.float16, (8, 0), True, id="f16_sm80"),
        pytest.param(torch.float16, None, True, id="f16_unknown_target"),
    ],
)
@skipUnlessBackends(["cute"])
def test_bf16_products_narrow_only_from_sm90(
    dtype: torch.dtype, capability: tuple[int, int] | None, narrowed: bool
) -> None:
    """``mul.rn.bf16`` is an sm_90+ PTX form: below that capability, or when
    the target is unknown, a bfloat16 product keeps the sm_80-legal FP32
    ``mul.rn.f32`` plus its cast.  ``mul.rn.f16`` needs only sm_53."""
    args = tuple(torch.empty(64, dtype=dtype) for _ in range(3))
    source = _render(_rounded_product_stored_and_added, args, capability=capability)
    narrow = {torch.float16: "mul.rn.f16 ", torch.bfloat16: "mul.rn.bf16 "}[dtype]
    assert source.count(narrow) == (1 if narrowed else 0)
    assert source.count("mul.rn.f32") == (0 if narrowed else 1)
    assert "mul.rn.bf16" not in source or dtype is torch.bfloat16


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrowed_product_shifted_dot_operand(
    a: torch.Tensor, b: torch.Tensor, s: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            rounded = (a[tile_m, tile_k].float() * s[tile_k][None, :].float()).to(
                a.dtype
            )
            # The subtraction can fuse the product: it is flagged and, being
            # a product of two 16-bit upcasts cast back, narrowed.
            acc = hl.dot(rounded - c[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_narrowed_product_stays_a_collective_operand_recipe(
    dtype: torch.dtype,
) -> None:
    """A collective matmul rematerializes its operand from a scalar recipe,
    which admits inline asm only in the rounded-product spellings
    (``scalar_recipe.is_rounded_multiply``).  The 16-bit ``mul.rn`` of
    ``narrow_exact_products`` must stay a recipe like the FP32 form it
    replaces, or the collective lowering rejects the operand."""
    args = (
        torch.empty((64, 32), dtype=dtype),
        torch.empty((32, 32), dtype=dtype),
        torch.empty(32, dtype=dtype),
        torch.empty((64, 32), dtype=dtype),
    )
    config = helion.Config(
        block_sizes=[64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
    )
    source = _render(_narrowed_product_shifted_dot_operand, args, config=config)
    narrow = {torch.float16: "mul.rn.f16 ", torch.bfloat16: "mul.rn.bf16 "}[dtype]
    assert "cute.gemm(" in source
    assert narrow in source and "mul.rn.f32" not in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrowed_products_then_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = (a[tile].float() * b[tile].float()).to(a.dtype)
        second = (c[tile].float() * d[tile].float()).to(a.dtype)
        out[tile] = first - second
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrowed_squares_then_add(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size(0)):
        first = a[tile].float()
        second = c[tile].float()
        out[tile] = (first**2).to(a.dtype) - (second * second).to(a.dtype)
    return out


def _exactness_operands(
    dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``a, b, c, d`` for ``(a * b).to(dtype) - (c * d).to(dtype)``.

    Random normals span the dtype's exponent range, then explicit sections
    put the products of ``a * b`` into the dtype's subnormal range and across
    its flush-to-zero boundary, across the overflow threshold, at NaN (a NaN
    operand and ``inf * 0``) and at signed zeros with ``c * d == +0`` so the
    difference keeps the product's sign; every fourth row is a cancelling
    pair (``c, d = a, b``) that is zero only if both products are rounded
    before the subtraction (the ``examples/rope.py`` hazard).
    """
    torch.manual_seed(0)
    n = 1 << 16
    info = torch.finfo(dtype)
    # Products of two values below 2^16 (f16) / 2^70 (bf16) reach the
    # dtype's overflow threshold; the lower end reaches its subnormals.
    span = {torch.float16: 16, torch.bfloat16: 70}[dtype]
    scale = torch.exp2(torch.randint(-span, span + 1, (n,), device=device).float())
    a = (torch.randn(n, device=device) * scale).to(dtype)
    b = (torch.randn(n, device=device) * scale).to(dtype)
    c = a.clone()
    d = (b.float() * torch.exp2(torch.randint(-2, 3, (n,), device=device).float())).to(
        dtype
    )

    def section(
        rows: slice, product_exponent_range: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(range(*rows.indices(n)))
        low, high = product_exponent_range
        exponent = torch.randint(low, high + 1, (count,), device=device)
        split = exponent // 2 + torch.randint(-3, 4, (count,), device=device)
        signs = torch.where(torch.rand(2, count, device=device) < 0.5, -1.0, 1.0)
        mantissa = 1 + torch.rand(2, count, device=device)
        left = (signs[0] * mantissa[0] * torch.exp2(split.float())).to(dtype)
        right = (signs[1] * mantissa[1] * torch.exp2((exponent - split).float())).to(
            dtype
        )
        return left, right

    tiny_exponent = int(torch.log2(torch.tensor(info.tiny)).item())
    max_exponent = int(torch.log2(torch.tensor(info.max)).item())
    # Subnormal products down to below the flush-to-zero boundary.
    a[0:4096], b[0:4096] = section(
        slice(0, 4096), (tiny_exponent - 12, tiny_exponent + 2)
    )
    # Products around the overflow threshold.
    a[4096:8192], b[4096:8192] = section(
        slice(4096, 8192), (max_exponent - 2, max_exponent + 6)
    )
    # NaN operands and ``inf * 0``.
    a[8192:8256] = float("nan")
    a[8256:8320] = float("inf")
    b[8256:8320] = 0.0
    # Signed zeros: ``-0 * x``, ``0 * -x``, ``-x * 0`` and ``-0 * -x``, with
    # ``c * d == +0`` so the difference is the product itself.
    zeros = slice(8320, 8576)
    a[8320:8384] = -0.0
    b[8320:8384] = 3.5
    a[8384:8448] = 0.0
    b[8384:8448] = -2.0
    a[8448:8512] = -1.25
    b[8448:8512] = 0.0
    a[8512:8576] = -0.0
    b[8512:8576] = -2.0
    c[zeros] = 0.0
    d[zeros] = 1.0
    # Cancelling pairs outside the explicit sections.
    c[8576::4] = a[8576::4]
    d[8576::4] = b[8576::4]
    return a, b, c, d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_narrowed_products_are_rounded_once_on_cuda(dtype: torch.dtype) -> None:
    """``(a.float() * b.float()).to(dtype)`` as ``mul.rn`` in ``dtype``: the
    product of two 16-bit floats is exact in FP32, so both forms round it
    once, and the explicit rounding keeps ptxas from contracting the
    multiply with the subtraction that follows (which the DSL's own
    ``mul.bf16`` would allow).  Inputs span normals, the dtype's subnormal
    products and flush-to-zero boundary, overflow, NaN, signed zeros and
    halfway cases (``_exactness_operands``); the result matches eager bit
    for bit, signs of zeros included.  Self-products (``first ** 2`` and
    ``second * second``) narrow the same way."""
    a, b, c, d = _exactness_operands(dtype, CUDA_DEVICE)
    info = torch.finfo(dtype)
    first = (a.float() * b.float()).to(dtype)
    # The sections exercise what they claim.
    assert bool(((first != 0) & (first.abs() < info.tiny)).sum() > 256)
    assert bool(((first == 0) & (a != 0) & (b != 0) & torch.isfinite(a)).sum() > 64)
    assert bool(torch.isinf(first).sum() > 256) and bool(
        (torch.isfinite(first) & (first.abs() > info.max / 2)).sum() > 64
    )
    assert bool(torch.isnan(first[8192:8320]).all())
    assert bool(((first == 0) & torch.signbit(first)).sum() >= 192)

    cpu_args = tuple(x.cpu() for x in (a, b, c, d))
    narrow = {torch.float16: "mul.rn.f16 ", torch.bfloat16: "mul.rn.bf16 "}[dtype]
    source = _render(_narrowed_products_then_add, cpu_args)
    assert source.count(narrow) == 2 and "mul.rn.f32" not in source
    out = _narrowed_products_then_add(a, b, c, d)
    expected = first - (c.float() * d.float()).to(dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert bool(torch.equal(torch.signbit(out[zero]), torch.signbit(expected[zero])))
    finite = torch.isfinite(expected[8576::4])
    assert bool(finite.any()) and bool((out[8576::4][finite] == 0).all())

    source = _render(_narrowed_squares_then_add, cpu_args[::2])
    assert source.count(narrow) == 2 and "mul.rn.f32" not in source
    squares = _narrowed_squares_then_add(a, c)
    expected = (a.float() ** 2).to(dtype) - (c.float() * c.float()).to(dtype)
    torch.testing.assert_close(squares, expected, atol=0, rtol=0, equal_nan=True)
    zero = expected == 0
    assert bool(
        torch.equal(torch.signbit(squares[zero]), torch.signbit(expected[zero]))
    )


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
