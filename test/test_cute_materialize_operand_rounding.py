"""Rounding boundaries must survive moving an operand to a separate launch."""

from __future__ import annotations

import ast
import struct
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_materialize_operand import _computed_rhs
from test.test_cute_materialize_operand import _reference

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _linear_cancellation_rhs(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            first = b[inner, column].float() * 1.44269504
            second = c[inner, column].float() * 1.44269504
            operand = (first - second + 1.00390625).to(a.dtype)
            acc = torch.addmm(acc, a[row, inner], operand)
        out[row, column] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _same_dtype_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            operand = (b[inner, column] * 0.25 + 2).to(a.dtype)
            acc = torch.addmm(acc, a[row, inner], operand)
        out[row, column] = acc.to(out.dtype)
    return out


def _stage_sources(source: str) -> list[str]:
    return [
        ast.literal_eval(node.args[0])
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "PyCodeCache"
        and node.func.attr == "load"
    ]


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@pytest.mark.parametrize("static_shapes", (False, True))
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("fast_math", (False, True))
def test_materialized_fp32_products_keep_the_halfway_cast_boundary(
    cpu_b200: None,
    static_shapes: bool,
    input_dtype: torch.dtype,
    fast_math: bool,
) -> None:
    args = (
        torch.ones((13, 40), dtype=torch.bfloat16),
        torch.full((40, 19), -12.125, dtype=input_dtype),
        torch.full((40, 19), -12.125, dtype=input_dtype),
    )
    kernel = helion.kernel(
        _linear_cancellation_rhs.fn,
        backend="cute",
        static_shapes=static_shapes,
        fast_math=fast_math,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(args)
    assert bound.env.cute_fission_plan is not None
    stages = _stage_sources(bound.to_code(bound.config_spec.default_config()))
    assert len(stages) == 2
    assert stages[0].count("mul.rn.f32") == (0 if fast_math else 2)
    assert "mul.rn.f32" not in stages[1]
    if not fast_math:
        assert "from helion._compiler.cute.inline_asm_helpers import" in stages[0]
    actual = _reference(bound, args, transformed=True, materialized_operand=True)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)

    # Exact linear witness: a newly fused product/subtraction moves an exact
    # BF16 halfway value above the tie. Binary64 represents this FP32 product
    # exactly, so this calculation does not depend on math.fma or exp accuracy.
    product = -12.125 * _f32(1.44269504)
    separate = _f32(_f32(product) - _f32(product))
    contracted = _f32(product - _f32(product))
    halfway = 1.00390625
    assert separate == 0.0
    assert float(torch.tensor(_f32(separate + halfway), dtype=torch.bfloat16)) == 1.0
    assert (
        float(torch.tensor(_f32(contracted + halfway), dtype=torch.bfloat16))
        == 1.0078125
    )


@pytest.mark.parametrize(
    "input_dtype", (torch.int32, torch.float16, torch.bfloat16, torch.float64)
)
def test_materialization_uses_the_product_result_dtype_for_rounding(
    cpu_b200: None, input_dtype: torch.dtype
) -> None:
    # A floating scalar promotes int32 inputs to FP32; input dtype alone is
    # insufficient to determine whether a product needs a rounding boundary.
    kernel = helion.kernel(
        _same_dtype_rhs.fn,
        backend="cute",
        static_shapes=False,
        fast_math=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    args = (
        torch.empty((16, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=input_dtype),
    )
    bound = kernel._bind_isolated(args)
    assert bound.env.cute_fission_plan is not None
    stages = _stage_sources(bound.to_code(bound.config_spec.default_config()))
    assert stages[0].count("mul.rn.f32") == (1 if input_dtype is torch.int32 else 0)


def test_ordinary_pointwise_kernel_keeps_existing_emission(cpu_b200: None) -> None:
    args = (
        torch.empty((16, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=torch.bfloat16),
    )
    kernel = helion.kernel(
        _computed_rhs.fn,
        backend="cute",
        static_shapes=False,
        fast_math=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(args)
    assert bound.env.cute_fission_plan is None
    source = bound.to_code(bound.config_spec.default_config())
    assert "mul.rn.f32" not in source


def test_vectorized_materialization_keeps_separate_fp32_products(
    cpu_b200: None,
) -> None:
    args = (
        torch.empty((13, 40), dtype=torch.bfloat16),
        torch.empty((40, 19), dtype=torch.bfloat16),
        torch.empty((40, 19), dtype=torch.bfloat16),
    )
    kernel = helion.kernel(
        _linear_cancellation_rhs.fn,
        backend="cute",
        static_shapes=False,
        fast_math=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(args)
    config = helion.Config.from_dict(
        bound.config_spec.default_config().config
        | {
            "block_sizes": [1, 32, 16, 16, 16],
            "num_threads": [1, 4, 0, 0, 0],
            "cute_vector_widths": [1, 8, 1, 1, 1],
        }
    )
    stages = _stage_sources(bound.to_code(config))
    assert "cutlass.range_constexpr(8)" in stages[0]
    assert stages[0].count("mul.rn.f32") == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
def test_materialized_halfway_operand_is_exact_on_cuda(
    input_dtype: torch.dtype,
) -> None:
    args = (
        torch.ones((65, 40), dtype=torch.bfloat16, device=CUDA_DEVICE),
        torch.full((40, 19), -12.125, dtype=input_dtype, device=CUDA_DEVICE),
        torch.full((40, 19), -12.125, dtype=input_dtype, device=CUDA_DEVICE),
    )
    kernel = helion.kernel(
        _linear_cancellation_rhs.fn,
        backend="cute",
        static_shapes=False,
        fast_math=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(args)
    assert bound.env.cute_fission_plan is not None
    bound.set_config(bound.config_spec.default_config())
    expected = torch.full((65, 19), 40, dtype=torch.bfloat16, device=CUDA_DEVICE)
    torch.testing.assert_close(bound(*args), expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            bound(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = bound(*args)
    for value in (-12.125, 3.5, -12.125):
        args[1].fill_(value)
        args[2].fill_(value)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
