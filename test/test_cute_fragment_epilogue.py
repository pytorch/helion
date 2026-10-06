from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.fragment_epilogue import _has_fresh_output_allocation
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.host_function import HostFunction
from helion._testing import DEVICE
import helion.language as hl
from helion.language import _tracing_ops


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("out = torch.empty((128, 64))\nreturn out", True),
        ("out = torch.empty((128, 64), out=None)\nreturn out", True),
        ("out = torch.empty((128, 64), out=x)\nreturn out", False),
        ("out = torch.empty((128, 64), out=destination)\nreturn out", False),
        ("out = torch.empty((128, 64), **options)\nreturn out", False),
        ("out = torch.empty_like(x, **options)\nreturn out", False),
        ("out = torch.empty_like(x)\nreturn out.reshape(-1)", True),
        (
            "out = torch.empty_like(x)\nfor i in hl.tile(64):\n    out[i] = x[i]\nreturn out",
            True,
        ),
        ("return out", False),
        ("out = x\nreturn out", False),
        ("out = x.reshape(128, 64)\nreturn out", False),
        ("if flag:\n    out = torch.empty_like(x)\nreturn out", False),
        (
            "out = torch.empty_like(x)\nif flag:\n    out = x\nreturn out",
            False,
        ),
        (
            "out = torch.empty_like(x)\nif flag:\n    scale = out.reshape(-1)\nreturn out",
            False,
        ),
        (
            "out = torch.empty_like(x)\nfor i in range(1):\n    scale = out\nreturn out",
            False,
        ),
    ],
)
def test_fresh_output_allocation_proof(body: str, expected: bool) -> None:
    graph = torch.fx.Graph()
    output = graph.call_function(_tracing_ops._host_tensor, ("out",))
    with patch.object(
        HostFunction,
        "current",
        return_value=SimpleNamespace(body=ast.parse(body).body),
    ):
        assert _has_fresh_output_allocation(output) is expected


@helion.kernel(backend="cute", static_shapes=True)
def _indexed_scale_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    column_scale: torch.Tensor,
    offsets: torch.Tensor | None,
) -> torch.Tensor:
    batches, m, k = lhs.shape
    n = rhs.shape[2]
    groups = column_scale.numel()
    out = torch.empty_like(residual)
    for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
        acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
        for ki in hl.tile(k):
            acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
        group = ni.index // (n // groups)
        if offsets is not None:
            group = offsets[ni]
        address = (
            bi.index[:, None, None] * m + mi.index[None, :, None]
        ) * groups + group[None, None, :]
        weight = torch.exp(scale[address])
        out[bi, mi, ni] = (
            acc * weight
            + residual[bi, mi, ni].float() * column_scale[group][None, None, :].float()
        ).to(residual.dtype)
    return out


@pytest.mark.skipif(
    not get_cute_mma_support().tcgen05_f16bf16,
    reason="tcgen05 F16/BF16 MMA is unavailable",
)
@pytest.mark.parametrize("n", [64, 128])
def test_fragment_epilogue_computed_broadcast_loads(n: int) -> None:
    # The flattened scale address combines all three tile axes, while the
    # column scale broadcasts a computed index into a smaller source extent.
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    batches, m, k, groups = 2, 256, 64, 8

    def random(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return (
            torch.randn(shape, device=DEVICE, dtype=dtype, generator=generator) * 0.125
        )

    lhs = random((batches, m, k), torch.bfloat16)
    rhs = random((batches, k, n), torch.bfloat16)
    scale = random((batches * m * groups,), torch.float32)
    residual = random((batches, m, n), torch.bfloat16)
    column_scale = random((groups,), torch.bfloat16)
    args = (lhs, rhs, scale, residual, column_scale, None)
    before = tuple(value.clone() for value in args[:-1])
    _indexed_scale_matmul.reset()
    bound = _indexed_scale_matmul.bind(args)
    assert bound.config_spec.cute_tcgen05_search_enabled
    config = helion.Config(block_sizes=[128, 64, 64], pid_type="flat")
    code = bound.to_code(config)
    assert "cute.gemm(" in code
    fn = bound.compile_config(config)
    actual = fn(*args)
    group = torch.arange(n, device=DEVICE) // (n // groups)
    expected = (
        torch.bmm(lhs.double(), rhs.double())
        * torch.exp(scale.double().reshape(batches, m, groups)[:, :, group])
        + residual.double() * column_scale.double()[group]
    ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual, fn(*args), atol=0, rtol=0)
    for value, previous in zip(args[:-1], before, strict=True):
        torch.testing.assert_close(value, previous, atol=0, rtol=0)


@pytest.mark.skipif(
    not get_cute_mma_support().tcgen05_f16bf16,
    reason="tcgen05 F16/BF16 MMA is unavailable",
)
def test_fragment_epilogue_rejects_data_dependent_loads() -> None:
    args = (
        torch.empty((2, 128, 64), device=DEVICE, dtype=torch.bfloat16),
        torch.empty((2, 64, 64), device=DEVICE, dtype=torch.bfloat16),
        torch.empty((2 * 128 * 8,), device=DEVICE),
        torch.empty((2, 128, 64), device=DEVICE, dtype=torch.bfloat16),
        torch.empty((8,), device=DEVICE, dtype=torch.bfloat16),
        torch.zeros((64,), device=DEVICE, dtype=torch.int32),
    )
    _indexed_scale_matmul.reset()
    bound = _indexed_scale_matmul.bind(args)
    assert not bound.config_spec.cute_tcgen05_search_enabled


@pytest.mark.skipif(
    not get_cute_mma_support().tcgen05_f16bf16,
    reason="tcgen05 F16/BF16 MMA is unavailable",
)
@pytest.mark.parametrize("reassign", [False, True])
def test_fragment_epilogue_rejects_computed_output_alias(reassign: bool) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def with_output(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
        reassign: hl.constexpr,
    ) -> torch.Tensor:
        batches, m, k = lhs.shape
        n = rhs.shape[2]
        original = out
        if reassign:
            out = torch.empty_like(out)
            if original is not None:
                out = original
        for bi, mi, ni in hl.tile([batches, m, n], block_size=[1, None, None]):
            acc = hl.zeros([bi, mi, ni], dtype=torch.float32)
            for ki in hl.tile(k):
                acc = torch.baddbmm(acc, lhs[bi, mi, ki], rhs[bi, ki, ni])
            address = (bi.index[:, None, None] * m + mi.index[None, :, None]) * n + (
                ni.index[None, None, :] + 1
            ) % n
            out[bi, mi, ni] = (acc * torch.exp(scale[address])).to(out.dtype)
        return out

    output = torch.empty((2, 128, 64), device=DEVICE, dtype=torch.bfloat16)
    args = (
        torch.empty((2, 128, 64), device=DEVICE, dtype=torch.bfloat16),
        torch.empty((2, 64, 64), device=DEVICE, dtype=torch.bfloat16),
        output.reshape(-1),
        output,
        reassign,
    )
    bound = with_output.bind(args)
    assert not bound.config_spec.cute_tcgen05_search_enabled
