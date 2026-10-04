from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64 import _batched
from .test_cute_chained_m64 import _batched_narrow
from .test_cute_chained_m64 import _m64_config
from .test_cute_chained_m64 import _m64_single
from .test_cute_chained_plain_root_runtime import _bits_equal
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

CASES = (
    ("plain", 32, False, False, 2),
    ("plain", 96, True, True, 1),
    ("scan", 128, False, True, 2),
    ("scan", 256, True, False, 2),
    ("narrow", 64, False, False, 2),
    ("narrow", 128, True, True, 2),
)


def _fixture(device, dtype, mode, width, direct, early, unroll):
    generator = torch.Generator(device=device).manual_seed(64291)
    shape = (128, 128) if mode == "plain" else (2, 128, 128)
    b_shape = (128, width) if mode == "plain" else (2, 128, width)
    values = tuple(
        torch.randn(size, device=device, dtype=dtype, generator=generator) * 0.05
        for size in (shape, b_shape)
    )
    if mode != "plain":
        # Exact binary FP32 coefficients isolate the original typed operand and
        # sparse result mapping from differences between prefix-sum trees.
        coefficient = (
            torch.randint(-8, 9, (128,), device=device, generator=generator).float()
            / 256
        )
        values = (*values, coefficient)
    kernel = {"plain": _m64_single, "scan": _batched, "narrow": _batched_narrow}[mode]
    config = _m64_config(
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=unroll,
        cute_chained_direct_output=direct,
        cute_chained_tmem_early_release=early,
    )
    return kernel, values, config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode,width,direct,early,unroll", CASES)
def test_native_m64_runtime_fixture_uses_exact_shared_sparse_stage(
    dtype, mode, width, direct, early, unroll
):
    kernel, values, config = _fixture("cpu", dtype, mode, width, direct, early, unroll)
    with (
        _cpu_codegen(),
        patch.object(roots, "supports_independent_root", return_value=False),
    ):
        before = kernel._bind_isolated(values).to_code(config)
    with (
        _cpu_codegen(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = kernel._bind_isolated(values).to_code(config)
    assert after == before
    assert emitted.call_count == 1
    assert emitted.call_args.args[4].physical == (64, width, 128)
    assert emitted.call_args.args[4].native_rows == 64
    assert emitted.call_args.kwargs["terminal_fragment"] is True


def _results(value):
    return value if isinstance(value, tuple) else (value,)


def _reference(values, mode):
    a, b = values[:2]
    if mode == "plain":
        weighted = (b.float() * 0.75).to(b.dtype)
        return ((a.double() @ weighted.double() + 0.125).float(),)
    prefix = values[2].float().cumsum(0)
    weighted = (b.float() * prefix[None, :, None]).to(a.dtype)
    value = (a.double() @ weighted.double() + prefix[-1].double()).float()
    if mode == "narrow":
        return (value.to(a.dtype),)
    return value, prefix[-1].expand(a.shape[0])


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode,width,direct,early,unroll", CASES)
def test_native_m64_original_arithmetic_sparse_outputs_and_replay_gpu(
    dtype, mode, width, direct, early, unroll
):
    kernel, canonical, config = _fixture(
        DEVICE, dtype, mode, width, direct, early, unroll
    )
    with patch.object(roots, "supports_independent_root", return_value=False):
        ordinary = kernel._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = kernel._bind_isolated(canonical).compile_config(config)
    assert emitted.call_count == 1
    assert emitted.call_args.args[4].native_rows == 64
    for layout in ("contiguous", "misaligned", "strided"):
        values = []
        for original in canonical:
            if layout == "contiguous":
                value = original.clone()
            elif layout == "misaligned":
                storage = torch.empty(
                    original.numel() + 1, device=DEVICE, dtype=original.dtype
                )
                value = storage[1:].view(original.shape)
                value.copy_(original)
                assert value.data_ptr() % 16 != 0
            else:
                storage = torch.empty(
                    (*original.shape[:-1], original.shape[-1] * 2),
                    device=DEVICE,
                    dtype=original.dtype,
                )
                value = storage[..., ::2]
                value.copy_(original)
                assert value.stride(-1) == 2
            values.append(value)
        saved = tuple(value.clone() for value in values)
        actual = _results(shared(*values))
        reference = _reference(values, mode)
        for value, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(value, expected, rtol=2e-3, atol=2e-3)
        for value, expected in zip(actual, _results(ordinary(*values)), strict=True):
            _bits_equal(value, expected)
        for _ in range(3):
            for value, expected in zip(_results(shared(*values)), actual, strict=True):
                _bits_equal(value, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _results(shared(*values))
        for _ in range(3):
            graph.replay()
            for value, expected in zip(captured, actual, strict=True):
                _bits_equal(value, expected)
        for value, before in zip(values, saved, strict=True):
            _bits_equal(value, before)
