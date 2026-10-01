from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_accumulator import _late_rhs_args
from .test_cute_chained_accumulator import _late_rhs_config
from .test_cute_chained_accumulator import _late_rhs_pair
from .test_cute_chained_plain_root_runtime import _bits_equal
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

CASES = (
    ("full", 64, 0, False),
    ("full", 128, 32, True),
    ("full", 192, 64, True),
    ("serial64", 64, 0, True),
    ("serial64", 128, 32, False),
    ("serial64", 192, 64, False),
    ("overlap64", 64, 0, False),
    ("overlap64", 128, 32, True),
)


def _fixture(device, dtype, schedule, width, columns, early):
    generator = torch.Generator(device=device).manual_seed(65173)
    original = _late_rhs_args(dtype=dtype, n=width)
    values = tuple(
        torch.randn(value.shape, dtype=value.dtype, device=device, generator=generator)
        * 0.05
        for value in original[:4]
    )
    # Exact binary coefficients keep the prefix-sum tree out of this transport
    # comparison. The original exp, typed operands and FP32 seed remain intact.
    coefficients = tuple(
        torch.randint(-8, 9, value.shape, device=device, generator=generator).float()
        / 512
        for value in original[4:6]
    )
    config = _late_rhs_config(
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_pointwise_read_cache=True,
        cute_chained_k_schedule=schedule,
        cute_chained_seed_tile_columns=columns,
        cute_chained_tmem_early_release=early,
    )
    a, b, c, d = values
    scale, weights = coefficients
    return (a, b, c, d, scale, weights, original[6]), config


def _reference(values):
    a, b, c, d, scale, weights, mode = values
    assert mode == "plain"
    first = (a.double() @ b.double()).float()
    seed = first * torch.exp(scale.float())[:, None]
    prefix = weights.float().cumsum(0)
    left = (c.float() * torch.exp(prefix)[None, :]).to(c.dtype)
    # The selected initialized-accumulator policy seeds the second FP32 MMA;
    # this independent high-precision oracle is not a different rounding policy
    # for the compiled path. Old/new compiled results must also match bytewise.
    result = (seed.double() + left.double() @ d.double()).float()
    return (result + d.float()).to(a.dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule,width,columns,early", CASES)
def test_initialized_root_runtime_fixture_preserves_original_source(
    dtype, schedule, width, columns, early
):
    values, config = _fixture("cpu", dtype, schedule, width, columns, early)
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = _late_rhs_pair._bind_isolated(values).to_code(config)
    with (
        _cpu(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _late_rhs_pair._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    first_k = values[0].shape[1]
    padded_k = ((first_k + 127) // 128) * 128
    assert emitted.call_args_list[0].args[4].physical == (128, width, padded_k)
    assert emitted.call_args_list[0].kwargs["terminal_fragment"] is False
    assert ("chain_0_values =" not in after) is bool(columns)


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule,width,columns,early", CASES)
def test_initialized_root_original_seed_deferred_rhs_and_replay_gpu(
    dtype, schedule, width, columns, early
):
    canonical, config = _fixture(DEVICE, dtype, schedule, width, columns, early)
    with patch.object(roots, "codegen_plain_root", return_value=False):
        ordinary = _late_rhs_pair._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = _late_rhs_pair._bind_isolated(canonical).compile_config(config)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    for generation in range(2):
        tensors = tuple(value.clone() for value in canonical[:6])
        if generation:
            for value in tensors:
                value.mul_(0.5)
        values = (*tensors, canonical[6])
        saved = tuple(value.clone() for value in tensors)
        actual = shared(*values)
        torch.testing.assert_close(actual, _reference(values), rtol=2e-3, atol=2e-3)
        _bits_equal(actual, ordinary(*values))
        for _ in range(3):
            _bits_equal(actual, shared(*values))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = shared(*values)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            _bits_equal(captured, actual)
        for value, before in zip(tensors, saved, strict=True):
            _bits_equal(value, before)
