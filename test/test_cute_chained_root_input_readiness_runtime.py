from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_accumulator import _initialized_args
from .test_cute_chained_accumulator import _initialized_config
from .test_cute_chained_accumulator import _initialized_pair
from .test_cute_chained_accumulator import _late_rhs_args
from .test_cute_chained_accumulator import _late_rhs_config
from .test_cute_chained_accumulator import _late_rhs_pair
from .test_cute_chained_plain_root_runtime import _bits_equal
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _fixture(device, dtype, mode, columns, alternate):
    if mode == "local":
        originals = _initialized_args(
            dtype=dtype, n=128, kind="major" if alternate else "dense"
        )
        kernel = _initialized_pair
        config = _initialized_config(128, cute_chained_seed_tile_columns=columns)
    else:
        originals = _late_rhs_args(
            dtype=dtype, n=128, kind="stride" if alternate else "dense"
        )
        kernel = _late_rhs_pair
        config = _late_rhs_config(value=False, cute_chained_seed_tile_columns=columns)
    generator = torch.Generator(device=device).manual_seed(65291)
    values = []
    for index, original in enumerate(originals[:-1]):
        value = torch.empty_strided(
            original.shape, original.stride(), dtype=original.dtype, device=device
        )
        if index < 4:
            data = (
                torch.randn(original.shape, device=device, generator=generator) * 0.05
            )
        else:
            # Exactly representable coefficients keep the prefix-sum tree out
            # of this transport regression, without removing the source scan.
            data = (
                torch.randint(
                    -8, 9, original.shape, device=device, generator=generator
                ).float()
                / 512
            )
        value.copy_(data)
        values.append(value)
    return kernel, (*values, originals[-1]), config


def _reference(values, mode):
    a, b, c, d, scale = values[:5]
    if mode == "local":
        first_left = (a.float() * 1.01).to(a.dtype)
        first = (first_left.double() @ b.double()).float()
        second_left = c
        second_right = (d.float() * scale[:, None]).to(a.dtype)
    else:
        first = (a.double() @ b.double()).float()
        prefix = values[5].float().cumsum(0)
        second_left = (c.float() * torch.exp(prefix)[None, :]).to(c.dtype)
        second_right = d
    seed = first * torch.exp(scale.float())[:, None]
    # Preserve the source operand narrowings and FP32 seed. The independent
    # FP64 contraction reference is complemented by exact old/new code bits.
    result = (seed.double() + second_left.double() @ second_right.double()).float()
    if mode == "upfront":
        result = result + d.float()
    return result.to(a.dtype)


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
@pytest.mark.parametrize("alternate", (False, True))
def test_input_readiness_runtime_fixture_keeps_original_source(
    mode, dtype, columns, alternate
):
    initialized = torch.cuda.is_initialized()
    kernel, values, config = _fixture("cpu", dtype, mode, columns, alternate)
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = kernel._bind_isolated(values).to_code(config)
    with (
        _cpu(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = kernel._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    sequence = emitted.call_args.kwargs["root_actions"].sequence
    assert sequence.seeded_inputs.mode == mode
    assert sequence.input_completion.stage == 1
    assert sequence.next_stage == 2
    assert torch.cuda.is_initialized() == initialized


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
@pytest.mark.parametrize("alternate", (False, True))
def test_input_readiness_original_local_and_upfront_copies_gpu(
    mode, dtype, columns, alternate
):
    kernel, canonical, config = _fixture(DEVICE, dtype, mode, columns, alternate)
    with patch.object(roots, "codegen_plain_root", return_value=False):
        original = kernel._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = kernel._bind_isolated(canonical).compile_config(config)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    assert emitted.call_args.kwargs["root_actions"].sequence.seeded_inputs.mode == mode
    for generation in range(2):
        tensors = []
        for value in canonical[:-1]:
            tensor = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device
            )
            tensor.copy_(value)
            if generation:
                tensor.mul_(0.5)
            tensors.append(tensor)
        values = (*tensors, canonical[-1])
        saved = tuple(value.clone() for value in tensors)
        actual = shared(*values)
        torch.testing.assert_close(
            actual, _reference(values, mode), rtol=2e-3, atol=2e-3
        )
        _bits_equal(actual, original(*values))
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
