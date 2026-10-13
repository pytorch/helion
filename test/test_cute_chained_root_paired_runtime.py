from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_pipeline import args as leaf_args
from .test_cute_chained_pipeline import leaf_config
from .test_cute_chained_pipeline import pair
from .test_cute_chained_plain_root_runtime import _bits_equal
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

CASES = ((64, 0), (64, 32), (128, 64))


def _fixture(device, dtype, schedule, width, columns):
    generator = torch.Generator(device=device).manual_seed(92741)
    original = leaf_args(dtype=dtype, n=width)
    a, b, raw, d = (
        torch.randn(value.shape, dtype=value.dtype, device=device, generator=generator)
        * 0.05
        for value in original[:4]
    )
    # Exact binary coefficients make all original prefix-tree sums exact.
    # This isolates transport while retaining the original nonlinear expression.
    weights = (
        torch.randint(
            -8, 9, original[4].shape, device=device, generator=generator
        ).float()
        / 512
    )
    dt = (
        torch.randn(original[5].shape, dtype=dtype, device=device, generator=generator)
        * 0.05
    )
    config = leaf_config(
        schedule=schedule,
        cute_chained_seed_tile_columns=columns,
        cute_chained_tmem_early_release=bool(columns),
        cute_chained_tmem_free="last_read",
    )
    return (a, b, raw, d, weights, dt), config


def _reference(values):
    a, b, raw, d, weights, dt = values
    first = (a.double() @ b.double()).float()
    prefix = weights.float().cumsum(0)
    seed = (first * torch.exp(prefix)[:, None]).float()
    difference = (prefix[:, None] - prefix[None, :]).float()
    decay = torch.exp(difference.clamp(max=0.0)).float()
    weighted = (raw.float() * decay).float()
    weighted = (weighted * dt.float()[None, :]).float()
    row = torch.arange(raw.shape[0], device=raw.device)[:, None]
    col = torch.arange(raw.shape[1], device=raw.device)[None, :]
    left = torch.where(row >= col, weighted, 0.0).to(a.dtype)
    # Existing initialized-accumulator policy seeds the second FP32 MMA.
    # Preserve its typed boundary; independent products/sums use FP64.
    accumulated = (seed.double() + left.double() @ d.double()).float()
    return (accumulated + d.float()).to(a.dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
@pytest.mark.parametrize("width,columns", CASES)
def test_paired_root_runtime_fixture_preserves_original_source(
    dtype, schedule, width, columns
):
    initial = torch.cuda.is_initialized()
    values, config = _fixture("cpu", dtype, schedule, width, columns)
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = pair._bind_isolated(values).to_code(config)
    with (
        _cpu(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = pair._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    assert emitted.call_args.args[4].physical == (128, width, 128)
    assert emitted.call_args.kwargs["root_actions"].sequence.paired_issued
    assert "chained_paired_leaf_tma" in after
    assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in after
    assert "mbarrier_wait(chain_bars + 1, 0)" not in after
    assert ("chain_0_values =" not in after) is bool(columns)
    assert torch.cuda.is_initialized() == initial


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
@pytest.mark.parametrize("width,columns", CASES)
def test_paired_root_original_seed_tma_phases_and_replay_gpu(
    dtype, schedule, width, columns
):
    canonical, config = _fixture(DEVICE, dtype, schedule, width, columns)
    with patch.object(roots, "codegen_plain_root", return_value=False):
        ordinary = pair._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = pair._bind_isolated(canonical).compile_config(config)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    # Keep both generations alive: each descriptor must use a genuinely fresh
    # address, not an allocator-recycled pointer or a cached tensor value.
    generations = [tuple(value.clone() for value in canonical) for _ in range(2)]
    for index in range(len(canonical)):
        assert (
            len(
                {
                    canonical[index].data_ptr(),
                    *(values[index].data_ptr() for values in generations),
                }
            )
            == 3
        )
    for generation, values in enumerate(generations):
        if generation:
            for value in values:
                value.mul_(0.5)
        saved = tuple(value.clone() for value in values)
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
        for value, before in zip(values, saved, strict=True):
            _bits_equal(value, before)
