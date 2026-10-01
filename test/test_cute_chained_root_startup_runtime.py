from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_native_m64_runtime import _fixture
from .test_cute_chained_native_m64_runtime import _reference
from .test_cute_chained_native_m64_runtime import _results
from .test_cute_chained_plain_root_runtime import _bits_equal
from helion import exc
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

CASES = (
    ("plain", 64, "KK", False, False),
    ("plain", 128, "KM", False, True),
    ("plain", 64, "MK", True, False),
    ("plain", 128, "MM", True, True),
    ("scan", 64, "KM", True, False),
    ("scan", 128, "MM", True, True),
    ("narrow", 64, "MK", False, True),
    ("narrow", 128, "MM", True, False),
)


def _startup_fixture(device, dtype, mode, width, major, direct, early):
    kernel, values, config = _fixture(device, dtype, mode, width, direct, early, 2)
    a, b = values[:2]
    if major[0] == "M":
        a = a.transpose(-2, -1).contiguous().transpose(-2, -1)
    if major[1] == "K":
        b = b.transpose(-2, -1).contiguous().transpose(-2, -1)
    config.config.update(
        cute_chained_startup_transfer="tma",
        cute_chained_pointwise_inplace_async=True,
    )
    return kernel, (a, b, *values[2:]), config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode,width,major,direct,early", CASES)
def test_startup_runtime_fixture_preserves_complete_original_source(
    dtype, mode, width, major, direct, early
):
    kernel, values, config = _startup_fixture(
        "cpu", dtype, mode, width, major, direct, early
    )
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
    assert after.count("cute.arch.mbarrier_wait(chain_start_bar, 0)") == 2
    assert "chain_0_b_raw_partition" in after


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode,width,major,direct,early", CASES)
def test_startup_original_values_replay_and_current_pointer_validation_gpu(
    dtype, mode, width, major, direct, early
):
    kernel, canonical, config = _startup_fixture(
        DEVICE, dtype, mode, width, major, direct, early
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
    live_inputs: list[tuple[torch.Tensor, ...]] = [canonical]
    for generation in range(2):
        # Rebuild both TMA descriptors from fresh pointers with the same dense
        # permutation. The compiled wrapper must not retain the prior inputs.
        values = tuple(value.clone() for value in canonical)
        assert all(
            new.data_ptr() != old.data_ptr()
            for previous in live_inputs
            for new, old in zip(values, previous, strict=True)
        )
        live_inputs.append(values)
        if generation:
            for value in values:
                value.mul_(0.5)
        saved = tuple(value.clone() for value in values)
        actual = _results(shared(*values))
        for value, expected in zip(actual, _reference(values, mode), strict=True):
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

        # Keep shape and stride unchanged, but invalidate the current pointer
        # alignment after the wrapper/cache has already been exercised.
        storage = torch.empty(
            values[0].numel() + 1, device=DEVICE, dtype=values[0].dtype
        )
        misaligned = storage[1:].as_strided(values[0].shape, values[0].stride())
        misaligned.copy_(values[0])
        assert misaligned.data_ptr() % 16 != 0
        for implementation in (ordinary, shared):
            with pytest.raises(exc.BackendUnsupported, match="layout/alignment"):
                implementation(misaligned, *values[1:])
