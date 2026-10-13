from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_frontier_ownership_runtime import _config
from .test_cute_chained_frontier_ownership_runtime import _inputs
from .test_cute_chained_frontier_ownership_runtime import _mixed_recurrence
from .test_cute_chained_frontier_ownership_runtime import _reference
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_preparation_cut import _runtime_values
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _store_config(cohorts, enabled):
    config = _config(0, cohorts)
    config.config["cute_chained_frontier_stmatrix"] = enabled
    return config


def _check(actual, expected):
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cohorts", (1, 3))
@pytest.mark.parametrize("steps", (0, 1, 7))
@pytest.mark.parametrize("masked", (False, True))
def test_native_frontier_store_runtime_preflight(dtype, cohorts, steps, masked):
    values = _inputs(dtype, steps, masked)
    with _cpu_codegen():
        bound = _mixed_recurrence._bind_isolated(values)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_mixed_recurrence, values)
        ):
            source = bound.to_code(_store_config(cohorts, True))
    assert source.count("StMatrix8x8x16bOp") == 1
    assert "num_matrices=4, transpose=True" in source
    assert "chain_sync.arrive_mbarrier" in source


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cohorts", (1, 3))
@pytest.mark.parametrize("steps", (0, 1, 7))
@pytest.mark.parametrize("masked", (False, True))
def test_native_frontier_store_values_generations_and_replay_gpu(
    dtype, cohorts, steps, masked
):
    canonical = _inputs(dtype, steps, masked, DEVICE)
    implementations = []
    for enabled in (False, True):
        bound = _mixed_recurrence._bind_isolated(canonical)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_mixed_recurrence, canonical)
        ):
            implementations.append(
                bound.compile_config(_store_config(cohorts, enabled))
            )
    generations = [[value.clone() for value in canonical[:5]] for _ in range(2)]
    for index, value in enumerate(canonical[:5]):
        if value.numel():
            assert (
                len(
                    {
                        value.data_ptr(),
                        *(args[index].data_ptr() for args in generations),
                    }
                )
                == 3
            )
    for generation, tensors in enumerate(generations):
        values = (*tensors, *canonical[5:])
        if generation:
            for value in tensors:
                value.mul_(0.5)
        saved = [value.clone() for value in tensors]
        expected = _reference(values)
        previous = None
        for _ in range(3):
            before, after = (
                implementation(*values) for implementation in implementations
            )
            _check(before, expected)
            _check(after, expected)
            for left, right in zip(before, after, strict=True):
                _bits_equal(left, right)
            if steps == 0:
                for actual in (before, after):
                    for output, initial in zip(actual, saved[-2:], strict=True):
                        _bits_equal(output, initial)
            if previous is not None:
                for output, reference in zip(after, previous, strict=True):
                    _bits_equal(output, reference)
            previous = [value.clone() for value in after]
        assert previous is not None
        for implementation in implementations:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = implementation(*values)
            for _ in range(3):
                for output in captured:
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                _check(captured, expected)
                for output, reference in zip(captured, previous, strict=True):
                    _bits_equal(output, reference)
        for value, original in zip(tensors, saved, strict=True):
            _bits_equal(value, original)
