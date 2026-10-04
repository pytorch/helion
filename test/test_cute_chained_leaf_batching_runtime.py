from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_leaf_batching import KEY
from .test_cute_chained_leaf_set_runtime import _assert_bits
from .test_cute_chained_leaf_set_runtime import _config as _leaf_config
from .test_cute_chained_leaf_set_runtime import _inputs
from .test_cute_chained_leaf_set_runtime import _reference
from .test_cute_chained_leaf_sets import _reused_raw
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_leaves import _source
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

_CASES = tuple(
    (dtype, steps, masked)
    for dtype in (torch.bfloat16, torch.float16)
    for steps in (0, 1, 7)
    for masked in (False, True)
)


def _config(enabled):
    config = _leaf_config(3, 4)
    config.config.update(cute_chained_compact_preparation=True)
    config.config[KEY] = enabled
    return config


@pytest.mark.parametrize("dtype,steps,masked", _CASES)
def test_public_batching_runtime_preflight(dtype, steps, masked):
    args = _inputs(dtype, steps, masked)
    saved = tuple(value.clone() for value in args[:5])
    expected = _reference(args)
    captured = []
    original = storage.finalize_pipeline_storage

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        body = result.preparation.physical.scheduled_body
        assert body is not None and body.schedule.max_pending == 2
        captured.append(body)
        return result

    with patch.object(storage, "finalize_pipeline_storage", observe):
        source = _source(_reused_raw, args, _config(True))
    assert len(captured) == 1
    assert "chain_generation & 1" in source and "tma_partition(" in source
    assert expected[0].shape == (steps, 16, 128)
    assert all(torch.isfinite(value).all() for value in expected)
    if not steps:
        _assert_bits(expected[1], saved[4])
    for value, old in zip(args[:5], saved, strict=True):
        _assert_bits(value, old)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,steps,masked", _CASES)
def test_public_batching_gpu_original_reference_bits_replay(dtype, steps, masked):
    args = _inputs(dtype, steps, masked, device=DEVICE)
    compiled = []
    for enabled in (False, True):
        bound = _reused_raw._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_reused_raw, args)):
            compiled.append(bound.compile_config(_config(enabled)))
    generations = [args, _inputs(dtype, steps, masked, device=DEVICE)]
    # Fresh descriptors also see changed input values, not a memoized output.
    generations[1][0].mul_(0.75)
    for values in generations:
        saved = tuple(value.clone() for value in values[:5])
        reference = _reference(values)
        control, actual = (kernel(*values) for kernel in compiled)
        torch.cuda.synchronize()
        snapshots = tuple(value.clone() for value in actual)
        for value, old, expected in zip(actual, control, reference, strict=True):
            _assert_bits(value, old)
            torch.testing.assert_close(value.cpu(), expected, atol=2e-3, rtol=2e-3)
        for _ in range(3):
            repeated = compiled[1](*values)
            for value, old in zip(repeated, snapshots, strict=True):
                _assert_bits(value, old)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled[1](*values)
        for _ in range(3):
            for value in captured:
                value.fill_(float("nan"))
            graph.replay()
            for value, old in zip(captured, snapshots, strict=True):
                _assert_bits(value, old)
        if not steps:
            _assert_bits(actual[1], saved[4])
            _assert_bits(captured[1], saved[4])
        for value, old in zip(values[:5], saved, strict=True):
            _assert_bits(value, old)
