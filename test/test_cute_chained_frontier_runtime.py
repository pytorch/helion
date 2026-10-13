from __future__ import annotations

import pytest
import torch

from .test_cute_chained_frontier_emission import _group_loops
from .test_cute_chained_frontier_emission import _sequence_config
from .test_cute_chained_frontier_emission import _sequence_inputs
from .test_cute_chained_frontier_emission import _shared_frontier_sequence
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_register_runtime import _assert_bits
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _reference(args):
    a, b, initial, narrow_shared = args
    state = initial.double()
    history = []
    for step in range(a.shape[0]):
        # Keep the source's explicit storage snapshots while computing the
        # independent contraction reference in FP64.
        product = (a[step].double() @ b[step].double()).float().double()
        shared = product * torch.exp(a[step].double() * 0.125)
        first = shared.to(a.dtype).double()
        if narrow_shared:
            shared = first
        second = (shared + 0.125).to(a.dtype).double()
        operand = state.to(a.dtype).double()
        state = ((operand @ first).float().double() + operand @ second).float()
        history.append(state)
    return (
        torch.stack(history)
        if history
        else torch.empty((0, 32, 32), dtype=torch.float32, device=a.device),
        state.float(),
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("steps", [0, 1, 4, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("narrow_shared", [False, True], ids=["fp32", "snapshot"])
def test_frontier_group_fp64_replay_and_inputs_gpu(
    dtype, steps, cohorts, narrow_shared
):
    args = _sequence_inputs(dtype, steps, DEVICE, narrow_shared)
    inputs = args[:3]
    saved = tuple(value.clone() for value in inputs)
    compiled = []
    for enabled in (False, True):
        config = _sequence_config(cohorts)
        config.config["cute_chained_vector_group"] = enabled
        bound = _shared_frontier_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_shared_frontier_sequence, args)
        ):
            source = bound.to_code(config)
            assert len(_group_loops(source)) == int(enabled)
            compiled.append(bound.compile_config(config))
    assert compiled[0] is not compiled[1]
    ordinary = tuple(value.clone() for value in compiled[0](*args))
    reference_outputs = _reference(args)
    for implementation in compiled:
        expected = tuple(value.clone() for value in implementation(*args))
        if steps == 0:
            _assert_bits(expected[1], saved[2])
        torch.testing.assert_close(expected, reference_outputs, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(expected, ordinary, rtol=2e-3, atol=2e-3)
        if narrow_shared:
            # Without this explicit source snapshot, sharing the FP32 product
            # can change default NVVM FMA contraction before the second cast.
            # Replay remains bitwise in both cases; only this source contract
            # additionally fixes the rounding boundary across configurations.
            for value, baseline in zip(expected, ordinary, strict=True):
                _assert_bits(value, baseline)
        for _ in range(3):
            actual = implementation(*args)
            assert len(actual) == 2
            for value, reference in zip(actual, expected, strict=True):
                _assert_bits(value, reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(2):
            graph.replay()
            for value, reference in zip(captured, expected, strict=True):
                _assert_bits(value, reference)
        for value, original in zip(inputs, saved, strict=True):
            _assert_bits(value, original)
