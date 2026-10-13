from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_output_lease import _config
from .test_cute_chained_output_lease import _output_metadata_loop
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_preparation_cut import _runtime_values
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _fixture(device, dtype, steps):
    generator = torch.Generator(device=device).manual_seed(93127)
    a = (
        torch.randn((steps, 16, 16), dtype=dtype, device=device, generator=generator)
        * 0.25
    )
    b = (
        torch.randn((steps, 16, 16), dtype=dtype, device=device, generator=generator)
        * 0.25
    )
    initial = (
        torch.randn((16, 16), dtype=torch.float32, device=device, generator=generator)
        * 0.25
    )
    row = torch.arange(16, dtype=torch.int32, device=device)[None, :]
    step = torch.arange(steps, dtype=torch.int32, device=device)[:, None]
    destination = (row * 5 + step * 3) % 16
    valid = (row + step) % 3 != 0
    return a, b, initial, destination, valid


def _reference(values):
    a, b, initial, destination, valid = values
    state = initial.float()
    history = torch.full(
        (a.shape[0], 16, 16), float("nan"), dtype=torch.float32, device=a.device
    )
    written = torch.zeros_like(history, dtype=torch.bool)
    for step in range(a.shape[0]):
        prepared = (a[step].double() @ b[step].double()).float()
        operand = (prepared * 0.125).to(a.dtype)
        state = (state.to(a.dtype).double() @ operand.double()).float()
        rows = destination[step, valid[step]].long()
        history[step, rows, :] = state[valid[step], :]
        written[step, rows, :] = True
    return history, state, written


def _check_bits(actual, expected, written):
    # The original example intentionally leaves masked output cells unwritten.
    # Their preservation is tested below by poisoning captured output storage.
    _bits_equal(actual[0][written], expected[0][written])
    _bits_equal(actual[1], expected[1])


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cohorts", (1, 3))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_output_lease_runtime_fixture_source_preflight(dtype, cohorts, steps):
    values = _fixture("cpu", dtype, steps)
    with _cpu_codegen():
        bound = _output_metadata_loop._bind_isolated(values)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_output_metadata_loop, values)
        ):
            source = bound.to_code(
                _config(
                    num_warps=16,
                    cute_chained_preparation_cohorts=cohorts,
                    cute_chained_output_lease_snapshot=True,
                )
            )
    assert "chain_output_snapshot_0_index =" in source
    assert "chain_output_snapshot_1_index =" in source
    assert "chain_output_snapshot_2_index =" in source
    slots = 2 if cohorts == 1 else cohorts
    assert (
        source.count(
            f"chain_sync.arrive_mbarrier(chain_slot_bars + {slots} + chain_slot)"
        )
        == 1
    )


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("cohorts", (1, 3))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_output_lease_typed_indices_masks_generations_and_replay_gpu(
    dtype, cohorts, steps
):
    canonical = _fixture(DEVICE, dtype, steps)
    options = {
        "num_warps": 16,
        "cute_chained_preparation_cohorts": cohorts,
    }
    ordinary = _output_metadata_loop._bind_isolated(canonical).compile_config(
        _config(**options)
    )
    detached = _output_metadata_loop._bind_isolated(canonical).compile_config(
        _config(**options, cute_chained_output_lease_snapshot=True)
    )
    generations = [tuple(value.clone() for value in canonical) for _ in range(2)]
    for index, value in enumerate(canonical):
        if value.numel():
            assert (
                len(
                    {
                        value.data_ptr(),
                        *(generation[index].data_ptr() for generation in generations),
                    }
                )
                == 3
            )
    for generation, values in enumerate(generations):
        if generation:
            for value in values[:3]:
                value.mul_(0.5)
            values[3].copy_((values[3] + 7) % 16)
            values[4].logical_not_()
        saved = tuple(value.clone() for value in values)
        reference_history, reference_final, written = _reference(values)
        actual = detached(*values)
        torch.testing.assert_close(
            actual[0][written], reference_history[written], rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(actual[1], reference_final, rtol=2e-3, atol=2e-3)
        ordinary_actual = ordinary(*values)
        _check_bits(actual, ordinary_actual, written)
        if steps == 0:
            _bits_equal(actual[1], saved[2])
            _bits_equal(ordinary_actual[1], saved[2])
        for _ in range(3):
            _check_bits(actual, detached(*values), written)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = detached(*values)
        for _ in range(3):
            for output in captured:
                output.fill_(float("nan"))
            graph.replay()
            _check_bits(captured, actual, written)
            assert torch.isnan(captured[0][~written]).all()
        for value, before in zip(values, saved, strict=True):
            _bits_equal(value, before)
