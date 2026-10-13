from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config as _pipeline_config
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence(a, b, c, d, initial, valid_steps: hl.constexpr):
    _, rows, reduction = a.shape
    steps: int = valid_steps  # pyrefly: ignore [bad-assignment]
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            projected = hl.dot(state.to(a.dtype), prepared, out_dtype=torch.float32)
            state = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state * 0.5)
            history[step.id, rr, jj] = projected
        final[rr, :] = state
    return history, final


def _inputs(dtype, rows, steps, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(51073)
    physical_steps = max(1, steps)
    values = tuple(
        (torch.randn(shape, device=device, generator=generator) * 0.125).to(dtype)
        for shape in (
            (physical_steps, rows, 16),
            (physical_steps, 16, 64),
            (physical_steps, 64, 16),
            (physical_steps, 16, 64),
        )
    )
    initial = torch.randn((rows, 64), device=device, generator=generator) * 0.125
    return (*values, initial, steps)


def _config(warps, columns):
    config = _pipeline_config(16, pipeline=True, consumer_warps=warps)
    config.config.update(
        cute_chained_warp_mma_rows=64,
        cute_chained_snapshot_tile_columns=columns,
    )
    return config


def _reference(values):
    a, b, c, d, initial, steps = values
    state = initial.clone()
    history = torch.empty((steps, *state.shape), device=a.device)
    for step in range(steps):
        prepared = (c[step].double() @ d[step].double()).float().to(a.dtype)
        history[step] = (state.to(a.dtype).double() @ prepared.double()).float()
        state = (a[step].double() @ b[step].double() + (state * 0.5).double()).float()
    return history, state


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("rows", (128, 129))
@pytest.mark.parametrize("steps", (0, 1, 7))
@pytest.mark.parametrize("warps", (4, 8))
def test_streamed_snapshot_runtime_preflight(dtype, rows, steps, warps):
    values = _inputs(dtype, rows, steps)
    source = _source(_sequence, values, _config(warps, 32))
    assert "_snapshot_panel in cutlass.range(2, unroll=1)" in source
    assert "chain_resident_carry_" in source
    assert "chain_tmem_barrier.arrive_and_wait()" in source


def test_streamed_snapshot_cannot_succeed_without_a_resident_carry():
    with (
        patch(
            "helion._compiler.cute.chained_loop_tmem_carry_transport.prepare_loop_tmem_carry",
            return_value=None,
        ),
        pytest.raises(exc.BackendUnsupported, match="proven resident carry"),
    ):
        _source(_sequence, _inputs(torch.bfloat16, 128, 1), _config(4, 32))


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("rows", (128, 129))
@pytest.mark.parametrize("steps", (0, 1, 7))
@pytest.mark.parametrize("warps", (4, 8))
def test_streamed_snapshot_exact_values_state_masks_generations_gpu(
    dtype, rows, steps, warps
):
    canonical = _inputs(dtype, rows, steps, DEVICE)
    implementations = []
    for columns in (0, 32):
        bound = _sequence._bind_isolated(canonical)
        with bound.env.use_runtime_arg_values(_runtime_values(_sequence, canonical)):
            implementations.append(bound.compile_config(_config(warps, columns)))
    generations = [[value.clone() for value in canonical[:5]] for _ in range(2)]
    for index, value in enumerate(canonical[:5]):
        assert (
            len({value.data_ptr(), *(args[index].data_ptr() for args in generations)})
            == 3
        )
    for generation, tensors in enumerate(generations):
        if generation:
            for value in tensors:
                value.mul_(0.5)
        values = (*tensors, steps)
        saved = [value.clone() for value in tensors]
        expected = _reference(values)
        previous = None
        for _ in range(3):
            before, after = (
                implementation(*values) for implementation in implementations
            )
            torch.testing.assert_close(before, expected, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(after, expected, atol=2e-3, rtol=2e-3)
            for left, right in zip(before, after, strict=True):
                _bits_equal(left, right)
            if steps == 0:
                _bits_equal(before[1], saved[-1])
                _bits_equal(after[1], saved[-1])
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
                torch.testing.assert_close(captured, expected, atol=2e-3, rtol=2e-3)
                for output, reference in zip(captured, previous, strict=True):
                    _bits_equal(output, reference)
        for value, original in zip(tensors, saved, strict=True):
            _bits_equal(value, original)
