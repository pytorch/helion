from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_config
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_preparation_storage as storage_module
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _fixture(device, dtype, steps, selected):
    generator = torch.Generator(device=device).manual_seed(541203)

    def values(shape, target_dtype):
        return (
            torch.randint(-4, 5, shape, device=device, generator=generator).to(
                target_dtype
            )
            / 256
        )

    return (
        values((max(1, steps), 32, 64), dtype),
        values((max(1, steps), 32, 64), dtype),
        values((max(1, steps), 32, 32), dtype),
        values((32, 64), torch.float32),
        selected,
        steps,
    )


def _reference(values):
    x, y, z, initial, selected, steps = values
    state = initial.clone()
    history = torch.empty((steps, 32, 64), dtype=torch.float32, device=x.device)
    for step in range(steps):
        raw = x[step].float()
        prefix = (raw * 0.125).cumsum(dim=0)
        # Binary-small inputs make both prefix and energy sums exactly
        # representable in FP32, independent of the reference reduction tree.
        weights = y[step].float()
        energy = (weights * weights).sum(dim=1)
        left = (torch.sigmoid(prefix) + energy[:, None] + raw * 0.0625).to(x.dtype)
        right = (energy[:, None] + prefix * 0.25).to(x.dtype)
        prepared = (left.double() @ right.double().T).float()
        rhs = (prepared * 0.125 + z[step].float()).to(x.dtype)
        state = (
            rhs.double() @ state.to(x.dtype).double() + state.double()
        ).float() + prefix[selected][None, :]
        history[step] = state
    return history, state


def _check(actual, expected):
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)


def _compile(values, compact, *, source_only=False):
    original_emit = pipeline_module.emit_preparation_pipeline
    original_bind = storage_module.bind_preparation_storage
    bindings = []

    def emit(*args, **kwargs):
        return original_emit(*args, **kwargs, compact_preparation=compact)

    def bind(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        assert result is not None
        bindings.append(result)
        return result

    with (
        patch.object(pipeline_module, "emit_preparation_pipeline", emit),
        patch.object(storage_module, "bind_preparation_storage", bind),
    ):
        bound = _scan_sequence._bind_isolated(values)
        if source_only:
            with bound.env.use_runtime_arg_values(
                _runtime_values(_scan_sequence, values)
            ):
                result = bound.to_code(_sequence_config())
        else:
            result = bound.compile_config(_sequence_config())
    assert len(bindings) == int(compact)
    if compact:
        physical = bindings[0]
        assert physical._state.consumed
        assert (
            physical.stride
            < physical.physical.accepted.pipeline.frame.layout.allocated_bytes
        )
        storage = physical._state.finalized
        assert storage is not None
        assert (
            dict(storage.allocations)["frames"]
            == physical.stride * physical.physical.accepted.pipeline.slots
        )
    return result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_preparation_storage_runtime_fixture_source_preflight(dtype, selected, steps):
    values = _fixture("cpu", dtype, steps, selected)
    with _cpu_codegen():
        source = _compile(values, True, source_only=True)
    assert isinstance(source, str)
    assert "chain_scan_producer_" in source
    assert "shuffle_sync_up" in source
    assert "chain_sync.arrive_mbarrier" in source
    assert "chain_preparation_workspace_" in source


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("selected", (0, 17, 31))
@pytest.mark.parametrize("steps", (0, 1, 7))
def test_preparation_storage_original_values_generations_and_replay_gpu(
    dtype, selected, steps
):
    canonical = _fixture(DEVICE, dtype, steps, selected)
    ordinary = _compile(canonical, False)
    retained = _compile(canonical, True)
    assert callable(ordinary) and callable(retained)
    generations = [
        (*[value.clone() for value in canonical[:4]], selected, steps) for _ in range(2)
    ]
    for index, value in enumerate(canonical[:4]):
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
            for value in values[:4]:
                value.mul_(0.5)
        saved = [value.clone() for value in values[:4]]
        expected = _reference(values)
        previous = None
        for _ in range(3):
            before, after = ordinary(*values), retained(*values)
            _check(before, expected)
            _check(after, expected)
            for left, right in zip(before, after, strict=True):
                _bits_equal(left, right)
            if steps == 0:
                _bits_equal(before[1], saved[3])
                _bits_equal(after[1], saved[3])
            if previous is not None:
                for left, right in zip(after, previous, strict=True):
                    _bits_equal(left, right)
            previous = [value.clone() for value in after]
        assert previous is not None
        for implementation in (ordinary, retained):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = implementation(*values)
            for _ in range(3):
                for output in captured:
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                _check(captured, expected)
                for actual, reference in zip(captured, previous, strict=True):
                    _bits_equal(actual, reference)
        for value, original in zip(values[:4], saved, strict=True):
            _bits_equal(value, original)
