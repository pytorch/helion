from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_broadcast_expressions import _broadcast_loop
from .test_cute_chained_broadcast_retention import _config as _broadcast_config
from .test_cute_chained_leaf_set_runtime import _assert_bits
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_leaves import _source
import helion
from helion import exc
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

_CASES = tuple(
    (dtype, steps, rounded, mode)
    for dtype in (torch.bfloat16, torch.float16)
    for steps, rounded in ((0, False), (1, True), (7, False))
    for mode in ("single", "group", "scan", "scan_batch")
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_runtime_loop(
    a, b, z, initial, rounded: hl.constexpr, steps: hl.constexpr
):
    """Original broadcast fixture with a logical bound independent of TMA backing."""
    _, height, width = a.shape
    result = torch.empty_like(initial)
    history = torch.empty((steps, height, width), dtype=torch.float32, device=a.device)
    for _rows in hl.tile(height, block_size=32):
        initial_rows = hl.arange(32)
        initial_columns = hl.arange(width)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            rows = hl.arange(32)
            columns = hl.arange(width)
            raw = a[step.id, rows, columns].float()
            prefix = hl.cumsum(raw * 0.125, dim=0)
            endpoint = torch.sum(torch.where((rows == 17)[:, None], prefix, 0.0), dim=0)
            values = b[step.id, rows, columns].float()
            energy = torch.sum(values * values, dim=1)
            retained = torch.tanh(energy + 0.125)
            if rounded:
                retained = retained.to(torch.float16).float()
            broadcast = retained[:, None].expand(32, width)
            left = (prefix + broadcast + raw * 0.0625).to(a.dtype)
            right = (prefix * 0.25 + broadcast).to(a.dtype)
            prepared = hl.dot(left, right.T, out_dtype=torch.float32)
            other = hl.arange(32)
            rhs = (prepared * 0.125 + z[step.id, rows, other].float()).to(a.dtype)
            state = (
                hl.dot(rhs, state.to(a.dtype), acc=state, out_dtype=torch.float32)
                + endpoint[None, :]
            )
            history[step.id, rows, columns] = state
        result[initial_rows, initial_columns] = state
    return history, result


def _config(enabled, mode):
    scan = mode.startswith("scan")
    config = _broadcast_config(enabled, scan=scan, batching=mode == "scan_batch")
    if not scan:
        config.config.update(
            cute_chained_vector_group=mode == "group",
            cute_chained_leaf_pipeline="legacy",
            cute_chained_leaf_count=1,
        )
    return config


def _inputs(dtype, steps, rounded, *, device="cpu", seed=20260926):
    generator = torch.Generator().manual_seed(seed)

    def values(shape, scale, kind):
        return (torch.randn(shape, generator=generator) * scale).to(
            device=device, dtype=kind
        )

    return (
        values((max(1, steps), 32, 64), 0.125, dtype),
        values((max(1, steps), 32, 64), 0.03125, dtype),
        values((max(1, steps), 32, 32), 0.03125, dtype),
        values((32, 64), 0.03125, torch.float32),
        rounded,
        steps,
    )


def _reference(args):
    """FP64 arithmetic with every original explicit tensor narrowing retained."""
    a, b, z, initial, rounded, steps = args
    a, b, z, state = (value.cpu().double() for value in (a, b, z, initial))
    dtype = args[0].dtype
    history = []
    for step in range(steps):
        prefix = torch.cumsum(a[step] * 0.125, dim=0)
        endpoint = prefix[17]
        energy = (b[step] * b[step]).sum(dim=1)
        retained = torch.tanh(energy + 0.125)
        if rounded:
            retained = retained.to(torch.float16).double()
        broadcast = retained[:, None].expand(32, 64)
        left = (prefix + broadcast + a[step] * 0.0625).to(dtype).double()
        right = (prefix * 0.25 + broadcast).to(dtype).double()
        prepared = (left @ right.T).float().double()
        rhs = (prepared * 0.125 + z[step]).to(dtype).double()
        state = (rhs @ state.to(dtype).double() + state).float().double()
        state = (state + endpoint[None, :]).float().double()
        history.append(state.float())
    return (
        torch.stack(history) if history else torch.empty((0, 32, 64)),
        state.float(),
    )


@pytest.mark.parametrize("dtype,steps,rounded,mode", _CASES)
def test_broadcast_runtime_preflight(dtype, steps, rounded, mode):
    args = _inputs(dtype, steps, rounded)
    saved = tuple(value.clone() for value in args[:4])
    reference = _reference(args)
    captured = []
    original = storage.finalize_pipeline_storage

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None and result.preparation is not None
        bound = result.preparation
        assert bound.broadcast_transfers
        if mode == "scan_batch":
            assert bound.physical.scheduled_body is not None
        captured.append(bound)
        return result

    with patch.object(storage, "finalize_pipeline_storage", observe):
        source = _source(_broadcast_runtime_loop, args, _config(True, mode))
    assert len(captured) == 1 and "chain_broadcast" in source
    assert all(torch.isfinite(value).all() for value in reference)
    assert reference[0].shape == (steps, 32, 64)
    if not steps:
        _assert_bits(reference[1], saved[3])
    for value, old in zip(args[:4], saved, strict=True):
        _assert_bits(value, old)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["scan", "scan_batch"])
def test_empty_physical_tma_backing_still_rejects(dtype, mode):
    args = (
        torch.empty((0, 32, 64), dtype=dtype),
        torch.empty((0, 32, 64), dtype=dtype),
        torch.empty((0, 32, 32), dtype=dtype),
        torch.empty((32, 64)),
        False,
        False,
    )
    with pytest.raises(exc.InternalError, match="rectangular TMA requires"):
        _source(_broadcast_loop, args, _config(True, mode))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,steps,rounded,mode", _CASES)
def test_broadcast_gpu_fp64_bits_replay(dtype, steps, rounded, mode):
    args = _inputs(dtype, steps, rounded, device=DEVICE)
    compiled = []
    for enabled in (False, True):
        bound = _broadcast_runtime_loop._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_broadcast_runtime_loop, args)
        ):
            compiled.append(bound.compile_config(_config(enabled, mode)))
    generations = [
        args,
        _inputs(dtype, steps, rounded, device=DEVICE, seed=20260927),
    ]
    for values in generations:
        saved = tuple(value.clone() for value in values[:4])
        reference = _reference(values)
        control, actual = (kernel(*values) for kernel in compiled)
        torch.cuda.synchronize()
        snapshots = tuple(value.clone() for value in actual)
        for value, old, expected in zip(actual, control, reference, strict=True):
            _assert_bits(value, old)
            torch.testing.assert_close(value.cpu(), expected, atol=2e-2, rtol=2e-2)
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
            _assert_bits(actual[1], saved[3])
            _assert_bits(captured[1], saved[3])
        for value, old in zip(values[:4], saved, strict=True):
            _assert_bits(value, old)
