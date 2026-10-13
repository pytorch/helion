from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_leaf_sets import _reused_raw
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

# Bounded representatives, not a Cartesian timing/sweep harness. Positive
# five/seven-trip cases reuse every independent slot, including odd phases.
_CASES = (
    (torch.bfloat16, 0, False, 1, 2),
    (torch.float16, 0, True, 3, 4),
    (torch.float32, 1, False, 3, 2),
    (torch.bfloat16, 1, True, 1, 4),
    (torch.float16, 5, False, 1, 2),
    (torch.float32, 5, True, 3, 4),
    (torch.bfloat16, 7, False, 3, 4),
    (torch.float16, 7, True, 1, 2),
)


def _inputs(dtype, steps, masked, *, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(48021)
    origin = 3 if masked else 0
    rows = origin + max(steps, 1) * 16

    def random(shape, dtype):
        return (
            torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            * 0.125
        ).to(dtype)

    initial = random((16, 128), torch.float32)
    initial[0, 0], initial[0, 1] = -0.0, 0.0
    return (
        random((rows, 64), dtype),
        random((rows, 64), dtype),
        random((64, 64), torch.bfloat16),
        random((max(steps, 1), 64, 128), torch.bfloat16),
        initial,
        steps,
        origin,
        origin + steps * 16 - (7 if masked and steps else 0),
    )


def _config(cohorts, count):
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=count,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
    )


def _reference(args):
    """FP64 products with every original FP32/BF16 version retained.

    Raw FP16/BF16 values widen exactly. Pointwise add/subtract first round to
    FP32, then to the original BF16 operand dtype. Each matmul result rounds
    to FP32; the sum of the two results separately rounds to FP32 before its
    BF16 snapshot. Recurrence publishes another FP32 accumulator version.
    No cancellation of the repeated raw operands or fusion across casts.
    """
    a, b, weight, common, initial = (value.detach().cpu() for value in args[:5])
    steps, origin, end = args[5:]
    state = initial.clone()
    history = []
    for step in range(steps):
        start = origin + step * 16
        mask = (torch.arange(start, start + 16) < end)[:, None]
        av = torch.where(mask, a[start : start + 16].double(), 0.0)
        bv = torch.where(mask, b[start : start + 16].double(), 0.0)
        summed = (av + bv).float().to(torch.bfloat16)
        difference = (av - bv).float().to(torch.bfloat16)
        first = (summed.double() @ weight.double()).float()
        second = (difference.double() @ weight.double()).float()
        snapshot = (first.double() + second.double()).float().to(torch.bfloat16)
        state = (snapshot.double() @ common[step].double() + state.double()).float()
        history.append(state.clone())
    return (
        torch.stack(history) if history else torch.empty((0, 16, 128)),
        state,
    )


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype,steps,masked,cohorts,count", _CASES)
def test_leaf_set_runtime_route_and_reference_preflight_cpu(
    dtype, steps, masked, cohorts, count
):
    args = _inputs(dtype, steps, masked)
    saved = tuple(value.clone() for value in args[:5])
    expected = _reference(args)
    assert expected[0].shape == (steps, 16, 128)
    assert expected[1].dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in expected)
    if not steps:
        _assert_bits(expected[1], args[4])
        assert args[4].view(torch.int32)[0, 0] == -(1 << 31)
    initialized = torch.cuda.is_initialized()
    with _cpu_codegen():
        bound = _reused_raw._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_reused_raw, args)):
            source = bound.to_code(_config(cohorts, count))
    assert "cute.nvgpu.cpasync.tma_partition(" in source
    assert (
        "chain_generation & 1" in source
        if cohorts > 1
        else "chain_iteration & 1" in source
    )
    assert source.count("cute.arch.mbarrier_arrive_and_expect_tx(") >= 2
    assert torch.cuda.is_initialized() == initialized
    for value, old in zip(args[:5], saved, strict=True):
        _assert_bits(value, old)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,steps,masked,cohorts,count", _CASES)
def test_leaf_sets_gpu_reference_graph_replay_and_input_bits(
    dtype, steps, masked, cohorts, count
):
    args = _inputs(dtype, steps, masked, device=DEVICE)
    saved = tuple(value.clone() for value in args[:5])
    expected = _reference(args)
    bound = _reused_raw._bind_isolated(args)
    config = _config(cohorts, count)
    with bound.env.use_runtime_arg_values(_runtime_values(_reused_raw, args)):
        source = bound.to_code(config)
        assert source.count("cute.arch.mbarrier_arrive_and_expect_tx(") >= 2
        compiled = bound.compile_config(config)
    actual = compiled(*args)
    torch.cuda.synchronize()
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.cpu(), reference, atol=2e-3, rtol=2e-3)
    snapshots = tuple(value.clone() for value in actual)
    for _ in range(3):
        repeated = compiled(*args)
        for value, old in zip(repeated, snapshots, strict=True):
            _assert_bits(value, old)
    # Descriptor construction has been warmed by the original calls above;
    # graph capture/replay must use the same immutable compiled configuration.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    for _ in range(3):
        graph.replay()
        for value, old in zip(captured, snapshots, strict=True):
            _assert_bits(value, old)
    if not steps:
        _assert_bits(actual[1], saved[4])
        _assert_bits(captured[1], saved[4])
    for value, old in zip(args[:5], saved, strict=True):
        _assert_bits(value, old)
