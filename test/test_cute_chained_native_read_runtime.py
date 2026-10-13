from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_runtime import _assert_bits
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._compiler.cute import chained_native_read_inputs as native
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _retained_sequence(a, b, initial, masked: hl.constexpr):
    steps, size, width = a.shape
    history = torch.empty((steps, size, width), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=32):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(width)
            if masked:
                raw = hl.load(
                    a,
                    [step.id, rows, kk],
                    extra_mask=(rows.index % 3 != 0)[:, None],
                ).float()
            else:
                raw = a[step.id, rows, kk].float()
            x = torch.exp(raw * 0.125).to(a.dtype)
            y = (b[step.id, rows, kk].float() + 0.25).to(a.dtype)
            product = hl.dot(x, y.T, out_dtype=torch.float32)
            shifted = (y.float() + 0.125).to(a.dtype)
            other = hl.dot(x, shifted.T, out_dtype=torch.float32)
            basis = (product * 0.001953125).to(a.dtype)
            coefficient = torch.sum(product + other, dim=1) * 0.000244140625
            shared = (x.float() + y.float()) * coefficient[:, None]
            first = (shared * 0.125).to(a.dtype)
            second = (shared * 0.25 + 0.0078125).to(a.dtype)
            state = hl.dot(basis, (state + first).to(a.dtype), out_dtype=torch.float32)
            state = hl.dot(basis, (state + second).to(a.dtype), out_dtype=torch.float32)
            history[step.id, rows, :] = state
        final[rows, :] = state
    return history, final


def _config(enabled, cohorts):
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=32,
        cute_chained_group_contractions=True,
        cute_chained_preparation_pipeline=True,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
        cute_chained_operand_retention=True,
        cute_chained_pointwise_vectorize=True,
        cute_chained_vector_group=True,
        cute_chained_native_vector_reads=enabled,
    )


def _inputs(dtype, steps, width, masked, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(76331)

    def random(shape, scale, dtype):
        return (torch.randn(shape, device=device, generator=generator) * scale).to(
            dtype
        )

    initial = random((32, width), 2**-16, torch.float32)
    initial[0, 0], initial[0, 1] = -0.0, 0.0
    return (
        random((steps, 32, width), 0.25, dtype),
        random((steps, 32, width), 0.125, dtype),
        initial,
        masked,
    )


def _reference(args):
    a, b, initial, masked = args
    state = initial.cpu().clone()
    history = []
    for step in range(a.shape[0]):
        raw = a[step].cpu().float()
        if masked:
            raw = torch.where((torch.arange(32) % 3 != 0)[:, None], raw, 0.0)
        x = torch.exp(raw * 0.125).to(a.dtype)
        y = (b[step].cpu().float() + 0.25).to(a.dtype)
        product = (x.double() @ y.double().T).float()
        shifted = (y.float() + 0.125).to(a.dtype)
        other = (x.double() @ shifted.double().T).float()
        basis = (product * 0.001953125).to(a.dtype)
        coefficient = (product + other).double().sum(1).float() * 0.000244140625
        shared = (x.float() + y.float()) * coefficient[:, None]
        first = (shared * 0.125).to(a.dtype)
        second = (shared * 0.25 + 0.0078125).to(a.dtype)
        state = (basis.double() @ (state + first).to(a.dtype).double()).float()
        state = (basis.double() @ (state + second).to(a.dtype).double()).float()
        history.append(state.clone())
    return (
        torch.stack(history) if history else torch.empty((0, *initial.shape)),
        state,
    )


_CASES = ((0, 64, False, 3), (1, 128, True, 1), (4, 64, True, 3))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps,width,masked,cohorts", _CASES)
def test_native_read_original_sequence_cpu(dtype, steps, width, masked, cohorts):
    args = _inputs(dtype, steps, width, masked)
    with patch.object(
        native, "bind_retained_native_inputs", side_effect=AssertionError("default")
    ):
        before = _source(_retained_sequence, args, _config(False, cohorts))
        missing = _config(False, cohorts)
        missing.config.pop("cute_chained_native_vector_reads")
        assert _source(_retained_sequence, args, missing) == before
    after = _source(_retained_sequence, args, _config(True, cohorts))
    assert "_group_input_0_values" in after and "_group_input_1_values" in after
    assert "_group_input_0_values" not in before
    for marker in (
        "chain_slot_bars",
        "chain_sync.arrive_mbarrier",
        "chain_allocator.allocate",
        "cute.arch.fence_view_async_shared()",
    ):
        assert before.count(marker) == after.count(marker)
    expected = _reference(args)
    assert all(torch.isfinite(value).all() for value in expected)
    if not steps:
        _assert_bits(expected[1], args[2])


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps,width,masked,cohorts", _CASES)
def test_native_read_sequence_gpu_bitwise_oracle_replay(
    dtype, steps, width, masked, cohorts
):
    args = _inputs(dtype, steps, width, masked, DEVICE)
    saved = tuple(value.clone() for value in args if isinstance(value, torch.Tensor))
    reference = _reference(args)
    implementations = []
    for enabled in (False, True):
        bound = _retained_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_retained_sequence, args)
        ):
            implementations.append(bound.compile_config(_config(enabled, cohorts)))
    ordinary = tuple(value.clone() for value in implementations[0](*args))
    for value, expected in zip(ordinary, reference, strict=True):
        torch.testing.assert_close(value.cpu(), expected, rtol=2e-3, atol=2e-3)
    for implementation in implementations:
        for _ in range(3):
            actual = implementation(*args)
            for value, expected in zip(actual, ordinary, strict=True):
                _assert_bits(value, expected)
            if not steps:
                _assert_bits(actual[1], saved[2])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(3):
            for value in captured:
                value.fill_(float("nan"))
            graph.replay()
            for value, expected in zip(captured, ordinary, strict=True):
                _assert_bits(value, expected)
    tensors = tuple(value for value in args if isinstance(value, torch.Tensor))
    for value, expected in zip(tensors, saved, strict=True):
        _assert_bits(value, expected)
