from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_frontier_emission import _sequence_config
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_register_runtime import _assert_bits
import helion
from helion import exc
from helion._compiler.cute.chained_frontier_ownership import FrontierOwnership
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

_CASES = [(0, True, 3), (1, False, 1), (4, True, 3), (7, False, 1)]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_recurrence(
    a,
    b,
    c,
    initial_left,
    initial_right,
    masked: hl.constexpr,
    dependent: hl.constexpr,
):
    steps, height, width = a.shape
    left = torch.empty_like(initial_left)
    right = torch.empty_like(initial_right)
    # Both short matrix axes share the same explicit tile domain. The two
    # carried outputs still have opposite orientations and distinct storage.
    for short_rows, rows in hl.tile([width, height], block_size=[32, 128]):
        state_left = initial_left[short_rows, rows]
        state_right = initial_right[rows, short_rows]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(width)
            product = hl.dot(
                b[step.id, short_rows, kk],
                c[step.id, kk, short_rows],
                out_dtype=torch.float32,
            )
            if masked:
                raw = hl.load(
                    b,
                    [step.id, short_rows, short_rows],
                    extra_mask=(short_rows.index % 3 != 0)[:, None],
                )
            else:
                raw = b[step.id, short_rows, short_rows]
            shared = product * torch.exp(raw.float() * 0.125)
            first = shared.to(a.dtype)
            if dependent:
                shared = first.float()
            second = (shared + 0.125).to(a.dtype)
            common = a[step.id, rows, short_rows]
            state_left = hl.dot(
                first, common.T, acc=state_left, out_dtype=torch.float32
            )
            state_right = hl.dot(
                common, second, acc=state_right, out_dtype=torch.float32
            )
            # Publish each recurrence result, including its final zero-trip
            # value below, as required by the admitted lexical-loop contract.
            left[short_rows, rows] = state_left
            right[rows, short_rows] = state_right
        left[short_rows, rows] = state_left
        right[rows, short_rows] = state_right
    return left, right


def _inputs(dtype, steps, masked, device="cpu", *, dependent=False):
    generator = torch.Generator(device=device).manual_seed(55173)

    def random(shape, dtype):
        return (torch.randn(shape, device=device, generator=generator) * 0.125).to(
            dtype
        )

    return (
        random((steps, 128, 32), dtype),
        random((steps, 32, 32), dtype),
        random((steps, 32, 32), dtype),
        random((32, 128), torch.float32),
        random((128, 32), torch.float32),
        masked,
        dependent,
    )


def _config(columns, cohorts):
    config = _sequence_config(cohorts)
    config.config.update(
        cute_chained_group_contractions=True,
        cute_chained_frontier_tile_columns=columns,
    )
    return config


def _reference(args):
    a, b, c, left, right, masked, dependent = args
    left, right = left.clone(), right.clone()
    for step in range(a.shape[0]):
        product = (b[step].double() @ c[step].double()).float()
        raw = b[step].float()
        if masked:
            raw = torch.where(
                (torch.arange(32, device=a.device) % 3 != 0)[:, None], raw, 0
            )
        shared = product * torch.exp(raw * 0.125)
        first = shared.to(a.dtype)
        if dependent:
            shared = first.float()
        second = (shared + 0.125).to(a.dtype)
        left = (first.double() @ a[step].double().T + left.double()).float()
        right = (a[step].double() @ second.double() + right.double()).float()
    return left, right


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps,masked,cohorts", _CASES)
def test_mixed_native_recurrence_selects_general_frontier_ownership_cpu(
    dtype, steps, masked, cohorts
):
    args = _inputs(dtype, steps, masked)
    before = _source(_mixed_recurrence, args, _config(0, cohorts))
    selected = []
    original = FrontierOwnership.select

    def select(policy, frame, group, prepared_operands, prepared_groups, threads):
        result = original(
            policy, frame, group, prepared_operands, prepared_groups, threads
        )
        if result is not None:
            nodes = {buffer.node for buffer in group.buffers}
            members = [
                member
                for binding in prepared_groups
                for member in binding.candidate.members
                if member.buffer.node in nodes
            ]
            assert len(members) == len(nodes) == 2
            assert {member.logical_modes for member in members} == {(0, 1), (1, 0)}
            assert all(member.buffer.dtype == dtype for member in members)
            assert result.matches((32, 32), threads)
            assert result.tile_columns == 16 and result.changed
            selected.append(result)
        return result

    initialized = torch.cuda.is_initialized()
    with patch.object(FrontierOwnership, "select", select):
        after = _source(_mixed_recurrence, args, _config(16, cohorts))
    assert torch.cuda.is_initialized() == initialized
    assert selected
    assert before != after
    assert "_group_step * 16" in after
    assert "stride=(2, 1)" in after
    assert "chain_prepared_group_" in after
    assert all(torch.isfinite(value).all() for value in _reference(args))


def test_snapshot_dependent_frontier_remains_ineligible_cpu():
    args = _inputs(torch.bfloat16, 3, False, dependent=True)
    assert _source(_mixed_recurrence, args, _config(0, 1))
    with pytest.raises(exc.BackendUnsupported, match="mixed-native frontier group"):
        _source(_mixed_recurrence, args, _config(16, 1))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps,masked,cohorts", _CASES)
def test_mixed_native_recurrence_gpu_bits_reference_graph_and_inputs(
    dtype, steps, masked, cohorts
):
    args = _inputs(dtype, steps, masked, DEVICE)
    saved = tuple(value.clone() for value in args[:5])
    implementations = []
    for columns in (0, 16):
        bound = _mixed_recurrence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_mixed_recurrence, args)):
            implementations.append(bound.compile_config(_config(columns, cohorts)))
    expected = tuple(value.clone() for value in implementations[0](*args))
    torch.testing.assert_close(expected, _reference(args), rtol=2e-3, atol=2e-3)
    assert all(torch.isfinite(value).all() for value in expected)
    for implementation in implementations:
        for _ in range(3):
            actual = implementation(*args)
            for value, reference in zip(actual, expected, strict=True):
                _assert_bits(value, reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(3):
            graph.replay()
            for value, reference in zip(captured, expected, strict=True):
                _assert_bits(value, reference)
        if steps == 0:
            for value, original in zip(captured, saved[-2:], strict=True):
                _assert_bits(value, original)
        for value, original in zip(args[:5], saved, strict=True):
            _assert_bits(value, original)
