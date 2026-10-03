from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_row_collective_emission import _row_loop
import helion
from helion import exc
from helion._compiler.cute import chained_row_collective_emission as emission
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

# Nonlinear cases use one step: this original, deliberately unscaled recurrence
# otherwise overflows FP16. Linear five/seven-step cases exercise slot reuse.
_CASES = (
    (torch.bfloat16, 0, 48, False, False, False, 1),
    (torch.float16, 0, 64, True, False, True, 3),
    (torch.bfloat16, 1, 16, True, True, False, 3),
    (torch.float16, 1, 16, False, True, True, 1),
    (torch.bfloat16, 5, 48, False, False, True, 1),
    (torch.float16, 5, 64, True, False, False, 3),
    (torch.bfloat16, 7, 64, True, False, False, 3),
    (torch.float16, 7, 48, False, False, True, 1),
)


def _inputs(
    dtype, steps, width, keepdim, nonlinear, masked, *, device="cpu", cross=False
):
    generator = torch.Generator(device=device).manual_seed(72031)

    def random(shape, dtype):
        return (torch.randn(shape, generator=generator, device=device) * 0.0625).to(
            dtype
        )

    initial = random((16, 16), torch.float32) * 0.25
    initial[0, 0], initial[0, 1] = -0.0, 0.0
    return (
        random((steps, 16, width), dtype),
        random((steps, 16, width), dtype),
        initial,
        keepdim,
        nonlinear,
        cross,
        masked,
    )


def _config(cohorts, enabled):
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
        cute_chained_collective_retention=enabled,
    )


def _reference(args):
    """Independent FP64 products with original typed pointwise/dot snapshots.

    Sum accuracy is checked with the existing 2e-3 oracle, not cross-config
    bitwise equality: default NVVM contraction remains the original policy.
    Every explicit FP16/BF16 cast and each FP32 accumulator version is kept.
    """
    a, b, initial = (value.detach().cpu() for value in args[:3])
    keepdim, nonlinear, cross, masked = args[3:]
    state = initial.clone()
    history = []
    for step in range(a.shape[0]):
        raw_x = a[step].float()
        if masked:
            raw_x = torch.where((torch.arange(16) % 3 != 0)[:, None], raw_x, 0.0)
        raw_y = b[step].float()
        x = (raw_x.double() * 0.125).float().to(torch.float16).float()
        y = (raw_y.double() * 0.25).float().to(torch.bfloat16).float()
        if nonlinear:
            x = torch.sigmoid(x.double()).float()
            y = torch.exp(y.double()).float()
        sx = (x.double() * x.double()).float().double().sum(1, keepdim=keepdim).float()
        sy = (y.double() + y.double()).float().double().sum(1, keepdim=keepdim).float()
        expanded_sx = sx if keepdim else sx[:, None]
        expanded_sy = sy if keepdim else sy[:, None]
        xout = x.T if cross else x
        left = (xout.double() + expanded_sx.double()).float().to(a.dtype)
        right = (y.double() + expanded_sy.double()).float().to(a.dtype)
        first = (left.double() @ right.double().T).float()
        # The second operand starts from the already narrowed original right.
        shifted_right = (right.double() + 1).float().to(a.dtype)
        second = (left.double() @ shifted_right.double().T).float()
        snapshot = (first.double() + second.double()).float().to(a.dtype)
        state = (
            state.to(a.dtype).double() @ snapshot.double() + state.double()
        ).float()
        history.append(state.clone())
    return (
        torch.stack(history)
        if history
        else torch.empty((0, 16, 16), dtype=torch.float32),
        state,
    )


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _retained_arrays(source):
    return tuple(
        assignment.targets[0].id
        for assignment in ast.walk(ast.parse(source))
        if isinstance(assignment, ast.Assign)
        and len(assignment.targets) == 1
        and isinstance(assignment.targets[0], ast.Name)
        and assignment.targets[0].id.startswith("chain_row_collective_")
        and "_retained_" in assignment.targets[0].id
        and isinstance(assignment.value, ast.Call)
        and ast.unparse(assignment.value.func) == "cute.make_rmem_tensor"
    )


def _source(args, cohorts, enabled):
    initialized = torch.cuda.is_initialized()
    with _cpu_codegen():
        bound = _row_loop._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_row_loop, args)):
            source = bound.to_code(_config(cohorts, enabled))
    assert torch.cuda.is_initialized() == initialized
    return source


@pytest.mark.parametrize("dtype,steps,width,keepdim,nonlinear,masked,cohorts", _CASES)
def test_row_retention_runtime_route_and_original_reference_preflight_cpu(
    dtype, steps, width, keepdim, nonlinear, masked, cohorts
):
    args = _inputs(dtype, steps, width, keepdim, nonlinear, masked)
    saved = tuple(value.clone() for value in args[:3])
    expected = _reference(args)
    assert expected[0].shape == (steps, 16, 16)
    assert all(torch.isfinite(value).all() for value in expected)
    if not steps:
        _assert_bits(expected[1], args[2])
        assert args[2].view(torch.int32)[0, 0] == -(1 << 31)
    ordinary = _source(args, cohorts, False)
    retained = _source(args, cohorts, True)
    assert not _retained_arrays(ordinary)
    assert len(_retained_arrays(retained)) == 2
    assert "chain_cohort" in retained if cohorts > 1 else "chain_cohort" not in retained
    for value, old in zip(args[:3], saved, strict=True):
        _assert_bits(value, old)


@pytest.mark.parametrize("enabled", [False, True])
def test_cross_row_frontend_domain_guard_is_not_bypassed_cpu(enabled):
    args = _inputs(torch.bfloat16, 1, 16, False, False, True, cross=True)
    # A concrete square shape does not make the tiled-row and constant-column
    # symbolic domains interchangeable. This existing frontend guard runs
    # before retention; do not repair the original graph to reach the emitter.
    with pytest.raises(exc.ControlFlowTensorMismatch, match="xout"):
        _source(args, 1, enabled)
    assert all(torch.isfinite(value).all() for value in _reference(args))


def test_ordinary_sparse_reduction_cannot_be_replaced_by_dense_retention_cpu():
    args = _inputs(torch.float16, 1, 32, False, False, False)
    with patch.object(
        emission, "emit_sparse_reduction", return_value=["original_sparse"]
    ):
        with pytest.raises(exc.BackendUnsupported, match="retention"):
            _source(args, 1, True)
        assert not _retained_arrays(_source(args, 1, False))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,steps,width,keepdim,nonlinear,masked,cohorts", _CASES)
def test_row_retention_gpu_fp64_replay_graph_and_input_bits(
    dtype, steps, width, keepdim, nonlinear, masked, cohorts
):
    args = _inputs(dtype, steps, width, keepdim, nonlinear, masked, device=DEVICE)
    saved = tuple(value.clone() for value in args[:3])
    reference = _reference(args)
    implementations = []
    for enabled in (False, True):
        # Isolated binding prevents one config from replacing another's cache.
        bound = _row_loop._bind_isolated(args)
        config = _config(cohorts, enabled)
        with bound.env.use_runtime_arg_values(_runtime_values(_row_loop, args)):
            source = bound.to_code(config)
            assert len(_retained_arrays(source)) == (2 if enabled else 0)
            implementations.append(bound.compile_config(config))
    assert implementations[0] is not implementations[1]
    ordinary = tuple(value.clone() for value in implementations[0](*args))
    for implementation in implementations:
        actual = implementation(*args)
        torch.cuda.synchronize()
        for value, expected, baseline in zip(actual, reference, ordinary, strict=True):
            torch.testing.assert_close(value.cpu(), expected, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(value, baseline, atol=2e-3, rtol=2e-3)
        snapshots = tuple(value.clone() for value in actual)
        for _ in range(3):
            repeated = implementation(*args)
            for value, old in zip(repeated, snapshots, strict=True):
                _assert_bits(value, old)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(3):
            graph.replay()
            for value, old in zip(captured, snapshots, strict=True):
                _assert_bits(value, old)
        if not steps:
            _assert_bits(actual[1], saved[2])
            _assert_bits(captured[1], saved[2])
        for value, old in zip(args[:3], saved, strict=True):
            _assert_bits(value, old)
