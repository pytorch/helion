from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_operand_retention_emission import _config as _base_config
from .test_cute_chained_operand_retention_emission import _typed_reuse
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _wide_typed_reuse(a, b, c, initial, masked: hl.constexpr):
    steps, size, width = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=32):
        state = initial[rows, rows]
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
            z = (c[step.id, rows, kk].float() - 0.125).to(a.dtype)
            first = hl.dot(x, y.T, out_dtype=torch.float32)
            second = hl.dot(x, z.T, out_dtype=torch.float32)
            coefficient = torch.sum(first, dim=1) * 0.000244140625
            reused = (
                (x.float() + y.float() + z.float()) * coefficient[:, None] * 0.0078125
            ).to(a.dtype)
            third = hl.dot(reused, x.T, out_dtype=torch.float32)
            combined = ((first + second + third) * 0.015625).to(a.dtype)
            state = hl.dot(state.to(a.dtype), combined, acc=state)
            history[step.id, rows, rows] = state
        final[rows, rows] = state
    return history, final


# Existing square fixture is unchanged. Short trips use independent, noncanceling
# b values; long trips use the symmetric original RHS pair to keep its unscaled
# recurrence finite in FP16. The separate wide fixture has a bounded original
# expression and exercises both K64 panels of the full grouped-B K128 image.
_CASES = (
    (False, torch.bfloat16, 0, False, False, 1),
    (False, torch.float16, 0, True, True, 3),
    (False, torch.bfloat16, 1, True, False, 3),
    (False, torch.float16, 1, False, True, 1),
    (False, torch.bfloat16, 4, False, True, 1),
    (False, torch.float16, 4, True, False, 3),
    (False, torch.bfloat16, 7, True, True, 3),
    (False, torch.float16, 7, False, False, 1),
    (True, torch.bfloat16, 0, False, False, 3),
    (True, torch.float16, 1, True, False, 1),
    (True, torch.bfloat16, 4, True, False, 1),
    (True, torch.float16, 7, False, False, 3),
)


def _inputs(wide, dtype, steps, masked, transpose, *, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(48211)

    def random(shape, scale, dtype):
        return (torch.randn(shape, generator=generator, device=device) * scale).to(
            dtype
        )

    shape = (steps, 32, 128 if wide else 32)
    a = random(shape, 0.25, dtype)
    b = random(shape, 0.125, dtype)
    initial = random((32, 32), 2**-16, torch.float32)
    initial[0, 0], initial[0, 1] = -0.0, 0.0
    if wide:
        return a, b, random(shape, 0.125, dtype), initial, masked
    if steps > 1:
        b.fill_(-0.75)
    return a, b, initial, masked, transpose


def _config(cohorts, enabled):
    return helion.Config.from_dict(
        {
            **_base_config().config,
            "cute_chained_preparation_cohorts": cohorts,
            "cute_chained_preparation_unroll": 1,
            "cute_chained_operand_retention": enabled,
        }
    )


def _reference(wide, args):
    """CPU finiteness check preserving all original typed snapshots.

    This does not establish a new numerical tolerance or replace the strict
    off/on runtime comparison. FP64 products feed the original FP32 snapshots.
    """
    inputs = tuple(
        value.detach().cpu() for value in args if isinstance(value, torch.Tensor)
    )
    a, b = inputs[:2]
    state = inputs[-1].clone()
    masked = args[-1] if wide else args[-2]
    history = []
    for step in range(a.shape[0]):
        raw = a[step].float()
        if masked:
            raw = torch.where((torch.arange(32) % 3 != 0)[:, None], raw, 0.0)
        x = torch.exp(raw * 0.125).to(a.dtype)
        y = (b[step].float() + 0.25).to(a.dtype)
        first = (x.double() @ y.double().T).float()
        if wide:
            z = (inputs[2][step].float() - 0.125).to(a.dtype)
            second = (x.double() @ z.double().T).float()
            coefficient = first.double().sum(1).float() * 0.000244140625
            reused = (
                (x.float() + y.float() + z.float()) * coefficient[:, None] * 0.0078125
            ).to(a.dtype)
            third = (reused.double() @ x.double().T).float()
            combined = ((first + second + third) * 0.015625).to(a.dtype)
        else:
            shifted = (y.float() + 1).to(a.dtype)
            second = (x.double() @ shifted.double().T).float()
            reused = x.T if args[-1] else x
            combined = (first + second + reused.float() + y.float()).to(a.dtype)
        state = (
            state.to(a.dtype).double() @ combined.double() + state.double()
        ).float()
        history.append(state.clone())
    return (
        torch.stack(history)
        if history
        else torch.empty((0, 32, 32), dtype=torch.float32),
        state,
    )


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _aliases(source):
    return {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("chain_retained_operand_")
    }


def _check_source(source, wide, enabled, cohorts):
    aliases = _aliases(source)
    if enabled:
        assert len(aliases) >= (3 if wide else 2)
        assert "chain_0_a" in aliases.values()
        assert "chain_0_b" in aliases.values()
        if wide:
            assert "cute.domain_offset((32, 0), chain_0_b)" in aliases.values()
    else:
        assert not aliases
    assert ("chain_cohort" in source) is (cohorts > 1)


@pytest.mark.parametrize("wide,dtype,steps,masked,transpose,cohorts", _CASES)
def test_operand_retention_runtime_route_and_finite_inputs_cpu(
    wide, dtype, steps, masked, transpose, cohorts
):
    args = _inputs(wide, dtype, steps, masked, transpose)
    saved = tuple(value.clone() for value in args if isinstance(value, torch.Tensor))
    expected = _reference(wide, args)
    assert expected[0].shape == (steps, 32, 32)
    assert all(torch.isfinite(value).all() for value in expected)
    if not steps:
        _assert_bits(expected[1], saved[-1])
        assert saved[-1].view(torch.int32)[0, 0] == -(1 << 31)
    kernel = _wide_typed_reuse if wide else _typed_reuse
    initialized = torch.cuda.is_initialized()
    for enabled in (False, True):
        with _cpu_codegen():
            bound = kernel._bind_isolated(args)
            with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
                source = bound.to_code(_config(cohorts, enabled))
        _check_source(source, wide, enabled, cohorts)
    assert torch.cuda.is_initialized() == initialized
    tensors = tuple(value for value in args if isinstance(value, torch.Tensor))
    for actual, original in zip(tensors, saved, strict=True):
        _assert_bits(actual, original)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("wide,dtype,steps,masked,transpose,cohorts", _CASES)
def test_operand_retention_gpu_exact_outputs_replay_graph_and_inputs(
    wide, dtype, steps, masked, transpose, cohorts
):
    args = _inputs(wide, dtype, steps, masked, transpose, device=DEVICE)
    tensors = tuple(value for value in args if isinstance(value, torch.Tensor))
    saved = tuple(value.clone() for value in tensors)
    kernel = _wide_typed_reuse if wide else _typed_reuse
    implementations = []
    for enabled in (False, True):
        # Isolated binding prevents one configuration from overwriting the other.
        bound = kernel._bind_isolated(args)
        config = _config(cohorts, enabled)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            _check_source(bound.to_code(config), wide, enabled, cohorts)
            implementations.append(bound.compile_config(config))
    assert implementations[0] is not implementations[1]
    ordinary = tuple(value.clone() for value in implementations[0](*args))
    assert all(torch.isfinite(value).all() for value in ordinary)
    for implementation in implementations:
        actual = implementation(*args)
        torch.cuda.synchronize()
        for value, expected in zip(actual, ordinary, strict=True):
            _assert_bits(value, expected)
        snapshots = tuple(value.clone() for value in actual)
        for _ in range(3):
            repeated = implementation(*args)
            for value, expected in zip(repeated, snapshots, strict=True):
                _assert_bits(value, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(3):
            graph.replay()
            for value, expected in zip(captured, snapshots, strict=True):
                _assert_bits(value, expected)
        if not steps:
            _assert_bits(actual[1], saved[-1])
            _assert_bits(captured[1], saved[-1])
        for value, original in zip(tensors, saved, strict=True):
            _assert_bits(value, original)
