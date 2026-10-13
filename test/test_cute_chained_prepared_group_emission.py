from __future__ import annotations

import ast
from contextlib import contextmanager
import re
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_prepared_groups as groups
from helion._compiler.cute import chained_prepared_values as values
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _group_sequence(a0, b0, a1, b1, common, initial0, initial1):
    steps = common.shape[0]
    history0 = torch.empty((steps, 32, 128), device=common.device, dtype=torch.float32)
    history1 = torch.empty((steps, 128, 64), device=common.device, dtype=torch.float32)
    final0, final1 = torch.empty_like(initial0), torch.empty_like(initial1)
    for rr, cc in hl.tile([32, 128], block_size=[32, 128]):
        state0, state1 = initial0[rr, cc], initial1[cc, :]
        for step in hl.tile(steps, block_size=1):
            jj, kk, nn = hl.arange(32), hl.arange(16), hl.arange(64)
            wide = hl.arange(256)
            first = hl.dot(a0[step.id, rr, wide], b0[step.id, wide, jj]).to(a0.dtype)
            second = hl.dot(a1[step.id, jj, kk], b1[step.id, kk, nn]).to(a1.dtype)
            shared = common[step.id, cc, jj]
            state0 = hl.dot(first, shared.T, acc=state0)
            state1 = hl.dot(shared, second, acc=state1)
            history0[step.id, rr, cc] = state0
            history1[step.id, cc, nn] = state1
        final0[rr, cc], final1[cc, :] = state0, state1
    return history0, history1, final0, final1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tail_group_sequence(a0, b0, a1, b1, common, initial0, initial1):
    steps = common.shape[0]
    history0 = torch.empty(
        ((steps + 31) // 32, 32, 128), device=common.device, dtype=torch.float32
    )
    history1 = torch.empty(
        ((steps + 31) // 32, 128, 64), device=common.device, dtype=torch.float32
    )
    final0, final1 = torch.empty_like(initial0), torch.empty_like(initial1)
    for rr, cc in hl.tile([32, 128], block_size=[32, 128]):
        state0, state1 = initial0[rr, cc], initial1[cc, :]
        for time in hl.tile(steps, block_size=32):
            kk, nn = hl.arange(16), hl.arange(64)
            wide = hl.arange(256)
            first = torch.reciprocal(hl.dot(a0[rr, wide], b0[wide, time])).to(a0.dtype)
            second = hl.dot(a1[time, kk], b1[kk, nn]).to(a1.dtype)
            shared = common[time, cc].T
            state0 = hl.dot(first, shared.T, acc=state0)
            state1 = hl.dot(shared, second, acc=state1)
            history0[time.id, rr, cc] = state0
            history1[time.id, cc, nn] = state1
        final0[rr, cc], final1[cc, :] = state0, state1
    return history0, history1, final0, final1


def _inputs(device, dtype, steps=3, *, tail=False):
    torch.manual_seed(9741)
    shapes = (
        ((32, 256), (256, steps), (steps, 16), (16, 64), (steps, 128))
        if tail
        else (
            (steps, 32, 256),
            (steps, 256, 32),
            (steps, 32, 16),
            (steps, 16, 64),
            (steps, 128, 32),
        )
    )
    operands = tuple(
        torch.rand(shape, device=device, dtype=dtype) + 0.25
        if tail
        else torch.randn(shape, device=device, dtype=dtype) * 0.125
        for shape in shapes
    )
    return (
        *operands,
        torch.randn((32, 128), device=device) * 0.125,
        torch.randn((128, 64), device=device) * 0.125,
    )


def _group_config(consumer_warps=4):
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    config.config["cute_chained_group_contractions"] = True
    return config


@contextmanager
def _capture_groups():
    original_candidates, original_place = (
        groups.prepared_group_candidates,
        groups.coallocate_prepared_groups,
    )
    original_bind = values.bind_frame_buffers
    captured = {"candidates": [], "placements": [], "frames": []}

    def candidates(*args, **kwargs):
        result = original_candidates(*args, **kwargs)
        assert result is not None
        captured["candidates"].extend(result)
        return result

    def place(*args, **kwargs):
        result = original_place(*args, **kwargs)
        captured["placements"].append((args[1], result))
        return result

    def bind(*args, **kwargs):
        captured["frames"].append((args[0], kwargs.get("prepared_groups", ())))
        return original_bind(*args, **kwargs)

    with (
        patch.object(groups, "prepared_group_candidates", candidates),
        patch.object(groups, "coallocate_prepared_groups", place),
        patch.object(values, "bind_frame_buffers", bind),
    ):
        yield captured


def _source(kernel, args, config):
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            return bound.to_code(config)


def _assignment(source, name):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]


def _stable_source(source):
    # A rejected late proof may consume fresh temporary names but must not
    # change emitted operations, masks, layouts, barriers or loop structure.
    names = {}
    return re.sub(
        r"\b(?:chain_value|v)_\d+\b",
        lambda match: names.setdefault(match[0], f"chain_value_{len(names)}"),
        source,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("consumer_warps", [4, 8])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_complete_group_cpu_aliases_both_orientations_and_rebinds_entire_frame(
    dtype, consumer_warps, steps
):
    with _capture_groups() as captured:
        source = _source(
            _group_sequence, _inputs("cpu", dtype, steps), _group_config(consumer_warps)
        )
    assert captured["candidates"]
    before, placement = captured["placements"][-1]
    assert placement is not None and len(placement.groups) == 1
    binding = placement.groups[0]
    candidate = binding.candidate
    assert tuple(member.logical_modes for member in candidate.members) == (
        (0, 1),
        (1, 0),
    )
    assert candidate.physical_shape == (96, 32)
    assert placement.frame.actions is before.actions
    assert placement.frame.frontier_order is before.frontier_order
    assert placement.frame.buffers is before.buffers
    assert all(
        frame is placement.frame and bindings == placement.groups
        for frame, bindings in captured["frames"]
    )
    for stage in placement.frame.stages:
        assert stage.a is placement.frame.layout.region(stage.a.name)
        assert stage.b is placement.frame.layout.region(stage.b.name)
    prefix = f"chain_{candidate.group.stages[0]}"
    assert f"{prefix}_b = {candidate.name}" in source
    assert f"{prefix}_b_ptr" not in source
    assert all(
        f"{prefix}_b_{member.stage}_step" not in source for member in candidate.members
    )
    assert (
        f"chain_frames = cute.arch.alloc_smem(cutlass.Uint8, {2 * placement.frame.layout.allocated_bytes}"
        in source
    )
    for member in candidate.members:
        region = placement.frame.layout.region(member.buffer.name)
        assert region.byte_offset == binding.byte_offset + member.byte_offset
        assignments = _assignment(source, f"{member.buffer.name}_ptr")
        assert len(assignments) == 2
        assert all(
            f"chain_frame + {region.byte_offset}" in ast.unparse(item.value)
            for item in assignments
        )
        assert member.buffer.name + "_layout.inner" in source
    transposed = candidate.members[1].buffer.name
    assert f"cute.select({transposed}_layout.outer, mode=[1, 0])" in source
    roles = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == f"chain_thread < {512 - 32 * consumer_warps}"
    )
    producer = ast.unparse(ast.Module(body=roles.body, type_ignores=[]))
    assert (
        producer.rindex("cute.arch.fence_view_async_shared()")
        < producer.rindex("chain_prep_barrier.arrive_and_wait()")
        < producer.rindex("chain_sync.arrive_mbarrier(")
    )


@pytest.mark.parametrize("reason", ["mask", "unsupported", "capacity"])
def test_whole_group_cpu_fallback_never_partially_rewrites_members(reason):
    args, config = _inputs("cpu", torch.bfloat16), _group_config()
    with patch.object(groups, "prepared_group_candidates", return_value=()):
        ordinary = _source(_group_sequence, args, config)
    original_domain = chain._operand_domain
    rejected = []

    def domain(cg, node, coords, plan):
        if coords == ("chain_prepared_k", "chain_prepared_row"):
            rejected.append(node)
            if reason == "unsupported":
                raise chain._UnsupportedChain("test unproved complete group domain")
            return "chain_prepared_k < 31"
        return original_domain(cg, node, coords, plan)

    with (
        patch.object(chain, "_operand_domain", domain)
        if reason != "capacity"
        else patch.object(groups, "coallocate_prepared_groups", return_value=None)
    ):
        source = _source(_group_sequence, args, config)
    assert reason == "capacity" or rejected
    assert _stable_source(source) == _stable_source(ordinary)
    assert "chain_prepared_group_" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_real_inherited_tail_cpu_rejects_complete_group_and_keeps_original_mask(dtype):
    args, config = _inputs("cpu", dtype, 5, tail=True), _group_config()
    with _capture_groups() as captured:
        source = _source(_tail_group_sequence, args, config)
    assert captured["candidates"]
    candidate = captured["candidates"][0]
    assert captured["placements"][-1][1] is not None
    assert captured["placements"][-1][1].groups == ()
    assert "chain_prepared_group_" not in source
    prefix = f"chain_{candidate.group.stages[0]}"
    assert f"{prefix}_b_ptr = chain_b_workspace" in source
    assert all(
        f"{prefix}_b_{member.stage}_step" in source for member in candidate.members
    )
    with patch.object(groups, "prepared_group_candidates", return_value=()):
        ordinary = _source(_tail_group_sequence, args, config)
    assert _stable_source(source) == _stable_source(ordinary)


def _compile_pair(kernel, args, config, *, activates):
    bound = kernel._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
        patch.object(groups, "prepared_group_candidates", return_value=()),
    ):
        ordinary = bound.compile_config(config)
    bound = kernel._bind_isolated(args)
    with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
        source = bound.to_code(config)
        assert ("chain_prepared_group_" in source) is activates
        native = bound.compile_config(config)
    return ordinary, native


def _reference(args):
    a0, b0, a1, b1, common, initial0, initial1 = args
    state0, state1 = initial0.clone(), initial1.clone()
    history0, history1 = [], []
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for step in range(common.shape[0]):
            first = (a0[step].float() @ b0[step].float()).to(a0.dtype)
            second = (a1[step].float() @ b1[step].float()).to(a1.dtype)
            state0 = first.float() @ common[step].float().T + state0
            state1 = common[step].float() @ second.float() + state1
            history0.append(state0)
            history1.append(state1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return (
        torch.stack(history0) if history0 else initial0.new_empty((0, *initial0.shape)),
        torch.stack(history1) if history1 else initial1.new_empty((0, *initial1.shape)),
        state0,
        state1,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1])
def test_native_precision_reference_cpu_restores_context_and_handles_empty_loop(
    dtype, steps
):
    args = _inputs("cpu", dtype, steps)
    saved = tuple(arg.clone() for arg in args)
    previous = torch.backends.cuda.matmul.allow_tf32
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = _reference(args)
        torch.testing.assert_close(_reference(args), result, atol=0, rtol=0)
    assert torch.backends.cuda.matmul.allow_tf32 == previous
    assert result[0].shape == (steps, 32, 128)
    assert result[1].shape == (steps, 128, 64)
    if not steps:
        torch.testing.assert_close(result[2:], args[-2:], atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_complete_group_gpu_preserves_typed_recurrence_replay_and_inputs(
    dtype, steps, consumer_warps
):
    args = _inputs(DEVICE, dtype, steps)
    saved = tuple(arg.clone() for arg in args)
    ordinary, native = _compile_pair(
        _group_sequence, args, _group_config(consumer_warps), activates=True
    )
    assert ordinary is not native
    expected, actual = ordinary(*args), native(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(native(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
    torch.testing.assert_close(actual, _reference(args), atol=2e-4, rtol=2e-4)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype,steps", [(torch.bfloat16, 5), (torch.float16, 97)])
def test_inherited_tail_group_gpu_fallback_retains_finite_masked_values(dtype, steps):
    args = _inputs(DEVICE, dtype, steps, tail=True)
    saved = tuple(arg.clone() for arg in args)
    ordinary, candidate = _compile_pair(
        _tail_group_sequence, args, _group_config(), activates=False
    )
    expected, actual = ordinary(*args), candidate(*args)
    assert all(torch.isfinite(value).all() for value in expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(candidate(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
