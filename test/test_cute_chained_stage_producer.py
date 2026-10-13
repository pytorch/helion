from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_execution import _LEGACY_DIGESTS
from .test_cute_chained_execution import _digest
from .test_cute_chained_execution import _emissions
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_stage import _config as _transposed_config
from .test_cute_chained_preparation_stage import _transposed_group
from .test_cute_chained_preparation_stage import _transposed_inputs
from .test_cute_chained_register_emission import _register_config
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage as stage_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_mma_selection import warp_mma_shape
from helion._compiler.cute.chained_vector_stage import emit_vector_stage_group
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _stage_sequence(x, initial):
    steps, size, width = x.shape
    history = torch.empty((steps, size, width), dtype=torch.float32, device=x.device)
    final = torch.empty_like(initial)
    for rows, columns in hl.tile([size, width], block_size=[32, 32]):
        state = initial[rows, columns]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(width)
            raw = hl.load(
                x, [step.id, rows, kk], extra_mask=(rows.index % 3 != 1)[:, None]
            )
            shared = torch.exp(raw.float() * 0.125)
            common = (shared * 0.0625).to(x.dtype)
            first = hl.dot(
                common, (shared + 0.5).to(x.dtype).T, out_dtype=torch.float32
            )
            second = hl.dot(
                common, (shared * 0.5).to(x.dtype).T, out_dtype=torch.float32
            )
            state = state * 0.5 + first + second
            history[step.id, rows, columns] = state
        final[rows, columns] = state
    return history, final


def _descriptors(operands):
    return tuple(
        (
            item.node.name,
            item.geometry.logical,
            item.geometry.transpose,
            item.role,
            item.shape,
            item.target,
            item.offset,
        )
        for item in operands
    )


def _generated(mode, *, grouped=True, dtype=torch.bfloat16, threads=128):
    original = stage_module.emit_stage
    groups = stage_module.emit_vector_stage_group
    records = []
    rejections = []
    execution = ChainedExecution(
        threads,
        thread="owner_thread",
        warp="owner_warp",
        sync="owner_barrier.arrive_and_wait()",
        a_workspace="frame_a",
        b_workspace="frame_b",
    )

    def record_group(cg, plan, boundaries, operands, vector, **kwargs):
        records.append(_descriptors(operands))
        return groups(cg, plan, boundaries, operands, vector, **kwargs)

    def record(*args, **kwargs):
        cg, plan, boundaries, stage, geometry, phase, group, vector, unroll = args
        assert stage in plan.warp_mma_stages
        kwargs.update(
            execution=execution, prepared_shape=warp_mma_shape(geometry, group)
        )

        def delegate(operands):
            records.append(_descriptors(operands))
            assert isinstance(operands, tuple)
            with pytest.raises(FrozenInstanceError):
                operands[0].offset = 1  # pyrefly: ignore[read-only]
            return emit_vector_stage_group(
                cg,
                plan,
                boundaries,
                operands,
                vector,
                tag=f"chain_{stage}_vector_group",
                producer_unroll=unroll,
                execution=execution,
            )

        if mode == "delegate":
            return original(*args, **kwargs, operand_producer=delegate)
        if mode == "explicit_none":
            return original(*args, **kwargs, operand_producer=None)
        if mode in ("null", "empty"):
            before = dict(boundaries)
            before_flags = dict(vars(vector)), dict(vars(unroll))
            called = []

            def reject(operands):
                called.append(_descriptors(operands))
                return None if mode == "null" else []

            with (
                patch.object(
                    stage_module,
                    "_publish_result",
                    side_effect=AssertionError("result published"),
                ),
                pytest.raises(
                    chain._UnsupportedChain, match="complete operand producer rejected"
                ),
            ):
                original(*args, **kwargs, operand_producer=reject)
            assert len(called) == 1 and boundaries == before
            assert (vars(vector), vars(unroll)) == before_flags
            rejections.append(called[0])
        return original(*args, **kwargs)

    config = helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=32,
        cute_chained_group_contractions=grouped,
        cute_chained_pointwise_vectorize=True,
        cute_chained_vector_group=True,
    )
    with (
        _cpu_codegen(),
        patch.object(stage_module, "emit_stage", record),
        patch.object(stage_module, "emit_vector_stage_group", record_group),
    ):
        source = _stage_sequence._bind_isolated(
            (torch.empty((3, 35, 32), dtype=dtype), torch.empty((35, 32)))
        ).to_code(config)
    return source, records, rejections


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("threads", [128, 384])
def test_complete_producer_delegate_preserves_full_stage_source(
    dtype, grouped, threads
):
    ordinary, old_descriptors, _ = _generated(
        "ordinary", dtype=dtype, grouped=grouped, threads=threads
    )
    delegated, new_descriptors, _ = _generated(
        "delegate", dtype=dtype, grouped=grouped, threads=threads
    )
    assert delegated == ordinary and new_descriptors == old_descriptors
    assert len(new_descriptors) == (1 if grouped else 2)
    first = new_descriptors[0]
    assert [item[3] for item in first] == (["a", "b", "b"] if grouped else ["a", "b"])
    assert all(item[4] == (32, 32) for item in first)
    assert [item[6] for item in first] == ([0, 0, 32] if grouped else [0, 0])
    tree = ast.parse(delegated)
    producer = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.For)
        and ast.unparse(item.target) == "chain_0_vector_group_step"
    )
    assert "barrier" not in ast.unparse(producer)
    parent = next(
        item.body
        for item in ast.walk(tree)
        if isinstance(item, (ast.If, ast.For, ast.FunctionDef))
        and producer in item.body
    )
    index = parent.index(producer)
    assert ast.unparse(parent[index + 1]) == "cute.arch.fence_view_async_shared()"
    assert ast.unparse(parent[index + 2]) == "owner_barrier.arrive_and_wait()"
    assert "frame_a" in delegated and "frame_b" in delegated
    assert "chain_0_c[" in delegated and "cute.gemm(" in delegated


@pytest.mark.parametrize("mode", ["null", "empty"])
def test_failed_complete_producer_installs_no_result_boundary(mode):
    rejected, _, records = _generated(mode)
    ordinary, _, _ = _generated("ordinary")
    assert records and rejected == ordinary


@pytest.mark.parametrize("case", tuple(_LEGACY_DIGESTS))
def test_absent_hook_keeps_all_frozen_stage_default_bytes(case):
    original = stage_module.emit_stage

    def explicit_none(*args, **kwargs):
        return original(*args, **kwargs, operand_producer=None)

    with patch.object(stage_module, "emit_stage", explicit_none):
        assert _digest(_emissions(case)) == _LEGACY_DIGESTS[case]


def test_explicit_none_keeps_prepared_warp_source_identical():
    assert _generated("ordinary")[0] == _generated("explicit_none")[0]


@pytest.mark.parametrize("mode", ["unprepared", "tcgen", "tmem_accumulator"])
def test_nonordinary_ownership_rejects_before_invoking_producer(mode):
    original = stage_module.emit_stage
    seen = []

    def check(*args, **kwargs):
        plan, boundaries, stage, geometry = args[1:5]
        group = args[6]
        if not seen:
            attempted = list(args)
            changes = dict(kwargs)
            if mode == "tcgen":
                attempted[1] = replace(plan, warp_mma_stages=frozenset())
            elif mode == "tmem_accumulator":
                changes.update(
                    prepared_shape=warp_mma_shape(geometry, group),
                    tmem_accumulator="another_tmem_arena",
                )
            hook = Mock(side_effect=AssertionError("invalid hook invoked"))
            before = dict(boundaries)
            with pytest.raises(
                chain._UnsupportedChain, match="ordinary prepared warp stage"
            ):
                original(*attempted, **changes, operand_producer=hook)
            hook.assert_not_called()
            assert boundaries == before
            seen.append(stage)
        return original(*args, **kwargs)

    config = _transposed_config(vector=True)
    config.config["cute_chained_pointwise_unroll"] = 1
    with _cpu_codegen(), patch.object(stage_module, "emit_stage", check):
        _transposed_group._bind_isolated(_transposed_inputs(16)).to_code(config)
    assert seen


def test_transposed_group_descriptors_keep_original_modes_and_member_offsets():
    original = stage_module.emit_stage
    captured = []

    def check(*args, **kwargs):
        plan, boundaries, stage, geometry = args[1:5]
        group = args[6]
        if stage in plan.warp_mma_stages and not captured:
            before = dict(boundaries)

            def reject(operands):
                captured.append(operands)
                return None

            with pytest.raises(
                chain._UnsupportedChain, match="complete operand producer rejected"
            ):
                original(
                    *args,
                    **kwargs,
                    prepared_shape=warp_mma_shape(geometry, group),
                    operand_producer=reject,
                )
            assert boundaries == before
        return original(*args, **kwargs)

    config = _transposed_config(vector=True)
    config.config["cute_chained_pointwise_unroll"] = 1
    with _cpu_codegen(), patch.object(stage_module, "emit_stage", check):
        _transposed_group._bind_isolated(_transposed_inputs(16)).to_code(config)
    (operands,) = captured
    assert all(item.geometry.transpose for item in operands)
    assert [item.role for item in operands] == ["a", "b", "b"]
    assert [item.shape for item in operands] == [(16, 16), (16, 16), (32, 16)]
    assert [item.offset for item in operands] == [0, 0, 16]


def test_live_prepared_B_and_tmem_transports_cannot_install_complete_producer():
    original = stage_module.emit_stage
    rejected = set()

    def check(*args, **kwargs):
        selected = {
            name
            for name in (
                "prepared_operand",
                "prepared_group",
                "tmem_input",
                "tmem_carry",
            )
            if kwargs.get(name) is not None
        }
        if selected:
            boundaries = args[2]
            before = dict(boundaries)
            hook = Mock(side_effect=AssertionError("transport hook invoked"))
            with pytest.raises(
                chain._UnsupportedChain, match="ordinary prepared warp stage"
            ):
                original(*args, **kwargs, operand_producer=hook)
            hook.assert_not_called()
            assert boundaries == before
            rejected.update(selected)
        return original(*args, **kwargs)

    kernel, args = _kda_fixture()
    config = _register_config(3, cute_chained_vector_group=True)
    with patch.object(stage_module, "emit_stage", check):
        _source(kernel, args, config)
    assert {"prepared_group", "tmem_input", "tmem_carry"} <= rejected
