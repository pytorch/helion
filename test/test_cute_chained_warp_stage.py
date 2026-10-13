from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_execution import _emissions
from .test_cute_chained_matmul import _offset_scaled_operand
from .test_cute_chained_matmul import _plain_chain
from .test_cute_chained_matmul import _scaled_operand
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_warp_stage as roots
from helion._compiler.cute import chained_warp_mma as mma
from helion._compiler.cute import chained_warp_stage as shared


def _code(dtype=torch.bfloat16, schedule="cp_async", warps=8, columns=16):
    with _cpu_codegen():
        bound = _offset_scaled_operand._bind_isolated(
            (
                torch.empty((2, 256, 32), dtype=dtype),
                torch.empty((2, 256, 64), dtype=dtype),
                torch.empty((2, 2, 128), dtype=torch.float32),
            )
        )
        return bound.to_code(
            helion.Config(
                block_sizes=[16, columns],
                num_warps=warps,
                cute_chained_mma_schedule=schedule,
            )
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("cp_async", "cp_async_register_reuse"))
@pytest.mark.parametrize("warps,columns", ((2, 16), (8, 16), (8, 64)))
def test_root_source_exact_and_shared_executor(dtype, schedule, warps, columns):
    initialized = torch.cuda.is_initialized()
    with patch.object(roots, "root_warp_stage_action", return_value=None):
        old = _code(dtype, schedule, warps, columns)
    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as call:
        new = _code(dtype, schedule, warps, columns)
    assert old == new
    assert call.call_count == 1
    prepared = call.call_args.args[3]
    assert prepared.shape == (16, columns, 128)
    assert prepared.completion.execution.threads == warps * 32
    assert prepared.threads == 32 * min(warps, 2 ** (columns.bit_length() - 4))
    assert prepared.completion._consumed
    assert torch.cuda.is_initialized() == initialized


def test_loop_uses_same_executor_and_recorded_emission_digest():
    import hashlib
    import json

    from .test_cute_chained_execution import _LEGACY_DIGESTS

    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as call:
        records = _emissions("warp")
    assert call.call_count > 0
    assert all(
        isinstance(item.args[3].result, shared.WarpMemberResult)
        for item in call.call_args_list
    )
    assert (
        hashlib.sha256(json.dumps(records).encode()).hexdigest()
        == _LEGACY_DIGESTS["warp"]
    )


def test_actual_copy_result_controls_wait_not_schedule():
    def code():
        with _cpu_codegen():
            bound = _scaled_operand._bind_isolated(
                (
                    torch.empty((2, 128, 64), dtype=torch.bfloat16),
                    torch.empty((2, 128, 64), dtype=torch.bfloat16),
                    torch.empty((2, 128), dtype=torch.float32),
                )
            )
            return bound.to_code(
                helion.Config(num_warps=4, cute_chained_mma_schedule="cp_async")
            )

    for copied in (False, True):
        with (
            patch.object(chain, "_async_copy", return_value=None)
            if not copied
            else patch.object(chain, "_async_copy", wraps=chain._async_copy)
        ):
            with patch.object(roots, "root_warp_stage_action", return_value=None):
                old = code()
            new = code()
        assert old == new
        assert ("cp_async_wait_group(0)" in new) == copied
        assert "cute.arch.sync_threads()" in new


@pytest.mark.parametrize(
    "schedule", ("coalesced", "cp_async_register", "k_major", "k_major_padded")
)
def test_unported_schedules_keep_original_route(schedule):
    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as call:
        code = _code(schedule=schedule, warps=4)
    assert "chain_0_mma" in code
    assert call.call_count == 0


def test_multistage_register_bridge_preserves_original_source():
    from helion._compiler.cute import chained_warp_bridge

    with (
        _cpu_codegen(),
        patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as call,
    ):
        bound = _plain_chain._bind_isolated(
            (
                torch.empty((2, 32, 128), dtype=torch.bfloat16),
                torch.empty((2, 128, 128), dtype=torch.bfloat16),
                torch.empty((2, 128, 32), dtype=torch.bfloat16),
            )
        )
        config = helion.Config(
            num_warps=4, cute_chained_mma_schedule="cp_async_register"
        )
        with patch.object(
            chained_warp_bridge, "root_warp_bridge_sequence", return_value=None
        ):
            old = bound.to_code(config)
        call.reset_mock()
        code = bound.to_code(config)
    assert "chain_1_mma" in code
    assert code == old
    assert call.call_count == 2
    assert "chain_0_c_ptr =" not in code
    assert "chain_0_sc =" not in code


@pytest.mark.parametrize("change", ("map", "cg", "axis", "kwargs", "shape", "config"))
def test_actual_root_context_rejected_before_builder(change):
    original = roots.RootWarpStageAction.emit

    def changed(self, cg, plan, boundaries, *args):
        if change == "map":
            boundaries = {**boundaries, plan.dots[0].args[0]: "unapproved"}
        elif change == "cg":
            cg = object()
        elif change == "axis":
            object.__setattr__(plan, "axes", (*plan.axes, (999, 32, 16)))
        elif change == "kwargs":
            plan.dots[0].kwargs = {**plan.dots[0].kwargs, "unapproved": True}
        elif change == "shape":
            object.__setattr__(plan, "shapes", ((16, 32, 128),))
        else:
            cg.device_function.config.config["unapproved"] = True
        return original(self, cg, plan, boundaries, *args)

    with (
        patch.object(roots.RootWarpStageAction, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="root warp action changed"),
    ):
        _code()


@pytest.mark.parametrize(
    "change",
    (
        "thread",
        "warp",
        "threads",
        "sync",
        "a_workspace",
        "b_workspace",
        "tmem",
        "barriers",
        "wait",
        "result",
        "active",
        "axis",
    ),
)
def test_prepared_context_and_complete_input_prefix_reject(change):
    original = shared.emit_prepared_warp_stage

    def changed(cg, plan, boundaries, prepared, prefix):
        if change == "wait":
            prefix = prefix[:-1]
        elif change == "result":
            object.__setattr__(prepared.result, "elements", 1)
        elif change == "active":
            prepared = replace(prepared, threads=32)
        elif change == "axis":
            prepared = replace(prepared, axes=tuple(1 - axis for axis in prepared.axes))
        else:
            value = 128 if change == "threads" else "unapproved"
            object.__setattr__(prepared.completion.execution, change, value)
        return original(cg, plan, boundaries, prepared, prefix)

    with (
        patch.object(shared, "emit_prepared_warp_stage", changed),
        pytest.raises(exc.BackendUnsupported, match="prepared warp"),
    ):
        _code()


def test_completion_is_once_only_and_failure_does_not_publish():
    original = shared.emit_prepared_warp_stage
    checks = []

    def checked(cg, plan, boundaries, prepared, prefix):
        before = dict(boundaries)
        with pytest.raises(chain._UnsupportedChain):
            original(cg, plan, boundaries, prepared, prefix[:-1])
        assert boundaries == before and not prepared.completion._consumed
        lines = original(cg, plan, boundaries, prepared, prefix)
        with pytest.raises(chain._UnsupportedChain):
            original(cg, plan, before, prepared, prefix)
        checks.append(True)
        return lines

    with patch.object(shared, "emit_prepared_warp_stage", checked):
        _code()
    assert checks == [True]


@pytest.mark.parametrize(
    "change", ("lines", "copies", "axes", "asynchronous", "stride")
)
def test_original_operand_receipt_rejects_mutation_before_join(change):
    original = roots.emit_original_warp_operands

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "stride":
            object.__setattr__(result.layouts[0], "stride", (1, 1))
        else:
            values = {
                "lines": (),
                "copies": (("unapproved",),),
                "axes": (),
                "asynchronous": not result.asynchronous,
            }
            object.__setattr__(result, change, values[change])
        return result

    with (
        patch.object(roots, "emit_original_warp_operands", changed),
        pytest.raises(exc.BackendUnsupported, match="root warp operand selection"),
    ):
        _code()


def test_equal_valued_replacement_completion_latch_rejects():
    original = shared.emit_prepared_warp_stage

    def changed(cg, plan, boundaries, prepared, prefix):
        object.__setattr__(prepared.completion, "_state", shared._CompletionState())
        return original(cg, plan, boundaries, prepared, prefix)

    with (
        patch.object(shared, "emit_prepared_warp_stage", changed),
        pytest.raises(exc.BackendUnsupported, match="prepared warp stage changed"),
    ):
        _code()


@pytest.mark.parametrize("route", ("root", "loop"))
@pytest.mark.parametrize("phase", ("before", "during"))
@pytest.mark.parametrize("field", ("schedule", "blocks"))
def test_late_config_mutation_rejects_without_publication(route, phase, field):
    original = shared.emit_prepared_warp_stage
    checked = []

    def check(cg, plan, boundaries, prepared, prefix):
        before = dict(boundaries)
        config = cg.device_function.config.config
        saved = deepcopy(config)

        def mutate():
            if field == "schedule":
                config["cute_chained_mma_schedule"] = "coalesced"
            else:
                blocks = config["block_sizes"]
                assert isinstance(blocks, list)
                blocks.append(16)

        raw = mma.emit_warp_mma

        def changed(*args, **kwargs):
            result = raw(*args, **kwargs)
            mutate()
            return result

        try:
            if phase == "before":
                mutate()
            with (
                patch.object(
                    mma, "emit_warp_mma", changed if phase == "during" else raw
                ),
                pytest.raises(chain._UnsupportedChain, match="prepared warp"),
            ):
                original(cg, plan, boundaries, prepared, prefix)
            assert boundaries == before
            assert prepared.completion._consumed is False
            assert prepared.completion._state.consumed is False
            checked.append(True)
        finally:
            config.clear()
            config.update(saved)
        return original(cg, plan, boundaries, prepared, prefix)

    with patch.object(shared, "emit_prepared_warp_stage", check):
        _code() if route == "root" else _emissions("warp")
    assert checked


@pytest.mark.parametrize("dtype_name", ("BFloat16", "Float16"))
@pytest.mark.parametrize("columns,threads", ((16, 64), (64, 256)))
@pytest.mark.parametrize("axes", ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_actual_original_padded_warp_partitions(dtype_name, columns, threads, axes):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
                ),
                atom_layout_mnk=(1, threads // 32, 1),
            )
            covered = set()
            a_seen, b_seen = set(), set()
            for thread in range(threads):
                partition = mma.get_slice(thread)
                c = partition.partition_C(cute.make_identity_tensor((16, columns)))
                cells = [tuple(map(int, c[i])) for i in range(cute.size(c))]
                assert not covered.intersection(cells)
                covered.update(cells)
                for role, shape, axis, seen in (
                    ("a", (16, 128), axes[0], a_seen),
                    ("b", (columns, 128), axes[1], b_seen),
                ):
                    part = (
                        partition.partition_A if role == "a" else partition.partition_B
                    )(cute.make_identity_tensor(shape))
                    assert cute.size(part, mode=[2]) == 8
                    stride = (shape[1] + 8, 1) if axis == 1 else (1, shape[0] + 8)
                    layout = cute.make_layout(shape, stride=stride)
                    for k in range(8):
                        panel = part[None, None, k]
                        for i in range(cute.size(panel)):
                            row, column = map(int, panel[i])
                            assert (
                                0 <= row < shape[0] and k * 16 <= column < (k + 1) * 16
                            )
                            assert (
                                int(layout((row, column)))
                                == row * stride[0] + column * stride[1]
                            )
                            seen.add((row, column))
            assert covered == {(r, c) for r in range(16) for c in range(columns)}
            assert a_seen == {(r, k) for r in range(16) for k in range(128)}
            assert b_seen == {(r, k) for r in range(columns) for k in range(128)}
        assert module.operation.verify()
