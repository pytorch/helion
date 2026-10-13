from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_execution import ChainedExecution
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _three(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    q, n = v.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([m, n]):
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[row, kk], b[qq, kk].T)
        second = hl.dot(first.to(a.dtype), b[qq, kk])
        out[row, col] = hl.dot(second.to(a.dtype), v[kk, col]).to(a.dtype)
    return out


def test_multiple_original_bridges_reuse_one_immutable_revision() -> None:
    arguments = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((128, 128), (128, 128), (128, 64))
    )
    with (
        _cpu_codegen(),
        patch.object(
            chain, "_register_bridge_revision", wraps=chain._register_bridge_revision
        ) as revision,
    ):
        source = _three._bind_isolated(arguments).to_code(
            helion.Config(
                block_sizes=[128, 64],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
        )
    assert source.count("OperandSource.TMEM") == 2
    assert "chain_1_bridge_values" in source and "chain_2_bridge_values" in source
    assert revision.call_count == 1


def _source(dtype=torch.bfloat16, mode="scan", columns=64):
    arguments = (*_tcgen_inputs("cpu", dtype, n=columns), mode)
    config = helion.Config(
        block_sizes=[128, columns],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=False,
        cute_chained_auxiliary_cache=False,
    )
    with _cpu_codegen():
        return _tcgen_chain._bind_isolated(arguments).to_code(config)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("columns", [32, 64, 128])
def test_pair_routes_both_stages_without_changing_original_source(dtype, mode, columns):
    initialized = torch.cuda.is_initialized()
    with patch.object(roots, "supports_root_pair", return_value=False):
        original = _source(dtype, mode, columns)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old entry")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as calls,
    ):
        actual = _source(dtype, mode, columns)
    assert actual == original
    assert calls.call_count == 2
    assert [call.args[3] for call in calls.call_args_list] == [0, 1]
    assert [call.kwargs["terminal_fragment"] for call in calls.call_args_list] == [
        False,
        True,
    ]
    assert all(call.kwargs["root_actions"] is not None for call in calls.call_args_list)
    assert "chain_0_c =" not in actual
    assert "OperandSource.TMEM" in actual
    assert "OperandMajorMode.MN" in actual
    assert actual.index("chain_1_b_ptr =") < actual.index("chain_0_mma =")
    assert actual.index("chain_1_bridge_values[") < actual.index("chain_1_acc =")
    if mode == "scan":
        commit = actual.index("cp_async_commit_group()")
        scan = actual.index("chain_scan_0_pointer =")
        wait = actual.index("cp_async_wait_group(0)")
        assert commit < scan < wait
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize(
    "case",
    [
        "stage",
        "before_ready",
        "geometry",
        "plan",
        "participants",
        "thread",
        "warp",
        "barriers",
        "native_axes",
        "allocation_policy",
        "accumulator_overlap",
        "missing_packed",
        "second_allocation",
        "premature_terminal",
        "stale_graph",
        "stale_dtype",
        "duplicate_complete",
    ],
)
def test_pair_rejects_changed_provenance_or_lifetime_before_writes(case):
    original = stages.emit_stage
    checked = []

    def observe(*args, **kwargs):
        stage = args[3]
        if checked or stage != (
            1
            if case in ("accumulator_overlap", "missing_packed", "second_allocation")
            else 0
        ):
            return original(*args, **kwargs)
        bad_args, bad = list(args), dict(kwargs)
        action = bad["root_actions"]
        sequence = action.sequence
        before = (
            sequence.next_stage,
            list(sequence.staged),
            dict(args[2]),
            list(args[0].device_function.body),
        )
        old_args = args[1].dots[0].args
        old_value = args[1].dots[0].meta["val"]
        old_axes = sequence.inner_axes
        try:
            if case == "stage":
                bad["root_actions"] = replace(action, stage=1)
            elif case == "before_ready":
                action = sequence.action(1)
                bad_args[3:5] = [1, sequence.geometries[1]]
                bad.update(
                    root_actions=action,
                    tmem_input=action.tmem_input,
                    tmem_accumulator=action.accumulator,
                    pending_allocation=None,
                    terminal_fragment=True,
                )
            elif case == "geometry":
                bad_args[4] = stages.StageGeometry((128, 32, 128), False)
            elif case == "plan":
                bad_args[1] = replace(args[1])
            elif case == "participants":
                bad["execution"] = ChainedExecution(256)
            elif case == "thread":
                bad["execution"] = ChainedExecution(128, thread="different_thread")
            elif case == "warp":
                bad["execution"] = ChainedExecution(128, warp="different_warp")
            elif case == "barriers":
                bad["execution"] = ChainedExecution(128, barriers="different_bars")
            elif case == "native_axes":
                sequence.inner_axes = ((1, 1), (1, 1))
            elif case == "allocation_policy":
                bad["pending_allocation"] = not bad["pending_allocation"]
            elif case == "accumulator_overlap":
                bad["tmem_accumulator"] = "chain_tptr + 0"
            elif case == "missing_packed":
                bad["tmem_input"] = None
            elif case == "second_allocation":
                bad["pending_allocation"] = False
            elif case == "premature_terminal":
                bad["terminal_fragment"] = True
            elif case == "stale_graph":
                args[1].dots[0].args = (old_args[1], old_args[0], *old_args[2:])
            elif case == "stale_dtype":
                args[1].dots[0].meta["val"] = torch.empty(
                    old_value.shape, dtype=torch.float16
                )
            else:
                output = original(*args, **kwargs)
                with pytest.raises(chain._UnsupportedChain, match="readiness"):
                    original(*args, **kwargs)
                checked.append(case)
                return output
            with pytest.raises(chain._UnsupportedChain):
                original(*bad_args, **bad)
            assert (
                sequence.next_stage,
                sequence.staged,
                args[2],
                args[0].device_function.body,
            ) == before
        finally:
            args[1].dots[0].args = old_args
            args[1].dots[0].meta["val"] = old_value
            sequence.inner_axes = old_axes
        checked.append(case)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=observe):
        _source()
    assert checked == [case]


@pytest.mark.parametrize(
    "case",
    [
        "no_prefetch",
        "bridge_stage",
        "bridge_source",
        "bridge_operand",
        "bridge_revision",
        "bridge_role",
    ],
)
def test_failed_late_pair_proof_keeps_original_scheduler_transaction(case):
    with patch.object(roots, "supports_root_pair", return_value=False):
        reference = _source()
    factory = roots.plan_root_stage_sequence
    rejected = []

    def decline(cg, plan, bridges, axes, scans, scan_lines, **kwargs):
        altered = dict(bridges)
        if case == "no_prefetch":
            kwargs["prefetched"] = False
        else:
            bridge = altered[1]
            changes = {
                "bridge_stage": {"stage": 0},
                "bridge_source": {"source": plan.dots[1]},
                "bridge_operand": {"operand": plan.dots[1].args[1]},
                "bridge_revision": {"revision": ()},
                "bridge_role": {"role": "b"},
            }
            altered[1] = replace(bridge, **changes[case])
        result = factory(cg, plan, altered, axes, scans, scan_lines, **kwargs)
        assert result is None
        rejected.append(case)
        return result

    with (
        patch.object(roots, "plan_root_stage_sequence", side_effect=decline),
        patch.object(
            stages, "emit_stage", side_effect=AssertionError("unproved stage")
        ),
    ):
        assert _source() == reference
    assert rejected == [case]
