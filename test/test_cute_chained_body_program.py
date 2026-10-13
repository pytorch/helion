from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test._cute_aux import _cpu_codegen
from test.test_cute_chained_tcgen05 import _both_operands_chain

import helion
from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen05 as roots
from helion._compiler.cute import chained_tcgen_stage as stages


def _source(dtype=torch.bfloat16, early=False):
    with _cpu_codegen():
        return _both_operands_chain._bind_isolated(
            (torch.empty((2, 128, 128), dtype=dtype),) * 2
        ).to_code(
            helion.Config(
                block_sizes=[],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_tmem_early_release=early,
            )
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("early", [False, True])
def test_materialized_body_retires_original_stage_walk_exactly(dtype, early):
    with patch.object(body, "supports_materialized_root", return_value=False):
        original = _source(dtype, early)
    with (
        patch.object(body, "emit_body_program", wraps=body.emit_body_program) as walk,
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as shared,
        patch.object(roots, "_stage", wraps=roots._stage) as producers,
    ):
        candidate = _source(dtype, early)
    assert candidate == original
    assert walk.call_count == 1
    assert [call.args[3] for call in shared.call_args_list] == [0, 1]
    assert [call.kwargs["root_body"].result for call in shared.call_args_list] == [
        "allocated",
        "terminal",
    ]
    assert [(c.args[4], c.args[5], c.args[6]) for c in producers.call_args_list] == [
        (0, "a", 1),
        (0, "b", 1),
        (1, "a", 1),
        (1, "b", 0),
    ]
    assert "chain_0_c =" in candidate and "chain_1_c =" not in candidate


@pytest.mark.parametrize(
    "mutation",
    [
        "actions",
        "event",
        "reads",
        "writes",
        "axes",
        "threads",
        "thread",
        "warp",
        "sync",
        "tmem",
        "barriers",
        "a_workspace",
        "b_workspace",
        "config",
        "dtype",
        "shape",
        "stride",
        "node_args",
        "boundary",
        "scratch",
        "unroll",
        "cache",
        "inplace",
        "early_release",
        "consumed",
        "cursor",
        "store",
        "strategy",
    ],
)
def test_accepted_body_rejects_same_object_mutation_before_stage(mutation):
    original = body.plan_root_body
    accepted = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        accepted.append(result)
        if mutation in {"actions", "event", "reads", "writes"}:
            program = result.program
            if mutation == "actions":
                object.__setattr__(program, "actions", program.actions[::-1])
            else:
                value = {"event": 7, "reads": ("unpublished",), "writes": ()}[mutation]
                object.__setattr__(program.actions[0], mutation, value)
        elif mutation == "axes":
            result.axes = ((0, 1), (1, 0))
        elif mutation in {
            "threads",
            "thread",
            "warp",
            "sync",
            "tmem",
            "barriers",
            "a_workspace",
            "b_workspace",
        }:
            object.__setattr__(
                result.execution, mutation, 256 if mutation == "threads" else "changed"
            )
        elif mutation == "config":
            result.codegen.device_function.config.config[
                "cute_chained_pointwise_unroll"
            ] = 2
        elif mutation in {"dtype", "shape", "stride"}:
            node = result.plan.dots[0]
            value = node.meta["val"]
            if mutation == "dtype":
                value = value.to(torch.float16)
            elif mutation == "shape":
                value = value[:64]
            else:
                value = value.T
            node.meta["val"] = value
        elif mutation == "node_args":
            node = result.plan.dots[1]
            node.args = (*node.args[:2], result.plan.dots[0], *node.args[3:])
        elif mutation == "boundary":
            result.boundaries[result.plan.dots[0]] = "forged"
        elif mutation == "scratch":
            result.scratch.mode = "xor"
        elif mutation == "unroll":
            assert result.unroll is not None
            result.unroll.factor = 2
        elif mutation in {"cache", "inplace"}:
            getattr(result, mutation).enabled = True
        elif mutation == "early_release":
            result.early_release = not result.early_release
        elif mutation == "consumed":
            result.consumed = True
        elif mutation == "store":
            object.__setattr__(result.plan, "store", result.plan.dots[0])
        elif mutation == "strategy":
            object.__setattr__(result.plan, "strategy", "warp")
        else:
            result.next_stage = 1
        return result

    with (
        patch.object(body, "plan_root_body", capture),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as stage,
        pytest.raises(Exception, match="ordered root body changed"),
    ):
        _source()
    assert accepted and stage.call_count == 0


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("change", ["codegen", "geometry", "result", "boundary"])
def test_actual_stage_arguments_and_result_policy_are_checked(ordinal, change):
    original = stages.emit_stage
    reached = []

    def intercept(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        if stage == ordinal:
            reached.append(stage)
            if change == "codegen":
                cg = object()
            elif change == "geometry":
                geometry = replace(geometry, transpose=True)
            elif change == "result":
                object.__setattr__(
                    kwargs["root_body"],
                    "result",
                    "allocated" if ordinal else "terminal",
                )
            else:
                boundaries[plan.dots[0]] = "forged"
        return original(cg, plan, boundaries, stage, geometry, *args, **kwargs)

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(Exception, match="ordered root|stage policy"),
    ):
        _source()
    assert reached == [ordinal]


def test_late_failure_does_not_install_or_retry_a_partial_body():
    original = roots._stage
    calls = []
    captures = []

    def fail(cg, plan, boundaries, scans, stage, role, *args, **kwargs):
        calls.append((stage, role))
        if stage == 1:
            captures.append((cg.device_function, tuple(cg.device_function.body)))
            raise chain._UnsupportedChain("bounded injected late producer failure")
        return original(cg, plan, boundaries, scans, stage, role, *args, **kwargs)

    with (
        patch.object(roots, "_stage", fail),
        pytest.raises(Exception, match="bounded injected late producer failure"),
    ):
        _source()
    assert calls == [(0, "a"), (0, "b"), (1, "a")]
    assert len(captures) == 1
    df, before = captures[0]
    assert tuple(df.body) == before


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["drop_receipt", "changed_body", "duplicate", "alias"]
)
def test_completion_inventory_and_alias_history_are_required(ordinal, mutation):
    original = stages.emit_stage
    reached = []

    def intercept(cg, plan, boundaries, stage, geometry, *args, **kwargs):
        policy = kwargs["root_body"]
        if stage == ordinal and mutation == "alias":
            first = next(iter(plan.tensor_aliases))
            plan.tensor_aliases[first] += "_changed"
        result = original(cg, plan, boundaries, stage, geometry, *args, **kwargs)
        if stage == ordinal:
            reached.append(stage)
            if mutation == "drop_receipt":
                policy.body.completion = None
            elif mutation == "changed_body":
                result.pop()
            elif mutation == "duplicate":
                policy.record_completion(result)
        return result

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(Exception, match="ordered root|completed body span"),
    ):
        _source()
    assert reached == ([] if mutation == "alias" else [ordinal])


@pytest.mark.parametrize("ordinal", [0, 1])
def test_post_completion_alias_growth_rejects_without_consumption(ordinal):
    original = stages.emit_stage
    reached = []

    def intercept(cg, plan, boundaries, stage, *args, **kwargs):
        result = original(cg, plan, boundaries, stage, *args, **kwargs)
        if stage == ordinal:
            policy = kwargs["root_body"]
            assert policy.body.completion == (stage, policy.result, tuple(result))
            plan.tensor_aliases["unowned_tensor"] = "unpublished_alias"
            reached.append(policy.body)
        return result

    with (
        patch.object(stages, "emit_stage", intercept),
        pytest.raises(Exception, match="changed completed body span"),
    ):
        _source()
    assert len(reached) == 1 and not reached[0].consumed
    assert reached[0].next_stage == ordinal


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("role", ["a", "b"])
def test_actual_operand_dtype_is_original_before_lowering(dtype, ordinal, role):
    original = body.RootBodyStage.operand_lines
    reached = []

    def intercept(policy, cg, boundaries, actual_role, actual_dtype):
        if policy.ordinal == ordinal and actual_role == role:
            reached.append(policy.body)
            actual_dtype = (
                "cutlass.Float16" if dtype == torch.bfloat16 else "cutlass.BFloat16"
            )
        return original(policy, cg, boundaries, actual_role, actual_dtype)

    with (
        patch.object(body.RootBodyStage, "operand_lines", intercept),
        pytest.raises(Exception, match="operand role or dtype changed"),
    ):
        _source(dtype)
    assert len(reached) == 1 and not reached[0].consumed
    assert reached[0].next_stage == ordinal
    assert reached[0].operand_count == (0 if role == "a" else 1)


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("change", ["swap", "duplicate", "unknown", "omit"])
def test_operand_order_and_complete_pair_required(ordinal, change):
    original = body.RootBodyStage.operand_lines
    reached = []

    def intercept(policy, cg, boundaries, role, dtype):
        if policy.ordinal == ordinal and role == "a":
            reached.append(policy.body)
            if change == "omit":
                return []
            if change == "duplicate":
                original(policy, cg, boundaries, role, dtype)
            elif change == "swap":
                role = "b"
            else:
                role = "unknown"
        return original(policy, cg, boundaries, role, dtype)

    with (
        patch.object(body.RootBodyStage, "operand_lines", intercept),
        pytest.raises(Exception, match="operand role or dtype changed"),
    ):
        _source()
    assert len(reached) == 1 and not reached[0].consumed
    assert reached[0].next_stage == ordinal


@pytest.mark.parametrize("ordinal", [0, 1])
def test_completion_requires_both_actual_operand_calls(ordinal):
    original = body.RootBodyStage.operand_lines
    reached = []

    def intercept(policy, cg, boundaries, role, dtype):
        if policy.ordinal == ordinal:
            reached.append(policy.body)
            return []
        return original(policy, cg, boundaries, role, dtype)

    with (
        patch.object(body.RootBodyStage, "operand_lines", intercept),
        pytest.raises(Exception, match="ordered root operands incomplete"),
    ):
        _source()
    assert len(reached) == 2 and not reached[0].consumed
    assert reached[0].next_stage == ordinal
