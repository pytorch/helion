from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_register_emission import _register_config
from .test_cute_chained_row_collective_emission import _args as _row_args
from .test_cute_chained_row_collective_emission import _config as _row_config
from .test_cute_chained_row_collective_emission import _row_loop
from .test_cute_chained_row_collectives import _case
import helion
from helion import exc
from helion._compiler.cute import chained_collectives
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute import chained_row_collective_emission as row_emission
from helion._compiler.cute import chained_row_collectives as row_planner
from helion._compiler.cute import chained_tcgen_stage as stage_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_pipeline import PreparationPipeline
from helion._compiler.cute.chained_recurrence_workspace import RecurrenceWorkspace
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_vector_stage import VectorStageOperand
from helion._compiler.cute.chained_vector_stage import VectorStaging
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan


def _execution():
    return ChainedExecution(
        128,
        thread="prep_thread",
        warp="prep_warp",
        sync="prep_barrier.arrive_and_wait()",
    )


def _operands(plan, candidate):
    group = candidate.stage.group
    result = []
    members = tuple(zip(group.stages, group.geometries, group.offsets, strict=True))
    for role in ("a", "b"):
        for stage, geometry, offset in members[:1] if role == "a" else members:
            index, _ = geometry.operand(role, "row", "column")
            result.append(
                VectorStageOperand(
                    plan.dots[stage].args[index],
                    geometry,
                    role,
                    candidate.shape,
                    f"chain_{group.stages[0]}_{role}",
                    offset if role == "b" else 0,
                )
            )
    return tuple(result)


@pytest.mark.parametrize(
    "mode", ["producer_none", "producer_unsupported", "stage_unsupported"]
)
def test_failed_row_attempt_discards_every_temporary_boundary(mode):
    plan, frame, shapes = _case()
    candidate = row_planner.plan_row_collective_group(plan, frame, 1, shapes)
    assert candidate is not None
    input_node = frame.buffers[0].node
    assert input_node is not None
    boundaries = {input_node: "original_input"}
    before = dict(boundaries)
    unroll = BoundedProducerUnroll(1)
    attempted = []

    def producer(
        cg, actual_plan, actual_frame, actual_candidate, trial, operands, **kwargs
    ):
        assert actual_plan is plan and actual_frame is frame
        assert actual_candidate == candidate and trial is not boundaries
        assert operands == _operands(plan, candidate)
        trial[plan.dots[0]] = "temporary_producer_result"
        trial[next(iter(boundaries))] = "temporary_overwrite"
        if mode == "producer_unsupported":
            raise chain._UnsupportedChain("producer declined")
        return None

    def stage(cg, actual_plan, trial, *args, operand_producer, **kwargs):
        assert trial is not boundaries and trial == before
        attempted.append(trial)
        trial[plan.dots[-1]] = "temporary_stage_result"
        if mode == "stage_unsupported":
            raise chain._UnsupportedChain("stage declined after writes")
        result = operand_producer(_operands(plan, candidate))
        assert result is None
        raise chain._UnsupportedChain("complete operand producer rejected")

    with (
        patch.object(chain, "_shape", side_effect=shapes.__getitem__),
        patch.object(row_emission, "emit_row_collective_group", side_effect=producer),
        patch.object(stage_module, "emit_stage", side_effect=stage),
    ):
        result = pipeline_module._emit_row_collective_stage(
            Mock(), plan, frame, 1, boundaries, _execution(), unroll
        )
    assert result is None and len(attempted) == 1
    assert boundaries == before
    assert not unroll.activated and not unroll.eliminated


def test_missing_candidate_never_calls_stage_or_mutates_boundaries():
    plan, frame, shapes = _case()
    boundaries = {plan.dots[0]: "existing_result"}
    before = dict(boundaries)
    with (
        patch.object(chain, "_shape", side_effect=shapes.__getitem__),
        patch.object(row_planner, "plan_row_collective_group", return_value=None),
        patch.object(
            stage_module, "emit_stage", side_effect=AssertionError("stage called")
        ),
    ):
        assert (
            pipeline_module._emit_row_collective_stage(
                Mock(),
                plan,
                frame,
                1,
                boundaries,
                _execution(),
                BoundedProducerUnroll(1),
            )
            is None
        )
    assert boundaries == before


def test_success_commits_sums_and_complete_stage_results_atomically():
    plan, frame, shapes = _case()
    candidate = row_planner.plan_row_collective_group(plan, frame, 1, shapes)
    assert candidate is not None
    input_node = frame.buffers[0].node
    assert input_node is not None
    boundaries = {input_node: "existing_input"}
    before = dict(boundaries)
    execution = _execution()
    unroll = BoundedProducerUnroll(1)

    def stage(cg, actual_plan, trial, stage, geometry, phase, group, **kwargs):
        assert actual_plan is plan and group is candidate.stage.group
        assert boundaries == before and trial is not boundaries
        assert kwargs["prepared_shape"] == candidate.stage.shape
        local = kwargs["execution"]
        assert local.thread == execution.thread and local.sync == execution.sync
        assert (
            local.a_workspace
            == f"cute.recast_ptr(chain_frame + {candidate.stage.a.byte_offset}, dtype=cutlass.BFloat16)"
        )
        assert (
            local.b_workspace
            == f"cute.recast_ptr(chain_frame + {candidate.stage.b.byte_offset}, dtype=cutlass.BFloat16)"
        )
        assert kwargs["producer_unroll"] is unroll
        assert kwargs["operand_producer"](_operands(plan, candidate)) == [
            "complete_fill()"
        ]
        assert boundaries == before
        for index, node in enumerate(plan.dots):
            trial[node] = f"published_c_{index}"
        return ["complete_stage()", execution.sync]

    with (
        patch.object(chain, "_shape", side_effect=shapes.__getitem__),
        patch.object(
            row_emission, "emit_row_collective_group", return_value=["complete_fill()"]
        ),
        patch.object(stage_module, "emit_stage", side_effect=stage),
    ):
        result = pipeline_module._emit_row_collective_stage(
            Mock(), plan, frame, 1, boundaries, execution, unroll
        )
    assert result == (["complete_stage()", execution.sync], candidate.stop_event)
    assert frame.actions[candidate.stop_event].kind == "mma"
    assert boundaries == {
        **before,
        **{node: f"published_c_{index}" for index, node in enumerate(plan.dots)},
        **{
            operation.node: buffer.name
            for operation, buffer in zip(
                candidate.collectives, candidate.buffers, strict=True
            )
        },
    }


def _control_frame():
    plan, frame, shapes = _case()
    candidate = row_planner.plan_row_collective_group(plan, frame, 1, shapes)
    assert candidate is not None
    node = plan.dots[0]
    image = PreparationBuffer(
        "later_image", "frontier", node, node.meta["val"].dtype, shapes[node]
    )
    # Control-flow-only fixture begins after the initial raw publication; its
    # mocked emitters do not read storage. Keep the following MMA and a later
    # frontier action so an over-broad skip is observable.
    ready = frame.actions[-1]
    actions = (
        *frame.actions[1:-1],
        PreparationAction(
            "frontier", ready.event, (node,), (), None, (), (image.name,)
        ),
        replace(ready, event=ready.event + 1),
    )
    frame = replace(frame, actions=actions, buffers=(*frame.buffers, image))
    recurrence = RecurrenceWorkspace(
        frame.cut, SharedMemoryLayoutPlan((), 0), (), (), 0, 0, 0, 0
    )
    return (
        plan,
        PreparationPipeline(frame, recurrence, None, 2, 128, 128, 128),
        candidate,
    )


@pytest.mark.parametrize("enabled", [None, False, True])
def test_prepare_skips_exact_sums_and_fill_and_preserves_later_actions(enabled):
    plan, pipeline, candidate = _control_frame()
    cg = Mock()
    cg.device_function.config = helion.Config()
    if enabled is not None:
        cg.device_function.config.config["cute_chained_collective_retention"] = enabled
    execution = _execution()
    calls = []

    def row(cg, actual_plan, frame, event, boundaries, *rest):
        assert enabled is True and event == candidate.first_event
        calls.append("retained")
        return ["retained_stage()", execution.sync], candidate.stop_event

    def collective(*args, selected, **kwargs):
        (node,) = selected
        index = [item.node for item in candidate.collectives].index(node)
        calls.append(f"sum_{index}")
        return [f"ordinary_sum_{index}()", execution.sync]

    def stage(*args, **kwargs):
        calls.append("fill_and_mma")
        return ["ordinary_stage()", execution.sync]

    def frontier(*args, **kwargs):
        calls.append("later_frontier")
        return ["later_frontier()", execution.sync]

    with (
        patch.object(pipeline_module, "_emit_row_collective_stage", side_effect=row),
        patch.object(
            chained_collectives, "emit_collectives_before", side_effect=collective
        ),
        patch.object(stage_module, "emit_stage", side_effect=stage),
        patch.object(
            chained_prepared_values, "emit_prepared_value", side_effect=frontier
        ),
    ):
        lines = pipeline_module._prepare(
            cg,
            plan,
            pipeline,
            execution,
            VectorStaging(False),
            BoundedProducerUnroll(1),
            ScratchLayouts(),
        )
    assert calls == (
        ["retained", "later_frontier"]
        if enabled
        else ["sum_0", "sum_1", "fill_and_mma", "later_frontier"]
    )
    assert lines[-3:] == ["later_frontier()", execution.sync, execution.sync]
    assert lines.count("retained_stage()") == int(enabled is True)


def test_explicit_true_without_actual_activation_rejects_after_ordinary_fallback():
    plan, pipeline, _ = _control_frame()
    cg = Mock()
    cg.device_function.config = helion.Config.from_dict(
        {"cute_chained_collective_retention": True}
    )
    with (
        patch.object(pipeline_module, "_emit_row_collective_stage", return_value=None),
        patch.object(chained_collectives, "emit_collectives_before", return_value=[]),
        patch.object(stage_module, "emit_stage", return_value=[]),
        patch.object(chained_prepared_values, "emit_prepared_value", return_value=[]),
        pytest.raises(exc.BackendUnsupported, match="collective retention requires"),
    ):
        pipeline_module._prepare(
            cg,
            plan,
            pipeline,
            _execution(),
            VectorStaging(False),
            BoundedProducerUnroll(1),
            ScratchLayouts(),
        )


def _retained_source(kernel, args, config):
    attempts = []
    stages = []
    ordinary_sums = []
    original_row = pipeline_module._emit_row_collective_stage
    original_stage = stage_module.emit_stage
    original_sum = chained_collectives.emit_collectives_before

    def row(cg, plan, frame, event, boundaries, execution, unroll):
        actions = frame.actions
        layout = frame.layout
        before = dict(boundaries)
        result = original_row(cg, plan, frame, event, boundaries, execution, unroll)
        assert frame.actions == actions and frame.layout == layout
        if result is None:
            assert boundaries == before
        else:
            lines, stop = result
            skipped = frame.actions[event:stop]
            assert [action.kind for action in skipped] == [
                "collective",
                "collective",
                "fill",
            ]
            assert frame.actions[stop].kind == "mma"
            for action in skipped[:-1]:
                assert boundaries[action.nodes[0]] == action.writes[0]
            assert all(node in boundaries for node in skipped[-1].nodes)
            attempts.append((event, stop, lines, execution, skipped))
        return result

    def stage(*args, **kwargs):
        stages.append((args[3], kwargs.get("operand_producer") is not None))
        return original_stage(*args, **kwargs)

    def collective(*args, **kwargs):
        ordinary_sums.extend(kwargs.get("selected") or ())
        return original_sum(*args, **kwargs)

    with (
        patch.object(pipeline_module, "_emit_row_collective_stage", side_effect=row),
        patch.object(stage_module, "emit_stage", side_effect=stage),
        patch.object(
            chained_collectives, "emit_collectives_before", side_effect=collective
        ),
    ):
        source = _source(kernel, args, config)
    assert len(attempts) == 1
    event, stop, lines, execution, skipped = attempts[0]
    target_stage = skipped[-1].stages[0]
    assert stages.count((target_stage, True)) == 1
    assert (target_stage, False) not in stages
    assert not set(ordinary_sums).intersection(
        action.nodes[0] for action in skipped[:-1]
    )
    assert source.count(f"for chain_row_collective_{event}_step ") == 1
    return source, lines, execution, target_stage


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("warps", [8, 16])
def test_real_emission_retains_complete_fill_and_unconditional_role_barriers(
    dtype, warps
):
    config = _row_config()
    config.config.update(
        num_warps=warps,
        cute_chained_collective_retention=True,
    )
    source, lines, execution, stage = _retained_source(
        _row_loop,
        _row_args(dtype, 48, keepdim=warps == 16, nonlinear=True),
        config,
    )
    assert execution.threads == (warps - 4) * 32
    tree = ast.parse("\n".join(lines))
    row_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and ast.unparse(node.target).startswith("chain_row_collective_")
    )
    assert execution.sync not in ast.unparse(row_loop)
    position = tree.body.index(row_loop)
    assert [ast.unparse(node) for node in tree.body[position + 1 : position + 5]] == [
        execution.sync,
        execution.sync,
        "cute.arch.fence_view_async_shared()",
        execution.sync,
    ]
    assert ast.unparse(tree.body[-1]) == execution.sync
    compute_name = "execute_prepared_warp_k" if dtype == torch.bfloat16 else "cute.gemm"
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == compute_name
    ]
    assert len(calls) == 1
    compute = [
        node
        for node in tree.body
        if any(
            isinstance(child, ast.Call) and ast.unparse(child.func) == compute_name
            for child in ast.walk(node)
        )
    ]
    assert len(compute) == 1
    assert execution.sync not in ast.unparse(compute[0])
    if dtype == torch.bfloat16:
        prefix = f"chain_{stage}_warp"
        expected_import = ast.parse(
            "from helion._compiler.cute.prepared_warp_contraction import execute_prepared_warp_k"
        ).body[0]
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == compute_name for alias in node.names)
        ]
        assert [ast.dump(node) for node in imports] == [ast.dump(expected_import)]
        expected = (
            f"{prefix}_acc = execute_prepared_warp_k("
            f"({prefix}_copy_a, {prefix}_copy_sa, {prefix}_copy_ra, {prefix}_ra), "
            f"({prefix}_copy_b, {prefix}_copy_sb, {prefix}_copy_rb, {prefix}_rb), "
            f"{prefix}_acc, {prefix}_mma, cute.size({prefix}_sa, mode=[2]), False, 1)"
        )
        assignments = [
            node
            for node in ast.walk(compute[0])
            if isinstance(node, ast.Assign) and node.value is calls[0]
        ]
        assert [ast.dump(node) for node in assignments] == [
            ast.dump(ast.parse(expected).body[0])
        ]
    assert f"chain_{stage}_a[" in ast.unparse(row_loop)
    assert f"chain_{stage}_b[" in ast.unparse(row_loop)
    assert f"chain_{stage}_c[" in source
    assert "chain_sm100.make_trivial_tiled_mma(" in source


def test_absent_and_false_preserve_full_original_source_without_discovery():
    config = _row_config()
    disabled = helion.Config.from_dict(
        {**config.config, "cute_chained_collective_retention": False}
    )
    args = _row_args()
    with patch.object(
        pipeline_module,
        "_emit_row_collective_stage",
        side_effect=AssertionError("disabled retention discovery"),
    ):
        source = _source(_row_loop, args, config)
        assert _source(_row_loop, args, disabled) == source
    assert "chain_row_collective_" not in source


@pytest.mark.parametrize("cohorts", [1, 3])
def test_retention_and_existing_register_island_both_activate(cohorts):
    kernel, args = _kda_fixture()
    config = _register_config(
        cohorts,
        cute_chained_register_islands=True,
        cute_chained_collective_retention=True,
    )
    source, _, execution, _ = _retained_source(kernel, args, config)
    assert execution.threads == (384 if cohorts == 1 else 128)
    tree = ast.parse(source)
    islands = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "chain_prep_thread < 64"
        and "chain_register_island" in ast.unparse(node)
    ]
    assert len(islands) == 1
    island = ast.unparse(islands[0])
    assert island.count("cute.gemm(") == 6
    assert execution.sync not in island
    assert "chain_2_a_ptr" not in source
