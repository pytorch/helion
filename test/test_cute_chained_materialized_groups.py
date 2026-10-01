from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_frontier_groups as frontier_fixture
from . import test_cute_chained_vector_group as fixture
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_frontier_groups as frontiers
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_vector_group as groups
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_vector_stage import VectorStaging

# Original root-pair-gate-Njwy2St3 snapshot, captured independently before the
# two-module overlay. These cover exact source bytes, not normalized ASTs.
_FROZEN_DIGESTS = {
    (
        torch.bfloat16,
        False,
    ): "a644d8820b8a6c87dfeb24b7e88ed6d8dcaec2ae27377ee598a1463d5b23a9de",
    (
        torch.bfloat16,
        True,
    ): "ef678766976a178ceb7560565fd7254f141b1ec03818ee5d9dd6f76e0187b1d6",
    (
        torch.float16,
        False,
    ): "3a0fa953eeda3eabfa77d29bb5292d6f061462536f4ba215be2a8d3a43164bc6",
    (
        torch.float16,
        True,
    ): "2e8c3227c7bf695495d9125ffc841bac0bf2349a370eb981ab8153f64d4a05c9",
}


def _capture(*, mode="roots", alter=None, **options):
    captured = {}

    def observe(cg, plan, boundaries, outputs, **kwargs):
        if mode == "mixed":
            outputs = [
                outputs[0],
                replace(
                    outputs[1],
                    node=outputs[1].node.args[0].args[0],
                    coordinates=lambda row, column: (row, column),
                    final_value=None,
                ),
            ]
        selected = dict(boundaries)
        if mode == "raw":
            for item in outputs:
                probe = chain._Expression(cg, plan, boundaries)
                probe.coordinate_names.update(
                    ("joined_row", "joined_base", "joined_element")
                )
                probe.value(
                    item.node,
                    item.coordinates("joined_row", "(joined_base + joined_element)"),
                )
                for leaf, _, _, _ in probe.loaded_inputs:
                    selected.setdefault(leaf, f"retained_{len(selected)}")
        elif mode != "host":
            for index, item in enumerate(outputs):
                if mode != "partial" or index == 0:
                    selected[item.node] = f"retained_{index}"
        if alter is not None:
            outputs, selected = alter(outputs, selected, plan)
        before = dict(selected)
        if mode == "roots":
            assert (
                groups.emit_vector_group(cg, plan, selected, outputs, **kwargs) is None
            )
        lines = groups.emit_materialized_group(cg, plan, selected, outputs, **kwargs)
        assert selected == before
        captured.update(lines=lines, outputs=outputs, boundaries=selected)
        if lines is not None:
            expected = []
            # Independent original scalar lowering, without the group memo or
            # any invented expression/cast/mask. Compare recursively expanded
            # trees, not temporary names allocated during separate probes.
            for item in outputs:
                expr = chain._Expression(cg, plan, selected)
                coords = item.coordinates(
                    "joined_row", "(joined_base + joined_element)"
                )
                expr.coordinate_names.update(
                    ("joined_row", "joined_base", "joined_element")
                )
                value = expr.value(item.node, coords)
                dtype = CompileEnvironment.current().backend.dtype_str(
                    item.node.meta["val"].dtype
                )
                value = (
                    f"{dtype}({value})"
                    if item.final_value is None
                    else item.final_value(value, dtype, coords)
                )
                scalar = [
                    "for joined_element in range(8):",
                    "    if True:",
                    chain._indent(
                        [*expr.lines, f"expected[joined_element] = {value}"], 8
                    ),
                ]
                expected.extend(fixture._expanded_outputs(scalar, "joined_element"))
            captured["expected"] = expected
        return lines

    with patch.object(fixture, "emit_vector_group", observe):
        _, _, unroll = fixture._capture(**options)
    captured["unroll"] = unroll
    return captured


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["roots", "raw", "mixed"])
@pytest.mark.parametrize("scalar", [False, True])
def test_original_typed_boundary_and_expression_publication(dtype, mode, scalar):
    result = _capture(dtype=dtype, mode=mode, scalar=scalar)
    lines = result["lines"]
    assert lines is not None
    assert fixture._expanded_outputs(lines, "joined_element") == result["expected"]
    source = "\n".join(lines)
    assert "_leaf_" not in source and "_vectorized" not in source
    assert "input_tensor" not in source and "alloc_smem" not in source
    assert "sync_threads" not in source and "arrive_and_wait" not in source
    assert ("cute.copy(" in source) is not scalar
    assert "cutlass.range_constexpr(8)" in source
    if mode == "raw":
        assert "cute.math.exp2" in source
        assert "cutlass.Float16" in source and "cutlass.Float32" in source
    if mode == "mixed":
        assert result["outputs"][0].node.meta["val"].dtype == dtype
        assert result["outputs"][1].node.meta["val"].dtype == torch.float32


@pytest.mark.parametrize("mode", ["host", "partial"])
def test_any_remaining_host_input_rejects_before_unroll(mode):
    result = _capture(mode=mode)
    assert result["lines"] is None and not result["unroll"].activated


@pytest.mark.parametrize(
    "reason", ["overlap", "input_alias", "coords", "negative", "future_graph"]
)
def test_original_geometry_and_disjoint_output_guards(reason):
    def alter(outputs, boundaries, plan):
        left, right = outputs
        if reason == "overlap":
            right = replace(right, target=left.target)
        elif reason == "input_alias":
            right = replace(right, target=boundaries[left.node])
        elif reason == "coords":
            right = replace(right, coordinates=lambda row, col: (row, f"({col} + 1)"))
        elif reason == "negative":
            right = replace(right, offset=-1)
        else:
            from torch.fx import Graph

            node = Graph().placeholder("foreign")
            node.meta.update(right.node.meta)
            right = replace(right, node=node)
            boundaries[node] = "foreign_image"
        return [left, right], boundaries

    result = _capture(alter=alter)
    assert result["lines"] is None and not result["unroll"].activated


@pytest.mark.parametrize("width", [0, 7, 24, 2048])
def test_existing_eight_value_geometry_rejects_invalid_width(width):
    result = _capture(shape=(128, width))
    assert result["lines"] is None and not result["unroll"].activated


def test_materialized_output_does_not_relax_original_domain_equality():
    original = chain._operand_domain

    def different(cg, node, coords, plan):
        domain = original(cg, node, coords, plan)
        if node is plan.dots[0].args[1]:
            return [*domain, "different_extent < 7"]
        return domain

    with patch.object(chain, "_operand_domain", different):
        result = _capture()
    assert result["lines"] is None and not result["unroll"].activated


def test_no_actual_boundary_read_does_not_grant_ownership_schedule():
    original = chain._Expression.value

    def value(self, node, coords):
        if node in self.boundaries:
            # Models a pure constant output: no host load, but also no
            # materialized input actually visited in the coordinate memo.
            return "cutlass.BFloat16(1)"
        return original(self, node, coords)

    with patch.object(chain._Expression, "value", value):
        result = _capture()
    assert result["lines"] is None and not result["unroll"].activated


@pytest.mark.parametrize("row", [0, 127, 128, 135])
@pytest.mark.parametrize("scalar", [False, True])
def test_typed_boundary_padding_and_original_target_offset(row, scalar):
    def offset(outputs, boundaries, plan):
        return [outputs[0], replace(outputs[1], offset=128)], boundaries

    result = _capture(shape=(136, 128), scalar=scalar, alter=offset)
    assert result["lines"] is not None
    source = "\n".join(result["lines"])
    pointwise = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For) and ast.unparse(node.target) == "joined_element"
    )
    first = [float("nan")] * 8
    second = [float("nan")] * 8
    a, b = {}, {}
    namespace = {
        "cutlass": SimpleNamespace(BFloat16=float, range_constexpr=range),
        "retained_0": torch.ones((128, 128)),
        "retained_1": torch.full((128, 128), 2.0),
        "joined_output_0_values": first,
        "joined_output_1_values": second,
        "chain_0_a": a,
        "chain_0_b": b,
        "joined_row": row,
        "joined_base": 0,
        **{
            node.id: 0
            for node in ast.walk(pointwise)
            if isinstance(node, ast.Name) and node.id.startswith("chain_origin_")
        },
    }
    exec(
        compile(
            ast.Module(body=[pointwise], type_ignores=[]), "<materialized>", "exec"
        ),
        namespace,
    )
    expected = (1.0, 2.0) if row < 128 else (0.0, 0.0)
    if scalar:
        assert a == {(row, column): expected[0] for column in range(8)}
        assert b == {(row + 128, column): expected[1] for column in range(8)}
    else:
        assert first == [expected[0]] * 8 and second == [expected[1]] * 8
        assert "cute.domain_offset((128, 0), chain_0_b)" in source


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("row", [0, 128])
def test_masked_boundary_zero_feeds_original_nonlinearity_before_padding(scalar, row):
    result = _capture(mode="raw", scalar=scalar, final_domain=False, shape=(136, 128))
    assert result["lines"] is not None
    pointwise = next(
        node
        for node in ast.walk(ast.parse("\n".join(result["lines"])))
        if isinstance(node, ast.For) and ast.unparse(node.target) == "joined_element"
    )
    first, second = [float("nan")] * 8, [float("nan")] * 8
    a, b = {}, {}
    namespace = {
        "cutlass": SimpleNamespace(
            Float16=lambda x: torch.tensor(x, dtype=torch.float16).item(),
            Float32=lambda x: torch.tensor(x, dtype=torch.float32).item(),
            BFloat16=lambda x: torch.tensor(x, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(math=SimpleNamespace(exp2=lambda x, **kwargs: 2.0**x)),
        **{name: torch.zeros((128, 128)) for name in result["boundaries"].values()},
        "joined_output_0_values": first,
        "joined_output_1_values": second,
        "chain_0_a": a,
        "chain_0_b": b,
        "joined_row": row,
        "joined_base": 0,
    }
    exec(
        compile(
            ast.Module(body=[pointwise], type_ignores=[]),
            "<masked-materialized>",
            "exec",
        ),
        namespace,
    )
    expected = 2.0 if row < 128 else 0.0
    if scalar:
        assert a == b == {(row, column): expected for column in range(8)}
    else:
        assert first == second == [expected] * 8


@pytest.mark.parametrize("threads", [128, 384])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("previously_active", [False, True])
def test_materialized_frontier_joins_all_outputs_without_faking_activation(
    threads, scalar, previously_active
):
    plan, frame = frontier_fixture._frame()
    execution = ChainedExecution(
        threads, thread="prep_thread", sync="role.arrive_and_wait()"
    )
    vector = VectorStaging(True, previously_active, True, previously_active)
    with (
        patch.object(frontiers, "emit_materialized_group", return_value=["publish()"]),
        patch.object(
            frontiers, "emit_vector_group", side_effect=AssertionError("not CSE")
        ),
    ):
        result = frontiers.emit_frontier_group(
            Mock(),
            plan,
            frame,
            2,
            {},
            execution,
            vector,
            BoundedProducerUnroll(1),
            scalar_targets={buffer.name for buffer in frame.buffers}
            if scalar
            else set(),
            materialized=True,
        )
    assert result is not None
    lines, stop = result
    assert stop == 5 and lines[-1] == execution.sync
    assert lines.count(execution.sync) == 1
    assert (
        vector.activated is previously_active
        and vector.group_activated is previously_active
    )
    tree = ast.parse("\n".join(lines))
    assert ast.unparse(tree.body[-1]) == execution.sync
    assert isinstance(tree.body[0], ast.If) is (threads == 384 and not scalar)


@pytest.mark.parametrize(
    "enabled,group_enabled", [(False, False), (False, True), (True, False)]
)
def test_explicit_materialization_keeps_existing_feature_gates(enabled, group_enabled):
    plan, frame = frontier_fixture._frame()
    with patch.object(
        frontiers, "plan_frontier_group", side_effect=AssertionError("disabled")
    ):
        assert (
            frontiers.emit_frontier_group(
                Mock(),
                plan,
                frame,
                2,
                {},
                ChainedExecution(128),
                VectorStaging(enabled, group_enabled=group_enabled),
                BoundedProducerUnroll(1),
                scalar_targets=set(),
                materialized=True,
            )
            is None
        )


def test_earlier_reservation_is_preserved_and_frame_unchanged():
    _, frame = frontier_fixture._frame()
    for buffer in frame.buffers:
        frame = frontier_fixture._region(frame, buffer.name, live_from=1)
    before = repr(frame)
    result = frontiers.plan_frontier_group(frame, 2)
    assert result is not None and result.stop_event == 5
    assert repr(frame) == before


def test_earlier_reservation_must_not_be_shortened_to_hide_alias():
    _, frame = frontier_fixture._frame()
    frame = frontier_fixture._region(frame, "image_2", byte_offset=0, live_from=0)
    frame = frontier_fixture._region(frame, "prefix", live_from=0, live_until=1)
    frame = replace(
        frame, actions=tuple(replace(action, reads=()) for action in frame.actions)
    )
    result = frontiers.plan_frontier_group(frame, 2)
    assert result is not None and result.stop_event == 4


@pytest.mark.parametrize("index", [0, 1, 2])
def test_future_allocation_start_is_fail_closed(index):
    _, frame = frontier_fixture._frame()
    frame = frontier_fixture._region(frame, f"image_{index}", live_from=index + 3)
    result = frontiers.plan_frontier_group(frame, 2)
    assert (result is None) is (index < 2)
    if result is not None:
        assert result.stop_event == 4


@pytest.mark.parametrize("writer", [0, 1, 2])
def test_early_storage_reservation_is_not_actual_publication(writer):
    _, frame = frontier_fixture._frame()
    source = f"image_{writer}"
    frame = frontier_fixture._region(frame, source, live_from=1)
    frame = replace(
        frame,
        actions=tuple(
            replace(action, reads=(source,)) if action.event == 3 else action
            for action in frame.actions
        ),
    )
    # Even image_0 is not available *before* this group's first action2.
    # Moving its reservation to1 does not grant a different execution order.
    assert frontiers.plan_frontier_group(frame, 2) is None


def test_failed_materialization_does_not_publish_or_activate():
    plan, frame = frontier_fixture._frame()
    vector = VectorStaging(True, group_enabled=True)
    tracker = BoundedProducerUnroll(2)
    with patch.object(frontiers, "emit_materialized_group", return_value=None):
        assert (
            frontiers.emit_frontier_group(
                Mock(),
                plan,
                frame,
                2,
                {},
                ChainedExecution(128),
                vector,
                tracker,
                scalar_targets=set(),
                materialized=True,
            )
            is None
        )
    assert not vector.activated and not vector.group_activated and not tracker.activated


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalar", [False, True])
def test_default_vector_group_source_is_identical_to_frozen_snapshot(dtype, scalar):
    after = fixture._capture(dtype=dtype, scalar=scalar)[0]
    assert after is not None
    assert (
        hashlib.sha256(json.dumps(after).encode()).hexdigest()
        == _FROZEN_DIGESTS[dtype, scalar]
    )
