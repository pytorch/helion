from __future__ import annotations

from dataclasses import replace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_scan_producer import _config
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
from .test_cute_chained_scan_producer import _sequence_config
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_scan_producer_emission as emission_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_native_read_inputs import (
    bind_preparation_leaf_native_inputs,
)
from helion._compiler.cute.chained_native_reads import plan_native_vector_read
from helion._compiler.cute.chained_native_stores import plan_native_stmatrix_store
from helion._compiler.cute.chained_vector_expression import emit_vector_expression
from helion._compiler.cute.chained_vector_group import VectorGroupOutput
from helion._compiler.cute.chained_vector_group import emit_vector_group
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership


def _capture(kernel, values, config):
    stages, records = [], []
    stage_original = pipeline_module._emit_scan_producer_stage
    emit_original = emission_module.emit_scan_producer

    def stage(cg, plan, pipeline, *args, **kwargs):
        stages.append((plan, pipeline))
        return stage_original(cg, plan, pipeline, *args, **kwargs)

    def emit(cg, plan, scan, boundaries, *args, **kwargs):
        assert stages[-1][0] is plan
        pipeline = stages[-1][1]
        shapes = dict(scan.revision.shapes)
        inputs = bind_preparation_leaf_native_inputs(
            plan, pipeline, scan, boundaries, shapes
        )
        records.append((plan, pipeline, scan, dict(boundaries), shapes, inputs))
        return emit_original(cg, plan, scan, boundaries, *args, **kwargs)

    with (
        patch.object(pipeline_module, "_emit_scan_producer_stage", stage),
        patch.object(emission_module, "emit_scan_producer", emit),
    ):
        source = _source(kernel, values, config)
    assert len(records) == 1
    return records[0], source


@pytest.fixture(scope="module")
def generic_cases():
    # No CUDA-forbidden context is held open across the fixture yield/lifetime.
    return tuple(
        _capture(_scan_sequence, _sequence_args(dtype), _sequence_config())[0]
        for dtype in (torch.bfloat16, torch.float16)
    )


def _bind(case, **changes: Any):
    arguments: dict[str, Any] = dict(
        zip(("plan", "pipeline", "scan", "published", "shapes"), case[:5], strict=True)
    )
    arguments.update(changes)
    return bind_preparation_leaf_native_inputs(**arguments)


def test_published_original_leaf_inputs_and_actual_kda_three_leaves(generic_cases):
    for case in generic_cases:
        plan, pipeline, scan, published, _, inputs = case
        assert len(inputs) == 1  # Energy's raw input is not reread by the fill.
        source = inputs[0]
        assert source.matches(plan, published)
        leaf = next(
            leaf for leaf in pipeline.prepared_leaves if leaf.node is source.node
        )
        assert source.tensor == leaf.name
        assert source.shape == source.full_shape == scan.shape == (32, 64)
        assert source.row_offset == 0
    kernel, values = _kda_fixture()
    case, _ = _capture(kernel, values, _config())
    plan, pipeline, scan, published, _, inputs = case
    assert len(inputs) == 3
    assert len({source.node for source in inputs}) == 3
    assert all(source.matches(plan, published) for source in inputs)
    assert all(source.shape == (32, 128) for source in inputs)
    assert {source.tensor for source in inputs} <= {
        leaf.name for leaf in pipeline.prepared_leaves
    }
    assert scan.prelude


@pytest.mark.parametrize("index", (0, 1))
def test_no_invented_alias_or_unpublished_authority(generic_cases, index):
    case = generic_cases[index]
    plan, _, _, published, _, inputs = case
    source = inputs[0]
    assert _bind(case, published={}) == ()
    assert _bind(case, published={source.node: "invented_native_alias"}) == ()
    assert not source.matches(plan, {source.node: "invented_native_alias"})
    for changes in (
        {"tensor": "invented_native_alias"},
        {"full_shape": (64, 64)},
        {"row_offset": 1},
        {"index": False},
        {"dtype": torch.float32},
    ):
        assert not replace(source, **changes).matches(plan, published)


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("field", ("kernel_args", "proof", "dtype", "shape", "stride"))
def test_bound_leaf_witness_rechecks_mutable_and_typed_facts(
    generic_cases, index, field
):
    case = generic_cases[index]
    plan, pipeline, _, published, _, inputs = case
    source = inputs[0]
    leaf = next(leaf for leaf in pipeline.prepared_leaves if leaf.node is source.node)
    if field == "kernel_args":
        old = leaf.wrapper["kernel_args"]
        leaf.wrapper["kernel_args"] = ["changed_atom", "changed_tensor"]
        try:
            assert not source.matches(plan, published)
        finally:
            leaf.wrapper["kernel_args"] = old
    elif field == "proof":
        changed = replace(leaf, proof=replace(leaf.proof, mask="False"))
        altered = replace(
            pipeline,
            prepared_leaves=tuple(
                changed if item is leaf else item for item in pipeline.prepared_leaves
            ),
        )
        assert _bind(case, pipeline=altered) == ()
    else:
        old = source.node.meta["val"]
        if field == "dtype":
            value = torch.empty((32, 64), dtype=torch.float32)
        elif field == "shape":
            value = torch.empty((16, 64), dtype=source.dtype)
        else:
            value = torch.empty((64, 32), dtype=source.dtype).T
        source.node.meta["val"] = value
        try:
            assert not source.matches(plan, published)
            assert _bind(case) == ()
        finally:
            source.node.meta["val"] = old
    assert source.matches(plan, published)


@pytest.mark.parametrize("field", ("phases", "regions", "stage", "leaf_read"))
def test_changed_effective_phase_owner_or_destination_rejects(generic_cases, field):
    case = generic_cases[0]
    _, pipeline, scan, _, _, inputs = case
    if field == "phases":
        changed = replace(scan, phases=tuple(0 for _ in scan.phases))
    elif field == "regions":
        changed = replace(
            scan,
            regions=tuple(replace(region, live_until=0) for region in scan.regions),
        )
    elif field == "stage":
        owner = pipeline.frame.layout.region(inputs[0].tensor)
        changed = replace(scan, stage=replace(scan.stage, a=owner))
    else:
        leaves = tuple(
            replace(leaf, read_events=()) for leaf in pipeline.prepared_leaves
        )
        assert _bind(case, pipeline=replace(pipeline, prepared_leaves=leaves)) == ()
        return
    assert (
        _bind(case, scan=changed, pipeline=replace(pipeline, scan_producer=changed))
        == ()
    )


@pytest.mark.parametrize("order", (None, False, 0, "row", "column", (), []))
def test_strict_thread_order(order):
    assert (
        plan_vector_ownership((32, 128), 128, tile_columns=32, thread_order=order)
        is None
    )


def test_unproved_publication_order_rejects_before_expression(generic_cases):
    owner = plan_vector_ownership(
        (32, 128), 128, tile_columns=32, thread_order="column_major"
    )
    old = plan_vector_ownership((32, 128), 128, tile_columns=32)
    assert owner is not None and old is not None
    assert (
        plan_native_stmatrix_store((128, 32), (32, 128), 0, torch.bfloat16, old)
        is not None
    )
    assert (
        plan_native_stmatrix_store((128, 32), (32, 128), 0, torch.bfloat16, owner)
        is None
    )
    plan = generic_cases[0][0]

    def coordinates(row, column):
        return row, column

    with patch(
        "helion._compiler.cute.chained_matmul._Expression",
        side_effect=AssertionError("must reject before expression emission"),
    ):
        assert (
            emit_vector_expression(
                cast("Any", None),
                plan,
                {},
                plan.dots[0],
                shape=(32, 128),
                coordinates=coordinates,
                offset=0,
                tag="review",
                target="target",
                execution=ChainedExecution(128),
                ownership=owner,
            )
            is None
        )
        outputs = tuple(
            VectorGroupOutput(node, f"target_{index}", coordinates)
            for index, node in enumerate(plan.dots[:2])
        )
        assert (
            emit_vector_group(
                cast("Any", None),
                plan,
                {},
                outputs,
                shape=(32, 128),
                tag="review",
                execution=ChainedExecution(128),
                ownership=owner,
            )
            is None
        )


@pytest.mark.parametrize(
    "shape,threads,columns",
    (
        ((32, 128), 128, 32),
        ((33, 64), 64, 16),
        ((7, 192), 32, 64),
        ((97, 256), 512, 64),
    ),
)
def test_column_major_exact_coverage_and_stale_fields(shape, threads, columns):
    owner = plan_vector_ownership(
        shape, threads, tile_columns=columns, thread_order="column_major"
    )
    assert owner is not None and owner.matches(shape, threads)
    cells = []
    for step in range(owner.trips):
        for thread in range(threads):
            coordinates = {"thread": thread, "step": step}
            row = eval(owner.row_expression("thread", "step"), {}, coordinates)
            base = eval(owner.base_expression("thread", "step"), {}, coordinates)
            if row < shape[0]:
                cells.extend((row, base + element) for element in range(8))
    assert len(cells) == len(set(cells)) == shape[0] * shape[1]
    assert set(cells) == {
        (row, column) for row in range(shape[0]) for column in range(shape[1])
    }
    assert not replace(owner, thread_rows=owner.thread_rows + 1).matches(shape, threads)
    read = plan_native_vector_read(
        ((shape[0] + 7) // 8 * 8, shape[1]), shape, 0, torch.bfloat16, owner
    )
    assert read is not None
    assert (
        f"stride={owner.thread_strides!r}"
        in read.emit("native", "input", "thread", "step").setup[0]
    )
    assert (
        plan_native_stmatrix_store((128, 32), (32, 128), 0, torch.bfloat16, owner)
        is None
    )
