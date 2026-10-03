from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_register_binding as binding_module
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _shape
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_prepared_groups import _valid_frame
from helion._compiler.cute.chained_register_binding import bind_preparation_island
from helion._compiler.cute.chained_register_islands import plan_register_islands
from helion._compiler.cute.chained_tcgen_stage import stage_geometry
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _polynomial_loop(a, b, initial, keep_entry: hl.constexpr):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=16):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            i, j = hl.arange(size), hl.arange(size)
            dense = hl.dot(
                a[step.id, rows, j], b[step.id, i, j], out_dtype=torch.float32
            )
            base = hl.dot(b[step.id, i, j], b[step.id, i, j], out_dtype=torch.float32)
            factor = base.to(a.dtype)
            block = dense.to(a.dtype)
            square = hl.dot(block, factor, out_dtype=torch.float32)
            rounded = square.to(a.dtype)
            cube = hl.dot(rounded, factor, out_dtype=torch.float32)
            fourth = hl.dot(cube.to(a.dtype), factor, out_dtype=torch.float32)
            seed = state + fourth
            if keep_entry:
                seed = seed + dense
            state = hl.dot(
                state.to(a.dtype),
                factor,
                acc=seed,
                out_dtype=torch.float32,
            )
            history[step.id, rows, j] = state
        final[rows, :] = state
    return history, final


def _capture_binding(kernel, args, config, check=None, *, expect_binding=True):
    original = pipeline_module._prepare
    results = []

    def observe(cg, plan, pipeline, execution, vector, unroll, scratch):
        assert plan.region is not None
        frame = pipeline.frame
        shapes = {
            node: _shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        entries = (
            frozenset(entry.node for entry in plan.pointwise_cache.entries)
            if plan.pointwise_cache is not None
            else frozenset()
        )
        groups = plan.contraction_groups
        if groups is None:
            geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
            assert all(geometry is not None for geometry in geometries)
            groups = tuple(
                ContractionGroup((index,), (geometry,))
                for index, geometry in enumerate(geometries)
                if geometry is not None
            )
        candidates = plan_register_islands(
            plan.region,
            groups,
            shapes,
            fast_math=True,
            entry_boundaries=entries,
        )
        assert candidates
        for candidate in candidates:
            first = next(
                action.event
                for action in frame.actions
                if action.kind == "fill" and action.stages == candidate.groups[0].stages
            )
            published = {}
            for action in frame.actions[:first]:
                for name in action.writes:
                    buffer = next(
                        buffer for buffer in frame.buffers if buffer.name == name
                    )
                    if buffer.node is not None:
                        published[buffer.node] = name
            bound = bind_preparation_island(
                cg, plan, frame, candidate, execution, published
            )
            if not expect_binding:
                assert bound is None
                results.append(None)
                continue
            assert bound is not None
            assert bound.first_event == first
            assert bound.stop_event == first + len(candidate.groups) * 2
            assert bound.matches(plan, frame, execution, published)
            assert tuple(export.node for export in bound.exports) == candidate.exports
            assert all(
                export.region is frame.layout.region(export.name)
                for export in bound.exports
            )
            if check is not None:
                check(cg, plan, frame, candidate, execution, published, bound)
            results.append(bound)
        return original(cg, plan, pipeline, execution, vector, unroll, scratch)

    with patch.object(pipeline_module, "_prepare", observe):
        source = _source(kernel, args, config)
    return results, source


def test_actual_kda_binds_final_frame_and_original_expression_domains_cpu():
    kernel, args = _kda_fixture()
    config = _config(16, pipeline=True)
    config.config.update(
        block_sizes=[128],
        num_stages=2,
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
    )
    bindings, _ = _capture_binding(kernel, args, config, _check_rejections)
    assert len(bindings) == 1
    bound = bindings[0]
    assert bound is not None
    assert tuple(issue.stage for issue in bound.island.components[0].issues) == (
        2,
        3,
        4,
        5,
        6,
        7,
    )
    assert len(bound.exports) == 3
    for origin in bound.origins:
        assert origin.axes == ((0, 16), (0, 16))
        assert bound.coordinates(origin.node, "probe") == (
            "(0 + 16 * probe_warp + probe_coords[probe_index][0])",
            "(0 + 16 * probe_warp + probe_coords[probe_index][1])",
        )


def _check_rejections(cg, plan, frame, island, execution, published, bound):
    def rejected(
        *,
        candidate=island,
        actual_plan=plan,
        actual_frame=frame,
        actual_execution=execution,
        actual_published=published,
    ):
        assert (
            bind_preparation_island(
                cg,
                actual_plan,
                actual_frame,
                candidate,
                actual_execution,
                actual_published,
            )
            is None
        )

    entry = island.entries[0]
    rejected(
        actual_published={
            node: name for node, name in published.items() if node is not entry
        }
    )
    rejected(actual_published={**published, entry: "wrong_view"})
    rejected(
        actual_published={
            **published,
            island.components[0].issues[0].node: "already_published",
        }
    )
    rejected(actual_execution=ChainedExecution(32))
    rejected(actual_plan=replace(plan, warp_mma_stages=frozenset()))
    rejected(
        actual_plan=replace(
            plan, contraction_groups=tuple(reversed(plan.contraction_groups))
        )
    )
    rejected(candidate=replace(island, exports=()))
    record = island.values[0]
    rejected(
        candidate=replace(
            island,
            values=(
                replace(record, tiles=((0, (16, 0)), (1, (0, 16)))),
                *island.values[1:],
            ),
        )
    )
    rejected(
        candidate=replace(
            island,
            values=(
                replace(record, support_rows=(0,) * len(record.support_rows)),
                *island.values[1:],
            ),
        )
    )
    actions = list(frame.actions)
    actions[bound.first_event + 1] = replace(
        actions[bound.first_event + 1], kind="frontier"
    )
    rejected(actual_frame=replace(frame, actions=tuple(actions)))
    assert not bound.matches(plan, replace(frame), execution, published)
    assert not bound.matches(replace(plan), frame, execution, published)
    assert not bound.matches(plan, frame, execution, {**published, entry: "changed"})
    # Construct an otherwise valid final frame with a pending reader that dies
    # before this export's ORIGINAL definition, but after the new early zero.
    export = next(
        export
        for export in bound.exports
        if export.region.live_from > bound.first_event + 1
    )
    old = export.region
    pending = SharedBufferRegion(
        "pending_reader",
        frame.layout.allocated_bytes,
        old.byte_size,
        0,
        bound.first_event + 1,
        128,
    )
    regions = tuple(
        replace(region, byte_offset=pending.byte_offset)
        if region.name == export.name
        else region
        for region in frame.layout.regions
    )
    unsafe = replace(
        frame,
        layout=replace(
            frame.layout,
            regions=(*regions, pending),
            allocated_bytes=frame.layout.allocated_bytes + pending.byte_size,
        ),
        buffers=(
            *frame.buffers,
            PreparationBuffer(pending.name, "a", None, torch.float32, export.shape),
        ),
    )
    assert _valid_frame(unsafe)
    rejected(actual_frame=unsafe)
    with patch.object(binding_module, "_operand_domain", return_value=["not_proven"]):
        rejected()
    with patch.object(
        binding_module,
        "_operand_domain",
        side_effect=_UnsupportedChain("unsupported original domain"),
    ):
        rejected()
    coordinate_method = type(bound).coordinates

    def different_origin(self, node, prefix):
        row, column = coordinate_method(self, node, prefix)
        return (
            (f"({row} + 16)", column)
            if node is island.values[1].node
            else (row, column)
        )

    with patch.object(type(bound), "coordinates", different_origin):
        rejected()
    with patch.object(
        binding_module.CompileEnvironment.current().settings, "fast_math", False
    ):
        rejected()
        assert not bound.matches(plan, frame, execution, published)
    with (
        patch.object(
            binding_module,
            "_operand_domain",
            side_effect=ValueError("do not hide implementation errors"),
        ),
        pytest.raises(ValueError, match="do not hide"),
    ):
        bind_preparation_island(cg, plan, frame, island, execution, published)
    assert bound.matches(plan, frame, execution, published)


@pytest.mark.parametrize("keep_entry", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unrelated_polynomial_respects_actual_early_zero_alias_cpu(keep_entry, dtype):
    args = (
        torch.empty((3, 16, 16), dtype=dtype),
        torch.empty((3, 16, 16), dtype=dtype),
        torch.empty((16, 16), dtype=torch.float32),
        keep_entry,
    )
    config = _config(8, pipeline=True)
    bindings, source = _capture_binding(
        _polynomial_loop, args, config, expect_binding=keep_entry
    )
    assert bindings
    assert "chain_prep_barrier.arrive_and_wait()" in source
    if not keep_entry:
        # Original packing legally reuses entry C0 at C4's later publication.
        # It is NOT safe to zero that shared range before the island reads C0.
        assert bindings == [None]
        return
    assert bindings[0] is not None
    assert tuple(issue.stage for issue in bindings[0].island.components[0].issues) == (
        2,
        3,
        4,
    )
