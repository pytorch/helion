from __future__ import annotations

from dataclasses import replace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention import _case
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_pipeline import _config as _pipeline_config
import helion
from helion._compiler.cute import chained_operand_retention_emission as proof
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute.chained_frontier_groups import plan_frontier_group
from helion._compiler.cute.chained_matmul import _shape
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_operand_retention import discover_operand_retention
from helion._compiler.cute.chained_operand_retention import plan_operand_retention_frame
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def _typed_reuse(a, b, initial, masked: hl.constexpr, transpose: hl.constexpr):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=32):
        state = initial[rows, rows]
        for step in hl.tile(steps, block_size=1):
            if masked:
                raw = hl.load(
                    a,
                    [step.id, rows, rows],
                    extra_mask=(rows.index % 3 != 0)[:, None],
                ).float()
            else:
                raw = a[step.id, rows, rows].float()
            x = torch.exp(raw * 0.125).to(a.dtype)
            y = (b[step.id, rows, rows].float() + 0.25).to(a.dtype)
            first = hl.dot(x, y.T, out_dtype=torch.float32)
            second = hl.dot(x, (y.float() + 1).to(a.dtype).T, out_dtype=torch.float32)
            reused = x
            if transpose:
                reused = x.T
            combined = (first + second + reused.float() + y.float()).to(a.dtype)
            state = hl.dot(state.to(a.dtype), combined, acc=state)
            history[step.id, rows, rows] = state
        final[rows, rows] = state
    return history, final


def _config():
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
    )


def _discover(plan, frame):
    assert plan.region is not None
    shapes = {
        node: _shape(node)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    return candidates, shapes


def _capture(kernel, args, config, check):
    original = pipeline_module._prepare
    seen = []
    before = torch.cuda.is_initialized()

    def observe(cg, plan, pipeline, *rest):
        candidates, shapes = _discover(plan, pipeline.frame)
        check(cg, plan, pipeline, candidates, shapes)
        seen.append(True)
        return original(cg, plan, pipeline, *rest)

    with patch.object(pipeline_module, "_prepare", observe):
        source = _source(kernel, args, config)
    assert seen == [True]
    assert torch.cuda.is_initialized() == before
    return source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "masked,transpose", [(False, False), (True, False), (True, True)]
)
def test_original_typed_masks_and_consumer_axis_maps_cpu(dtype, masked, transpose):
    args = (
        torch.empty((3, 32, 32), dtype=dtype),
        torch.empty((3, 32, 32), dtype=dtype),
        torch.empty((32, 32), dtype=torch.float32),
        masked,
        transpose,
    )

    def check(cg, plan, pipeline, candidates, shapes):
        assert len(candidates) == 2
        body, aliases = list(cg.statements_stack[-1]), dict(plan.tensor_aliases)
        accepted = proof.admit_operand_retention(cg, plan, pipeline.frame, candidates)
        assert accepted == candidates
        assert all(item.dtype == dtype and item.row_offset == 0 for item in accepted)
        assert list(cg.statements_stack[-1]) == body
        assert all(plan.tensor_aliases[key] == value for key, value in aliases.items())
        assert not any(
            "chain_retained_proof" in str(line) for line in cg.statements_stack[-1]
        )
        consumer_roots = {
            node
            for action in pipeline.frame.actions
            if action.kind == "frontier"
            for node in action.nodes
        }
        assert all(item.operand not in consumer_roots for item in candidates)
        original_domain = proof._operand_domain

        def padded_consumer(cg, node, coordinates, plan):
            return (
                ["coord_0 < 31"]
                if node in consumer_roots
                else original_domain(cg, node, coordinates, plan)
            )

        with patch.object(proof, "_operand_domain", padded_consumer):
            assert (
                proof.admit_operand_retention(cg, plan, pipeline.frame, candidates)
                == ()
            )

    source = _capture(_typed_reuse, args, _config(), check)
    assert "chain_retained_proof" not in source


def test_unsupported_consumer_actions_and_extra_use_cannot_be_ignored_cpu():
    args = (
        torch.empty((1, 32, 32), dtype=torch.bfloat16),
        torch.empty((1, 32, 32), dtype=torch.bfloat16),
        torch.empty((32, 32), dtype=torch.float32),
        True,
        False,
    )

    def check(cg, plan, pipeline, candidates, shapes):
        frame = pipeline.frame
        event = candidates[0].consumer_events[0]
        assert frame.actions[event].kind == "frontier"
        for kind in ("cache", "collective", "leaf"):
            changed = replace(
                frame,
                actions=tuple(
                    replace(action, kind=kind) if action.event == event else action
                    for action in frame.actions
                ),
            )
            discovered, _ = _discover(plan, changed)
            assert discovered
            assert proof.admit_operand_retention(cg, plan, changed, discovered) == ()

    _capture(_typed_reuse, args, _config(), check)


@pytest.mark.parametrize(
    "coordinates,shape,extents,expected",
    [
        (("r", "k"), (32, 128), {"r": 32, "k": 128}, True),
        (("k", "r"), (128, 32), {"r": 32, "k": 128}, True),
        (("0", "k"), (1, 128), {"k": 128}, True),
        (("r", "k"), (31, 128), {"r": 32, "k": 128}, False),
        (("r", "k"), (32, 127), {"r": 32, "k": 128}, False),
        (("r+1", "k"), (32, 128), {"r": 31, "k": 128}, False),
        (("Int32(r)", "k"), (32, 128), {"r": 32, "k": 128}, False),
        (("r%32", "k"), (32, 128), {"r": 64, "k": 128}, False),
        (("indices[r]", "k"), (32, 128), {"r": 32, "k": 128}, False),
        (("r", "k"), (32, 128), {"r": 0, "k": 128}, False),
        (("0",), (0,), {}, False),
        (("r",), (32, 128), {"r": 32}, False),
    ],
)
def test_only_exact_bounded_coordinate_maps(coordinates, shape, extents, expected):
    assert proof._in_bounds(coordinates, shape, extents) is expected


def test_early_reserved_bytes_are_not_published_values():
    plan, frame, _, _ = _case()
    early = replace(
        frame,
        layout=replace(
            frame.layout,
            regions=tuple(
                replace(region, live_from=0) if region.name == "result_0" else region
                for region in frame.layout.regions
            ),
        ),
    )
    node = plan.dots[0]
    publications = proof._publications(early)
    assert node not in publications[0]
    assert node not in publications[1]
    assert node not in publications[2]
    # Existing result lifetime ends at event3, so it is never available later.
    assert node not in publications[3]
    raw = frame.buffers[0].node
    assert raw is not None
    assert raw not in publications[0]
    assert publications[1][raw] == "raw_0"


def test_frontier_reservations_use_original_safe_prefix_and_preserve_sizes():
    plan, frame, _, _ = _case()
    requests = proof.operand_retention_reservations(
        plan, frame, register_islands=False, frontier_groups=True
    )
    group = plan_frontier_group(frame, 3)
    assert group is not None and group.stop_event == 6
    assert tuple(request.name for request in requests) == tuple(
        buffer.name for buffer in group.buffers
    )
    for request in requests:
        region = frame.layout.region(request.name)
        assert request.live_from == 3
        assert request.live_until == region.live_until
        assert (request.byte_size, request.alignment) == (
            region.byte_size,
            region.alignment,
        )
    assert (
        proof.operand_retention_reservations(
            plan, frame, register_islands=False, frontier_groups=False
        )
        == ()
    )


@pytest.mark.parametrize("bad", [1, None, "true"])
def test_reservation_and_admission_flags_are_strict(bad):
    plan, frame, _, _ = _case()
    with pytest.raises(TypeError):
        proof.operand_retention_reservations(
            plan, frame, register_islands=False, frontier_groups=bad
        )
    with pytest.raises(TypeError):
        proof.admit_operand_retention(
            cast("Any", None), plan, frame, (), register_islands=bad
        )


def test_invalid_frame_cannot_omit_required_reservations():
    plan, frame, _, _ = _case()
    broken = replace(frame, actions=frame.actions[:-1])
    with pytest.raises(_UnsupportedChain):
        proof.operand_retention_reservations(
            plan, broken, register_islands=True, frontier_groups=True
        )


def test_actual_kda_three_typed_images_and_conservative_reservations_cpu():
    kernel, args = _kda_fixture()
    config = _pipeline_config(16, pipeline=True)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=4,
        cute_chained_preparation_cohorts=3,
        cute_chained_register_islands=True,
        cute_chained_vector_group=True,
    )

    def check(cg, plan, pipeline, candidates, shapes):
        frame = pipeline.frame
        accepted = proof.admit_operand_retention(
            cg, plan, frame, candidates, register_islands=True
        )
        first = tuple(item for item in accepted if item.group == frame.stages[0].group)
        assert len(first) == 3
        assert tuple(item.row_offset for item in first) == (0, 0, 32)
        requests = proof.operand_retention_reservations(
            plan, frame, register_islands=True, frontier_groups=True
        )
        retained = plan_operand_retention_frame(
            plan,
            frame,
            first,
            shapes,
            prepared_groups=pipeline.prepared_groups,
            reservations=requests,
            capacity_bytes=frame.layout.allocated_bytes,
        )
        assert retained is not None
        assert retained.frame.layout.allocated_bytes <= frame.layout.allocated_bytes
        assert retained.matches(
            plan,
            frame,
            shapes,
            prepared_groups=pipeline.prepared_groups,
            reservations=requests,
        )
        assert (
            proof.admit_operand_retention(
                cg, plan, replace(frame), first, register_islands=True
            )
            == ()
        )
        bad = (replace(first[0], row_offset=1), *first[1:])
        assert proof.admit_operand_retention(cg, plan, frame, bad) == ()
        with patch.object(proof, "_operand_domain", return_value=["row < 1"]):
            assert proof.admit_operand_retention(cg, plan, frame, first) == ()
        with patch.object(proof, "_finite_consumers", return_value=False):
            assert proof.admit_operand_retention(cg, plan, frame, first) == ()
        with patch.object(
            proof,
            "_publications",
            return_value={action.event: {} for action in frame.actions},
        ):
            assert proof.admit_operand_retention(cg, plan, frame, first) == ()
        island_stages = {
            index
            for island in proof._islands(plan, True)
            for group in island.groups
            for index in group.stages
        }
        assert all(
            not island_stages.intersection(item.group.stages) for item in accepted
        )

    _capture(kernel, args, config, check)
