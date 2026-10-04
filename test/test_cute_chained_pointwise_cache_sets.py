from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_pointwise_residency import _repeated
from .test_cute_chained_pointwise_residency import _unary
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute import chained_pointwise_residency as residency
from helion._compiler.cute.chained_pointwise_residency import allocate_pointwise_cache
from helion._compiler.cute.chained_pointwise_residency import plan_pointwise_cache
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts


def _independent(extents=(16, 16, 16), counts=(2, 2, 2)):
    graph = Graph()
    operands, dots = [], []
    for index, (extent, uses) in enumerate(zip(extents, counts, strict=True)):
        source = _input(graph, f"input_{index}", (extent, extent), torch.float32)
        computed = _unary(graph, torch.ops.aten.exp.default, source)
        operand = _convert(graph, computed, torch.bfloat16)
        right = _input(graph, f"rhs_{index}", (extent, extent), torch.bfloat16)
        operands.append(operand)
        dots.extend(_dot(graph, operand, right) for _ in range(uses))
    return _plan(graph, tuple(dots)), operands


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rank", (1, 2))
@pytest.mark.parametrize("budget", (128, 512, 4096))
def test_default_single_entry_exactly_matches_explicit_limit(dtype, rank, budget):
    plan, _ = _repeated(dtype, rank=rank)
    implicit = plan_pointwise_cache(plan, budget)
    explicit = plan_pointwise_cache(plan, budget, max_entries=1)
    assert implicit == explicit
    assert allocate_pointwise_cache(
        implicit, ScratchLayouts()
    ) == allocate_pointwise_cache(explicit, ScratchLayouts())


@pytest.mark.parametrize("limit", (1, 2, 4))
def test_independent_candidates_keep_original_score_and_later_node_tie_order(limit):
    plan, operands = _independent()
    cache = plan_pointwise_cache(plan, 4096, max_entries=limit)
    selected = list(reversed(operands))[:limit]
    assert [entry.node for entry in cache.entries] == selected
    assert [entry.name for entry in cache.entries] == [
        f"chain_pointwise_cache_{index}" for index in range(len(selected))
    ]
    assert [entry.first_stage for entry in cache.entries] == [4, 2, 0][:limit]
    assert [entry.last_stage for entry in cache.entries] == [5, 3, 1][:limit]
    assert all(entry.dtype == torch.bfloat16 for entry in cache.entries)
    assert all(entry.shape == (16, 16) for entry in cache.entries)
    assert all(entry.estimated_saved_operations == 2304 for entry in cache.entries)
    assert cache.shared_bytes == 512 * len(selected)
    assert cache.entries[0] == plan_pointwise_cache(plan, 4096).entries[0]
    assert cache == plan_pointwise_cache(plan, 4096, max_entries=limit)


@pytest.mark.parametrize("limit", (2, 4))
@pytest.mark.parametrize("budget", (0, 127, 128, 511, 512, 1023, 1024, 1536))
def test_one_total_aligned_budget_for_all_entries(limit, budget):
    plan, _ = _independent()
    cache = plan_pointwise_cache(plan, budget, max_entries=limit)
    assert len(cache.entries) == min(limit, 3, budget // 512)
    assert cache.shared_bytes <= budget
    assert cache.shared_bytes == sum(entry.allocated_bytes for entry in cache.entries)


def test_greedy_budget_skip_does_not_prevent_a_later_smaller_candidate():
    plan, operands = _independent((16, 32, 8), (4, 2, 2))
    cache = plan_pointwise_cache(plan, 2304, max_entries=4)
    assert [entry.node for entry in cache.entries] == [operands[0], operands[2]]
    assert cache.shared_bytes == 512 + 128


@pytest.mark.parametrize("limit", (2, 4))
def test_nested_cast_arithmetic_candidates_are_not_selected_together(limit):
    plan, _ = _repeated()
    assert plan_pointwise_cache(
        plan, 1 << 20, max_entries=limit
    ) == plan_pointwise_cache(plan, 1 << 20)


@pytest.mark.parametrize("shared", ("host", "arithmetic", "boundary"))
def test_only_published_boundary_ancestry_may_be_shared(shared):
    graph = Graph()
    if shared == "boundary":
        lhs = _input(graph, "left", (16, 16), torch.bfloat16)
        rhs = _input(graph, "right", (16, 16), torch.bfloat16)
        source = _dot(graph, lhs, rhs)
    else:
        source = _input(graph, "host", (16, 16), torch.float32)
        rhs = _input(graph, "right", (16, 16), torch.bfloat16)
        if shared == "arithmetic":
            source = _unary(graph, torch.ops.aten.exp.default, source)
    operands = [
        _convert(graph, _unary(graph, target, source), torch.bfloat16)
        for target in (torch.ops.aten.exp.default, torch.ops.aten.tanh.default)
    ]
    dots = tuple(_dot(graph, operand, rhs) for operand in operands for _ in range(2))
    cache = plan_pointwise_cache(_plan(graph, dots), 4096, max_entries=4)
    assert len(cache.entries) == (2 if shared == "boundary" else 1)
    if shared == "boundary":
        assert {entry.node for entry in cache.entries} == set(operands)
        assert all(entry.dependencies == (source,) for entry in cache.entries)
        assert [entry.first_stage for entry in cache.entries] == [3, 1]


def test_declining_an_overlapping_sibling_does_not_hide_independent_work():
    graph = Graph()
    source = _input(graph, "source", (16, 16), torch.float32)
    other = _input(graph, "other", (16, 16), torch.float32)
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    roots = (
        _unary(graph, torch.ops.aten.exp.default, source),
        _unary(graph, torch.ops.aten.tanh.default, source),
        _unary(graph, torch.ops.aten.exp.default, other),
    )
    operands = tuple(_convert(graph, node, torch.bfloat16) for node in roots)
    dots = tuple(_dot(graph, operand, rhs) for operand in operands for _ in range(2))
    cache = plan_pointwise_cache(_plan(graph, dots), 4096, max_entries=4)
    assert [entry.node for entry in cache.entries] == [operands[2], operands[1]]


def test_shared_zero_cost_coordinate_nodes_are_not_treated_as_independent():
    graph = Graph()
    coord = _input(graph, "shared_coordinate", (16, 16), torch.float32)
    rhs = _input(graph, "right", (16, 16), torch.bfloat16)
    operands = []
    for name in ("one", "two"):
        source = _input(graph, name, (16, 16), torch.float32)
        value = _call(
            graph,
            torch.ops.aten.add.Tensor,
            (source, coord),
            (16, 16),
            torch.float32,
        )
        value = _unary(graph, torch.ops.aten.exp.default, value)
        operands.append(_convert(graph, value, torch.bfloat16))
    dots = tuple(_dot(graph, operand, rhs) for operand in operands for _ in range(2))
    cache = plan_pointwise_cache(_plan(graph, dots), 4096, max_entries=4)
    assert len(cache.entries) == 1
    assert cache.entries[0].node is operands[1]


@pytest.mark.parametrize("limit", (True, False, 0, -1, 3, 8, 2.0, "2", None))
def test_limit_is_a_strict_supported_upper_bound(limit):
    plan, _ = _independent()
    with pytest.raises(ValueError, match="max_entries"):
        plan_pointwise_cache(plan, 4096, max_entries=limit)


def test_max_entries_does_not_require_an_effective_multi_entry_request():
    plan, _ = _repeated()
    assert not plan_pointwise_cache(plan, 0, max_entries=4).entries
    assert len(plan_pointwise_cache(plan, 4096, max_entries=4).entries) == 1


def test_selected_set_keeps_each_original_dtype_and_epilogue_lifetime():
    graph = Graph()
    computed, dots = [], []
    for index, dtype in enumerate((torch.float32, torch.float16, torch.bfloat16)):
        source = _input(graph, f"source_{index}", (16, 16), dtype)
        value = _unary(graph, torch.ops.aten.exp.default, source)
        rhs = _input(graph, f"rhs_{index}", (16, 16), dtype)
        computed.append(value)
        dots.extend((_dot(graph, value, rhs), _dot(graph, value, rhs)))
    plan = _plan(graph, (*dots, *computed))
    cache = plan_pointwise_cache(plan, 4096, max_entries=4)
    assert [entry.node for entry in cache.entries] == list(reversed(computed))
    assert [entry.dtype for entry in cache.entries] == [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ]
    assert [entry.allocated_bytes for entry in cache.entries] == [512, 512, 1024]
    assert all(entry.last_stage == len(dots) for entry in cache.entries)
    assert all(
        entry.shape == tuple(entry.node.meta["val"].shape) for entry in cache.entries
    )


def test_actual_kda_independent_reciprocal_norms_and_original_cache_fit():
    kernel, args = _kda_fixture()
    plan, cut, shapes = _capture(
        kernel,
        args,
        helion.Config(
            block_sizes=[128],
            num_warps=16,
            num_stages=2,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=True,
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_scan_schedule="warp",
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_pointwise_unroll=8,
            cute_chained_warp_mma_rows=32,
        ),
    )
    with patch.object(residency, "_shape", side_effect=shapes.__getitem__):
        original = plan_pointwise_cache(plan, 4096)
        two = plan_pointwise_cache(plan, 4096, max_entries=2)
        four = plan_pointwise_cache(plan, 4096, max_entries=4)
    assert len(original.entries) == 1 and original.shared_bytes == 2048
    assert two.entries[0] == four.entries[0] == original.entries[0]
    assert len(two.entries) == 2 and two.shared_bytes == 2176
    assert len(four.entries) == 3 and four.shared_bytes == 2304
    assert all(
        entry.node.target is torch.ops.aten.rsqrt.default
        and entry.dtype == torch.float32
        and entry.shape == (32,)
        and entry.first_stage == 0
        for entry in four.entries[1:]
    )
    assert four.entries[1].dependencies != four.entries[2].dependencies
    frame = plan_preparation_frame(replace(plan, pointwise_cache=four), cut, shapes)
    assert frame is not None
    cache_actions = [action for action in frame.actions if action.kind == "cache"]
    assert [action.nodes[0] for action in cache_actions] == [
        four.entries[1].node,
        four.entries[2].node,
        four.entries[0].node,
    ]
    for action in cache_actions:
        assert all(
            frame.layout.region(name).live_from < action.event for name in action.reads
        )
    for left, right in combinations(frame.layout.regions, 2):
        assert not (left.overlaps_lifetime(right) and left.overlaps_storage(right))
