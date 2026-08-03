from __future__ import annotations

from unittest.mock import Mock
from unittest.mock import patch

import torch

from helion import Config
from helion._compiler.pallas.memory_access import MemoryAccessKind
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import IndexingPattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.tensorcore_plan import OneHotGatherPlan
from helion._compiler.pallas.tensorcore_plan import ProjectionScatterPlan
from helion._compiler.pallas.tensorcore_plan import select_tensorcore_plan
from helion.language import memory_ops
from helion.language.atomic_ops import atomic_add


def _placeholder(
    graph: torch.fx.Graph, name: str, value: torch.Tensor
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def test_memory_access_snapshots_semantic_patterns() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    fx_subscript = (index_node, slice(None))
    subscript: list[object] = list(fx_subscript)
    load = graph.call_function(memory_ops.load, (table_node, fx_subscript))
    load.meta["val"] = torch.empty(16, 32)
    patterns: list[IndexingPattern] = [
        TensorIndexPattern(),
        ArbitrarySlicePattern(slice(None)),
    ]

    access = build_memory_access(load, table, subscript, patterns)
    # Later tiling changes must not alter recorded memory semantics.
    patterns[0] = TilePattern(0)

    assert access.kind is MemoryAccessKind.LOAD
    assert access.tensor is table
    assert access.value_node is None
    assert isinstance(access.patterns[0], TensorIndexPattern)


def test_store_and_atomic_memory_accesses_record_values() -> None:
    graph = torch.fx.Graph()
    output = torch.empty(64, 32)
    value = torch.empty(16, 32)
    output_node = _placeholder(graph, "output", output)
    value_node = _placeholder(graph, "value", value)
    fx_subscript = (slice(None), slice(None))
    subscript: list[object] = list(fx_subscript)
    patterns: list[IndexingPattern] = [
        TilePattern(0),
        ArbitrarySlicePattern(slice(None)),
    ]

    store = graph.call_function(
        memory_ops.store, (output_node, fx_subscript, value_node)
    )
    atomic = graph.call_function(atomic_add, (output_node, fx_subscript, value_node))

    store_access = build_memory_access(store, output, subscript, patterns)
    atomic_access = build_memory_access(atomic, output, subscript, patterns)

    assert store_access.kind is MemoryAccessKind.STORE
    assert atomic_access.kind is MemoryAccessKind.ATOMIC
    assert store_access.value_node is value_node
    assert atomic_access.value_node is value_node


def test_tensorcore_plan_owns_indirect_fallbacks() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    value = torch.empty(16, 32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    value_node = _placeholder(graph, "value", value)
    fx_subscript = (index_node, slice(None))
    subscript: list[object] = list(fx_subscript)
    patterns: list[IndexingPattern] = [
        TensorIndexPattern(),
        ArbitrarySlicePattern(slice(None)),
    ]
    load = graph.call_function(memory_ops.load, (table_node, fx_subscript))
    load.meta["val"] = value
    store = graph.call_function(
        memory_ops.store, (table_node, fx_subscript, value_node)
    )
    load_access = build_memory_access(load, table, subscript, patterns)
    store_access = build_memory_access(store, table, subscript, patterns)

    gather_fallback = object()
    scatter_fallback = object()
    native_gather = Mock(return_value=None)
    with (
        patch(
            "helion._compiler.pallas.tensorcore_plan._NATIVE_LOAD_CANDIDATES",
            (native_gather,),
        ),
        patch(
            "helion._compiler.pallas.gather.build_gather_plan",
            return_value=gather_fallback,
        ),
        patch(
            "helion._compiler.pallas.gather.build_scatter_plan",
            return_value=scatter_fallback,
        ),
    ):
        gather = select_tensorcore_plan(load_access, Config())
        scatter = select_tensorcore_plan(store_access, Config())

    assert isinstance(gather, OneHotGatherPlan)
    assert isinstance(scatter, ProjectionScatterPlan)
    assert gather.plan is gather_fallback
    assert scatter.plan is scatter_fallback
    native_gather.assert_called_once()
    assert all(isinstance(pattern, TensorIndexPattern) for pattern in patterns[:1])
