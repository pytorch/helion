from __future__ import annotations

from unittest.mock import patch

import torch

from helion import Config
from helion._compiler.pallas.access import AccessKind
from helion._compiler.pallas.access import make_access_site
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.tensorcore_access import OneHotGatherAccess
from helion._compiler.pallas.tensorcore_access import ProjectionScatterAccess
from helion._compiler.pallas.tensorcore_access import select_tensorcore_access
from helion.language import memory_ops
from helion.language.atomic_ops import atomic_add


def _placeholder(
    graph: torch.fx.Graph, name: str, value: torch.Tensor
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def test_load_access_site_preserves_semantic_patterns() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    load = graph.call_function(memory_ops.load, (table_node, (index_node, slice(None))))
    load.meta["val"] = torch.empty(16, 32)
    patterns = [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))]

    site = make_access_site(load, table, list(load.args[1]), patterns)
    patterns[0] = TilePattern(0)

    assert site.kind is AccessKind.LOAD
    assert site.tensor is table
    assert site.value_node is None
    assert isinstance(site.patterns[0], TensorIndexPattern)


def test_store_and_atomic_access_sites_record_values() -> None:
    graph = torch.fx.Graph()
    output = torch.empty(64, 32)
    value = torch.empty(16, 32)
    output_node = _placeholder(graph, "output", output)
    value_node = _placeholder(graph, "value", value)
    patterns = [TilePattern(0), ArbitrarySlicePattern(slice(None))]

    store = graph.call_function(
        memory_ops.store, (output_node, (slice(None), slice(None)), value_node)
    )
    atomic = graph.call_function(
        atomic_add, (output_node, (slice(None), slice(None)), value_node)
    )

    store_site = make_access_site(store, output, list(store.args[1]), patterns)
    atomic_site = make_access_site(atomic, output, list(atomic.args[1]), patterns)

    assert store_site.kind is AccessKind.STORE
    assert atomic_site.kind is AccessKind.ATOMIC
    assert store_site.value_node is value_node
    assert atomic_site.value_node is value_node


def test_tensorcore_policy_owns_indirect_fallbacks() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    value = torch.empty(16, 32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    value_node = _placeholder(graph, "value", value)
    patterns = [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))]
    load = graph.call_function(memory_ops.load, (table_node, (index_node, slice(None))))
    load.meta["val"] = value
    store = graph.call_function(
        memory_ops.store, (table_node, (index_node, slice(None)), value_node)
    )
    load_site = make_access_site(load, table, list(load.args[1]), patterns)
    store_site = make_access_site(store, table, list(store.args[1]), patterns)

    gather_fallback = object()
    scatter_fallback = object()
    with (
        patch(
            "helion._compiler.pallas.gather.build_gather_plan",
            return_value=gather_fallback,
        ),
        patch(
            "helion._compiler.pallas.gather.build_scatter_plan",
            return_value=scatter_fallback,
        ),
    ):
        gather = select_tensorcore_access(load_site, Config())
        scatter = select_tensorcore_access(store_site, Config())

    assert isinstance(gather, OneHotGatherAccess)
    assert isinstance(scatter, ProjectionScatterAccess)
    assert gather.fallback is gather_fallback
    assert scatter.fallback is scatter_fallback
    assert all(isinstance(pattern, TensorIndexPattern) for pattern in patterns[:1])
