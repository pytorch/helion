from __future__ import annotations

import pytest
import torch

from helion import exc
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.sparsecore_plan import DirectLoadPlan
from helion._compiler.pallas.sparsecore_plan import IndirectLoadPlan
from helion._compiler.pallas.sparsecore_plan import SparseCorePlanContext
from helion._compiler.pallas.sparsecore_plan import build_sparsecore_memory_plan
from helion.language import memory_ops


def _placeholder(
    graph: torch.fx.Graph, name: str, value: torch.Tensor
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def test_direct_stream_is_lowered_without_graph_topology() -> None:
    graph = torch.fx.Graph()
    source = torch.empty(1024, 64)
    result = torch.empty(32, 64)
    source_node = _placeholder(graph, "source", source)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    load = graph.call_function(memory_ops.load, (source_node, (tile_node, slice(None))))
    load.meta["val"] = result
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [TilePattern(0), ArbitrarySlicePattern(slice(None))],
    )

    plan = build_sparsecore_memory_plan(
        access,
        SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(plan, DirectLoadPlan)
    assert plan.stream is not None
    assert plan.stream.group == 0
    assert plan.stream.group_count == 1
    assert plan.stream.elements_per_item == 64
    assert plan.layout.storage_shape == (8, 64)


def test_indirect_load_retains_its_local_dependency() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(4096, 64)
    index = torch.empty(32, 4, dtype=torch.int32)
    result = torch.empty(32, 4, 64)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    load = graph.call_function(memory_ops.load, (table_node, (index_node, slice(None))))
    load.meta["val"] = result
    access = build_memory_access(
        load,
        table,
        list(load.args[1]),
        [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))],
    )

    plan = build_sparsecore_memory_plan(
        access,
        SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(plan, IndirectLoadPlan)
    assert plan.index_node is index_node
    assert plan.dependencies == frozenset({index_node})
    assert plan.layout.value_size == 256


def test_several_indirect_accesses_do_not_require_a_stack() -> None:
    graph = torch.fx.Graph()
    index = torch.empty(32, dtype=torch.int32)
    index_node = _placeholder(graph, "index", index)
    context = SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8)
    plans = []
    for number in range(3):
        table = torch.empty(4096, 64)
        table_node = _placeholder(graph, f"table{number}", table)
        load = graph.call_function(
            memory_ops.load, (table_node, (index_node, slice(None)))
        )
        load.meta["val"] = torch.empty(32, 64)
        access = build_memory_access(
            load,
            table,
            list(load.args[1]),
            [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))],
        )
        plans.append(build_sparsecore_memory_plan(access, context))

    assert all(isinstance(plan, IndirectLoadPlan) for plan in plans)
    assert [plan.index_node for plan in plans] == [index_node] * 3


def test_indirect_access_rejects_a_leading_tensor_dimension() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(2, 4096, 64)
    index = torch.empty(32, dtype=torch.int32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    load = graph.call_function(
        memory_ops.load, (table_node, (1, index_node, slice(None)))
    )
    load.meta["val"] = torch.empty(32, 64)
    access = build_memory_access(
        load,
        table,
        list(load.args[1]),
        [
            ArbitraryIndexPattern(1),
            TensorIndexPattern(),
            ArbitrarySlicePattern(slice(None)),
        ],
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="indirect DMA requires the index to address the first tensor dimension",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )


def test_direct_store_rejects_a_leading_tensor_dimension() -> None:
    graph = torch.fx.Graph()
    output = torch.empty(2, 1024, 64)
    value = torch.empty(32, 64)
    output_node = _placeholder(graph, "output", output)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    value_node = _placeholder(graph, "value", value)
    store = graph.call_function(
        memory_ops.store,
        (output_node, (1, tile_node, slice(None)), value_node),
    )
    access = build_memory_access(
        store,
        output,
        list(store.args[1]),
        [
            ArbitraryIndexPattern(1),
            TilePattern(0),
            ArbitrarySlicePattern(slice(None)),
        ],
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="direct stores require the item axis to be the first tensor dimension",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )
