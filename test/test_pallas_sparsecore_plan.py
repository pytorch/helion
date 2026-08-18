from __future__ import annotations

import pytest
import torch

from helion import exc
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import NonePattern
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


def test_direct_load_plan() -> None:
    """Helion: `value = source[tile, :]`."""
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
    assert plan.transfer.prefix_index == 0
    assert plan.transfer.prefix_count == 1
    assert plan.transfer.elements_per_item == 64
    assert plan.layout.storage_shape == (8, 64)


def test_matching_suffix_tile_is_accepted() -> None:
    """Helion: `value = source[item_tile, value_tile]`."""
    graph = torch.fx.Graph()
    source = torch.empty(1024, 64)
    source_node = _placeholder(graph, "source", source)
    item_tile = _placeholder(graph, "item_tile", torch.empty(32, dtype=torch.int32))
    value_tile = _placeholder(graph, "value_tile", torch.empty(64, dtype=torch.int32))
    load = graph.call_function(memory_ops.load, (source_node, (item_tile, value_tile)))
    load.meta["val"] = torch.empty(32, 64)
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [TilePattern(0), TilePattern(1)],
    )

    plan = build_sparsecore_memory_plan(
        access,
        SparseCorePlanContext({0: 256, 1: 64}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(plan, DirectLoadPlan)
    assert plan.transfer.elements_per_item == 64


def test_large_suffix_tile_is_rejected() -> None:
    """Helion: a value tile wider than `source.size(1)`."""
    graph = torch.fx.Graph()
    source = torch.empty(1024, 64)
    source_node = _placeholder(graph, "source", source)
    item_tile = _placeholder(graph, "item_tile", torch.empty(32, dtype=torch.int32))
    value_tile = _placeholder(graph, "value_tile", torch.empty(128, dtype=torch.int32))
    load = graph.call_function(memory_ops.load, (source_node, (item_tile, value_tile)))
    load.meta["val"] = torch.empty(32, 128)
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [TilePattern(0), TilePattern(1)],
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="direct DMA requires all dimensions after the item axis",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext(
                {0: 256, 1: 128}, item_block_id=0, items_per_subcore=8
            ),
        )


def test_leading_broadcast_is_rejected() -> None:
    """Helion: `value = source[None, tile, :]`."""
    graph = torch.fx.Graph()
    source = torch.empty(1024, 64)
    source_node = _placeholder(graph, "source", source)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    load = graph.call_function(
        memory_ops.load, (source_node, (None, tile_node, slice(None)))
    )
    load.meta["val"] = torch.empty(1, 32, 64)
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [NonePattern(), TilePattern(0), ArbitrarySlicePattern(slice(None))],
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="broadcast dimensions before the item axis",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )


def test_negative_static_prefix_is_normalized() -> None:
    """Helion: `value = source[-1, tile, :]`."""
    graph = torch.fx.Graph()
    source = torch.empty(3, 1024, 64)
    source_node = _placeholder(graph, "source", source)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    load = graph.call_function(
        memory_ops.load, (source_node, (-1, tile_node, slice(None)))
    )
    load.meta["val"] = torch.empty(32, 64)
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [
            ArbitraryIndexPattern(-1),
            TilePattern(0),
            ArbitrarySlicePattern(slice(None)),
        ],
    )

    plan = build_sparsecore_memory_plan(
        access,
        SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(plan, DirectLoadPlan)
    assert plan.transfer.prefix_index == 2
    assert plan.transfer.prefix_count == 3


@pytest.mark.parametrize("index", (-4, 3))
def test_out_of_bounds_static_prefix_is_rejected(index: int) -> None:
    """Helion: `value = source[index, tile, :]` with an invalid static index."""
    graph = torch.fx.Graph()
    source = torch.empty(3, 1024, 64)
    source_node = _placeholder(graph, "source", source)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    load = graph.call_function(
        memory_ops.load, (source_node, (index, tile_node, slice(None)))
    )
    load.meta["val"] = torch.empty(32, 64)
    access = build_memory_access(
        load,
        source,
        list(load.args[1]),
        [
            ArbitraryIndexPattern(index),
            TilePattern(0),
            ArbitrarySlicePattern(slice(None)),
        ],
    )

    with pytest.raises(exc.InvalidConfig, match="static index .* is out of bounds"):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )


def test_broadcast_store_is_rejected() -> None:
    """Helion: `output[tile, :] = scalar`."""
    graph = torch.fx.Graph()
    output = torch.empty(1024, 64)
    output_node = _placeholder(graph, "output", output)
    tile_node = _placeholder(graph, "tile", torch.empty(32, dtype=torch.int32))
    value_node = _placeholder(graph, "value", torch.empty(()))
    store = graph.call_function(
        memory_ops.store,
        (output_node, (tile_node, slice(None)), value_node),
    )
    access = build_memory_access(
        store,
        output,
        list(store.args[1]),
        [TilePattern(0), ArbitrarySlicePattern(slice(None))],
    )

    with pytest.raises(
        exc.InvalidConfig,
        match="store value has 1 elements per item; output requires 64",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )


def test_indirect_load_plan() -> None:
    """Helion: `item_index = index[tile, :]`; `value = table[item_index, :]`."""
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
    assert plan.layout.elements_per_item == 256


def test_multiple_indirect_load_plans() -> None:
    """Helion: `value = left[index[tile], :] + right[index[tile], :]`."""
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


def test_indirect_load_rejects_leading_dimension() -> None:
    """Helion: `value = table[1, index[tile], :]`."""
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
        match="indirect DMA requires the index to be the first subscript",
    ):
        build_sparsecore_memory_plan(
            access,
            SparseCorePlanContext({0: 256}, item_block_id=0, items_per_subcore=8),
        )


def test_direct_store_rejects_leading_dimension() -> None:
    """Helion: `output[1, tile, :] = value`."""
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
