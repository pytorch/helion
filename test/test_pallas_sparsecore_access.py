from __future__ import annotations

import torch

from helion._compiler.pallas.access import make_access_site
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import NonePattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.sparsecore_access import AccessLoweringContext
from helion._compiler.pallas.sparsecore_access import CachedLoadAccess
from helion._compiler.pallas.sparsecore_access import DirectLoadAccess
from helion._compiler.pallas.sparsecore_access import IndirectLoadAccess
from helion._compiler.pallas.sparsecore_access import lower_sparsecore_access
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
    site = make_access_site(
        load,
        source,
        list(load.args[1]),
        [TilePattern(0), ArbitrarySlicePattern(slice(None))],
    )

    access = lower_sparsecore_access(
        site,
        AccessLoweringContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(access, DirectLoadAccess)
    assert access.stream is not None
    assert access.stream.group == 0
    assert access.stream.group_count == 1
    assert access.stream.elements_per_item == 64
    assert access.layout.storage_shape == (8, 64)


def test_indirect_load_retains_its_local_dependency() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(4096, 64, dtype=torch.bfloat16)
    index = torch.empty(32, 4, dtype=torch.int32)
    result = torch.empty(32, 4, 64, dtype=torch.bfloat16)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    load = graph.call_function(memory_ops.load, (table_node, (index_node, slice(None))))
    load.meta["val"] = result
    site = make_access_site(
        load,
        table,
        list(load.args[1]),
        [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))],
    )

    access = lower_sparsecore_access(
        site,
        AccessLoweringContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(access, IndirectLoadAccess)
    assert access.index_node is index_node
    assert access.dependencies == frozenset({index_node})
    assert access.layout.value_size == 256


def test_several_indirect_accesses_do_not_require_a_stack() -> None:
    graph = torch.fx.Graph()
    index = torch.empty(32, dtype=torch.int32)
    index_node = _placeholder(graph, "index", index)
    context = AccessLoweringContext({0: 256}, item_block_id=0, items_per_subcore=8)
    accesses = []
    for number in range(3):
        table = torch.empty(4096, 64)
        table_node = _placeholder(graph, f"table{number}", table)
        load = graph.call_function(
            memory_ops.load, (table_node, (index_node, slice(None)))
        )
        load.meta["val"] = torch.empty(32, 64)
        site = make_access_site(
            load,
            table,
            list(load.args[1]),
            [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))],
        )
        accesses.append(lower_sparsecore_access(site, context))

    assert all(isinstance(access, IndirectLoadAccess) for access in accesses)
    assert [access.index_node for access in accesses] == [index_node] * 3


def test_cached_load_accounts_for_the_copied_input() -> None:
    graph = torch.fx.Graph()
    source = torch.empty(2, 48)
    source_node = _placeholder(graph, "source", source)
    load = graph.call_function(
        memory_ops.load, (source_node, (None, slice(None), slice(None)))
    )
    load.meta["val"] = torch.empty(1, 2, 48)
    site = make_access_site(
        load,
        source,
        list(load.args[1]),
        [
            NonePattern(),
            ArbitrarySlicePattern(slice(None)),
            ArbitrarySlicePattern(slice(None)),
        ],
    )

    access = lower_sparsecore_access(
        site,
        AccessLoweringContext({0: 256}, item_block_id=0, items_per_subcore=8),
    )

    assert isinstance(access, CachedLoadAccess)
    assert access.layout.value_size == 96
    assert access.layout.storage_shape == (2, 48)
