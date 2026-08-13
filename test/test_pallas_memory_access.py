from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

from helion import Config
from helion._compiler.pallas.memory_access import MEMORY_ACCESS_META
from helion._compiler.pallas.memory_access import MemoryAccessKind
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import IndexingPattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.tensorcore_plan import OneHotGatherPlan
from helion._compiler.pallas.tensorcore_plan import OneHotScatterPlan
from helion._compiler.pallas.tensorcore_plan import select_tensorcore_plan
from helion._compiler.pallas.tracing_ops import _annotate_provable_sublane_alignment
from helion._compiler.pallas.tracing_ops import _descendant_memory_accesses
from helion._compiler.pallas.tracing_ops import _graph_memory_accesses
from helion._compiler.tile_strategy import LoopDimInfo
from helion.language import memory_ops
from helion.language._tracing_ops import _while_loop
from helion.language.atomic_ops import atomic_add

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


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


# This test intentionally exercises compiler internals, which tests normally
# avoid. Verifying alignment promises passed to Mosaic is worth that coupling:
# an incorrect promise can cause undefined behavior.
def test_loop_offset_uses_only_proven_window_alignment() -> None:
    block_id = 3
    loop = SimpleNamespace(
        block_id_to_info={
            block_id: LoopDimInfo(begin_var_name="runtime_begin", begin_expr=None)
        }
    )

    def make_state(aligned_tiles: dict[int, int], block_size: int) -> CodegenState:
        return cast(
            "CodegenState",
            SimpleNamespace(
                device_function=SimpleNamespace(
                    aligned_tiles=aligned_tiles,
                    resolved_block_size=lambda _block_id: block_size,
                ),
                codegen=SimpleNamespace(active_device_loops={block_id: [loop]}),
            ),
        )

    aligned_state = make_state({block_id: 16}, block_size=32)
    assert (
        _annotate_provable_sublane_alignment(aligned_state, block_id, "offset")
        == "pl.multiple_of(offset, 16)"
    )

    # A block size that is not a multiple of the window alignment proves
    # nothing: offsets 16, 40, 64, ... are not all multiples of 16.
    unstepped_state = make_state({block_id: 16}, block_size=24)
    assert (
        _annotate_provable_sublane_alignment(unstepped_state, block_id, "offset")
        == "offset"
    )

    # Without an aligned window, leave the offset unannotated.
    unaligned_state = make_state({}, block_size=32)
    assert (
        _annotate_provable_sublane_alignment(unaligned_state, block_id, "offset")
        == "offset"
    )


def test_alignment_accesses_keep_repeated_tensor_uses() -> None:
    graph = torch.fx.Graph()
    source = torch.empty(64, 256)
    source_node = _placeholder(graph, "source", source)
    first_subscript = (slice(None), slice(0, 128))
    second_subscript = (slice(None), slice(None))
    first = graph.call_function(memory_ops.load, (source_node, first_subscript))
    second = graph.call_function(memory_ops.load, (source_node, second_subscript))
    first.meta[MEMORY_ACCESS_META] = build_memory_access(
        first,
        source,
        list(first_subscript),
        [TilePattern(0), ArbitrarySlicePattern(slice(0, 128))],
    )
    second.meta[MEMORY_ACCESS_META] = build_memory_access(
        second,
        source,
        list(second_subscript),
        [TilePattern(0), TilePattern(1)],
    )

    accesses = _graph_memory_accesses(SimpleNamespace(graph=graph))

    assert len(accesses) == 2
    assert accesses[0].tensor is source
    assert accesses[1].tensor is source
    assert accesses[0].patterns != accesses[1].patterns


def test_alignment_accesses_follow_every_while_graph() -> None:
    parent_graph = torch.fx.Graph()
    parent_graph.call_function(_while_loop, (1, 2, [], 3))

    children: dict[int, SimpleNamespace] = {}
    expected = []
    for graph_id in (1, 2, 3):
        graph = torch.fx.Graph()
        source = torch.empty(64, 128)
        source_node = _placeholder(graph, f"source_{graph_id}", source)
        subscript = (slice(None), slice(None))
        load = graph.call_function(memory_ops.load, (source_node, subscript))
        access = build_memory_access(
            load,
            source,
            list(subscript),
            [TilePattern(0), ArbitrarySlicePattern(slice(None))],
        )
        load.meta[MEMORY_ACCESS_META] = access
        children[graph_id] = SimpleNamespace(graph=graph)
        expected.append(access)

    state = cast(
        "CodegenState",
        SimpleNamespace(get_graph=lambda graph_id: children[graph_id]),
    )
    accesses = _descendant_memory_accesses(SimpleNamespace(graph=parent_graph), state)

    assert accesses == expected


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
        gather = select_tensorcore_plan(load_access, Config())
        scatter = select_tensorcore_plan(store_access, Config())

    assert isinstance(gather, OneHotGatherPlan)
    assert isinstance(scatter, OneHotScatterPlan)
    assert gather.plan is gather_fallback
    assert scatter.plan is scatter_fallback
