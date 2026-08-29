from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

from helion._compiler.pallas.memory_access import MEMORY_ACCESS_META
from helion._compiler.pallas.memory_access import MemoryAccess
from helion._compiler.pallas.memory_access import MemoryAccessKind
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import IndexingPattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.tensorcore_plan import DmaAccessCandidate
from helion._compiler.pallas.tensorcore_plan import DmaAccessSpec
from helion._compiler.pallas.tensorcore_plan import build_dma_access_candidates
from helion._compiler.pallas.tensorcore_plan import dma_access_admission
from helion._compiler.pallas.tensorcore_plan import dma_autotuner_floor
from helion.language import distributed_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion._compiler.device_ir import GraphInfo


def _spec(
    graph: torch.fx.Graph, name: str, block_id: int, table_rows: int = 512
) -> DmaAccessSpec:
    tensor_node = graph.placeholder(f"{name}_tensor")
    node = graph.placeholder(name)
    index_tensor_node = graph.placeholder(f"{name}_indices")
    index_node = graph.placeholder(f"{name}_index")
    access = MemoryAccess(
        node,
        MemoryAccessKind.LOAD,
        tensor_node,
        torch.empty(table_rows, 2, 128),
        (),
        (),
        None,
    )
    index_access = MemoryAccess(
        index_node,
        MemoryAccessKind.LOAD,
        index_tensor_node,
        torch.empty(256, dtype=torch.int32),
        (),
        (),
        None,
    )
    return DmaAccessSpec(access, index_access, block_id, (0, 0), (2, 128))


def test_dma_access_candidate_block_admission() -> None:
    graph = torch.fx.Graph()
    load = _spec(graph, "load", 0)
    store = DmaAccessSpec(
        MemoryAccess(
            graph.placeholder("store"),
            MemoryAccessKind.STORE,
            load.access.tensor_node,
            load.access.tensor,
            (),
            (),
            None,
        ),
        load.index_access,
        load.index_block_id,
        load.selected_starts,
        load.selected_extents,
    )
    candidate = DmaAccessCandidate(7, load, store, frozenset())

    admitted, legal = dma_access_admission((candidate,), {7: {0: 256}}, {0: (128, 256)})
    assert admitted == {load.node, store.node}
    assert legal == {0: (128, 256)}
    assert not dma_access_admission((candidate,), {7: {0: 384}}, {0: (256, 256)})[0]
    small = DmaAccessCandidate(
        7, _spec(graph, "small", 0, table_rows=64), None, frozenset()
    )
    assert not dma_access_admission((small,), {7: {0: 256}}, {0: (128, 128)})[0]
    assert dma_autotuner_floor((128,), 64, 256) == 128
    assert dma_autotuner_floor((128, 256), 64, 64) == 128


def test_dma_access_candidate_requires_exact_store_pair() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(16, 2, 256)
    indices = torch.empty(128, dtype=torch.int32)
    table_arg = graph.placeholder("table")
    table_arg.meta["val"] = table
    indices_arg = graph.placeholder("indices")
    indices_arg.meta["val"] = indices
    index_load = graph.call_function(
        memory_ops.load, (indices_arg, [slice(None)], None)
    )
    index_load.meta["val"] = indices
    index_access = build_memory_access(
        index_load, indices, [slice(None)], [TilePattern(0)]
    )
    index_load.meta[MEMORY_ACCESS_META] = index_access
    fx_subscript = (index_load, slice(None), slice(0, 128))
    subscript: list[object] = list(fx_subscript)
    patterns: list[IndexingPattern] = [
        TensorIndexPattern(),
        ArbitrarySlicePattern(slice(None)),
        ArbitrarySlicePattern(slice(0, 128)),
    ]
    gather = graph.call_function(memory_ops.load, (table_arg, fx_subscript, None))
    gather.meta["val"] = torch.empty(128, 2, 128)
    load_access = build_memory_access(gather, table, subscript, patterns)
    gather.meta[MEMORY_ACCESS_META] = load_access
    store = graph.call_function(
        memory_ops.store, (table_arg, fx_subscript, gather, None)
    )
    store_access = build_memory_access(store, table, subscript, patterns)
    store.meta[MEMORY_ACCESS_META] = store_access
    load_spec = DmaAccessSpec(load_access, index_access, 0, (0, 0), (2, 128))
    store_spec = DmaAccessSpec(store_access, index_access, 0, (0, 0), (2, 128))
    owner = cast("GraphInfo", SimpleNamespace(graph=graph, graph_id=7))

    def spec(access: MemoryAccess) -> DmaAccessSpec | None:
        return {gather: load_spec, store: store_spec}.get(access.node)

    with patch(
        "helion._compiler.pallas.tensorcore_plan.build_dma_access_spec",
        side_effect=spec,
    ):
        assert build_dma_access_candidates([owner]) == (
            DmaAccessCandidate(7, load_spec, store_spec, frozenset({id(indices)})),
        )

        mismatched = DmaAccessSpec(store_access, index_access, 0, (0, 128), (2, 128))
        with patch(
            "helion._compiler.pallas.tensorcore_plan.build_dma_access_spec",
            side_effect=lambda access: {
                gather: load_spec,
                store: mismatched,
            }.get(access.node),
        ):
            assert not build_dma_access_candidates([owner])

        late_use = graph.call_function(torch.ops.aten.neg.default, (gather,))
        assert not build_dma_access_candidates([owner])
        graph.erase_node(late_use)
        graph.call_function(
            distributed_ops.make_async_remote_copy,
            (table_arg, [], 0, table_arg, []),
        )
        assert not build_dma_access_candidates([owner])


def test_dma_access_candidate_rejects_distinct_input_aliases() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(16, 2, 128)
    indices = torch.empty(128, dtype=torch.int32)
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    indices_arg = graph.placeholder("indices")
    left.meta["val"] = table
    right.meta["val"] = table
    indices_arg.meta["val"] = indices
    index_load = graph.call_function(memory_ops.load, (indices_arg, (slice(None),)))
    index_load.meta["val"] = indices
    index_access = build_memory_access(
        index_load, indices, [slice(None)], [TilePattern(0)]
    )
    index_load.meta[MEMORY_ACCESS_META] = index_access
    subscript = (index_load, slice(None), slice(None))
    load = graph.call_function(memory_ops.load, (left, subscript))
    load.meta["val"] = torch.empty(128, 2, 128)
    load_access = build_memory_access(load, table, list(subscript), [])
    load.meta[MEMORY_ACCESS_META] = load_access
    store = graph.call_function(memory_ops.store, (right, subscript, load))
    store_access = build_memory_access(store, table, list(subscript), [])
    store.meta[MEMORY_ACCESS_META] = store_access
    specs = {
        load: DmaAccessSpec(load_access, index_access, 0, (0, 0), (2, 128)),
        store: DmaAccessSpec(store_access, index_access, 0, (0, 0), (2, 128)),
    }
    owner = cast("GraphInfo", SimpleNamespace(graph=graph, graph_id=0))
    with patch(
        "helion._compiler.pallas.tensorcore_plan.build_dma_access_spec",
        side_effect=lambda access: specs.get(access.node),
    ):
        assert not build_dma_access_candidates([owner])
