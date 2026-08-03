from __future__ import annotations

import pytest
import torch

from helion import exc
from helion._compiler.pallas import sparsecore_program
from helion._compiler.pallas.memory_access import MEMORY_ACCESS_META
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.sparsecore_plan import SparseCorePlanContext
from helion._compiler.pallas.sparsecore_plan import build_sparsecore_memory_plan
from helion._compiler.pallas.sparsecore_program import SparseCoreGeometry
from helion._compiler.pallas.sparsecore_program import SparseCoreProgram
from helion._compiler.pallas.sparsecore_program import TaskKind
from helion._compiler.pallas.sparsecore_program import schedule_sparsecore_program
from helion.language import memory_ops


def _placeholder(
    graph: torch.fx.Graph, name: str, value: torch.Tensor
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _load(
    graph: torch.fx.Graph,
    source: torch.fx.Node,
    source_value: torch.Tensor,
    index: torch.fx.Node,
    result: torch.Tensor,
    *,
    indirect: bool,
) -> torch.fx.Node:
    node = graph.call_function(memory_ops.load, (source, (index, slice(None))))
    node.meta["val"] = result
    patterns = (
        [TensorIndexPattern(), ArbitrarySlicePattern(slice(None))]
        if indirect
        else [TilePattern(0), ArbitrarySlicePattern(slice(None))]
    )
    node.meta[MEMORY_ACCESS_META] = build_memory_access(
        node, source_value, list(node.args[1]), patterns
    )
    return node


def _program(graph: torch.fx.Graph, nodes: list[torch.fx.Node]) -> SparseCoreProgram:
    geometry = SparseCoreGeometry(0, 1024, 256, 2, 16)
    context = SparseCorePlanContext(
        {0: 256},
        item_block_id=0,
        items_per_subcore=geometry.items_per_subcore,
    )
    memory_plans = tuple(
        build_sparsecore_memory_plan(node.meta[MEMORY_ACCESS_META], context)
        for node in nodes
    )
    return SparseCoreProgram(graph, geometry, memory_plans)


def test_independent_gathers_share_stages_without_topology_rules() -> None:
    graph = torch.fx.Graph()
    indices = torch.empty(1024, dtype=torch.int32)
    index_arg = _placeholder(graph, "indices", indices)
    tile = _placeholder(graph, "tile", torch.empty(256, dtype=torch.int32))
    index_load = _load(
        graph,
        index_arg,
        indices,
        tile,
        torch.empty(256, dtype=torch.int32),
        indirect=False,
    )
    loads = [index_load]
    for number in range(3):
        table = torch.empty(4096, 64)
        table_arg = _placeholder(graph, f"table{number}", table)
        loads.append(
            _load(
                graph,
                table_arg,
                table,
                index_load,
                torch.empty(256, 64),
                indirect=True,
            )
        )

    schedule = schedule_sparsecore_program(_program(graph, loads))

    assert schedule.depth == 2
    assert [stage.lag for stage in schedule.steady_stages] == [0, 1]
    assert (
        sum(
            task.kind is TaskKind.ASYNC_START
            for stage in schedule.steady_stages
            for task in stage.tasks
        )
        == 3
    )


def test_dependent_gather_adds_a_stage_from_its_dependency() -> None:
    graph = torch.fx.Graph()
    roots = torch.empty(1024, dtype=torch.int32)
    roots_arg = _placeholder(graph, "roots", roots)
    tile = _placeholder(graph, "tile", torch.empty(256, dtype=torch.int32))
    root_load = _load(
        graph,
        roots_arg,
        roots,
        tile,
        torch.empty(256, dtype=torch.int32),
        indirect=False,
    )
    pointer = torch.empty(4096, 8, dtype=torch.int32)
    pointer_arg = _placeholder(graph, "pointer", pointer)
    pointer_load = _load(
        graph,
        pointer_arg,
        pointer,
        root_load,
        torch.empty(256, 8, dtype=torch.int32),
        indirect=True,
    )
    table = torch.empty(8192, 64)
    table_arg = _placeholder(graph, "table", table)
    data_load = _load(
        graph,
        table_arg,
        table,
        pointer_load,
        torch.empty(256, 8, 64),
        indirect=True,
    )

    schedule = schedule_sparsecore_program(
        _program(graph, [root_load, pointer_load, data_load])
    )

    assert schedule.depth == 3
    assert [stage.lag for stage in schedule.steady_stages] == [0, 1, 2]
    waits = [
        (stage.lag, task)
        for stage in schedule.steady_stages
        for task in stage.tasks
        if task.kind is TaskKind.ASYNC_WAIT
    ]
    assert [lag for lag, _ in waits] == [1, 2]


def test_index_stream_resource_uses_scalar_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = torch.fx.Graph()
    indices = torch.empty(1024, dtype=torch.int32)
    index_arg = _placeholder(graph, "indices", indices)
    tile = _placeholder(graph, "tile", torch.empty(256, dtype=torch.int32))
    index_load = _load(
        graph,
        index_arg,
        indices,
        tile,
        torch.empty(256, dtype=torch.int32),
        indirect=False,
    )
    table = torch.empty(4096, 64)
    table_arg = _placeholder(graph, "table", table)
    data_load = _load(
        graph,
        table_arg,
        table,
        index_load,
        torch.empty(256, 64),
        indirect=True,
    )
    program = _program(graph, [index_load, data_load])
    monkeypatch.setattr(sparsecore_program, "SC_VMEM_BYTES", 4500)
    monkeypatch.setattr(sparsecore_program, "SC_VMEM_MARGIN", 0)

    schedule_sparsecore_program(program)


def test_adjusted_index_cannot_feed_two_gathers() -> None:
    graph = torch.fx.Graph()
    indices = torch.empty(1024, dtype=torch.int32)
    index_arg = _placeholder(graph, "indices", indices)
    tile = _placeholder(graph, "tile", torch.empty(256, dtype=torch.int32))
    index_load = _load(
        graph,
        index_arg,
        indices,
        tile,
        torch.empty(256, dtype=torch.int32),
        indirect=False,
    )
    adjusted = graph.call_function(torch.ops.aten.add.Tensor, (index_load, 1))
    adjusted.meta["val"] = torch.empty(256, dtype=torch.int32)
    loads = [index_load]
    for number in range(2):
        table = torch.empty(4096, 64)
        table_arg = _placeholder(graph, f"table{number}", table)
        loads.append(
            _load(
                graph,
                table_arg,
                table,
                adjusted,
                torch.empty(256, 64),
                indirect=True,
            )
        )

    with pytest.raises(exc.InvalidConfig, match="code: schedule"):
        _program(graph, loads)
