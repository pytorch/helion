"""Optional original final-store recipe on a completed native FP32 carry drain."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast

import torch

from . import chained_matmul as chain
from .chained_completed_members import CompletedMemberMap
from .chained_completed_members import plan_completed_member_map
from .chained_loop_tmem_bridges import _ancestors
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_register_islands import _freeze
from .chained_store_expression import _node_facts
from .chained_store_expression import lower_store_point
from .chained_store_expression import materialized_store_read
from .chained_store_ownership import StoreOwnership
from .chained_store_ownership import plan_store_ownership

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import PipelineStorage


@dataclass
class _Progress:
    consumed: bool = False


@dataclass(frozen=True)
class FinalCarryStore:
    plan: ChainedMatmulPlan
    storage: PipelineStorage
    carry: LoopTmemCarryTransport
    store: Node
    source: Node
    ownership: StoreOwnership
    native: CompletedMemberMap
    lines: tuple[str, ...]
    facts: tuple[object, ...]
    _progress: _Progress = field(default_factory=_Progress)
    _selection: tuple[object, ...] = ()

    def _fields(self) -> tuple[object, ...]:
        return (
            self.plan,
            self.storage,
            self.carry,
            self.store,
            self.source,
            self.ownership,
            _freeze(vars(self.native)),
            self.lines,
            self.facts,
            self._progress,
        )

    def matches(
        self,
        plan: ChainedMatmulPlan,
        storage: PipelineStorage,
        carry: LoopTmemCarryTransport,
    ) -> bool:
        return (
            self._selection == self._fields()
            and plan is self.plan
            and storage is self.storage
            and carry is self.carry
            and plan.loop is not None
            and plan.loop.final_stores == (self.store,)
            and self.facts == _node_facts(tuple(plan.loop.root.graph.nodes))
            and plan.loop_workspace is storage.recurrence.layout
            and plan_store_ownership(plan, self.store) == self.ownership
        )

    def consume(self, store: Node) -> None:
        if store is not self.store or self._progress.consumed:
            raise chain._UnsupportedChain("final carry store consumed out of order")
        self._progress.consumed = True


def prepare_final_carry_store(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    storage: PipelineStorage,
    carry: LoopTmemCarryTransport,
) -> FinalCarryStore | None:
    """No permission to drain before the endpoint owner's complete-loop join."""
    from .chained_loop import loop_storage_is_proven
    from .chained_loop import loop_storage_matches_runtime
    from .chained_tcgen05 import _load_result

    loop = plan.loop
    geometry = carry.candidate.geometry
    if (
        loop is None
        or len(loop.final_stores) != 1
        or geometry.native_rows != 64
        or carry.drain_panels is not None
        or not loop_storage_is_proven(loop)
        or not loop_storage_matches_runtime(loop)
        or plan.loop_workspace is not storage.recurrence.layout
        or len(loop.region.carries) != 1
    ):
        return None
    store = loop.final_stores[0]
    ownership = plan_store_ownership(plan, store)
    if ownership is None:
        return None
    value = cast("Node", store.args[2])
    boundaries = loop.boundaries()
    frontier: set[Node] = set()
    pending = [value]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in boundaries:
            frontier.add(node)
        else:
            pending.extend(node.all_input_nodes)
    if len(frontier) != 1:
        return None
    source = next(iter(frontier))
    if (
        (source, carry.candidate.carry.input_index) not in loop.final_values
        or source.meta["val"].dtype is not torch.float32
        or chain._shape(source) != geometry.logical[:2]
    ):
        return None
    native = plan_completed_member_map(
        geometry.physical[:2],
        geometry.logical[:2],
        0,
        geometry.physical[1],
        geometry.transpose,
        carry.candidate.snapshot_dtype,
        reordered=True,
    )
    if native is None:
        return None
    prefix = carry.prefix
    row, column = f"{prefix}_row", f"{prefix}_column"
    source_coords = geometry.result_coordinates(row, column)
    ancestors = {node: _ancestors(node) for node in loop.root.graph.nodes}
    choices = [
        coords
        for coords, shape in (
            ((row, column), geometry.physical[:2]),
            ((column, row), geometry.physical[:2][::-1]),
        )
        if chain._shape(value) == shape
        and _coordinate_path(value, coords, source, source_coords, ancestors)
        is not None
    ]
    if len(choices) != 1:
        return None
    try:
        replacement = materialized_store_read(
            plan, boundaries, source, source_coords, f"{prefix}_values[{prefix}_index]"
        )
        point = lower_store_point(
            cg,
            plan,
            store,
            boundaries,
            [],
            [],
            coordinates=choices[0],
            coordinate_names=(row, column),
            completed_read=replacement,
        )
    except chain._UnsupportedChain:
        return None
    if (
        point.host_reads
        or any(node is not source for node, _ in point.reads)
        or not isinstance(point.target, str)
    ):
        return None
    body = [
        *_load_result(prefix, geometry.physical[:2]),
        f"for {prefix}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
        f"    {row}, {column} = {prefix}_coords[{prefix}_index]",
        chain._indent(point.lines),
        f"    if {' and '.join(point.bounds)}:",
        f"        {point.target}[{', '.join(point.indices)}] = {point.value}",
        "chain_tmem_barrier.arrive_and_wait()",
    ]
    lines = ("if chain_thread < 128:", chain._indent(body), "cute.arch.sync_threads()")
    result = FinalCarryStore(
        plan,
        storage,
        carry,
        store,
        source,
        ownership,
        native,
        lines,
        _node_facts(tuple(loop.root.graph.nodes)),
    )
    return replace(result, _selection=result._fields())
