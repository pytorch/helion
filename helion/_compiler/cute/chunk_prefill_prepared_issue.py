"""Bind the accepted centered recurrence to common native contraction leaves.

This is the existing M128 engine's physical adapter. Its five original dots
cover four native issues: the last concatenates the independent state/output
members. Preparation arithmetic and role-local events remain with that engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from . import chained_matmul as chain
from .chained_contraction_groups import _physical_left
from .chained_tcgen_stage import StageGeometry
from .contraction_region import collect_contraction_region
from .prepared_continuation import ContractionIssue
from .prepared_graph_schedule import ContractionGraph
from .prepared_graph_schedule import ContractionPartition
from .prepared_graph_schedule import IssuePlacement
from .prepared_graph_schedule import _input_reader
from .prepared_graph_schedule import order_issue_placements

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from .chunk_prefill import CuteChunkPrefillRegion
    from .contraction_region import ContractionSpec


@dataclass(frozen=True)
class FastIssuePort:
    placements: tuple[IssuePlacement, ...]
    # Native physical A source (node, orientation, rounding path).
    physical_left: tuple[Node, bool, tuple[torch.dtype, ...]]
    # A column, B shared offset, D column, M, N, B leading/stride/swizzle/major
    # and descriptor K advancement. K ordering comes from the common issues.
    descriptor: tuple[object, ...]
    ready: tuple[int, ...]
    complete: tuple[int, ...]

    def payload(self) -> tuple[object, ...]:
        first = self.placements[0].issue
        interval = (first.begin, first.end, first.initialized)
        if (
            any(
                (item.issue.begin, item.issue.end, item.issue.initialized) != interval
                for item in self.placements
            )
            or sum(item.member[1] - item.member[0] for item in self.placements)
            != self.descriptor[4]
        ):
            raise chain._UnsupportedChain(
                "changed grouped native issue interval or members"
            )
        return (*self.descriptor[:5], *interval, *self.descriptor[5:])


@dataclass(frozen=True)
class FastRecurrence:
    region: CuteChunkPrefillRegion
    graph: ContractionGraph
    partition: ContractionPartition
    projection: ContractionSpec
    query: ContractionSpec
    inverse_product: ContractionSpec
    state_update: ContractionSpec
    output_update: ContractionSpec
    ports: tuple[FastIssuePort, ...]

    @property
    def state_input(self) -> Node:
        return self.region.step.state_input

    @property
    def packed_state_nodes(self) -> tuple[Node, Node]:
        return self.projection.rhs, self.query.rhs

    @property
    def scaled_state(self) -> Node:
        result = self.state_update.accumulator
        assert isinstance(result, Node)
        return result

    def check(self) -> None:
        self.partition.check()
        if (
            self.partition.graph is not self.graph
            or self.region.step.state_output is not self.state_update.node
            or self.graph.region.graph is not self.state_input.graph
        ):
            raise chain._UnsupportedChain("foreign fast recurrence owner")
        placements = tuple(item for port in self.ports for item in port.placements)
        ordered = order_issue_placements(
            self.graph, placements, partition=self.partition
        )
        if len(ordered) != 1 or ordered[0] != placements:
            raise chain._UnsupportedChain("changed native recurrence order")

    def payload(self) -> tuple[tuple[object, ...], ...]:
        self.check()
        return tuple(port.payload() for port in self.ports)


def fast_issue_layouts() -> tuple[
    tuple[tuple[object, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    """Canonical physical descriptors and events shared by admission and issue."""
    from .chunk_prefill_bt32 import common as cm

    # These are the existing engine's descriptor layouts, not graph extents.
    # Each result shape comes from the accepted full-value-width CTA owner.
    native = (
        (
            cm.TMEM_PACKED_STATE,
            cm.KD,
            cm.TMEM_PROJECTION,
            cm.DV,
            cm.BT,
            16,
            1024,
            128,
            0,
            (2, 4, 32, 128),
        ),
        (
            cm.TMEM_PACKED_STATE,
            cm.QD,
            cm.TMEM_OUT,
            cm.DV,
            cm.BT,
            16,
            1024,
            128,
            0,
            (2, 4, 32, 128),
        ),
        (
            cm.TMEM_RHS_U,
            cm.INV,
            cm.TMEM_UPDATE,
            cm.DV,
            cm.BT,
            16,
            256,
            32,
            0,
            (2, 1, 32, 32),
        ),
        (
            cm.TMEM_RHS_U,
            cm.FINAL_TRANS,
            cm.TMEM_STATE,
            cm.DV,
            cm.DK + cm.BT,
            4096,
            1024,
            128,
            1,
            (2, 1, 16, 128),
        ),
    )
    return (
        native,
        (
            (cm.QK_FULL, cm.OUT_EMPTY, cm.STATE_INP_READY),
            (),
            (cm.U_INP_READY,),
            (cm.U2_INP_READY,),
        ),
        (
            (cm.OLD_OUT_READY,),
            (cm.RAW_INPUTS_FREE,),
            (cm.U2_ACC_READY,),
            (cm.FINAL_READY, cm.SMEM_FREE),
        ),
    )


def bind_fast_recurrence(
    region: CuteChunkPrefillRegion, semantic_loop: GraphInfo
) -> FastRecurrence:
    """Use accepted step outputs and original use-def paths, never node names."""
    from .chunk_prefill_bt32 import common as cm

    if (
        semantic_loop.graph_id != region.loop_graph_id
        or semantic_loop.graph is not region.step.state_output.graph
        or (region.chunk_size, region.key_width, region.value_width)
        != (cm.BT, cm.DK, cm.DV)
    ):
        raise chain._UnsupportedChain("foreign fast recurrence graph or geometry")
    contraction_region = collect_contraction_region(semantic_loop)
    if contraction_region is None:
        raise chain._UnsupportedChain("missing fast recurrence contractions")
    graph = ContractionGraph(contraction_region)
    by_node = {spec.node: spec for spec in contraction_region.contractions}
    inputs = _input_reader(frozenset(by_node))

    def nearest(value: Node) -> ContractionSpec:
        found = inputs(value)
        if len(found) != 1:
            raise chain._UnsupportedChain("ambiguous fast recurrence input")
        return by_node[next(iter(found))]

    state = by_node[region.step.state_output]
    output = nearest(region.step.output_store)
    if not isinstance(output.accumulator, Node):
        raise chain._UnsupportedChain("missing original output seed")
    query = by_node[output.accumulator]
    inverse = nearest(state.lhs)
    if nearest(output.rhs) is not inverse or inverse.lhs is not region.step.inverse:
        raise chain._UnsupportedChain("changed original solved-value binding")
    projection = nearest(inverse.rhs)
    selected = (projection, query, inverse, state, output)
    partition = ContractionPartition(graph, selected)
    partition.check()
    geometries = (
        StageGeometry((cm.BT, cm.DV, cm.DK), True),
        StageGeometry((cm.BT, cm.DV, cm.DK), True),
        StageGeometry((cm.BT, cm.DV, cm.BT), True),
        StageGeometry((cm.DV, cm.DK, cm.BT), False),
        StageGeometry((cm.BT, cm.DV, cm.BT), True),
    )
    left = tuple(
        _physical_left(spec.node, geo)
        for spec, geo in zip(selected, geometries, strict=True)
    )
    state_source = region.step.state_input
    while state_source.target is _tracing_ops._new_var:
        parent = state_source.args[0]
        assert isinstance(parent, Node)
        state_source = parent
    if (
        left[0] != left[1]
        or left[0] != (state_source, False, (torch.bfloat16,))
        or left[3] != left[4]
        or left[3][0] is not inverse.node
        or left[3][2] != (torch.bfloat16,)
        or any(spec.accumulator is not None for spec in selected[:3])
        or not isinstance(state.accumulator, Node)
        or any(
            spec.lhs.meta["val"].dtype is not torch.bfloat16
            or spec.rhs.meta["val"].dtype is not torch.bfloat16
            for spec in selected
        )
    ):
        raise chain._UnsupportedChain("changed original native operand image or seed")
    role = cm.PIPELINE_PLAN.role("service")
    placements = tuple(
        IssuePlacement(
            ContractionIssue(
                contraction_region,
                spec,
                0,
                geo.logical[2] // 16,
                16,
                spec.accumulator is not None,
                False,
            ),
            region,
            role,
            (0, geo.physical[1]),
        )
        for spec, geo in zip(selected, geometries, strict=True)
    )
    native, ready, complete = fast_issue_layouts()
    ports = tuple(
        starmap(
            FastIssuePort,
            zip(
                ((placements[0],), (placements[1],), (placements[2],), placements[3:]),
                (left[0], left[1], left[2], left[3]),
                native,
                ready,
                complete,
                strict=True,
            ),
        )
    )
    result = FastRecurrence(region, graph, partition, *selected, ports)
    result.check()
    return result
