"""Original accepted BT16 graph bindings for its existing physical schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from . import chained_matmul as chain
from .chained_contraction_groups import _physical_left
from .chained_tcgen_stage import StageGeometry
from .chunk_prefill import _block_inverse
from .chunk_prefill import _Capture
from .chunk_prefill import _match
from .chunk_prefill_prepared_inverse import SparseWarpProduct
from .contraction_region import collect_contraction_region
from .prepared_graph_schedule import ContractionGraph
from .prepared_graph_schedule import ContractionPartition
from .prepared_graph_schedule import _input_reader

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from .chunk_prefill import CuteChunkPrefillRegion
    from .contraction_region import ContractionSpec


def node_arg(node: Node, index: int) -> Node:
    value = node.args[index]
    if not isinstance(value, Node):
        raise chain._UnsupportedChain("changed original BT16 operand")
    return value


@dataclass(frozen=True)
class BT16Bindings:
    region: CuteChunkPrefillRegion
    graph: ContractionGraph
    partition: ContractionPartition
    pairwise_partition: ContractionPartition
    inverse_partition: ContractionPartition
    projection: ContractionSpec
    query: ContractionSpec
    inverse_product: ContractionSpec
    state_update: ContractionSpec
    output_update: ContractionSpec
    kk: ContractionSpec
    qk: ContractionSpec
    lower_value: Node
    products: tuple[SparseWarpProduct, ...]
    diagonal: Node
    n0: Node
    n1: Node
    n2: Node

    @property
    def state_input(self) -> Node:
        return self.region.step.state_input

    @property
    def packed_state_nodes(self) -> tuple[Node, Node]:
        return self.projection.rhs, self.query.rhs

    @property
    def scaled_state(self) -> Node:
        value = self.state_update.accumulator
        assert isinstance(value, Node)
        return value

    @property
    def typed_publication_edges(self) -> tuple[Node, ...]:
        # The common packed leaves encode these original casts. Keep both
        # occurrences of equivalent operands; graph identity owns their proof.
        return (
            node_arg(self.lower_value, 0),
            self.output_update.lhs,
            self.region.step.inverse,
            *(node for product in self.products for node in product.operands),
        )

    def check(self) -> None:
        for part in (self.partition, self.pairwise_partition, self.inverse_partition):
            part.check()
            if part.graph is not self.graph:
                raise chain._UnsupportedChain("foreign BT16 partition")
        for index, node in enumerate(self.typed_publication_edges):
            dtype = torch.bfloat16 if index < 3 else torch.float16
            if (
                node.target is not torch.ops.prims.convert_element_type.default
                or node.args[1] is not dtype
            ):
                raise chain._UnsupportedChain("changed BT16 packed publication cast")
        selected = (
            *self.partition.specs,
            *self.pairwise_partition.specs,
            *self.inverse_partition.specs,
        )
        if (
            len(selected) != len(self.graph.region.contractions)
            or {id(spec) for spec in selected}
            != {id(spec) for spec in self.graph.region.contractions}
            or self.state_update.node is not self.region.step.state_output
            or self.state_input.graph is not self.graph.region.graph
        ):
            raise chain._UnsupportedChain(
                "incomplete original BT16 contraction coverage"
            )

    def pairwise_payload(self) -> tuple[tuple[int, int, int], ...]:
        self.check()
        return tuple(
            (0, 2, chain._host_shape(spec.lhs.meta["val"])[1] // 16)
            for spec in (self.kk, self.qk)
        )

    def inverse_payload(self) -> tuple[tuple[int, int], ...]:
        self.check()
        return tuple(product.payload() for product in self.products)


def bind_bt16(region: CuteChunkPrefillRegion, semantic_loop: GraphInfo) -> BT16Bindings:
    if (
        semantic_loop.graph_id != region.loop_graph_id
        or semantic_loop.graph is not region.step.state_output.graph
        or (region.chunk_size, region.key_width, region.value_width) != (16, 128, 128)
        or region.numerical_policy != "native_bt16_bf16_rhs_v1"
    ):
        raise chain._UnsupportedChain("foreign BT16 graph or physical geometry")
    contractions = collect_contraction_region(semantic_loop)
    if contractions is None:
        raise chain._UnsupportedChain("missing original BT16 contractions")
    graph = ContractionGraph(contractions)
    by_node = {spec.node: spec for spec in contractions.contractions}
    inputs = _input_reader(frozenset(by_node))

    def nearest(value: Node) -> ContractionSpec:
        found = inputs(value)
        if len(found) != 1:
            raise chain._UnsupportedChain("ambiguous original BT16 contraction")
        return by_node[next(iter(found))]

    state = by_node[region.step.state_output]
    output = nearest(region.step.output_store)
    if not isinstance(output.accumulator, Node):
        raise chain._UnsupportedChain("missing BT16 output accumulator")
    query = by_node[output.accumulator]
    inverse_product = nearest(state.lhs)
    projection = nearest(inverse_product.rhs)
    if (
        nearest(output.rhs) is not inverse_product
        or inverse_product.lhs is not region.step.inverse
        or any(
            spec.accumulator is not None
            for spec in (projection, query, inverse_product)
        )
        or not isinstance(state.accumulator, Node)
    ):
        raise chain._UnsupportedChain("changed original BT16 recurrence ports")

    # Reuse the existing exact admission expression for every sparse support,
    # cast and additive snapshot. No numerical equivalence/reassociation claim.
    binding: dict[str, object] = {}
    if not _match(_block_inverse(_Capture("lower")), region.step.inverse, binding):
        raise chain._UnsupportedChain("changed original BT16 sparse inverse DAG")
    lower = binding["lower"]
    assert isinstance(lower, Node)
    kk, qk = nearest(lower), nearest(output.lhs)
    pairwise_geometry = StageGeometry((16, 16, 128), True)
    if (
        _physical_left(kk.node, pairwise_geometry)
        != _physical_left(qk.node, pairwise_geometry)
        or kk.lhs is not projection.lhs
        or qk.lhs is not query.lhs
        or any(
            spec.operand_dtypes != (torch.bfloat16, torch.bfloat16)
            for spec in (kk, qk, projection, query, inverse_product, state, output)
        )
    ):
        raise chain._UnsupportedChain("changed original BT16 pairwise images")

    selected = node_arg(region.step.inverse, 0)
    second = nearest(node_arg(selected, 1))
    first = nearest(second.lhs)
    n2 = node_arg(selected, 2)
    add4 = by_node[node_arg(n2, 1)]
    n1 = node_arg(add4.lhs, 0)
    add2 = by_node[node_arg(n1, 1)]
    n0 = node_arg(add2.lhs, 0)
    d2, d4 = nearest(add2.rhs), nearest(add4.rhs)
    diagonal = node_arg(d2.lhs, 0)
    specs = (d2, add2, d4, add4, first, second)
    diagonal_views = ((0, 0, 8, 8, 0, 8), (8, 8, 8, 8, 8, 16))
    views = (diagonal_views,) * 4 + (((8, 0, 8, 8, 8, 16),), ((8, 0, 8, 8, 0, 8),))
    products = tuple(
        SparseWarpProduct(spec, view, (16, 8, 16), (12,), (spec.lhs, spec.rhs))
        for spec, view in zip(specs, views, strict=True)
    )
    owner = BT16Bindings(
        region,
        graph,
        ContractionPartition(
            graph, (projection, query, inverse_product, state, output)
        ),
        ContractionPartition(graph, (kk, qk)),
        ContractionPartition(graph, specs),
        projection,
        query,
        inverse_product,
        state,
        output,
        kk,
        qk,
        lower,
        products,
        diagonal,
        n0,
        n1,
        n2,
    )
    owner.check()
    if owner.inverse_payload() != ((1, 1),) * 6:
        raise chain._UnsupportedChain("changed original BT16 inverse precision")
    return owner
