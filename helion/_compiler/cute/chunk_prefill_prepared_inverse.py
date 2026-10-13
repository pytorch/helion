"""Bind the accepted masked inverse DAG to its original packed warp images.

Logical zero regions are explicit. A native compressed block-diagonal issue
is not represented as a synthetic dense contraction or a shortened logical K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from . import chained_matmul as chain
from .prepared_graph_schedule import ContractionPartition
from .prepared_graph_schedule import _input_reader

if TYPE_CHECKING:
    from .chunk_prefill_prepared_issue import FastRecurrence
    from .chunk_prefill_prepared_pairwise import FastPairwise
    from .contraction_region import ContractionSpec


@dataclass(frozen=True)
class SparseWarpProduct:
    spec: ContractionSpec
    # Required nonzero result rectangles and their original logical K domains:
    # (row_begin, column_begin, rows, columns, k_begin, k_end).
    logical_views: tuple[tuple[int, int, int, int, int, int], ...]
    native_shape: tuple[int, int, int]
    local_warps: tuple[int, ...]
    # Original masked operands/casts, retained with the whole accepted graph.
    operands: tuple[Node, Node]

    def payload(self) -> tuple[int, int]:
        dtype = self.spec.operand_dtypes
        if dtype == (torch.float16, torch.float16):
            kind = 1
        elif dtype == (torch.bfloat16, torch.bfloat16):
            kind = 0
        else:
            raise chain._UnsupportedChain("changed sparse inverse operand dtype")
        if (
            self.spec.accumulator is not None
            or self.spec.result_dtype is not torch.float32
            or self.operands != (self.spec.lhs, self.spec.rhs)
        ):
            raise chain._UnsupportedChain("changed sparse inverse seed or inputs")
        return kind, self.native_shape[1] // 8


@dataclass(frozen=True)
class FastInverse:
    owner: FastRecurrence
    pairwise: FastPairwise
    partition: ContractionPartition
    products: tuple[SparseWarpProduct, ...]
    # This source snapshot must precede the degree-four diagonal correction.
    coupling_snapshot: Node
    diagonal_mask: Node
    inner_mask: Node
    outer_mask: Node
    inverse16_bf16: Node
    final_bf16: Node

    def check(self) -> None:
        self.owner.check()
        self.pairwise.check()
        self.partition.check()
        if tuple(item.spec for item in self.products) != self.partition.specs:
            raise chain._UnsupportedChain("changed inverse physical membership")
        selected = (
            *self.owner.partition.specs,
            *self.pairwise.partition.specs,
            *self.partition.specs,
        )
        if len(selected) != len(self.owner.graph.region.contractions) or {
            id(spec) for spec in selected
        } != {id(spec) for spec in self.owner.graph.region.contractions}:
            raise chain._UnsupportedChain(
                "incomplete original fast contraction coverage"
            )

    def payload(self) -> tuple[tuple[int, int], ...]:
        self.check()
        return tuple(item.payload() for item in self.products)


def bind_fast_inverse(owner: FastRecurrence, pairwise: FastPairwise) -> FastInverse:
    owner.check()
    by_node = {spec.node: spec for spec in owner.graph.region.contractions}
    inputs = _input_reader(frozenset(by_node))

    def node_arg(node: Node, index: int) -> Node:
        value = node.args[index]
        if not isinstance(value, Node):
            raise chain._UnsupportedChain("changed original sparse inverse input")
        return value

    def nearest(node: Node) -> ContractionSpec:
        found = inputs(node)
        if len(found) != 1:
            raise chain._UnsupportedChain("ambiguous sparse inverse contraction")
        return by_node[next(iter(found))]

    inverse = owner.region.step.inverse
    final_bf16 = node_arg(inverse, 1)
    inverse16 = node_arg(inverse, 2)
    outer_second = nearest(final_bf16)
    outer_first = nearest(outer_second.lhs)
    selected16 = node_arg(inverse16, 0)
    inner_second = by_node[node_arg(selected16, 1)]
    inner_first = nearest(inner_second.lhs)
    snapshot = inner_second.rhs
    n1 = node_arg(snapshot, 0)
    add2 = by_node[node_arg(n1, 1)]
    d2 = nearest(add2.rhs)
    n2 = node_arg(selected16, 2)
    add4 = by_node[node_arg(n2, 1)]
    d4 = nearest(add4.rhs)
    diagonal = node_arg(d2.lhs, 0)
    if (
        inverse.target is not torch.ops.aten.where.self
        or selected16.target is not torch.ops.aten.where.self
        or diagonal.target is not torch.ops.aten.where.self
        or outer_second.rhs is not inverse16
        or outer_first.lhs is not inverse16
        or inner_first.lhs is not snapshot
        or add4.lhs is not snapshot
        or nearest(d4.lhs) is not d2
        or nearest(d4.rhs) is not d2
        or node_arg(d2.rhs, 0) is not diagonal
        or node_arg(n2, 0) is not n1
    ):
        raise chain._UnsupportedChain("changed original inverse snapshot or masked DAG")
    specs = (d2, add2, d4, add4, inner_first, inner_second, outer_first, outer_second)
    partition = ContractionPartition(owner.graph, specs)
    partition.check()
    diagonal_views = tuple((i, i, 8, 8, i, i + 8) for i in range(0, 32, 8))
    inner_first_views = ((8, 0, 8, 8, 8, 16), (24, 16, 8, 8, 24, 32))
    inner_second_views = ((8, 0, 8, 8, 0, 8), (24, 16, 8, 8, 16, 24))
    views = (diagonal_views,) * 4 + (
        inner_first_views,
        inner_second_views,
        ((16, 0, 16, 16, 16, 32),),
        ((16, 0, 16, 16, 0, 16),),
    )
    products = tuple(
        SparseWarpProduct(
            spec,
            view,
            (16, 8 if index < 6 else 16, 16),
            (0, 3) if index < 6 else (0,),
            (spec.lhs, spec.rhs),
        )
        for index, (spec, view) in enumerate(zip(specs, views, strict=True))
    )
    result = FastInverse(
        owner,
        pairwise,
        partition,
        products,
        snapshot,
        node_arg(diagonal, 0),
        node_arg(selected16, 0),
        node_arg(inverse, 0),
        inverse16,
        final_bf16,
    )
    result.check()
    if result.payload() != ((1, 1),) * 6 + ((0, 2),) * 2:
        raise chain._UnsupportedChain("changed original inverse precision stages")
    return result
