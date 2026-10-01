"""Original pairwise contraction and typed-result cuts for the fast factor role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from . import chained_matmul as chain
from .chained_contraction_groups import _physical_left
from .chained_tcgen_stage import StageGeometry
from .prepared_graph_schedule import ContractionPartition
from .prepared_graph_schedule import _input_reader

if TYPE_CHECKING:
    from .chunk_prefill_prepared_issue import FastRecurrence
    from .contraction_region import ContractionSpec


@dataclass(frozen=True)
class FastPairwise:
    owner: FastRecurrence
    partition: ContractionPartition
    kk: ContractionSpec
    qk: ContractionSpec
    lower_value: Node
    qk_value: Node

    def check(self) -> None:
        self.owner.check()
        self.partition.check()
        if self.partition.graph is not self.owner.graph or self.partition.specs != (
            self.kk,
            self.qk,
        ):
            raise chain._UnsupportedChain("foreign pairwise partition")

    def payload(self) -> tuple[tuple[int, int, int, int, int], ...]:
        from .chunk_prefill_bt32 import common as cm

        self.check()
        # Full increasing K for each original 16x16 lower-result tile. The
        # original strict/causal masks own the unissued upper tile; this does
        # not claim a completed unmasked 32x32 contraction result.
        return tuple(
            (offset, cm.KI, chain._host_shape(spec.lhs.meta["val"])[1] // 16, 2, cm.BT)
            for spec, offset in ((self.kk, cm.KD), (self.qk, cm.QD))
        )


def bind_fast_pairwise(owner: FastRecurrence) -> FastPairwise:
    owner.check()
    graph = owner.graph
    by_node = {spec.node: spec for spec in graph.region.contractions}
    inputs = _input_reader(frozenset(by_node))
    qk_inputs = inputs(owner.output_update.lhs)
    if len(qk_inputs) != 1:
        raise chain._UnsupportedChain("ambiguous pairwise output input")
    qk = by_node[next(iter(qk_inputs))]
    inverse_nodes = chain._ancestors(owner.region.step.inverse)
    kk_candidates = tuple(
        spec
        for spec in graph.region.contractions
        if spec.node in inverse_nodes and spec.lhs is owner.projection.lhs
    )
    if len(kk_candidates) != 1:
        raise chain._UnsupportedChain("ambiguous pairwise inverse input")
    (kk,) = kk_candidates
    geo = StageGeometry(
        (owner.region.chunk_size, owner.region.chunk_size, owner.region.key_width), True
    )
    if _physical_left(kk.node, geo) != _physical_left(qk.node, geo) or any(
        spec.accumulator is not None
        or spec.operand_dtypes != (torch.bfloat16, torch.bfloat16)
        for spec in (kk, qk)
    ):
        raise chain._UnsupportedChain("changed original pairwise operands")

    def user(node: Node, target: object, dtype: torch.dtype | None = None) -> Node:
        found = tuple(
            item
            for item in node.users
            if item.target is target and (dtype is None or item.args[1] is dtype)
        )
        if len(found) != 1:
            raise chain._UnsupportedChain("changed original pairwise typed publication")
        return found[0]

    multiplied = user(kk.node, torch.ops.aten.mul.Tensor)
    masked = user(multiplied, torch.ops.aten.where.self)
    lower_bf16 = user(
        masked, torch.ops.prims.convert_element_type.default, torch.bfloat16
    )
    lower = user(
        lower_bf16, torch.ops.prims.convert_element_type.default, torch.float32
    )
    qk_masked = user(qk.node, torch.ops.aten.where.self)
    qk_value = user(
        qk_masked, torch.ops.prims.convert_element_type.default, torch.bfloat16
    )
    strict, causal = masked.args[0], qk_masked.args[0]
    if (
        not isinstance(strict, Node)
        or not isinstance(causal, Node)
        or strict.target is not torch.ops.aten.gt.Tensor
        or causal.target is not torch.ops.aten.ge.Tensor
        or qk_value is not owner.output_update.lhs
    ):
        raise chain._UnsupportedChain("changed original pairwise result mask")
    result = FastPairwise(
        owner, ContractionPartition(graph, (kk, qk)), kk, qk, lower, qk_value
    )
    result.check()
    return result
