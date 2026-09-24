from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo


def has_pointwise_vector_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Discover computed host operands, not a coordinate or alignment proof.

    Exact vector-leaf/geometry admission still happens in the existing emitter.
    Do not advertise this dimension for direct copies or dot-derived operands.
    """
    from . import chained_matmul as chain

    root = chain._root_graph(graphs)
    if root is None:
        return False
    dots = {node for node in root.graph.nodes if node.target is chain.dot}
    for dot in dots:
        for operand in dot.args[:2]:
            operand = cast("Node", operand)
            ancestors = chain._ancestors(operand)
            if chain._direct_operand(operand) or ancestors & dots:
                continue
            for node in ancestors:
                if node.target is not memory_ops.load:
                    continue
                source = cast("Node", node.args[0])
                if source.target is not _tracing_ops._host_tensor:
                    continue
                value = source.meta["val"]
                if value.dtype in (
                    torch.bfloat16,
                    torch.float16,
                    torch.float32,
                ) and any(
                    isinstance(stride, int) and stride == 1 for stride in value.stride()
                ):
                    return True
    return False


@dataclass
class PointwiseUnroll:
    """Per-codegen activation tracking; never shared across kernels/configs."""

    factor: int
    activated: bool = False

    def loop_factor(self, trips: int) -> int:
        # A single-trip loop has no additional iteration to schedule.
        if trips >= 2:
            # Preserve factors 1/2. Larger factors require complete unroll groups
            # in every admitted staging loop, never a silently substituted factor.
            if self.factor > 2 and (trips < self.factor or trips % self.factor):
                raise exc.BackendUnsupported(
                    "cute",
                    "pointwise unroll requires every admitted multi-trip vector "
                    "staging loop to contain a whole number of unroll groups",
                )
            self.activated = True
            return self.factor
        return 1

    def validate(self) -> None:
        if self.factor != 1 and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "pointwise unroll requires an admitted vector staging loop"
            )
