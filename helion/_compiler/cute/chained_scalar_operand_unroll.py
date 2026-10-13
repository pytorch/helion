"""Bounded scheduling of the original root scalar operand producers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo


def has_scalar_operand_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Find contraction-derived operands; exact emission still owns the loop.

    These operands cannot use the original host-leaf vector producer. Discovery
    adds no vector ownership or alignment claim and needs no compile context.
    """
    from . import chained_matmul as chain

    root = chain._root_graph(graphs)
    if root is None:
        return False
    dots = {node for node in root.graph.nodes if node.target is chain.dot}
    return len(dots) > 1 and any(
        chain._ancestors(cast("Node", operand)) & dots
        for node in dots
        for operand in node.args[:2]
    )


@dataclass
class ScalarOperandUnroll:
    """Track actual emitted scalar loops, separately from vector-loop proof.

    Zero leaves the original source unchanged. A positive factor is an upper
    bound: short loops cap it, and CuTe owns any residual iterations. Merely
    constructing an unused scalar fallback does not activate this choice.
    """

    factor: int
    activated: bool = False

    def loop_factor(self, trips: int) -> int:
        return min(max(1, self.factor), max(1, trips))

    def record(self, trips: int) -> None:
        self.activated |= self.loop_factor(trips) > 1

    def validate(self) -> None:
        if self.factor and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "scalar operand unroll requires an emitted multi-trip producer"
            )
