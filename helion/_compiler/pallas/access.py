"""Backend-neutral memory access sites for Pallas lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .plan_tiling import IndexingPattern


ACCESS_SITE_META = "pallas_access_site"


class AccessKind(Enum):
    """Logical effect of one Helion memory operation."""

    LOAD = "load"
    STORE = "store"
    ATOMIC = "atomic"


@dataclass(frozen=True)
class AccessSite:
    """One memory operation after shared Pallas indexing analysis."""

    node: torch.fx.Node
    kind: AccessKind
    target: object
    tensor_node: torch.fx.Node
    tensor: torch.Tensor
    subscripts: tuple[object, ...]
    patterns: tuple[IndexingPattern, ...]
    value_node: torch.fx.Node | None


def make_access_site(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: list[object],
    patterns: list[IndexingPattern],
) -> AccessSite:
    """Create target-independent access metadata for a memory FX node."""
    from ...language import memory_ops
    from ...language.atomic_ops import ATOMIC_OPS

    tensor_node = node.args[0]
    assert isinstance(tensor_node, torch.fx.Node)
    if node.target is memory_ops.load:
        kind = AccessKind.LOAD
        value_node = None
    elif node.target is memory_ops.store:
        kind = AccessKind.STORE
        value_node = node.args[2]
    elif node.target in ATOMIC_OPS:
        kind = AccessKind.ATOMIC
        value_node = node.args[2]
    else:
        raise AssertionError(f"not a memory access target: {node.target}")

    return AccessSite(
        node=node,
        kind=kind,
        target=node.target,
        tensor_node=tensor_node,
        tensor=tensor,
        subscripts=tuple(subscript),
        patterns=tuple(patterns),
        value_node=value_node if isinstance(value_node, torch.fx.Node) else None,
    )
