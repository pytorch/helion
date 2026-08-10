"""Backend-neutral Pallas memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .plan_tiling import IndexingPattern


MEMORY_ACCESS_META = "pallas_memory_access"


class MemoryAccessKind(Enum):
    """Logical effect of one Helion memory operation."""

    LOAD = "load"
    STORE = "store"
    ATOMIC = "atomic"


@dataclass(frozen=True)
class MemoryAccess:
    """One memory operation after shared Pallas indexing analysis."""

    node: torch.fx.Node
    kind: MemoryAccessKind
    tensor_node: torch.fx.Node
    tensor: torch.Tensor
    subscript: tuple[object, ...]
    patterns: tuple[IndexingPattern, ...]
    value_node: torch.fx.Node | None


def build_memory_access(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: list[object],
    patterns: list[IndexingPattern],
) -> MemoryAccess:
    """Build target-independent metadata for one memory operation."""
    from ...language import memory_ops
    from ...language.atomic_ops import ATOMIC_OPS

    tensor_node = node.args[0]
    assert isinstance(tensor_node, torch.fx.Node)
    if node.target is memory_ops.load:
        kind = MemoryAccessKind.LOAD
        value_node = None
    elif node.target is memory_ops.store:
        kind = MemoryAccessKind.STORE
        value_node = node.args[2]
    elif node.target in ATOMIC_OPS:
        kind = MemoryAccessKind.ATOMIC
        value_node = node.args[2]
    else:
        raise AssertionError(f"not a memory access target: {node.target}")

    return MemoryAccess(
        node=node,
        kind=kind,
        tensor_node=tensor_node,
        tensor=tensor,
        subscript=tuple(subscript),
        patterns=tuple(patterns),
        value_node=value_node if isinstance(value_node, torch.fx.Node) else None,
    )
