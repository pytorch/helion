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


@dataclass(frozen=True)
class IndirectAccess:
    """The single tensor-indexed dimension of a memory access."""

    access: MemoryAccess
    position: int
    index_node: torch.fx.Node


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


def memory_access_value(access: MemoryAccess) -> torch.Tensor | None:
    """Return the tensor loaded or stored by ``access`` when available."""
    value = (
        access.node.meta.get("val")
        if access.kind is MemoryAccessKind.LOAD
        else access.value_node.meta.get("val")
        if access.value_node is not None
        else None
    )
    return value if isinstance(value, torch.Tensor) else None


def memory_access_mask(access: MemoryAccess) -> object | None:
    """Return an explicit mask, excluding automatic tile-bound masks."""
    position = 2 if access.kind is MemoryAccessKind.LOAD else 3
    return access.node.args[position] if len(access.node.args) > position else None


def indirect_access(access: MemoryAccess) -> IndirectAccess | None:
    """Return the unique tensor-indexed dimension, if there is one."""
    positions = tensor_index_positions(access)
    if len(positions) != 1:
        return None
    position = positions[0]
    index = access.subscript[position]
    if not isinstance(index, torch.fx.Node):
        return None
    return IndirectAccess(access, position, index)


def tensor_index_positions(access: MemoryAccess) -> tuple[int, ...]:
    """Return all tensor-indexed subscript positions."""
    from .plan_tiling import TensorIndexPattern

    return tuple(
        position
        for position, pattern in enumerate(access.patterns)
        if isinstance(pattern, TensorIndexPattern)
    )
