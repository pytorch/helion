"""SparseCore memory plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from .memory_access import MemoryAccessKind
from .plan_tiling import ArbitraryIndexPattern
from .plan_tiling import ArbitrarySlicePattern
from .plan_tiling import NonePattern
from .plan_tiling import TensorIndexPattern
from .plan_tiling import TilePattern
from .sc_base import _INDIRECT_DTYPES
from .sc_base import SC_DMA_GRANULE_BYTES
from .sc_base import SC_LANES
from .sc_base import _reject

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .memory_access import MemoryAccess
    from .plan_tiling import IndexingPattern


@dataclass(frozen=True)
class ValueLayout:
    """Stored representation of one subcore value."""

    value_size: int
    storage_shape: tuple[int, ...]
    logical_dtype: torch.dtype
    storage_dtype: torch.dtype


@dataclass(frozen=True)
class StreamGeometry:
    """One tiled input or output stream."""

    group: int
    group_count: int
    elements_per_item: int


@dataclass(frozen=True)
class SparseCoreMemoryPlan:
    access: MemoryAccess
    layout: ValueLayout
    stream: StreamGeometry | None
    dependencies: frozenset[torch.fx.Node]


@dataclass(frozen=True)
class DirectLoadPlan(SparseCoreMemoryPlan):
    pass


@dataclass(frozen=True)
class IndirectLoadPlan(SparseCoreMemoryPlan):
    index_node: torch.fx.Node
    index_offset: int


@dataclass(frozen=True)
class DirectStorePlan(SparseCoreMemoryPlan):
    pass


@dataclass(frozen=True)
class IndirectStorePlan(SparseCoreMemoryPlan):
    index_node: torch.fx.Node
    index_offset: int


@dataclass(frozen=True)
class SparseCorePlanContext:
    """Geometry shared by independent memory plans."""

    block_sizes: Mapping[int, int]
    item_block_id: int
    items_per_subcore: int


def _value_tensor(access: MemoryAccess) -> torch.Tensor:
    if access.kind is MemoryAccessKind.LOAD:
        value = access.node.meta.get("val")
    else:
        value = (
            access.value_node.meta.get("val") if access.value_node is not None else None
        )
    if not isinstance(value, torch.Tensor):
        _reject("layout", "memory access value is not a tensor", node=access.node)
    return value


def _logical_shape(
    access: MemoryAccess, context: SparseCorePlanContext
) -> tuple[int, ...]:
    value = _value_tensor(access)
    shape = list(value.shape)
    if not shape:
        return (context.items_per_subcore,)
    # Access handlers use the leading value dimension as the item axis.
    shape[0] = context.items_per_subcore
    for index, dim in enumerate(shape):
        if isinstance(dim, int):
            continue
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        block_id = env.resolve_block_id(dim)
        block = context.block_sizes.get(block_id) if block_id is not None else None
        if block is None:
            _reject(
                "dynamic_shape",
                f"access result has dynamic trailing shape {tuple(value.shape)}",
                node=access.node,
            )
        shape[index] = block
    return tuple(shape)  # type: ignore[arg-type]


def _layout(access: MemoryAccess, context: SparseCorePlanContext) -> ValueLayout:
    value = _value_tensor(access)
    logical = _logical_shape(access, context)
    value_size = math.prod(logical[1:]) if len(logical) > 1 else 1
    stored_size = max(SC_LANES, math.ceil(value_size / SC_LANES) * SC_LANES)
    storage_dtype = (
        torch.int32
        if access.kind is not MemoryAccessKind.LOAD
        and value.dtype in (torch.int8, torch.bool)
        else value.dtype
    )
    return ValueLayout(
        value_size=value_size,
        storage_shape=(context.items_per_subcore, stored_size),
        logical_dtype=value.dtype,
        storage_dtype=storage_dtype,
    )


def _full_slice(
    pattern: IndexingPattern,
    index: object,
    tensor: torch.Tensor,
    dim: int,
    context: SparseCorePlanContext,
) -> bool:
    if isinstance(pattern, ArbitrarySlicePattern):
        return index == slice(None)
    if isinstance(pattern, TilePattern):
        size = tensor.shape[dim]
        block = context.block_sizes.get(pattern.block_id)
        return isinstance(size, int) and block is not None and block >= size
    return False


def _stream_group(
    access: MemoryAccess,
    stop: int,
) -> tuple[int, int]:
    group = 0
    count = 1
    tensor_dim = 0
    for pos, pattern in enumerate(access.patterns[:stop]):
        if isinstance(pattern, NonePattern):
            continue
        if not isinstance(pattern, ArbitraryIndexPattern):
            _reject(
                "access_pattern",
                "dimensions before the item axis must use static indices",
                node=access.node,
            )
        index = access.subscript[pos]
        if not isinstance(index, int):
            _reject(
                "access_pattern",
                "index before the item axis must be a static integer",
                node=access.node,
            )
        size = access.tensor.shape[tensor_dim]
        if not isinstance(size, int):
            _reject("dynamic_shape", "input group has a dynamic size", node=access.node)
        group = group * size + index
        count *= size
        tensor_dim += 1
    return group, count


def _stream_geometry(
    access: MemoryAccess,
    position: int,
    elements_per_item: int,
    context: SparseCorePlanContext,
) -> StreamGeometry:
    pattern = access.patterns[position]
    if not isinstance(pattern, TilePattern):
        _reject("access_pattern", "stream has no tiled item axis", node=access.node)
    group, count = _stream_group(access, position)
    return StreamGeometry(
        group=group,
        group_count=count,
        elements_per_item=elements_per_item,
    )


def _suffix_is_full(
    access: MemoryAccess,
    position: int,
    context: SparseCorePlanContext,
) -> bool:
    tensor_dim = sum(
        not isinstance(pattern, NonePattern)
        for pattern in access.patterns[: position + 1]
    )
    for pos in range(position + 1, len(access.patterns)):
        pattern = access.patterns[pos]
        if isinstance(pattern, NonePattern):
            continue
        if not _full_slice(
            pattern, access.subscript[pos], access.tensor, tensor_dim, context
        ):
            return False
        tensor_dim += 1
    return True


def _normalize_static_offset(node: torch.fx.Node) -> tuple[torch.fx.Node, int]:
    offset = 0
    while node.op == "call_function" and node.target is torch.ops.aten.add.Tensor:
        lhs, rhs = node.args[:2]
        if isinstance(lhs, torch.fx.Node) and isinstance(rhs, int):
            node, offset = lhs, offset + rhs
        elif isinstance(rhs, torch.fx.Node) and isinstance(lhs, int):
            node, offset = rhs, offset + lhs
        else:
            break
    return node, offset


def _build_direct_plan(
    access: MemoryAccess, context: SparseCorePlanContext
) -> SparseCoreMemoryPlan:
    positions = [
        pos
        for pos, pattern in enumerate(access.patterns)
        if isinstance(pattern, TilePattern)
        and pattern.block_id == context.item_block_id
    ]
    if len(positions) != 1:
        _reject(
            "access_pattern",
            "direct DMA requires exactly one item axis",
            node=access.node,
        )
    position = positions[0]
    if access.kind is not MemoryAccessKind.LOAD and any(
        not isinstance(pattern, NonePattern) for pattern in access.patterns[:position]
    ):
        _reject(
            "access_pattern",
            "direct stores require the item axis to be the first tensor dimension",
            node=access.node,
        )
    if not _suffix_is_full(access, position, context):
        _reject(
            "access_pattern",
            "direct DMA requires all dimensions after the item axis",
            node=access.node,
        )
    logical = _logical_shape(access, context)
    elements = math.prod(logical[1:]) if len(logical) > 1 else 1
    stream = _stream_geometry(access, position, elements, context)
    layout = _layout(access, context)
    if access.kind is MemoryAccessKind.LOAD:
        return DirectLoadPlan(access, layout, stream, frozenset())
    assert access.value_node is not None
    return DirectStorePlan(access, layout, stream, frozenset({access.value_node}))


def _build_indirect_plan(
    access: MemoryAccess, context: SparseCorePlanContext
) -> SparseCoreMemoryPlan:
    positions = [
        pos
        for pos, pattern in enumerate(access.patterns)
        if isinstance(pattern, TensorIndexPattern)
    ]
    if len(positions) != 1:
        _reject(
            "access_pattern",
            "indirect DMA over multiple tensor dimensions is not implemented",
            node=access.node,
        )
    position = positions[0]
    if any(
        not isinstance(pattern, NonePattern) for pattern in access.patterns[:position]
    ):
        _reject(
            "access_pattern",
            "indirect DMA requires the index to address the first tensor dimension",
            node=access.node,
        )
    if not _suffix_is_full(access, position, context):
        _reject(
            "access_pattern",
            "indirect DMA requires all dimensions after the index",
            node=access.node,
        )
    raw_index = access.subscript[position]
    if not isinstance(raw_index, torch.fx.Node):
        _reject("access_pattern", "indirect index is not an FX value", node=access.node)
    index_node, offset = _normalize_static_offset(raw_index)
    if access.kind is not MemoryAccessKind.LOAD and offset:
        _reject(
            "access_pattern",
            "static offsets for indirect stores are not implemented",
            node=access.node,
        )
    index_value = index_node.meta.get("val")
    if (
        not isinstance(index_value, torch.Tensor)
        or index_value.dtype is not torch.int32
    ):
        dtype = getattr(index_value, "dtype", None)
        _reject(
            "index_dtype",
            f"indirect index dtype must be int32, got {dtype}",
            node=access.node,
        )
    if access.tensor.dtype not in _INDIRECT_DTYPES:
        _reject(
            "access_dtype",
            f"indirect DMA dtype {access.tensor.dtype} is not implemented",
            node=access.node,
        )
    tensor_dim = sum(
        not isinstance(pattern, NonePattern) for pattern in access.patterns[:position]
    )
    value_size = math.prod(int(dim) for dim in access.tensor.shape[tensor_dim + 1 :])
    value_bytes = value_size * access.tensor.dtype.itemsize
    if value_bytes % SC_DMA_GRANULE_BYTES:
        _reject(
            "gather_granule",
            f"indirect value uses {value_bytes} bytes; it must be a multiple of "
            f"{SC_DMA_GRANULE_BYTES}",
            node=access.node,
        )
    index_shape = index_value.shape[1:]
    if any(not isinstance(dim, int) for dim in index_shape):
        _reject(
            "dynamic_shape",
            f"indirect index has dynamic trailing dimensions {tuple(index_value.shape)}",
            node=access.node,
        )
    entries = math.prod(index_shape) if index_shape else 1
    stream = StreamGeometry(
        group=0,
        group_count=1,
        elements_per_item=entries,
    )
    layout = _layout(access, context)
    value_dependency = (
        access.value_node if access.kind is not MemoryAccessKind.LOAD else None
    )
    dependencies = {index_node}
    if value_dependency is not None:
        dependencies.add(value_dependency)
    frozen_dependencies = frozenset(dependencies)
    if access.kind is MemoryAccessKind.LOAD:
        return IndirectLoadPlan(
            access, layout, stream, frozen_dependencies, index_node, offset
        )
    return IndirectStorePlan(
        access, layout, stream, frozen_dependencies, index_node, offset
    )


def build_sparsecore_memory_plan(
    access: MemoryAccess, context: SparseCorePlanContext
) -> SparseCoreMemoryPlan:
    """Map one memory access to its SparseCore transfer plan."""
    if access.kind is MemoryAccessKind.ATOMIC:
        _reject(
            "access_pattern",
            "SparseCore atomic operations are not implemented",
            node=access.node,
        )

    if any(isinstance(pattern, TensorIndexPattern) for pattern in access.patterns):
        return _build_indirect_plan(access, context)

    if any(isinstance(pattern, TilePattern) for pattern in access.patterns):
        return _build_direct_plan(access, context)

    patterns = ", ".join(type(pattern).__name__ for pattern in access.patterns)
    return _reject(
        "access_pattern",
        f"no SparseCore {access.kind.value} plan for [{patterns}]",
        node=access.node,
        operation=getattr(access.node.target, "__name__", str(access.node.target)),
    )
