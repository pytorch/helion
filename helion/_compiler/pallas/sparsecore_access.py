"""SparseCore lowering for backend-neutral Pallas access sites."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from .access import AccessKind
from .plan_tiling import ArbitraryIndexPattern
from .plan_tiling import ArbitrarySlicePattern
from .plan_tiling import NonePattern
from .plan_tiling import TensorIndexPattern
from .plan_tiling import TilePattern
from .sc_base import _INDIRECT_DTYPES
from .sc_base import SC_CACHED_INPUT_MAX_BYTES
from .sc_base import SC_DMA_GRANULE_BYTES
from .sc_base import SC_LANES
from .sc_base import _reject

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .access import AccessSite
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
class SparseCoreAccess:
    site: AccessSite
    layout: ValueLayout
    stream: StreamGeometry | None
    dependencies: frozenset[torch.fx.Node]


@dataclass(frozen=True)
class DirectLoadAccess(SparseCoreAccess):
    pass


@dataclass(frozen=True)
class IndirectLoadAccess(SparseCoreAccess):
    index_node: torch.fx.Node
    index_offset: int


@dataclass(frozen=True)
class CachedLoadAccess(SparseCoreAccess):
    pass


@dataclass(frozen=True)
class DirectStoreAccess(SparseCoreAccess):
    pass


@dataclass(frozen=True)
class IndirectStoreAccess(SparseCoreAccess):
    index_node: torch.fx.Node
    index_offset: int


@dataclass(frozen=True)
class AtomicAddAccess(SparseCoreAccess):
    index_node: torch.fx.Node
    index_offset: int


@dataclass(frozen=True)
class AccessLoweringContext:
    """Geometry shared by independent SC access lowerings."""

    block_sizes: Mapping[int, int]
    item_block_id: int
    items_per_subcore: int


def _static_shape(value: object, what: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        _reject("layout", f"{what} is not a tensor")
    result: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            _reject("dynamic_shape", f"{what} has dynamic shape {tuple(value.shape)}")
        result.append(dim)
    return tuple(result)


def _value_tensor(site: AccessSite) -> torch.Tensor:
    if site.kind is AccessKind.LOAD:
        value = site.node.meta.get("val")
    else:
        value = site.value_node.meta.get("val") if site.value_node is not None else None
    if not isinstance(value, torch.Tensor):
        _reject("layout", "memory access value is not a tensor", node=site.node)
    return value


def _logical_shape(site: AccessSite, context: AccessLoweringContext) -> tuple[int, ...]:
    value = _value_tensor(site)
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
                node=site.node,
            )
        shape[index] = block
    return tuple(shape)  # type: ignore[arg-type]


def _layout(site: AccessSite, context: AccessLoweringContext) -> ValueLayout:
    value = _value_tensor(site)
    logical = _logical_shape(site, context)
    value_size = math.prod(logical[1:]) if len(logical) > 1 else 1
    stored_size = max(SC_LANES, math.ceil(value_size / SC_LANES) * SC_LANES)
    storage_dtype = (
        torch.int32
        if site.kind is not AccessKind.LOAD and value.dtype in (torch.int8, torch.bool)
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
    context: AccessLoweringContext,
) -> bool:
    if isinstance(pattern, ArbitrarySlicePattern):
        return index == slice(None)
    if isinstance(pattern, TilePattern):
        size = tensor.shape[dim]
        block = context.block_sizes.get(pattern.block_id)
        return isinstance(size, int) and block is not None and block >= size
    return False


def _stream_group(
    site: AccessSite,
    stop: int,
) -> tuple[int, int]:
    group = 0
    count = 1
    tensor_dim = 0
    for pos, pattern in enumerate(site.patterns[:stop]):
        if isinstance(pattern, NonePattern):
            continue
        if not isinstance(pattern, ArbitraryIndexPattern):
            _reject(
                "access_pattern",
                "dimensions before the item axis must use static indices",
                node=site.node,
            )
        index = site.subscripts[pos]
        if not isinstance(index, int):
            _reject(
                "access_pattern",
                "index before the item axis must be a static integer",
                node=site.node,
            )
        size = site.tensor.shape[tensor_dim]
        if not isinstance(size, int):
            _reject("dynamic_shape", "input group has a dynamic size", node=site.node)
        group = group * size + index
        count *= size
        tensor_dim += 1
    return group, count


def _stream_geometry(
    site: AccessSite,
    position: int,
    elements_per_item: int,
    context: AccessLoweringContext,
) -> StreamGeometry:
    pattern = site.patterns[position]
    if not isinstance(pattern, TilePattern):
        _reject("access_pattern", "stream has no tiled item axis", node=site.node)
    group, count = _stream_group(site, position)
    return StreamGeometry(
        group=group,
        group_count=count,
        elements_per_item=elements_per_item,
    )


def _suffix_is_full(
    site: AccessSite,
    position: int,
    context: AccessLoweringContext,
) -> bool:
    tensor_dim = sum(
        not isinstance(pattern, NonePattern)
        for pattern in site.patterns[: position + 1]
    )
    for pos in range(position + 1, len(site.patterns)):
        pattern = site.patterns[pos]
        if isinstance(pattern, NonePattern):
            continue
        if not _full_slice(
            pattern, site.subscripts[pos], site.tensor, tensor_dim, context
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


def _direct_stream(
    site: AccessSite, context: AccessLoweringContext
) -> SparseCoreAccess | None:
    positions = [
        pos
        for pos, pattern in enumerate(site.patterns)
        if isinstance(pattern, TilePattern)
        and pattern.block_id == context.item_block_id
    ]
    if len(positions) != 1 or any(
        isinstance(pattern, TensorIndexPattern) for pattern in site.patterns
    ):
        return None
    position = positions[0]
    if not _suffix_is_full(site, position, context):
        return None
    logical = _logical_shape(site, context)
    elements = math.prod(logical[1:]) if len(logical) > 1 else 1
    stream = _stream_geometry(site, position, elements, context)
    layout = _layout(site, context)
    source = site.tensor_node if site.kind is AccessKind.LOAD else site.value_node
    assert source is not None
    dependencies = frozenset() if site.kind is AccessKind.LOAD else frozenset({source})
    cls = DirectLoadAccess if site.kind is AccessKind.LOAD else DirectStoreAccess
    return cls(site, layout, stream, dependencies)


def _indirect(
    site: AccessSite, context: AccessLoweringContext
) -> SparseCoreAccess | None:
    positions = [
        pos
        for pos, pattern in enumerate(site.patterns)
        if isinstance(pattern, TensorIndexPattern)
    ]
    if not positions:
        return None
    if len(positions) != 1:
        _reject(
            "access_pattern",
            "indirect DMA over multiple tensor dimensions is not implemented",
            node=site.node,
        )
    position = positions[0]
    if any(
        not isinstance(pattern, (ArbitraryIndexPattern, NonePattern))
        for pattern in site.patterns[:position]
    ):
        _reject(
            "access_pattern",
            "indirect DMA is implemented only on the first indexed dimension",
            node=site.node,
        )
    if not _suffix_is_full(site, position, context):
        _reject(
            "access_pattern",
            "indirect DMA requires all dimensions after the index",
            node=site.node,
        )
    raw_index = site.subscripts[position]
    if not isinstance(raw_index, torch.fx.Node):
        _reject("access_pattern", "indirect index is not an FX value", node=site.node)
    index_node, offset = _normalize_static_offset(raw_index)
    if site.kind is not AccessKind.LOAD and offset:
        _reject(
            "access_pattern",
            "static offsets for indirect stores are not implemented",
            node=site.node,
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
            node=site.node,
        )
    if site.tensor.dtype not in _INDIRECT_DTYPES:
        _reject(
            "access_dtype",
            f"indirect DMA dtype {site.tensor.dtype} is not implemented",
            node=site.node,
        )
    if site.kind is AccessKind.ATOMIC and site.tensor.dtype is not torch.float32:
        _reject(
            "atomic_dtype",
            "SparseCore shared-memory atomic add requires float32 values",
            node=site.node,
        )
    if site.kind is AccessKind.ATOMIC and context.items_per_subcore % SC_LANES:
        _reject(
            "atomic_lanes",
            "SparseCore shared-memory atomic add requires complete "
            f"{SC_LANES}-item lane groups per subcore",
            node=site.node,
        )
    tensor_dim = sum(
        not isinstance(pattern, NonePattern) for pattern in site.patterns[:position]
    )
    value_size = math.prod(int(dim) for dim in site.tensor.shape[tensor_dim + 1 :])
    value_bytes = value_size * site.tensor.dtype.itemsize
    if value_bytes % SC_DMA_GRANULE_BYTES:
        _reject(
            "gather_granule",
            f"indirect value uses {value_bytes} bytes; it must be a multiple of "
            f"{SC_DMA_GRANULE_BYTES}",
            node=site.node,
        )
    index_shape = index_value.shape[1:]
    if any(not isinstance(dim, int) for dim in index_shape):
        _reject(
            "dynamic_shape",
            f"indirect index has dynamic trailing dimensions {tuple(index_value.shape)}",
            node=site.node,
        )
    entries = math.prod(index_shape) if index_shape else 1
    stream = StreamGeometry(
        group=0,
        group_count=1,
        elements_per_item=entries,
    )
    layout = _layout(site, context)
    value_dependency = site.value_node if site.kind is not AccessKind.LOAD else None
    dependencies = {index_node}
    if value_dependency is not None:
        dependencies.add(value_dependency)
    args = (site, layout, stream, frozenset(dependencies), index_node, offset)
    if site.kind is AccessKind.LOAD:
        return IndirectLoadAccess(*args)
    if site.kind is AccessKind.STORE:
        return IndirectStoreAccess(*args)
    return AtomicAddAccess(*args)


def _cached_load(
    site: AccessSite, context: AccessLoweringContext
) -> CachedLoadAccess | None:
    if site.kind is not AccessKind.LOAD:
        return None
    if any(
        isinstance(pattern, (TilePattern, TensorIndexPattern))
        for pattern in site.patterns
    ):
        return None
    if not all(
        isinstance(pattern, (ArbitraryIndexPattern, ArbitrarySlicePattern, NonePattern))
        for pattern in site.patterns
    ):
        return None
    storage = [
        (pattern, index)
        for pattern, index in zip(site.patterns, site.subscripts, strict=True)
        if not isinstance(pattern, NonePattern)
    ]
    if (
        not site.patterns
        or not isinstance(site.patterns[0], NonePattern)
        or site.tensor.ndim not in (1, 2)
        or len(storage) != site.tensor.ndim
        or any(
            not isinstance(pattern, ArbitrarySlicePattern) or index != slice(None)
            for pattern, index in storage
        )
    ):
        _reject(
            "access_pattern",
            "cached inputs require a leading broadcast and a complete rank-1 or "
            "rank-2 tensor",
            node=site.node,
        )
    input_bytes = site.tensor.numel() * site.tensor.dtype.itemsize
    if input_bytes > SC_CACHED_INPUT_MAX_BYTES:
        _reject(
            "cached_input_size",
            f"cached input uses {input_bytes} bytes; limit is "
            f"{SC_CACHED_INPUT_MAX_BYTES}",
            node=site.node,
        )
    value = _value_tensor(site)
    value_shape = _static_shape(value, "cached value")
    input_shape = _static_shape(site.tensor, "cached input")
    layout = ValueLayout(
        value_size=math.prod(value_shape),
        storage_shape=input_shape,
        logical_dtype=value.dtype,
        storage_dtype=value.dtype,
    )
    return CachedLoadAccess(site, layout, None, frozenset())


def lower_sparsecore_access(
    site: AccessSite, context: AccessLoweringContext
) -> SparseCoreAccess:
    """Lower one access for SparseCore."""
    for candidate in (_indirect, _direct_stream, _cached_load):
        if result := candidate(site, context):
            return result
    patterns = ", ".join(type(pattern).__name__ for pattern in site.patterns)
    return _reject(
        "access_pattern",
        f"no SparseCore {site.kind.value} lowering for [{patterns}]",
        node=site.node,
        operation=getattr(site.target, "__name__", str(site.target)),
    )
