"""Tiling analysis pass for the Pallas backend.

Analyzes indexing expressions to determine which tensor dimensions can be tiled.
Sets 'dim_tilings' metadata on tensors based on indexing constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import sympy
import torch

from helion._compiler.backend import PallasBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import _symint_expr
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_function import PallasMemorySpace
from helion._compiler.host_function import HostFunction
from helion._compiler.host_function import SymbolOrigin
from helion._compiler.indexing_strategy import _get_tile_with_offset_info
from helion._compiler.variable_origin import GridOrigin
from helion._compiler.variable_origin import TileBeginOrigin
from helion._compiler.variable_origin import TileEndOrigin
from helion._compiler.variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch
    from .gather import GatherPlan
    from .gather import ScatterPlan


@dataclass
class IndexingPattern:
    """Base class for indexing patterns detected during tiling analysis."""


@dataclass
class TilePattern(IndexingPattern):
    """Vanilla tile pattern - translates to ':' when tiled."""

    block_id: int


@dataclass
class TileIndexWithOffsetPattern(IndexingPattern):
    """Tile index with offset - no tiling allowed."""

    block_id: int
    offset: int | torch.SymInt | object


@dataclass
class TileBeginWithOffsetPattern(IndexingPattern):
    """Tile begin with offset - allow/disallow tiling based on bounds."""

    block_id: int
    offset: int | torch.SymInt | object


@dataclass
class ArbitrarySlicePattern(IndexingPattern):
    slice: slice


def _is_full_slice(idx: slice) -> bool:
    return idx.start is None and idx.stop is None and idx.step in (None, 1)


def _static_slice_bound(value: object, env: CompileEnvironment) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, torch.SymInt):
        expr = env.specialize_expr(env.shape_env.replace(value._sympy_()))
        if isinstance(expr, sympy.Integer) and expr >= 0:
            return int(expr)
    return None


def _is_static_contiguous_slice(idx: slice, env: CompileEnvironment) -> bool:
    if idx.step not in (None, 1):
        return False
    return (idx.start is None or _static_slice_bound(idx.start, env) is not None) and (
        idx.stop is None or _static_slice_bound(idx.stop, env) is not None
    )


@dataclass
class ArbitraryIndexPattern(IndexingPattern):
    index: int | torch.SymInt | object | None


@dataclass
class NonePattern(IndexingPattern):
    """None index pattern (broadcasting dimension) - allow tiling."""


@dataclass
class TensorIndexPattern(IndexingPattern):
    """Tensor-valued index - no tiling. Resolved for indirect load/store codegen."""


@dataclass
class IndirectGatherPattern(IndexingPattern):
    """Indirect gather load ``table[idx, ...]`` - no tiling on this dim."""

    plan: GatherPlan


@dataclass
class IndirectScatterPattern(IndexingPattern):
    """Indirect scatter store ``table[idx, ...]`` - no tiling on this dim."""

    plan: ScatterPlan


@dataclass
class DimensionTiling:
    """Tiling decision for a specific dimension of a tensor

    can_tile: whether or not we can tile this dimension
    block_ids: which which block_ids we are indexing this dimension (there can be multiple, in which case we mustn't tile)
    needs_full_slice: whether this dimension is also read as a full slice
    """

    can_tile: bool = True
    block_ids: list[int] = field(default_factory=list)
    needs_full_slice: bool = False


def plan_tiling(
    graphs: list[GraphInfo],
    config: Config,
    tile_strategy: TileStrategyDispatch,
) -> None:
    for graph_info in graphs:
        _analyze_indexing_expressions(graph_info, config)


def _analyze_indexing_expressions(graph_info: GraphInfo, config: Config) -> None:
    from ...language import memory_ops
    from ...language.atomic_ops import ATOMIC_OPS

    indexing_targets = ATOMIC_OPS | {memory_ops.load, memory_ops.store}
    for node in graph_info.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in indexing_targets:
            _analyze_indexing(node, config)


def _analyze_indexing(node: torch.fx.Node, config: Config) -> None:
    tensor_arg = node.args[0]
    subscript = node.args[1]

    assert isinstance(subscript, (list, tuple))
    assert isinstance(tensor_arg, torch.fx.Node)
    tensor_val = tensor_arg.meta.get("val")
    assert isinstance(tensor_val, torch.Tensor)

    device_fn = DeviceFunction.current()
    if id(tensor_val) not in device_fn.pallas_tensor_dim_tilings:
        device_fn.pallas_tensor_dim_tilings[id(tensor_val)] = [
            DimensionTiling() for _ in range(tensor_val.ndim)
        ]
    dim_tilings = device_fn.pallas_tensor_dim_tilings[id(tensor_val)]

    # Store indexing patterns directly on the memory operation node
    indexing_patterns = _analyze_subscript_patterns(
        tensor_val, list(subscript), dim_tilings, node, config
    )
    _resolve_tensor_index_patterns(
        node, tensor_val, list(subscript), indexing_patterns, config
    )
    node.meta["indexing_patterns"] = indexing_patterns

    # Track SMEM eligibility (simplified — does not distinguish read vs write):
    #   SMEM: only scalar access.  VMEM: vector/slice + scalar reads.
    # A fully correct policy would check read vs write per access:
    #   - Scalar read-only tensors could stay in VMEM (no SMEM needed)
    #   - Scalar write requires SMEM
    #   - Mixed scalar-write + slice needs tensor duplication (unsupported)
    # For now we conservatively put all-scalar tensors in SMEM and
    # mixed tensors in VMEM. This is correct for the common cases
    # (scalar-only → SMEM, mixed scalar-read + slice → VMEM) but
    # over-allocates SMEM for scalar-read-only tensors.
    is_all_scalar = all(
        isinstance(p, (ArbitraryIndexPattern, TileBeginWithOffsetPattern, NonePattern))
        for p in indexing_patterns
    )
    tid = id(tensor_val)
    current = device_fn.pallas_memory_space.get(tid)
    if is_all_scalar:
        # Only mark for SMEM if not already assigned to VMEM or HBM
        if current is None:
            device_fn.pallas_memory_space[tid] = PallasMemorySpace.SMEM
    else:
        # Override SMEM → VMEM: this is intentional. When a tensor has
        # both scalar and slice accesses, we keep it in VMEM because
        # scalar *reads* work from VMEM (only scalar writes require
        # SMEM). We optimistically assume the scalar access is a read.
        # Don't override HBM (pipeline tensors).
        if current != PallasMemorySpace.HBM:
            device_fn.pallas_memory_space[tid] = PallasMemorySpace.VMEM


def _analyze_subscript_patterns(
    tensor: torch.Tensor,
    subscript: list[object],
    dim_tilings: list[DimensionTiling],
    node: torch.fx.Node,
    config: Config,
) -> list[IndexingPattern]:
    """Analyze subscript patterns and create indexing pattern metadata."""

    env = CompileEnvironment.current()
    patterns: list[IndexingPattern] = []
    tensor_dim = 0  # Track which tensor dimension we're indexing

    for i, idx in enumerate(subscript):
        if idx is None:
            # None adds an unsqueezed dimension but doesn't consume a tensor dimension
            patterns.append(NonePattern())
            continue

        if tensor_dim >= tensor.ndim:
            raise AssertionError(
                f"Indexing {tensor_dim}th dim but tensor only has {tensor.ndim} dims"
            )

        # Detect different indexing patterns
        pattern = _detect_indexing_pattern(idx, tensor, tensor_dim, node, i, env)
        patterns.append(pattern)

        # Update dim_tilings based on the detected pattern
        _update_tiling_decision(tensor, pattern, tensor_dim, dim_tilings, env, config)

        tensor_dim += 1

    return patterns


def _detect_indexing_pattern(
    idx: object,
    tensor: torch.Tensor,
    tensor_dim: int,
    node: torch.fx.Node,
    subscript_index: int,
    env: CompileEnvironment,
) -> IndexingPattern:
    """Detect the specific indexing pattern for a subscript element."""

    if isinstance(idx, torch.fx.Node):
        idx_val = idx.meta.get("val")
        if isinstance(idx_val, torch.SymInt):
            block_id = env.get_block_id(idx_val)
            if block_id is not None:
                symbol_origin = _maybe_get_symbol_origin(idx_val)
                is_hl_grid = symbol_origin is not None and isinstance(
                    symbol_origin.origin, GridOrigin
                )
                if not is_hl_grid:
                    return TilePattern(block_id=block_id)

        tile_with_offset = _get_tile_with_offset_info(idx_val, node, subscript_index)
        if tile_with_offset is not None:
            return TileIndexWithOffsetPattern(
                block_id=tile_with_offset.block_id, offset=tile_with_offset.offset
            )

        # Check for TileBeginWithOffset pattern (t.begin, t.end-1)
        tile_begin_with_offset = _maybe_get_tile_begin_with_offset_info(idx_val)
        if tile_begin_with_offset is not None:
            return TileBeginWithOffsetPattern(
                block_id=tile_begin_with_offset.block_id,
                offset=tile_begin_with_offset.offset,
            )
        # A tensor-valued index that didn't match any arithmetic-of-tile
        # pattern is an indirect gather (e.g. table[idx, :]).
        if isinstance(idx_val, torch.Tensor):
            return TensorIndexPattern()
        # Indices produced by other FX nodes, such as indices[tile] used in
        # tensor-indexed atomics, are legal but cannot participate in Pallas
        # tiling.
        return ArbitraryIndexPattern(idx)

    if isinstance(idx, slice):
        if not _is_static_contiguous_slice(idx, env):
            raise AssertionError(
                f"Only static contiguous slices are supported in Pallas backend, got {idx}"
            )
        return ArbitrarySlicePattern(idx)

    if isinstance(idx, (int, torch.SymInt)):
        return ArbitraryIndexPattern(idx)

    raise AssertionError(f"Unrecognized indexing pattern for pallas backend {idx}")


def _update_tiling_decision(
    tensor: torch.Tensor,
    pattern: IndexingPattern,
    tensor_dim: int,
    dim_tilings: list[DimensionTiling],
    env: CompileEnvironment,
    config: Config,
) -> None:
    """Update tiling decision based on the detected indexing pattern."""

    curr_dim_tiling = dim_tilings[tensor_dim]

    def _disallow_tiling() -> None:
        curr_dim_tiling.can_tile = False

    def _try_set_tiling_block_id(new_block_id: int) -> None:
        if new_block_id not in curr_dim_tiling.block_ids:
            curr_dim_tiling.block_ids.append(new_block_id)
            if len(curr_dim_tiling.block_ids) > 1:
                # we already need to tile this dim using a different block_id
                # so fallback to no-tiling so that we can access using both tiles
                _disallow_tiling()
        if curr_dim_tiling.needs_full_slice:
            _disallow_tiling()

    def _record_full_slice() -> None:
        curr_dim_tiling.needs_full_slice = True
        if curr_dim_tiling.block_ids:
            _disallow_tiling()

    if isinstance(pattern, TilePattern):
        _try_set_tiling_block_id(pattern.block_id)

    elif isinstance(pattern, TileIndexWithOffsetPattern):
        _disallow_tiling()

    elif isinstance(pattern, TileBeginWithOffsetPattern):
        _try_set_tiling_block_id(pattern.block_id)
        # check bounds
        if not isinstance(pattern.offset, int) or pattern.offset < 0:
            _disallow_tiling()
        else:
            block_size = env.block_sizes[pattern.block_id].from_config(config)
            if isinstance(block_size, int) and pattern.offset >= block_size:
                _disallow_tiling()

    elif isinstance(pattern, ArbitrarySlicePattern):
        if _is_full_slice(pattern.slice):
            _record_full_slice()
        else:
            # fow now we only support the `[:]` slice pattern
            _disallow_tiling()

    elif isinstance(pattern, (ArbitraryIndexPattern, TensorIndexPattern)):
        _disallow_tiling()

    elif isinstance(pattern, NonePattern):
        pass

    if isinstance(pattern, (TilePattern, TileBeginWithOffsetPattern)):
        block_size = env.block_sizes[pattern.block_id].from_config(config)
        if isinstance(block_size, int):
            backend = CompileEnvironment.current().backend

            assert isinstance(backend, PallasBackend)

            dim_from_end = tensor.ndim - tensor_dim - 1
            bitwidth = tensor.dtype.itemsize * 8
            required_alignment = backend._get_pallas_required_alignment(
                dim_from_end, tensor.ndim, bitwidth
            )

            if (
                block_size < tensor.shape[tensor_dim]
                and block_size % required_alignment != 0
            ):
                _disallow_tiling()


def resident_block_elements(
    tensor: torch.Tensor,
    patterns: list[IndexingPattern],
    config: Config,
) -> int | None:
    """Element count of the VMEM-resident block for one tensor access.

    Walks ``patterns`` alongside the tensor dims. Per-dim contribution:
      - ``NonePattern``: skipped (broadcast axis, no tensor dim consumed).
      - ``TilePattern`` / ``TileIndexWithOffsetPattern``: configured
        ``block_size``, clamped to the full dim extent.
      - ``TileBeginWithOffsetPattern`` / ``ArbitraryIndexPattern``: scalar
        index, contributes 1.
      - Anything else (full slice, indirect tensor index): the full dim
        extent.

    Returns ``None`` if any consumed dim is symbolic.
    """

    env = CompileEnvironment.current()
    elements = 1
    tdim = 0
    for p in patterns:
        if isinstance(p, NonePattern):
            continue
        dim_size = tensor.shape[tdim]
        if not isinstance(dim_size, int):
            # No support for dynamic shapes.
            return None
        if isinstance(p, (TilePattern, TileIndexWithOffsetPattern)):
            bs = env.block_sizes[p.block_id].from_config(config)
            if isinstance(bs, int):
                dim_size = min(bs, dim_size)
        elif isinstance(p, (TileBeginWithOffsetPattern, ArbitraryIndexPattern)):
            dim_size = 1
        elements *= dim_size
        # Advance only on patterns that consume a tensor dim; NonePattern doesn't.
        tdim += 1
    return elements


def _resolve_tensor_index_patterns(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: list[object],
    patterns: list[IndexingPattern],
    config: Config,
) -> None:
    """Replace TensorIndexPattern with Pallas indirect load/store patterns."""
    positions = [i for i, p in enumerate(patterns) if isinstance(p, TensorIndexPattern)]
    if not positions:
        return

    from ...language import memory_ops

    if node.target is memory_ops.load:
        from .gather import build_gather_plan

        plan = build_gather_plan(tensor, subscript, positions, patterns, config)
        for i in positions:
            patterns[i] = IndirectGatherPattern(plan=plan)
        return

    if node.target is memory_ops.store:
        from .gather import build_scatter_plan

        plan = build_scatter_plan(tensor, subscript, positions)
        for i in positions:
            patterns[i] = IndirectScatterPattern(plan=plan)
        return

    from ...language.atomic_ops import atomic_add

    if node.target is atomic_add:
        from .gather import build_scatter_plan

        plan = build_scatter_plan(tensor, subscript, positions)
        for i in positions:
            patterns[i] = IndirectScatterPattern(plan=plan)
        return

    op_name = getattr(node.target, "__name__", str(node.target))
    raise NotImplementedError(
        f"Pallas: tensor-indexed memory op is not supported for op={op_name}."
    )


# Helper functions moved from memory_ops.py
def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    """Get symbol origin for a subscript element."""

    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


def _maybe_get_tile_begin_with_offset_info(
    idx: object,
) -> TileBeginWithOffsetPattern | None:
    """Extended version that allows out-of-bounds and symbolic offsets.

    Matches expressions that resolve to a tile's start offset within the
    full loop extent (e.g. ``tile.begin``, ``tile.end - 1``, or affine
    combinations of those with integer constants).
    """

    idx_symbol_origin = _maybe_get_symbol_origin(idx)
    if isinstance(idx_symbol_origin, SymbolOrigin):
        if isinstance(idx_symbol_origin.origin, TileBeginOrigin):
            return TileBeginWithOffsetPattern(
                block_id=idx_symbol_origin.origin.block_id, offset=0
            )
        if isinstance(idx_symbol_origin.origin, GridOrigin) and not isinstance(
            idx_symbol_origin.origin, (TileEndOrigin, TileIdOrigin)
        ):
            return TileBeginWithOffsetPattern(
                block_id=idx_symbol_origin.origin.block_id, offset=0
            )

    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if not isinstance(expr, sympy.Expr):
        return None

    args = expr.args
    origin: TileBeginOrigin | TileEndOrigin | GridOrigin | None = None
    offset = 0

    for arg in args:
        assert isinstance(arg, sympy.Expr)
        if (
            symbol_origin := HostFunction.current().expr_to_origin.get(arg)
        ) is not None:
            if isinstance(
                symbol_origin.origin, (GridOrigin, TileBeginOrigin, TileEndOrigin)
            ):
                if origin is not None:
                    # Multiple tile offset expressions - result is out of current tile
                    return None
                origin = symbol_origin.origin
            else:
                return None
        elif arg.is_constant():
            evalf_result = arg.evalf()
            f_value = float(evalf_result)  # type: ignore[arg-type]
            if not f_value.is_integer():
                return None
            offset += int(f_value)
        else:
            # pyrefly: ignore [bad-argument-type]
            offset = torch.SymInt(arg)
            break

    env = CompileEnvironment.current()
    if origin is None:
        return None

    block_id = origin.block_id

    if isinstance(origin, TileEndOrigin):
        block_size = env.block_sizes[block_id].size
        if isinstance(block_size, int) and isinstance(offset, int):
            offset = block_size + offset  # Starting from end
        else:
            # For non-integer block sizes or offsets, fall back to symbolic offset
            offset = torch.SymInt(f"{block_size} + {offset}")  # type: ignore[arg-type]

    return TileBeginWithOffsetPattern(block_id=block_id, offset=offset)
