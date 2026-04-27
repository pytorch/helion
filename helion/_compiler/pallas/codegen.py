"""Pallas indexing codegen helpers."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


def _can_tile_dimension(state: CodegenState, tensor_dim: int) -> bool:
    assert state.fx_node is not None
    tensor_arg_node = state.fx_node.args[0]  # 0th argument to load/store is the tensor
    assert isinstance(tensor_arg_node, torch.fx.Node)

    tensor_val = tensor_arg_node.meta.get("val")
    assert isinstance(tensor_val, torch.Tensor)

    dim_tilings = state.device_function.pallas_tensor_dim_tilings.get(id(tensor_val))
    assert isinstance(dim_tilings, list)
    assert tensor_dim < len(dim_tilings)
    from helion._compiler.pallas.plan_tiling import DimensionTiling

    assert isinstance(dim_tilings[tensor_dim], DimensionTiling)
    return dim_tilings[tensor_dim].can_tile


def index_str(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    tensor: torch.Tensor,
) -> tuple[str, list[int]]:
    """Build a JAX/Pallas index string from a Helion subscript list.

    Uses ``pl.ds(offset, block_size)`` only for dimensions inside a looped
    reduction (``DeviceLoopState``).  Grid dimensions and persistent
    reduction dimensions use ``...`` — Pallas BlockSpecs in the launcher
    handle the grid-level tiling.

    For ``EmitPipelineLoopState`` or ``ForiLoopState``, pipeline-tiled
    dimensions also use ``...`` since the pipeline handles that tiling
    (via BlockSpecs or DMA copies respectively).

    Also returns positions of ``None`` indices so the caller can apply
    ``jnp.expand_dims`` after loading.
    """
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    if not subscript:
        return "...", []

    # Check if we're inside an emit_pipeline or fori_loop with DMA.
    # When fori_loop runs without DMA (use_dma=False), its block_ids
    # are NOT treated as pipeline dims — they get pl.ds() slicing instead.
    in_pipeline = False
    pipeline_block_ids: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, EmitPipelineLoopState) or (
                isinstance(loop, ForiLoopState) and loop.use_dma
            ):
                in_pipeline = True
                pipeline_block_ids.update(loop.block_ids)

    # Use pre-computed indexing patterns from plan_tiling analysis
    indexing_patterns = _get_indexing_patterns(state, tensor)

    # Build parts using the pre-computed patterns
    parts: list[str] = []
    none_dims: list[int] = []
    out_pos = 0
    tensor_dim = 0

    for i, (idx, pattern) in enumerate(zip(subscript, indexing_patterns, strict=True)):
        if idx is None:
            none_dims.append(out_pos)
            out_pos += 1
            continue

        # Generate code based on the pattern type
        index_code = _generated_index_code(
            pattern, idx, state, tensor, i, tensor_dim, in_pipeline, pipeline_block_ids
        )
        parts.append(index_code)

        out_pos += 1
        tensor_dim += 1

    return ", ".join(parts), none_dims


def _get_indexing_patterns(state: CodegenState, tensor: torch.Tensor) -> list[object]:
    assert state.fx_node is not None
    assert hasattr(state.fx_node, "meta")
    patterns = state.fx_node.meta.get("indexing_patterns")
    assert patterns is not None, f"No indexing patterns found for node {state.fx_node}"
    return patterns


def _generated_index_code(
    pattern: object,
    idx: object,
    state: CodegenState,
    tensor: torch.Tensor,
    subscript_index: int,
    tensor_dim: int,
    in_pipeline: bool,
    pipeline_block_ids: set[int],
) -> str:
    """Generate index code based on the indexing pattern."""
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    if isinstance(pattern, TilePattern):
        return _tile_pattern_code(
            pattern, idx, state, tensor, tensor_dim, in_pipeline, pipeline_block_ids
        )

    if isinstance(pattern, TileIndexWithOffsetPattern):
        return _tile_index_with_offset_pattern_code(pattern, state, tensor, tensor_dim)

    if isinstance(pattern, TileBeginWithOffsetPattern):
        return _tile_begin_with_offset_pattern_code(
            pattern, state, subscript_index, tensor_dim
        )

    if isinstance(pattern, ArbitrarySlicePattern):
        return _slice_code(idx, pattern, state, tensor, tensor_dim)

    if isinstance(pattern, ArbitraryIndexPattern):
        if isinstance(idx, int):
            return str(idx)
        return _index_expr_from_ast(state, subscript_index)

    raise RuntimeError(
        f"Unhandled indexing pattern type: {type(pattern).__name__}. "
        f"Pattern: {pattern}, idx: {idx}, subscript_index: {subscript_index}. "
        f"All indexing patterns should be handled by the tiling analysis system."
    )


def _tile_pattern_code(
    pattern: object,
    idx: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    in_pipeline: bool,
    pipeline_block_ids: set[int],
) -> str:
    from helion._compiler.pallas.plan_tiling import TilePattern
    from helion._compiler.tile_strategy import DeviceLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    assert isinstance(pattern, TilePattern)

    block_id = pattern.block_id

    # Pipeline-tiled dims are already sliced by emit_pipeline / fori_loop's
    # BlockSpec or DMA copy, so the body should use ``:`` regardless of
    # whether the planner marked the dim as tileable.
    # TODO(yifeixu): the long-term fix is making ``can_tile`` per-loop-scope
    # instead of per-tensor-dim so the planner doesn't mark this dim
    # untileable in pipeline mode in the first place.
    if in_pipeline and block_id in pipeline_block_ids:
        return ":"

    can_tile = _can_tile_dimension(state, tensor_dim)
    if not can_tile:
        return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)

    loops = state.codegen.active_device_loops.get(block_id)
    if loops and any(
        isinstance(loop, DeviceLoopState)
        or (isinstance(loop, ForiLoopState) and not loop.use_dma)
        for loop in loops
    ):
        return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)
    return ":"


def _tile_index_with_offset_pattern_code(
    pattern: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> str:
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern

    assert isinstance(pattern, TileIndexWithOffsetPattern)

    block_id = pattern.block_id
    offset_str = state.device_function.literal_expr(pattern.offset)
    return _ds_expr(state, block_id, offset_str, tensor=tensor, tensor_dim=tensor_dim)


def _tile_begin_with_offset_pattern_code(
    pattern: object,
    state: CodegenState,
    subscript_index: int,
    tensor_dim: int,
) -> str:
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, TileBeginWithOffsetPattern)

    can_tile = _can_tile_dimension(state, tensor_dim)

    if not can_tile:
        return _index_expr_from_ast(state, subscript_index)

    assert isinstance(pattern.offset, int)

    loops = state.codegen.active_device_loops.get(pattern.block_id)
    if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
        offset = state.codegen.offset_var(pattern.block_id)
        if pattern.offset != 0:
            offset = f"{offset} + {pattern.offset}"
        return offset

    return f"{pattern.offset}"


def _index_expr_from_ast(state: CodegenState, subscript_index: int) -> str:
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[subscript_index]
    assert isinstance(ast_idx, ast.AST)
    name = state.codegen.lift(ast_idx, dce=True, prefix="index")
    return name.id


def _slice_code(
    idx: object,
    pattern: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> str:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
    from helion._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, ArbitrarySlicePattern)

    if idx != slice(None):
        raise AssertionError(
            f"Arbitrary slice expr {slice} not supported in Pallas backend yet"
        )

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor.shape[tensor_dim])
    if block_id is not None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
            if block_id is not None:
                return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)

    return ":"


def _ds_expr(
    state: CodegenState,
    block_id: int,
    tile_offset: str = "",
    *,
    tensor: torch.Tensor | None = None,
    tensor_dim: int | None = None,
) -> str:
    """Return a ``pl.ds(offset, block_size)`` expression for *block_id*, offset by *tile_offset*.

    When *tensor* and *tensor_dim* are provided, records the dimension in
    ``pallas_pad_info`` so the launcher can zero-pad non-divisible dims.
    """
    offset = state.codegen.offset_var(block_id)
    if tile_offset:
        offset = f"{offset} + {tile_offset}"
    block_size = state.device_function.block_size_var(block_id)
    if block_size is None:
        return ":"
    if tensor is not None and tensor_dim is not None:
        from helion.language.memory_ops import _record_pad_info

        _record_pad_info(state, tensor, tensor_dim, block_id)
    return f"pl.ds({offset}, {block_size})"


def vmem_name(state: CodegenState, name: str) -> str:
    """Remap a tensor name to its VMEM ref name when inside emit_pipeline or fori_loop."""
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, (EmitPipelineLoopState, ForiLoopState)):
                mapping = getattr(loop, "_tensor_to_vmem", None)
                if mapping and name in mapping:
                    return mapping[name]
    return name
