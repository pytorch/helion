"""Pallas indexing codegen helpers."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from helion._compiler.ast_extension import expr_from_string

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.fx.node import Node

    from helion._compiler.aten_lowering import LoweringContext
    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.tile_strategy import DeviceLoopOrGridState


def load_expr(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
) -> ast.AST:
    """Pallas load codegen: normal path, or indirect gather if ``plan_tiling`` flagged it."""
    from helion._compiler.pallas.gather import emit_gather
    from helion._compiler.pallas.plan_tiling import IndirectGatherPattern

    name = state.device_function.tensor_arg(tensor).name
    name = vmem_name(state, name)
    device_fn = state.device_function
    device_fn.device_load_index += 1
    device_fn.device_memory_op_index += 1

    assert state.fx_node is not None
    patterns = state.fx_node.meta.get("indexing_patterns") or ()
    for pattern in patterns:
        if isinstance(pattern, IndirectGatherPattern):
            return emit_gather(state, pattern.plan, name)

    idx_str, none_dims = index_str(state, subscript, tensor)
    mask_expr = _load_mask_expr(state, subscript, tensor)
    if mask_expr is not None:
        result = expr_from_string(f"{name}[{idx_str}] * ({mask_expr})")
    else:
        result = expr_from_string(f"{name}[{idx_str}]")
    for dim in none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


def maybe_codegen_resident_prep_cache_read(
    ctx: LoweringContext, node: Node
) -> ast.AST | None:
    """Return a prep-cache read for an active resident-prep descriptor, if any."""
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.generate_ast import GenerateAST
    from helion._compiler.pallas.compact_worklist import metadata_ref_for_field

    if not isinstance(ctx.cg, GenerateAST):
        return None
    lowering = ctx.cg.resident_prep_lowering_for_node(node)
    if lowering is None:
        return None
    env = CompileEnvironment.current()
    decision = env.compact_worklist_resident_cache_decision
    plan = env.compact_worklist_plan
    if (
        decision is None
        or not decision.active
        or plan is None
        or plan.ordered_axis is None
    ):
        return None
    ordered_block_id = plan.ordered_axis.block_id
    block_size = ctx.cg.device_function.block_size_var(ordered_block_id)
    if block_size is None:
        return None
    assert decision.resident_key_fields == ("range_start",)
    offset = ctx.cg.offset_var(ordered_block_id)
    range_start_ref = f"{metadata_ref_for_field(plan, 'range_start')}[_wid]"
    local_ordered = f"pl.ds(({offset}) - ({range_start_ref}), {block_size})"
    rank = len(lowering.hoist.perm)
    window_elts = [local_ordered]
    window_elts.extend(":" for _ in range(rank - 1))
    read = f"{lowering.cache_name}[{', '.join(window_elts[p] for p in lowering.hoist.perm)}]"
    return expr_from_string(read)


def _load_mask_expr(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
) -> str | None:
    """Build a mask expression for a Pallas load to zero out-of-bounds data.

    Iterates over the indexing patterns for this load.  For each TilePattern
    whose loop range does not match the tensor's dimension size (e.g.
    data-dependent bounds, constexpr sub-ranges), generates a mask term so
    that out-of-tile positions are zeroed.

    Only applies to dimensions that are ds-padded (the ref is padded to a
    multiple of block_size).  Grid/tile dimensions where BlockSpecs size the
    ref to the actual remainder are not masked — a block-sized mask would
    cause a shape mismatch against the smaller ref.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    assert state.fx_node is not None
    output_val = state.fx_node.meta.get("val")
    if not isinstance(output_val, torch.Tensor):
        return None

    indexing_patterns = _get_indexing_patterns(state, tensor)
    env = CompileEnvironment.current()
    output_sizes = [*output_val.size()]
    # Dims whose mask has been deferred to a downstream ``_mask_to`` by
    # ``defer_pallas_load_masks`` -- masked later in the consumer layout instead.
    deferred = state.fx_node.meta.get("pallas_deferred_mask_block_ids") or frozenset()
    mask_exprs: list[str] = []
    dtype_str: str | None = None
    out_dim = 0
    tensor_dim = 0

    squeezing_patterns = (
        ArbitraryIndexPattern,
        TileIndexWithOffsetPattern,
        TileBeginWithOffsetPattern,
    )

    for idx, pattern in zip(subscript, indexing_patterns, strict=True):
        if idx is None:
            out_dim += 1
            continue

        if isinstance(pattern, TilePattern):
            block_id = pattern.block_id
            # Skip masking for size-1 (broadcast) dims: a single element is
            # always valid, and applying a block-sized mask would broadcast
            # the dim from 1 to block_size, causing shape mismatches.
            dim_size = tensor.shape[tensor_dim]
            if (
                block_id not in deferred
                and (not isinstance(dim_size, int) or dim_size > 1)
                and _tile_needs_mask(state, block_id, tensor, tensor_dim)
            ):
                mask_var = state.codegen.mask_var(block_id)
                if mask_var is not None:
                    if dtype_str is None:
                        dtype_str = env.backend.dtype_str(tensor.dtype)
                    expand = state.tile_strategy.expand_str(output_sizes, out_dim)
                    expr = f"({mask_var}.astype({dtype_str}){expand})"
                    mask_exprs.append(expr)

        # TODO(dunfanlu): Do other patterns beside TilePattern require masking?

        if not isinstance(pattern, squeezing_patterns):
            out_dim += 1
        tensor_dim += 1

    if not mask_exprs:
        return None
    return "*".join(mask_exprs)


def _iter_dma_scratch_loops(
    state: CodegenState, name: str
) -> Iterator[tuple[DeviceLoopOrGridState, str]]:
    """Yield ``(loop, scratch_ref_name)`` for every active loop that DMA-routes
    ``name`` through a VMEM scratch."""
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, (EmitPipelineLoopState, ForiLoopState)):
                mapping = getattr(loop, "_tensor_to_dma_scratch", None)
                if mapping and name in mapping:
                    yield loop, mapping[name]


def _find_dma_scratch_loop(
    state: CodegenState, name: str
) -> tuple[DeviceLoopOrGridState | None, str]:
    """Find the active loop that DMA-routes ``name`` and its scratch ref.

    Returns ``(loop, scratch_ref_name)`` for the first matching active loop, or
    ``(None, name)`` when the tensor is not scratch-routed.

    TODO: this returns the *first* matching active loop, which is wrong
    when nested fori_loops scratch-route the *same* tensor at multiple levels --
    a body access should resolve to the innermost (current) loop's scratch, not
    the first.  It silently miscompiles e.g.::

        for tile_m in hl.tile(m):
            out[tile_m, :] = x[tile_m, :] + 1.0
            for tile_n in hl.tile(n):
                out[tile_m, tile_n] = out[tile_m, tile_n] + 2.0

    where the inner RMW's load/store bind to the outer loop's scratch while its
    DMA uses the inner loop's scratch, producing wrong results with no error.
    This can be fixed by resolving against the current loop instead of first-match.
    """
    return next(_iter_dma_scratch_loops(state, name), (None, name))


def _tensor_routed_to_fori_scratch(state: CodegenState, tensor: torch.Tensor) -> bool:
    """True if ``tensor``'s store destination ref is a fori_loop DMA scratch."""
    from helion._compiler.tile_strategy import ForiLoopState

    name = state.codegen.device_function.tensor_arg(tensor).name
    loop, _ref = _find_dma_scratch_loop(state, name)
    return isinstance(loop, ForiLoopState)


def sliced_value_for_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_parts: list[str],
    value: ast.AST,
) -> ast.AST:
    """Slice the store value when the Pallas ref is smaller than the tile.

    The launcher clamps each BlockSpec dimension to
    ``min(block_size, tensor.shape[d])``.  When ``block_size > dim_size``
    the kernel ref is ``dim_size``-shaped but the computed value is
    ``block_size``-shaped, so we must slice the value before storing.

    This only applies to grid-tiled dimensions that produce ``:`` in the
    generated Pallas index.  Dimensions indexed via ``pl.ds()`` are padded
    instead of clamped, so they must keep their full block-size value.

    fori_loop-scratch stores are exempt: their destination is a block-sized
    VMEM scratch (not a clamped ref), so the value stays block-shaped and the
    writeback DMA clamps the extent instead (see ``_build_dma_slices``).
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import TilePattern

    assert state.fx_node is not None
    patterns = state.fx_node.meta.get("indexing_patterns")
    if patterns is None:
        return value

    if _tensor_routed_to_fori_scratch(state, tensor):
        return value

    env = CompileEnvironment.current()
    slices: list[str] = []
    needs_slice = False
    tensor_dim = 0

    index_part_idx = 0
    for idx, pattern in zip(subscript, patterns, strict=True):
        if idx is None:
            continue

        value_slice = ":"
        index_part = index_parts[index_part_idx]
        index_part_idx += 1
        if isinstance(pattern, TilePattern) and index_part == ":":
            block_size = env.block_sizes[pattern.block_id].from_config(state.config)
            dim_size = tensor.shape[tensor_dim]
            if (
                isinstance(block_size, int)
                and isinstance(dim_size, int)
                and dim_size < block_size
            ):
                value_slice = f":{dim_size}"
                needs_slice = True

        slices.append(value_slice)
        tensor_dim += 1

    if not needs_slice:
        return value

    return expr_from_string(
        f"{{value}}[{', '.join(slices)}]",
        value=value,
    )


def _tile_needs_mask(
    state: CodegenState,
    block_id: int,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> bool:
    """Return True when a TilePattern dimension needs load-time masking.

    A mask is needed when the tile loop's iteration range does not cover the
    full tensor dimension — i.e. the loop end differs from the tensor's
    symbolic size at *tensor_dim*.  This includes data-dependent bounds and
    constexpr sub-ranges.
    """
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return False
    info = loops[-1].block_id_to_info.get(block_id)
    if info is None:
        return False
    dim_size = tensor.shape[tensor_dim]
    if not info.is_end_matching(dim_size):
        return True
    return info.begin_expr is not None and info.begin_expr != 0


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
    parts, none_dims = index_parts(state, subscript, tensor)
    return ", ".join(parts), none_dims


def index_parts(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    tensor: torch.Tensor,
) -> tuple[list[str], list[int]]:
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
    if not subscript:
        return ["..."], []

    # Check if we're inside an emit_pipeline or fori_loop that pipelines
    # this specific tensor.  Both loop types take a per-tensor decision:
    # only tensors present in the loop's _tensor_to_dma_scratch mapping were
    # routed through the inner DMA / Buffered BlockSpec.  Others stay on
    # their outer BlockSpec and fall through to pl.ds().
    tensor_name = state.codegen.device_function.tensor_arg(tensor).name
    in_pipeline = False
    pipeline_block_ids: set[int] = set()
    for loop, _ref in _iter_dma_scratch_loops(state, tensor_name):
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

    return parts, none_dims


def _get_indexing_patterns(state: CodegenState, tensor: torch.Tensor) -> list[object]:
    assert state.fx_node is not None
    assert hasattr(state.fx_node, "meta")
    patterns = state.fx_node.meta.get("indexing_patterns")
    assert patterns is not None, f"No indexing patterns found for node {state.fx_node}"
    return patterns


def _arbitrary_index_pattern_code(
    pattern: object,
    idx: object,
    state: CodegenState,
    subscript_index: int,
    in_pipeline: bool,
) -> str:
    from helion._utils import is_scalar_index

    if in_pipeline and is_scalar_index(idx):
        return "0"
    if isinstance(idx, int):
        return str(idx)
    return _index_expr_from_ast(state, subscript_index)


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
    from helion._compiler.pallas.plan_tiling import IndirectGatherPattern
    from helion._compiler.pallas.plan_tiling import IndirectScatterPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    if isinstance(pattern, TilePattern):
        return _tile_pattern_code(
            pattern, idx, state, tensor, tensor_dim, in_pipeline, pipeline_block_ids
        )

    if isinstance(pattern, TileIndexWithOffsetPattern):
        return _tile_index_with_offset_pattern_code(
            pattern, state, tensor, tensor_dim, in_pipeline, pipeline_block_ids
        )

    if isinstance(pattern, TileBeginWithOffsetPattern):
        return _tile_begin_with_offset_pattern_code(
            pattern, state, subscript_index, tensor_dim, in_pipeline, pipeline_block_ids
        )

    if isinstance(pattern, ArbitrarySlicePattern):
        return _slice_code(idx, pattern, state, tensor, tensor_dim)

    if isinstance(pattern, ArbitraryIndexPattern):
        return _arbitrary_index_pattern_code(
            pattern, idx, state, subscript_index, in_pipeline
        )

    if isinstance(pattern, IndirectGatherPattern):
        # The gather emitter consumes the tensor index and projects the full
        # resident table axis through one-hot, so normal load codegen must
        # expose that axis instead of indexing it a second time.
        return ":"

    if isinstance(pattern, IndirectScatterPattern):
        # The scatter emitter consumes the tensor index and projects source lanes
        # through one-hot matrices, so normal store codegen must expose the full
        # resident target axis instead of indexing it a second time.
        return ":"

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
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    assert isinstance(pattern, TilePattern)

    block_id = pattern.block_id

    # Pipeline-tiled dims are already sliced by emit_pipeline / fori_loop's
    # BlockSpec or DMA copy, so the body should use ``:`` regardless of
    # whether the planner marked the dim as tileable.
    # TODO(yifeixu): the long-term fix is making ``can_tile`` per-loop-scope
    # instead of per-tensor-dim so the planner doesn't mark this dim
    # untileable in pipeline mode in the first place.
    if in_pipeline:
        return ":"

    can_tile = _can_tile_dimension(state, tensor_dim)
    if not can_tile:
        return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)

    # Non-pipelined inner-loop tensors: a pipeline/fori loop exists over
    # this block_id but this specific tensor was left on its outer
    # BlockSpec, so the kernel must slice it in VMEM with pl.ds().
    loops = state.codegen.active_device_loops.get(block_id)
    if loops and any(
        isinstance(loop, (DeviceLoopState, EmitPipelineLoopState, ForiLoopState))
        for loop in loops
    ):
        return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)
    return ":"


def _tile_index_with_offset_pattern_code(
    pattern: object,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    in_pipeline: bool,
    pipeline_block_ids: set[int],
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
    in_pipeline: bool,
    pipeline_block_ids: set[int],
) -> str:
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, TileBeginWithOffsetPattern)

    block_id = pattern.block_id
    offset_str = state.device_function.literal_expr(pattern.offset)

    if in_pipeline and block_id in pipeline_block_ids:
        return offset_str

    can_tile = _can_tile_dimension(state, tensor_dim)

    if not can_tile:
        return _index_expr_from_ast(state, subscript_index)

    assert isinstance(pattern.offset, int)

    loops = state.codegen.active_device_loops.get(block_id)
    if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
        offset = state.codegen.offset_var(block_id)
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
    assert isinstance(idx, slice)

    if idx != slice(None):
        # Bounded contiguous slice on an untiled dim: static slice into the
        # full-width block.
        start = (
            "" if idx.start is None else state.device_function.literal_expr(idx.start)
        )
        stop = "" if idx.stop is None else state.device_function.literal_expr(idx.stop)
        return f"{start}:{stop}"

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor.shape[tensor_dim])
    if block_id is not None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
            return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)

    return ":"


def _is_compact_aligned_load(
    state: CodegenState, block_id: int, tensor: torch.Tensor | None
) -> bool:
    """True if *tensor* is a compact-tile aligned-load or exact-store tensor.

    Both get a per-tile ``pl.Element`` BlockSpec sliced at ``tile_start`` (Pallas
    double-buffers the load's prefetch and the store's write-back), so the body
    accesses the whole sliced block at local offset 0.
    """
    from helion._compiler.compile_environment import CompileEnvironment

    if tensor is None:
        return False
    plan = CompileEnvironment.current().compact_worklist_plan
    if plan is None or block_id != plan.compact_axis.block_id:
        return False
    host = state.device_function.tensor_arg(tensor).host_str()
    return any(
        p.kind in ("compact_aligned_load", "compact_exact_store") and p.arg_name == host
        for p in plan.tensor_policies
    )


def _is_ordered_aligned_load(
    state: CodegenState, block_id: int, tensor: torch.Tensor | None
) -> bool:
    """True if *tensor* is a resident ordered reduction operand.

    Resident operands get a per-range ``pl.Element(C)`` window keyed on
    ``range_start`` (not tile_start), so the fori body reads at the LOCAL
    ordered-tile offset ``offset - range_start`` rather than the absolute offset.
    """
    from helion._compiler.compile_environment import CompileEnvironment

    if tensor is None:
        return False
    plan = CompileEnvironment.current().compact_worklist_plan
    if plan is None or plan.ordered_axis is None:
        return False
    if block_id != plan.ordered_axis.block_id:
        return False
    # Only active resident ordered operands read the resident window at the local
    # (offset - range_start) offset.  Consume the cached ResidentCacheDecision
    # the loop router also uses; inactive means the ordered loop streams and no
    # resident window exists.
    decision = CompileEnvironment.current().compact_worklist_resident_cache_decision
    if decision is None or not decision.active:
        return False
    host = state.device_function.tensor_arg(tensor).host_str()
    return host in decision.resident_operands


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
    # compact_aligned_load: the tensor is a per-tile sliced BlockSpec block (the
    # launcher slices it to one tile at tile_start via pl.Element, so Pallas
    # double-buffers it across work items).  The body therefore reads the whole
    # sliced block at local offset 0, not the absolute tile_start.
    if not tile_offset and _is_compact_aligned_load(state, block_id, tensor):
        return f"pl.ds(0, {block_size})"
    # Resident ordered operand: pl.Element(C) window at range_start, so read
    # at the LOCAL offset within the window (absolute offset - range_start).
    if not tile_offset and _is_ordered_aligned_load(state, block_id, tensor):
        from helion._compiler.compile_environment import CompileEnvironment
        from helion._compiler.pallas.compact_worklist import metadata_ref_for_field

        env = CompileEnvironment.current()
        plan = env.compact_worklist_plan
        assert plan is not None
        decision = env.compact_worklist_resident_cache_decision
        assert decision is not None
        assert decision.resident_key_fields == ("range_start",)
        begin_ref = f"{metadata_ref_for_field(plan, 'range_start')}[_wid]"
        return f"pl.ds(({offset}) - ({begin_ref}), {block_size})"
    if tensor is not None and tensor_dim is not None:
        from helion.language.memory_ops import _record_pad_info

        extra_pad = _loop_begin_extra_pad(block_id, state)
        _record_pad_info(state, tensor, tensor_dim, block_id, extra_pad)

        # Skip when tile_offset is set (e.g. offset + 64) — the shift
        # means the full expression may not be a multiple of block_size.
        if not tile_offset:
            alignment = _loop_offset_alignment(block_id, state)
            if alignment is not None:
                # Workaround for JAX <= 0.10.0 where AssumeMultipleOp
                # short-circuits divisibility analysis (fixed in
                # jax-ml/jax@33c38f50b): only apply when alignment meets
                # Mosaic's requirement, otherwise the hint could replace
                # a stronger proof Mosaic already has.
                from helion._compiler.backend import PallasBackend
                from helion._compiler.compile_environment import CompileEnvironment

                backend = CompileEnvironment.current().backend
                assert isinstance(backend, PallasBackend)
                dim_from_end = tensor.ndim - 1 - tensor_dim
                bitwidth = tensor.dtype.itemsize * 8
                required = backend._get_pallas_required_alignment(
                    dim_from_end, tensor.ndim, bitwidth
                )
                if alignment % required == 0:
                    # e.g. pl.ds(pl.multiple_of(offset_3, _BLOCK_SIZE_3), _BLOCK_SIZE_3)
                    offset = f"pl.multiple_of({offset}, {block_size})"

    return f"pl.ds({offset}, {block_size})"


def _loop_begin_extra_pad(block_id: int, state: CodegenState) -> int:
    """Return extra padding needed for a non-zero loop begin.

    A ``pl.ds(offset, block_size)`` read starting at a non-zero begin can
    overshoot the tensor boundary by up to ``begin % block_size`` elements
    beyond what ``(-N) % block_size`` accounts for.  Returns 0 when the
    loop starts at 0, ``begin % block_size`` for a provably constant begin,
    or ``block_size - 1`` for a data-dependent begin.
    """
    import sympy

    bs_value = state.device_function.resolved_block_size(block_id)
    if not isinstance(bs_value, int):
        return 0

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return 0

    info = loops[-1].block_id_to_info.get(block_id)
    if info is None or info.begin_expr is None:
        return 0

    begin = info.begin_expr
    if isinstance(begin, (int, sympy.Integer)):
        return int(begin) % bs_value

    return bs_value - 1


def _loop_offset_alignment(
    block_id: int,
    state: CodegenState,
) -> int | None:
    """Return the proven alignment of a loop's offset for *block_id*, or ``None``.

    A loop with step ``block_size`` produces offsets ``begin + i * block_size``,
    which are multiples of ``block_size`` iff ``begin`` is.  Returns
    ``block_size`` (int) when provable, ``None`` otherwise.
    """
    import sympy

    bs_value = state.device_function.resolved_block_size(block_id)
    if not isinstance(bs_value, int):
        return None

    # Check that the loop begins at a multiple of block_size.
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        info = loops[-1].block_id_to_info.get(block_id)
        if info is not None and info.begin_expr is not None:
            begin = info.begin_expr
            if not isinstance(begin, (int, sympy.Integer)):
                return None  # symbolic begin — can't prove alignment
            if int(begin) % bs_value != 0:
                return None

    return bs_value


def vmem_name(state: CodegenState, name: str) -> str:
    """Remap a tensor name to its VMEM ref name when inside emit_pipeline or fori_loop."""
    _loop, ref = _find_dma_scratch_loop(state, name)
    return ref
