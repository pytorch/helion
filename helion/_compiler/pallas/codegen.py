"""Pallas indexing codegen helpers."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from helion import exc
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.compile_environment import CompileEnvironment
from helion._utils import is_scalar_index

_FULL_LOAD_SELECT_MAX_BYTES = 256 << 10

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState


def _select_mask_dtype(dtype: torch.dtype) -> str | None:
    if dtype.is_floating_point:
        if dtype is torch.float64:
            return None
        if dtype.itemsize == 1:
            return "jnp.uint8"
        if dtype.itemsize == 2:
            return "jnp.uint16"
        if dtype.itemsize == 4:
            return "jnp.uint32"
        return None
    if dtype is torch.int8:
        return "jnp.int8"
    if dtype is torch.int16:
        return "jnp.int16"
    if dtype is torch.int32:
        return "jnp.int32"
    if dtype is torch.uint8:
        return "jnp.uint8"
    if dtype is torch.uint32:
        return "jnp.uint32"
    return None


def _mask_arithmetic_dtype(dtype: str) -> str:
    if dtype == "jnp.uint8":
        return "jnp.uint16"
    if dtype == "jnp.int8":
        return "jnp.int16"
    return dtype


def layout_tied_bf16_mask_expr(
    mask: ast.AST,
    layout_anchor: ast.AST,
) -> ast.AST:
    """Expand a numeric predicate through a value whose layout Mosaic already knows."""

    mask_value = expr_from_string(
        "lax.convert_element_type({mask}, jnp.bfloat16)",
        mask=mask,
    )
    return expr_from_string(
        "jnp.ones_like({layout_anchor}, dtype=jnp.bfloat16) * ({mask_value})",
        mask_value=mask_value,
        layout_anchor=layout_anchor,
    )


def numeric_mask_expr(mask: ast.AST, dtype: str = "jnp.int32") -> ast.AST:
    """Materialize a predicate as a numeric 0/1 tensor before shape relayouts."""

    return expr_from_string(
        f"lax.convert_element_type(({{mask}}) != 0, {dtype})",
        mask=mask,
    )


def numeric_mask_reshape_expr(
    mask: ast.AST, shape: str, dtype: str = "jnp.int32"
) -> ast.AST:
    """Cast a predicate to numeric, then add singleton dimensions by reshape."""

    return expr_from_string(
        f"jnp.reshape({{mask}}, {shape})",
        mask=numeric_mask_expr(mask, dtype),
    )


def numeric_where_expr(
    mask: ast.AST,
    true_value: ast.AST,
    false_value: ast.AST,
    output_dtype: torch.dtype,
    layout_anchor: ast.AST | None = None,
) -> ast.AST | None:
    """Select with a layout-tied numeric bit mask on TPU."""

    value_bit_dtype = _select_mask_dtype(output_dtype)
    if value_bit_dtype is None:
        return None
    mask_dtype = _mask_arithmetic_dtype(value_bit_dtype)
    value_dtype = CompileEnvironment.current().backend.dtype_str(output_dtype)
    if layout_anchor is None:
        layout_anchor = true_value
    mask_value = (
        "lax.convert_element_type("
        "jnp.ones_like({layout_anchor}, dtype=jnp.bfloat16) * "
        "lax.convert_element_type(({mask}) != 0, jnp.bfloat16), "
        f"{mask_dtype})"
    )
    mask_bits = (
        f"(jnp.zeros_like({{layout_anchor}}, dtype={mask_dtype}) - {mask_value})"
    )
    if output_dtype.is_floating_point:
        true_bits = (
            "lax.bitcast_convert_type("
            f"lax.convert_element_type({{true_value}}, {value_dtype}), "
            f"{value_bit_dtype})"
        )
        false_bits = (
            "lax.bitcast_convert_type("
            f"lax.convert_element_type({{false_value}}, {value_dtype}), "
            f"{value_bit_dtype})"
        )
        if mask_dtype != value_bit_dtype:
            true_bits = f"lax.convert_element_type({true_bits}, {mask_dtype})"
            false_bits = f"lax.convert_element_type({false_bits}, {mask_dtype})"
        selected_bits = (
            "jnp.bitwise_or("
            f"jnp.bitwise_and({mask_bits}, {true_bits}), "
            f"jnp.bitwise_and(jnp.bitwise_not({mask_bits}), {false_bits})"
            ")"
        )
        if mask_dtype != value_bit_dtype:
            selected_bits = (
                f"lax.convert_element_type({selected_bits}, {value_bit_dtype})"
            )
        return expr_from_string(
            f"lax.bitcast_convert_type({selected_bits}, {value_dtype})",
            mask=mask,
            layout_anchor=layout_anchor,
            true_value=true_value,
            false_value=false_value,
        )
    true_bits = (
        "lax.convert_element_type("
        f"lax.convert_element_type({{true_value}}, {value_dtype}), {mask_dtype})"
    )
    false_bits = (
        "lax.convert_element_type("
        f"lax.convert_element_type({{false_value}}, {value_dtype}), {mask_dtype})"
    )
    return expr_from_string(
        "lax.convert_element_type("
        "jnp.bitwise_or("
        f"jnp.bitwise_and({mask_bits}, {true_bits}), "
        f"jnp.bitwise_and(jnp.bitwise_not({mask_bits}), {false_bits})"
        f"), {value_dtype})",
        mask=mask,
        layout_anchor=layout_anchor,
        true_value=true_value,
        false_value=false_value,
    )


def numeric_mask_select_expr(
    mask: ast.AST,
    true_value: ast.AST,
    false_value: ast.AST,
    output_dtype: torch.dtype,
    layout_anchor: ast.AST | None = None,
) -> ast.AST:
    selected = numeric_where_expr(
        mask, true_value, false_value, output_dtype, layout_anchor
    )
    if selected is not None:
        return selected
    return expr_from_string(
        "jnp.where(({mask}) != 0, {true_value}, {false_value})",
        mask=mask,
        true_value=true_value,
        false_value=false_value,
    )


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
            result = emit_gather(state, pattern.plan, name)
            if (mask_expr := _indirect_gather_mask_expr(state, tensor)) is not None:
                dtype_str = _pallas_dtype_str(tensor)
                # Bind the gather once before using it as a layout anchor.
                result = state.codegen.lift(result, dce=True, prefix="gather")
                result = numeric_mask_select_expr(
                    expr_from_string(mask_expr),
                    result,
                    expr_from_string(f"jnp.array(0, dtype={dtype_str})"),
                    tensor.dtype,
                )
            return result

    parts, none_dims = index_parts(state, subscript, tensor)
    parts, post_load_takes = _widen_unaligned_tile_load_indices(
        state, subscript, tensor, parts
    )
    parts, post_load_selects, post_load_rank = _widen_small_aligned_scalar_load_indices(
        state, subscript, tensor, parts
    )
    idx_str = ", ".join(parts)
    mask_expr = _load_mask_expr(state, subscript, tensor)
    result = expr_from_string(f"{name}[{idx_str}]")
    result = _pad_clamped_load_expr(state, subscript, tensor, parts, result)
    promoted_scalar_dtype = _widened_scalar_load_promotion_dtype(
        state, tensor, post_load_selects, post_load_rank
    )
    result = _select_widened_scalar_load_indices(
        result, post_load_selects, post_load_rank, promoted_scalar_dtype
    )
    result = _select_unaligned_tile_load_indices(result, post_load_takes)
    if mask_expr is not None:
        result = expr_from_string(f"{{result}} * ({mask_expr})", result=result)
    for dim in none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


def _widen_unaligned_tile_load_indices(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
    index_parts: list[str],
) -> tuple[list[str], list[tuple[int, str, str, torch.dtype, int]]]:
    """Load small unaligned tile dims as full refs, then select lanes."""
    if _tensor_routed_to_dma_scratch(state, tensor):
        return index_parts, []

    from helion._compiler.device_function import PallasMemorySpace

    if (
        state.device_function.pallas_memory_space.get(id(tensor))
        == PallasMemorySpace.SMEM
    ):
        return index_parts, []

    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    adjusted = list(index_parts)
    takes: list[tuple[int, str]] = []
    widened_elements = _load_output_num_elements(state)
    value_dim = 0
    tensor_dim = 0
    index_part_idx = 0

    for idx, pattern in zip(
        subscript, _get_indexing_patterns(state, tensor), strict=True
    ):
        if idx is None:
            continue

        index_part = adjusted[index_part_idx]
        axis_dimensions = None
        if isinstance(
            pattern,
            (TilePattern, TileIndexWithOffsetPattern, TileBeginWithOffsetPattern),
        ):
            axis_dimensions = _unaligned_tile_axis_full_load_dimensions(
                state, tensor, tensor_dim, pattern, index_part
            )
        if axis_dimensions is not None:
            dim_size, block_size, local_owner = axis_dimensions
            if widened_elements is None or widened_elements % block_size != 0:
                raise exc.BackendUnsupported(
                    "pallas", "unaligned tile load with dynamic result shape"
                )
            candidate_elements = widened_elements // block_size * dim_size
            if candidate_elements * tensor.dtype.itemsize > _FULL_LOAD_SELECT_MAX_BYTES:
                raise exc.BackendUnsupported(
                    "pallas", "unaligned tile load exceeds full-load selection budget"
                )
            adjusted[index_part_idx] = ":"
            takes.append(
                (value_dim, _tile_lane_index_expr(state, pattern, local_owner))
            )
            widened_elements = candidate_elements

        if _index_part_produces_value_dim(adjusted[index_part_idx]):
            value_dim += 1
        tensor_dim += 1
        index_part_idx += 1

    dtype_str = _pallas_dtype_str(tensor)
    return adjusted, [
        (axis, index, dtype_str, tensor.dtype, value_dim) for axis, index in takes
    ]


def _unaligned_tile_axis_full_load_dimensions(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
    pattern: object,
    index_part: str,
) -> tuple[int, int, int | None] | None:
    """Return full-load metadata for an unsafe unaligned tile axis.

    Mosaic only needs extra proof for the last two VMEM dimensions.  When a
    dynamic ``pl.ds`` offset on such a dimension is not provably aligned, a
    small full-tensor load plus a register select is safer than asserting an
    alignment Helion cannot prove.
    """

    if not index_part.startswith("pl.ds("):
        return None
    if index_part.startswith("pl.ds(pl.multiple_of("):
        return None

    from helion._compiler.backend import PallasBackend
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    assert isinstance(
        pattern, (TilePattern, TileIndexWithOffsetPattern, TileBeginWithOffsetPattern)
    )
    env = CompileEnvironment.current()
    backend = env.backend
    assert isinstance(backend, PallasBackend)
    dim_from_end = tensor.ndim - tensor_dim - 1
    bitwidth = tensor.dtype.itemsize * 8
    required = backend._get_pallas_required_alignment(
        dim_from_end, tensor.ndim, bitwidth
    )
    if required <= 1:
        return None
    local_owner = _bounded_local_owner_block_id(
        state, pattern.block_id, tensor, tensor_dim
    )
    alignment = (
        state.device_function.resolved_block_size(pattern.block_id)
        if local_owner is not None
        else _loop_offset_alignment(pattern.block_id, state)
    )
    if (
        isinstance(pattern, (TileIndexWithOffsetPattern, TileBeginWithOffsetPattern))
        and pattern.offset != 0
    ):
        alignment = None
    if alignment is not None and alignment % required == 0:
        return None

    dim_size = (
        env.block_sizes[local_owner].from_config(state.config)
        if local_owner is not None
        else tensor.shape[tensor_dim]
    )
    block_size = env.block_sizes[pattern.block_id].from_config(state.config)
    if not isinstance(dim_size, int) or not isinstance(block_size, int):
        return None
    if block_size <= 0:
        raise exc.BackendUnsupported(
            "pallas", "unaligned tile load with invalid block size"
        )

    return dim_size, block_size, local_owner


def _load_output_num_elements(state: CodegenState) -> int | None:
    """Return the current load result element count for full-ref selection."""

    assert state.fx_node is not None
    output_val = state.fx_node.meta.get("val")
    if not isinstance(output_val, torch.Tensor):
        return None

    env = CompileEnvironment.current()
    elements = 1
    for size in output_val.size():
        if not isinstance(size, int):
            block_id = env.resolve_block_id(size)
            if block_id is None:
                return None
            resolved = env.block_sizes[block_id].from_config(state.config)
            if not isinstance(resolved, int):
                return None
            size = resolved
        elements *= size
    return elements


def _tile_lane_index_expr(
    state: CodegenState, pattern: object, local_owner: int | None = None
) -> str:
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TileIndexWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    assert isinstance(
        pattern, (TilePattern, TileIndexWithOffsetPattern, TileBeginWithOffsetPattern)
    )
    index = state.codegen.index_var(pattern.block_id)
    if (
        isinstance(pattern, (TileIndexWithOffsetPattern, TileBeginWithOffsetPattern))
        and pattern.offset != 0
    ):
        offset = state.device_function.literal_expr(pattern.offset)
        index = f"({index}) + ({offset})"
    if local_owner is not None:
        owner_offset = state.codegen.offset_var(local_owner)
        index = f"({index}) - ({owner_offset})"
    return index


def _select_unaligned_tile_load_indices(
    value: ast.AST, takes: list[tuple[int, str, str, torch.dtype, int]]
) -> ast.AST:
    """Select requested tiles from widened full-ref loads without bool relayout."""
    result = value
    for axis, index_expr, dtype_str, dtype, rank in takes:
        if axis == 0:
            mask_expr = (
                f"(({index_expr})[..., None] == "
                f"jnp.arange({{result}}.shape[0], dtype=({index_expr}).dtype))"
                f".astype({dtype_str})"
            )
            for _ in range(rank - 1):
                mask_expr = f"jnp.expand_dims({mask_expr}, axis=-1)"
            selected = numeric_mask_select_expr(
                expr_from_string(mask_expr, result=result),
                result,
                expr_from_string(f"jnp.array(0, dtype={dtype_str})"),
                dtype,
            )
            result = expr_from_string(
                (
                    "jnp.sum({selected}, axis=jnp.ndim("
                    f"{index_expr})).astype({dtype_str})"
                ),
                selected=selected,
            )
            continue
        mask_expr = (
            f"(({index_expr})[:, None] == "
            f"jnp.arange({{result}}.shape[{axis}], dtype=({index_expr}).dtype))"
            f".astype({dtype_str})"
        )
        for _ in range(axis):
            mask_expr = f"jnp.expand_dims({mask_expr}, axis=0)"
        for _ in range(rank - axis - 1):
            mask_expr = f"jnp.expand_dims({mask_expr}, axis=-1)"
        true_value = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={axis})",
            result=result,
        )
        selected = numeric_mask_select_expr(
            expr_from_string(mask_expr, result=result),
            true_value,
            expr_from_string(f"jnp.array(0, dtype={dtype_str})"),
            dtype,
            true_value,
        )
        result = expr_from_string(
            f"jnp.sum({{selected}}, axis={axis + 1}).astype({dtype_str})",
            selected=selected,
        )
    return result


def _pallas_dtype_str(tensor: torch.Tensor) -> str:
    from helion._compiler.compile_environment import CompileEnvironment

    return CompileEnvironment.current().backend.dtype_str(tensor.dtype)


def _indirect_gather_mask_expr(state: CodegenState, tensor: torch.Tensor) -> str | None:
    """Mask tensor-indexed gather outputs in their result layout."""
    from helion._compiler.compile_environment import CompileEnvironment

    assert state.fx_node is not None
    output_val = state.fx_node.meta.get("val")
    if not isinstance(output_val, torch.Tensor):
        return None

    env = CompileEnvironment.current()
    output_sizes = [*output_val.size()]
    dtype_str = _pallas_dtype_str(tensor)
    mask_exprs: list[str] = []
    for dim, size in enumerate(output_sizes):
        block_id = env.resolve_block_id(size)
        if block_id is None or (mask_var := state.codegen.mask_var(block_id)) is None:
            continue
        if env.is_jagged_tile(block_id):
            mask_shape = env.jagged_tile_mask_shapes[block_id]
            expand = state.tile_strategy.jagged_tile_expand_str(
                mask_shape, output_sizes
            )
        else:
            expand = state.tile_strategy.expand_str(output_sizes, dim)
        expr = f"({mask_var}.astype({dtype_str}){expand})"
        if expr not in mask_exprs:
            mask_exprs.append(expr)

    if not mask_exprs:
        return None
    return "*".join(mask_exprs)


def _index_part_produces_value_dim(index_part: str) -> bool:
    return index_part == ":" or index_part.startswith(("pl.ds(", "slice("))


def _widen_small_aligned_scalar_load_indices(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
    index_parts: list[str],
) -> tuple[list[str], list[tuple[int, str, int]], int]:
    """Load small scalar-indexed VMEM dims as full dims, then select in registers."""
    value_rank = sum(
        _index_part_produces_value_dim(index_part) for index_part in index_parts
    )
    if _tensor_routed_to_dma_scratch(state, tensor):
        return index_parts, [], value_rank

    from helion._compiler.device_function import PallasMemorySpace

    if (
        state.device_function.pallas_memory_space.get(id(tensor))
        == PallasMemorySpace.SMEM
    ):
        return index_parts, [], value_rank

    from helion._compiler.backend import PallasBackend
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, PallasBackend)
    patterns = _get_indexing_patterns(state, tensor)
    adjusted = list(index_parts)
    widened_selects: list[tuple[int, str, int]] = []
    value_dim = 0
    tensor_dim = 0
    index_part_idx = 0

    for idx, pattern in zip(subscript, patterns, strict=True):
        if idx is None:
            continue

        index_part = adjusted[index_part_idx]
        dim_size = tensor.shape[tensor_dim]
        is_scalar = not _index_part_produces_value_dim(index_part)
        can_widen = False
        if (
            is_scalar
            and isinstance(dim_size, int)
            and dim_size > 0
            and isinstance(pattern, (ArbitraryIndexPattern, TileBeginWithOffsetPattern))
        ):
            dim_from_end = tensor.ndim - tensor_dim - 1
            bitwidth = tensor.dtype.itemsize * 8
            required_alignment = backend._get_pallas_required_alignment(
                dim_from_end, tensor.ndim, bitwidth
            )
            can_widen = required_alignment > 1 and dim_size <= required_alignment
            if can_widen:
                index_part = _normalize_widened_scalar_index(index_part, dim_size)
                can_widen = index_part is not None

        if can_widen:
            adjusted[index_part_idx] = ":"
            assert index_part is not None
            widened_selects.append((value_dim, index_part, dim_size))
            value_dim += 1
        elif index_part is not None and _index_part_produces_value_dim(index_part):
            value_dim += 1

        tensor_dim += 1
        index_part_idx += 1

    return adjusted, widened_selects, value_dim


def _normalize_widened_scalar_index(index_part: str, dim_size: int) -> str | None:
    try:
        index = int(index_part)
    except ValueError:
        return (
            f"jnp.where(({index_part}) < 0, ({index_part}) + {dim_size}, {index_part})"
        )
    if index < 0:
        index += dim_size
    if 0 <= index < dim_size:
        return str(index)
    return None


def _select_widened_scalar_load_indices(
    value: ast.AST,
    widened_selects: list[tuple[int, str, int]],
    rank: int,
    scalar_result_dtype: str | None = None,
) -> ast.AST:
    result = value
    for axis, index_expr, dim_size in sorted(widened_selects, reverse=True):
        if rank == 1 and scalar_result_dtype is not None:
            result = expr_from_string(
                f"{{result}}.astype({scalar_result_dtype})",
                result=result,
            )
        lane_index = [":"] * rank
        lane_index[axis] = "0"
        selected = expr_from_string(
            f"{{result}}[{', '.join(lane_index)}]",
            result=result,
        )
        for lane in range(1, dim_size):
            lane_index[axis] = str(lane)
            lane_value = expr_from_string(
                f"{{result}}[{', '.join(lane_index)}]",
                result=result,
            )
            selected = expr_from_string(
                f"jnp.where(({index_expr}) == {lane}, {{lane_value}}, {{selected}})",
                lane_value=lane_value,
                selected=selected,
            )
        result = selected
        rank -= 1
    return result


def _widened_scalar_load_promotion_dtype(
    state: CodegenState,
    tensor: torch.Tensor,
    widened_selects: list[tuple[int, str, int]],
    rank: int,
) -> str | None:
    """Return fp32 when a widened sub-32-bit scalar load feeds only fp32 casts."""
    if tensor.dtype.itemsize >= 4 or rank != len(widened_selects):
        return None
    if not _load_feeds_only_immediate_fp32_converts(state):
        return None
    from helion._compiler.compile_environment import CompileEnvironment

    return CompileEnvironment.current().backend.dtype_str(torch.float32)


def _load_feeds_only_immediate_fp32_converts(state: CodegenState) -> bool:
    node = state.fx_node
    if node is None:
        return False
    users = list(node.users)
    if not users:
        return False
    convert = torch.ops.prims.convert_element_type.default
    for user in users:
        if (
            user.op != "call_function"
            or user.target is not convert
            or len(user.args) < 2
            or user.args[0] is not node
            or user.args[1] is not torch.float32
        ):
            return False
    return True


def _pad_clamped_load_expr(
    state: CodegenState,
    subscript: list[object],
    tensor: torch.Tensor,
    index_parts: list[str],
    value: ast.AST,
) -> ast.AST:
    """Pad BlockSpec-clamped Pallas loads back to their logical tile shape."""
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.pallas.plan_tiling import TilePattern

    if _tensor_routed_to_dma_scratch(state, tensor):
        return value

    env = CompileEnvironment.current()
    indexing_patterns = _get_indexing_patterns(state, tensor)
    pads: list[tuple[int, int]] = []
    needs_pad = False
    tensor_dim = 0
    index_part_idx = 0

    for idx, pattern in zip(subscript, indexing_patterns, strict=True):
        if idx is None:
            continue

        index_part = index_parts[index_part_idx]
        index_part_idx += 1

        if isinstance(pattern, TilePattern) and index_part == ":":
            block_size = env.block_sizes[pattern.block_id].from_config(state.config)
            dim_size = tensor.shape[tensor_dim]
            if (
                isinstance(block_size, int)
                and isinstance(dim_size, int)
                and 1 < dim_size < block_size
            ):
                pads.append((0, block_size - dim_size))
                needs_pad = True
            else:
                pads.append((0, 0))
        else:
            produces_value_dim = index_part == ":" or index_part.startswith("pl.ds(")
            produces_value_dim = produces_value_dim or not isinstance(
                pattern, (ArbitraryIndexPattern, TileBeginWithOffsetPattern)
            )
            if produces_value_dim:
                pads.append((0, 0))

        tensor_dim += 1

    if not needs_pad:
        return value

    return expr_from_string(f"jnp.pad({{value}}, {pads!r})", value=value)


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
) -> Iterator[tuple[EmitPipelineLoopState | ForiLoopState, str]]:
    """Yield ``(loop, scratch_ref_name)`` for every active loop that DMA-routes
    ``name`` through a VMEM scratch."""
    from helion._compiler.tile_strategy import EmitPipelineLoopState
    from helion._compiler.tile_strategy import ForiLoopState

    for loops in state.codegen.active_device_loops.values():
        for loop in loops:
            if isinstance(loop, (EmitPipelineLoopState, ForiLoopState)):
                mapping = loop._tensor_to_dma_scratch
                if name in mapping:
                    yield loop, mapping[name]


def _find_dma_scratch_loop(
    state: CodegenState, name: str
) -> tuple[EmitPipelineLoopState | ForiLoopState | None, str]:
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


def _tensor_routed_to_dma_scratch(state: CodegenState, tensor: torch.Tensor) -> bool:
    name = state.codegen.device_function.tensor_arg(tensor).name
    loop, _ref = _find_dma_scratch_loop(state, name)
    return loop is not None


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

    Pipeline/DMA scratch stores are exempt: their destination is a block-sized
    VMEM scratch (not a clamped ref), so the value stays block-shaped and the
    writeback path clamps the HBM extent instead.
    """
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.pallas.plan_tiling import TilePattern

    assert state.fx_node is not None
    patterns = state.fx_node.meta.get("indexing_patterns")
    if patterns is None:
        return value

    if _tensor_routed_to_dma_scratch(state, tensor):
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
    dma_block_ids: set[int] = set()
    for loop, _ref in _iter_dma_scratch_loops(state, tensor_name):
        in_pipeline = True
        dma_block_ids.update(loop._tensor_to_dma_block_ids[tensor_name])

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
            pattern, idx, state, tensor, i, tensor_dim, in_pipeline, dma_block_ids
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


def _pipeline_scalar_dim_is_local(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> bool:
    """Whether a pipelined ref made this scalar tensor dimension local.

    Pipeline BlockSpecs / DMA copies can shrink an arbitrary scalar dimension
    to extent 1, in which case the body must index that local ref at 0.  If the
    same dimension is also accessed through a tile or full slice in this loop,
    the merged DMA ref keeps a non-scalar extent and the original scalar index
    must be preserved.
    """

    if state.fx_node is None:
        return False

    access_count = 0
    dynamic_scalar_count = 0
    static_index: int | None = None
    for node in state.fx_node.graph.nodes:
        if not node.meta.get("indexing_patterns"):
            continue
        if not _node_indexes_tensor(node, tensor):
            continue
        idx = _node_tensor_dim_index(node, tensor_dim)
        if idx is _MISSING_INDEX:
            return False
        if not is_scalar_index(idx):
            return False
        access_count += 1
        if isinstance(idx, int):
            if static_index is None:
                static_index = idx
            elif static_index != idx:
                return False
        else:
            dynamic_scalar_count += 1

    if access_count == 0:
        return False
    if dynamic_scalar_count:
        return access_count == 1
    return True


_MISSING_INDEX = object()


def _node_indexes_tensor(node: torch.fx.Node, tensor: torch.Tensor) -> bool:
    if len(node.args) < 2:
        return False
    tensor_node = node.args[0]
    if not isinstance(tensor_node, torch.fx.Node):
        return False
    tensor_val = tensor_node.meta.get("val")
    return isinstance(tensor_val, torch.Tensor) and id(tensor_val) == id(tensor)


def _node_tensor_dim_index(
    node: torch.fx.Node,
    tensor_dim: int,
) -> object:
    subscript = node.args[1]
    if not isinstance(subscript, (list, tuple)):
        return _MISSING_INDEX
    current_dim = 0
    for idx in subscript:
        if isinstance(idx, torch.fx.Node):
            idx = idx.meta.get("val", idx)
        if idx is None:
            continue
        if current_dim == tensor_dim:
            return idx
        current_dim += 1
    return _MISSING_INDEX


def _arbitrary_index_pattern_code(
    idx: object,
    state: CodegenState,
    tensor: torch.Tensor,
    subscript_index: int,
    tensor_dim: int,
    in_pipeline: bool,
) -> str:
    if (
        in_pipeline
        and is_scalar_index(idx)
        and _pipeline_scalar_dim_is_local(state, tensor, tensor_dim)
    ):
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
    dma_block_ids: set[int],
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
            pattern, idx, state, tensor, tensor_dim, in_pipeline, dma_block_ids
        )

    if isinstance(pattern, TileIndexWithOffsetPattern):
        return _tile_index_with_offset_pattern_code(
            pattern, state, tensor, tensor_dim, in_pipeline, dma_block_ids
        )

    if isinstance(pattern, TileBeginWithOffsetPattern):
        return _tile_begin_with_offset_pattern_code(
            pattern, state, subscript_index, tensor_dim, in_pipeline, dma_block_ids
        )

    if isinstance(pattern, ArbitrarySlicePattern):
        return _slice_code(idx, pattern, state, tensor, tensor_dim)

    if isinstance(pattern, ArbitraryIndexPattern):
        return _arbitrary_index_pattern_code(
            idx, state, tensor, subscript_index, tensor_dim, in_pipeline
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
    dma_block_ids: set[int],
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
    if in_pipeline and block_id in dma_block_ids:
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
    dma_block_ids: set[int],
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
    dma_block_ids: set[int],
) -> str:
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern
    from helion._compiler.tile_strategy import DeviceLoopState

    assert isinstance(pattern, TileBeginWithOffsetPattern)

    block_id = pattern.block_id
    offset_str = state.device_function.literal_expr(pattern.offset)

    if in_pipeline and block_id in dma_block_ids:
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
    slice_idx = pattern.slice

    if slice_idx.step not in (None, 1):
        raise AssertionError(
            f"Only unit-step slices are supported in Pallas, got {slice_idx}"
        )
    full_slice = slice_idx.start is None and slice_idx.stop is None

    env = CompileEnvironment.current()
    block_id = env.resolve_block_id(tensor.shape[tensor_dim])
    if full_slice and block_id is not None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops and any(isinstance(loop, DeviceLoopState) for loop in loops):
            return _ds_expr(state, block_id, tensor=tensor, tensor_dim=tensor_dim)

    if full_slice:
        return ":"

    def bound_expr(value: object) -> str:
        if value is None:
            return "None"
        if isinstance(value, int) and value >= 0:
            return repr(value)
        if isinstance(value, torch.SymInt):
            expr = env.specialize_expr(env.shape_env.replace(value._sympy_()))
            if not expr.free_symbols and cast("Any", expr).is_integer is True:
                concrete = int(expr)
                if concrete >= 0:
                    return repr(concrete)
        raise AssertionError(
            f"Only static slice bounds are supported in Pallas, got {slice_idx}"
        )

    return f"slice({bound_expr(slice_idx.start)}, {bound_expr(slice_idx.stop)})"


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
    local_owner_block_id = (
        _bounded_local_owner_block_id(state, block_id, tensor, tensor_dim)
        if tensor is not None and tensor_dim is not None
        else None
    )
    if local_owner_block_id is not None:
        owner_offset = state.codegen.offset_var(local_owner_block_id)
        offset = f"({offset}) - ({owner_offset})"
    if tensor is not None and tensor_dim is not None:
        from helion.language.memory_ops import _record_pad_info

        pad_block_id = (
            local_owner_block_id if local_owner_block_id is not None else block_id
        )
        extra_pad = _loop_begin_extra_pad(pad_block_id, state)
        _record_pad_info(state, tensor, tensor_dim, pad_block_id, extra_pad)

        # Skip when tile_offset is set (e.g. offset + 64) — the shift
        # means the full expression may not be a multiple of block_size.
        if not tile_offset:
            alignment = (
                state.device_function.resolved_block_size(block_id)
                if local_owner_block_id is not None
                else _loop_offset_alignment(block_id, state)
            )
            # Workaround for JAX <= 0.10.0 where AssumeMultipleOp
            # short-circuits divisibility analysis (fixed in
            # jax-ml/jax@33c38f50b): only apply when the hint meets Mosaic's
            # requirement, otherwise it could replace a stronger proof Mosaic
            # already has.
            from helion._compiler.backend import PallasBackend
            from helion._compiler.compile_environment import CompileEnvironment

            backend = CompileEnvironment.current().backend
            assert isinstance(backend, PallasBackend)
            dim_from_end = tensor.ndim - 1 - tensor_dim
            bitwidth = tensor.dtype.itemsize * 8
            required = backend._get_pallas_required_alignment(
                dim_from_end, tensor.ndim, bitwidth
            )
            if alignment is not None and alignment % required == 0:
                hint = required
                bs_value = state.device_function.resolved_block_size(block_id)
                if isinstance(bs_value, int) and alignment % bs_value == 0:
                    hint = block_size
                # e.g. pl.ds(pl.multiple_of(offset_3, _BLOCK_SIZE_3), _BLOCK_SIZE_3)
                offset = f"pl.multiple_of({offset}, {hint})"
            elif _is_zero_begin_single_tile_dim(block_id, state, tensor, tensor_dim):
                offset = f"pl.multiple_of({offset}, {required})"

    return f"pl.ds({offset}, {block_size})"


def _bounded_local_owner_block_id(
    state: CodegenState,
    block_id: int,
    tensor: torch.Tensor | None,
    tensor_dim: int | None,
) -> int | None:
    """Return the outer grid block that owns a bounded inner tile's BlockSpec."""
    if tensor is None or tensor_dim is None:
        return None

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_function import PallasMemorySpace
    from helion._compiler.tile_strategy import DeviceGridState

    if (
        state.device_function.pallas_memory_space.get(id(tensor))
        == PallasMemorySpace.HBM
    ):
        return None

    dim_tilings = state.device_function.pallas_tensor_dim_tilings.get(id(tensor))
    if dim_tilings is None or tensor_dim >= len(dim_tilings):
        return None
    dim_tiling = dim_tilings[tensor_dim]
    if not dim_tiling.can_tile or dim_tiling.block_ids != [block_id]:
        return None

    env = CompileEnvironment.current()
    seen: set[int] = set()
    current = block_id
    while current not in seen:
        seen.add(current)
        try:
            spec = env.config_spec.block_sizes.block_id_lookup(current)
        except KeyError:
            return None
        parent = spec.owner_relative_bounded_by_block_id
        if parent is None:
            return None
        loops = state.codegen.active_device_loops.get(parent)
        if loops and any(isinstance(loop, DeviceGridState) for loop in loops):
            return parent
        current = parent
    return None


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
    if info is None:
        return 0

    begin = info.begin_expr
    if begin is None:
        alignment = info.offset_alignment
        if isinstance(alignment, int) and alignment > 0:
            return (bs_value - math.gcd(bs_value, alignment)) % bs_value
        return bs_value - 1
    if isinstance(begin, (int, sympy.Integer)):
        return int(begin) % bs_value

    return bs_value - 1


def _loop_offset_alignment(
    block_id: int,
    state: CodegenState,
) -> int | None:
    """Return the proven alignment of a loop's offset for *block_id*, or ``None``.

    A loop with step ``block_size`` produces offsets ``begin + i * block_size``.
    Returns a proven divisor of that offset expression when available.
    """
    import sympy

    bs_value = state.device_function.resolved_block_size(block_id)
    if not isinstance(bs_value, int):
        return None

    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        info = loops[-1].block_id_to_info.get(block_id)
        if info is None:
            return None
        offset_alignment = info.offset_alignment
        if isinstance(offset_alignment, int) and offset_alignment > 0:
            return offset_alignment
        begin = info.begin_expr
        if begin is None:
            return None
        if not isinstance(begin, (int, sympy.Integer)):
            return None  # symbolic begin — can't prove alignment
        return math.gcd(abs(int(begin)), bs_value)

    return bs_value


def _is_zero_begin_single_tile_dim(
    block_id: int,
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_dim: int,
) -> bool:
    """True when a zero-begin loop has only one in-bounds concrete tensor tile."""
    import sympy

    from helion._compiler.compile_environment import CompileEnvironment

    bs_value = state.device_function.resolved_block_size(block_id)
    if not isinstance(bs_value, int):
        return False

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return False

    info = loops[-1].block_id_to_info.get(block_id)
    if info is None or info.begin_expr not in (0, sympy.Integer(0)):
        return False

    env = CompileEnvironment.current()
    dim_size = tensor.shape[tensor_dim]
    if isinstance(dim_size, int):
        return env.settings.static_shapes and dim_size <= bs_value
    if not isinstance(dim_size, torch.SymInt):
        return False

    dim_expr = env.shape_env.replace(dim_size._sympy_())
    if not dim_expr.free_symbols <= env.specialized_vars:
        return False
    dim_expr = env.specialize_expr(dim_expr)
    if dim_expr.free_symbols or getattr(dim_expr, "is_integer", None) is not True:
        return False
    dim_value = int(dim_expr)
    return dim_value <= bs_value


def vmem_name(state: CodegenState, name: str) -> str:
    """Remap a tensor name to its VMEM ref name when inside emit_pipeline or fori_loop."""
    _loop, ref = _find_dma_scratch_loop(state, name)
    return ref
