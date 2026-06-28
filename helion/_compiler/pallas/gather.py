"""Pallas indirect-gather/scatter lowering.

No native gather in Pallas. Floating ``table[idx]`` emits
``one_hot(idx, V) @ table``; int32 ``table[idx]`` emits a boolean one-hot
select and reduction. Scatter store projects source lanes to target rows with
one-hot matrices, resolves duplicate lanes within one program, and merges
updates with the existing target block.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..inductor_lowering import CodegenState
    from .plan_tiling import IndexingPattern


# Fail early on oversized tables instead of a generic Mosaic OOM.
# Replace with real VMEM budget accounting once available.
_GATHER_VMEM_THRESHOLD_BYTES: int = 16 << 20  # 16 MiB


@dataclass(frozen=True)
class GatherPlan:
    indirect_pos: int
    none_dims: tuple[int, ...]
    jnp_dtype: str
    table_ndim: int
    index_ndim: int
    emit_select: bool
    use_highest_precision: bool


@dataclass(frozen=True)
class ScatterPlan:
    indirect_pos: int
    jnp_dtype: str
    target_ndim: int
    index_ndim: int


def build_gather_plan(
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_positions: list[int],
    patterns: list[IndexingPattern],
    config: Config,
) -> GatherPlan:
    """Validate the gather site and return its plan. Runs during plan_tiling."""
    from ..compile_environment import CompileEnvironment
    from .plan_tiling import resident_block_elements

    if len(indirect_positions) > 1:
        raise NotImplementedError(
            "Pallas gather: multiple indirect dims are not supported"
        )
    indirect_pos = indirect_positions[0]
    emit_select = not tensor.dtype.is_floating_point
    if emit_select and indirect_pos != 0:
        raise NotImplementedError(
            "Pallas gather: integer table gather on non-zero dim is not yet supported"
        )
    if emit_select and tensor.dtype != torch.int32:
        raise NotImplementedError(
            f"Pallas gather: integer table gather only supports torch.int32, "
            f"got {tensor.dtype}"
        )

    elements = resident_block_elements(tensor, patterns, config)
    if elements is not None:
        table_bytes = elements * tensor.dtype.itemsize
        if table_bytes > _GATHER_VMEM_THRESHOLD_BYTES:
            raise NotImplementedError(
                f"Pallas gather: resident block is {table_bytes} bytes, exceeds "
                f"the {_GATHER_VMEM_THRESHOLD_BYTES} byte VMEM threshold. The "
                "current codegen requires the full gather axis in VMEM; reduce "
                "V, tile the broadcast dims, or use a half-precision dtype."
            )

    # MXU truncates fp32 to bf16 without HIGHEST. For bf16/fp16 the truncation is a no-op.
    use_highest = tensor.dtype not in (torch.bfloat16, torch.float16)

    none_dims = tuple(i for i, idx in enumerate(subscript) if idx is None)
    jnp_dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    idx_element = subscript[indirect_pos]
    index_ndim = idx_element.meta["val"].ndim  # type: ignore[union-attr]

    return GatherPlan(
        indirect_pos=indirect_pos,
        none_dims=none_dims,
        jnp_dtype=jnp_dtype,
        table_ndim=tensor.ndim,
        index_ndim=index_ndim,
        emit_select=emit_select,
        use_highest_precision=use_highest,
    )


def build_scatter_plan(
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_positions: list[int],
) -> ScatterPlan:
    """Validate a Pallas scatter site and return its one-hot plan."""
    from ..compile_environment import CompileEnvironment

    if not tensor.dtype.is_floating_point:
        raise NotImplementedError(
            f"Pallas scatter: only floating-point output dtypes are supported, "
            f"got {tensor.dtype}"
        )
    if len(indirect_positions) > 1:
        raise NotImplementedError(
            "Pallas scatter: multiple indirect dims are not supported"
        )
    indirect_pos = indirect_positions[0]
    if indirect_pos != 0:
        raise NotImplementedError("Pallas scatter: only indirect dim 0 is supported")
    idx_element = subscript[indirect_pos]
    index_ndim = idx_element.meta["val"].ndim  # type: ignore[union-attr]
    if index_ndim != 1:
        raise NotImplementedError(
            "Pallas scatter: only rank-1 tensor indices are supported"
        )
    jnp_dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return ScatterPlan(
        indirect_pos=indirect_pos,
        jnp_dtype=jnp_dtype,
        target_ndim=tensor.ndim,
        index_ndim=index_ndim,
    )


def emit_gather(
    state: CodegenState,
    plan: GatherPlan,
    name: str,
) -> ast.AST:
    """Emit ``one_hot(idx, V) @ table``.

    MXU accumulates in fp32 via ``preferred_element_type``. fp32 tables need
    HIGHEST and fp32 one_hot; bf16/fp16 stay in the table dtype (MXU truncation
    is a no-op and we avoid a VMEM upcast).

    Contracting dim is ``jnp.ndim(idx)``: one_hot adds one trailing axis.
    """
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[plan.indirect_pos]
    assert isinstance(ast_idx, ast.AST)
    idx_name = state.codegen.lift(ast_idx, dce=False, prefix="index").id
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    from . import codegen as pallas_codegen

    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    parts, widened_selects, table_rank = (
        pallas_codegen._widen_small_aligned_scalar_load_indices(
            state, list(subscript), tensor, parts
        )
    )
    table_indirect_axis = _table_indirect_axis(subscript, parts, plan.indirect_pos)
    gather_rank = table_rank + plan.index_ndim - 1
    widened_selects = _remap_widened_selects_after_gather(
        widened_selects, table_indirect_axis, plan.index_ndim
    )
    base_index = ", ".join(parts)
    table_expr = f"{name}[{base_index}]"

    if plan.emit_select:
        mask_expr = (
            f"jax.nn.one_hot({idx_name}[...], {name}.shape[0], dtype={plan.jnp_dtype})"
        )
        for _ in range(table_rank - 1):
            mask_expr = f"jnp.expand_dims({mask_expr}, axis=-1)"
        result = expr_from_string(
            f"jnp.sum({table_expr} * {mask_expr}, "
            f"axis=jnp.ndim({idx_name}[...])"
            f").astype({plan.jnp_dtype})"
        )
        if widened_selects:
            result = state.codegen.lift(result, dce=True, prefix="gather")
        result = pallas_codegen._select_widened_scalar_load_indices(
            result, widened_selects, gather_rank
        )
        for dim in plan.none_dims:
            result = expr_from_string(
                f"jnp.expand_dims({{result}}, axis={dim})", result=result
            )
        return result

    if plan.use_highest_precision:
        oh_dtype = "jnp.float32"
        table_dot_expr = f"{table_expr}.astype(jnp.float32)"
        precision_arg = "precision=jax.lax.Precision.HIGHEST, "
    else:
        oh_dtype = plan.jnp_dtype
        table_dot_expr = table_expr
        precision_arg = ""

    result = expr_from_string(
        "jax.lax.dot_general("
        f"jax.nn.one_hot({idx_name}[...], "
        f"{name}.shape[{plan.indirect_pos}], dtype={oh_dtype}), "
        f"{table_dot_expr}, "
        f"(((jnp.ndim({idx_name}[...]),), ({table_indirect_axis},)), ((), ())), "
        "preferred_element_type=jnp.float32, "
        f"{precision_arg}"
        f").astype({plan.jnp_dtype})"
    )
    if table_indirect_axis > 0:
        n = plan.index_ndim
        src = tuple(range(n, n + table_indirect_axis))
        dst = tuple(range(table_indirect_axis))
        result = expr_from_string(
            f"jnp.moveaxis({{result}}, {src}, {dst})", result=result
        )
    if widened_selects:
        result = state.codegen.lift(result, dce=True, prefix="gather")
    result = pallas_codegen._select_widened_scalar_load_indices(
        result, widened_selects, gather_rank
    )
    for dim in plan.none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


def _table_indirect_axis(
    subscript: list[object] | tuple[object, ...],
    parts: list[str],
    indirect_pos: int,
) -> int:
    from . import codegen as pallas_codegen

    axis = 0
    part_idx = 0
    for pos, idx in enumerate(subscript):
        if idx is None:
            continue
        part = parts[part_idx]
        if pos == indirect_pos:
            return axis
        if pallas_codegen._index_part_produces_value_dim(part):
            axis += 1
        part_idx += 1
    raise AssertionError(f"indirect position {indirect_pos} not found in subscript")


def _remap_widened_selects_after_gather(
    widened_selects: list[tuple[int, str, int]],
    table_indirect_axis: int,
    index_ndim: int,
) -> list[tuple[int, str, int]]:
    remapped: list[tuple[int, str, int]] = []
    for axis, index_expr, dim_size in widened_selects:
        assert axis != table_indirect_axis
        if axis < table_indirect_axis:
            remapped_axis = axis
        else:
            remapped_axis = axis + index_ndim - 1
        remapped.append((remapped_axis, index_expr, dim_size))
    return remapped


def _scatter_one_hot_name(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
) -> str:
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[plan.indirect_pos]
    assert isinstance(ast_idx, ast.AST)
    idx_name = state.codegen.lift(ast_idx, dce=False, prefix="index").id
    # TODO(tcombes): investigate making the metadata into dtype,
    # currently hitting Mosaic issues with bf16 mask.
    return (
        f"jax.nn.one_hot({idx_name}[...], {name}.shape[{plan.indirect_pos}], "
        "dtype=jnp.float32)"
    )


def _scatter_source_mask_expr(
    state: CodegenState,
    plan: ScatterPlan,
) -> str | None:
    """Return a float mask for valid tensor-index source lanes."""
    from ..compile_environment import CompileEnvironment

    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    index = subscript[plan.indirect_pos]
    if not isinstance(index, torch.Tensor):
        return None

    env = CompileEnvironment.current()
    index_shape = [*index.size()]
    mask_exprs: list[str] = []
    for dim, size in enumerate(index_shape):
        block_id = env.resolve_block_id(size)
        if block_id is None or (mask_var := state.codegen.mask_var(block_id)) is None:
            continue
        if env.is_jagged_tile(block_id):
            mask_shape = env.jagged_tile_mask_shapes[block_id]
            expand = state.tile_strategy.jagged_tile_expand_str(mask_shape, index_shape)
        else:
            expand = state.tile_strategy.expand_str(index_shape, dim)
        expr = f"({mask_var}.astype(jnp.float32){expand})"
        if expr not in mask_exprs:
            mask_exprs.append(expr)

    if not mask_exprs:
        return None
    return "*".join(mask_exprs)


def _scatter_masked_one_hot(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
) -> str:
    oh = _scatter_one_hot_name(state, plan, name)
    source_mask = _scatter_source_mask_expr(state, plan)
    if source_mask is None:
        return oh
    return f"({oh}) * (({source_mask})[..., None])"


def emit_scatter_store(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
    base_index: str,
    value: ast.AST,
) -> ast.AST:
    """Emit one Pallas program's tensor-indexed store block.

    ``base_index`` is the target block with the indirect dimension replaced by
    ``:``. For ``target[idx, cols] = value`` this is ``target[:, cols]``.

    The lowering builds:
      - ``oh``: one-hot source-lane-to-target-row map, shape ``[M, V]``.
      - ``same_output_row``/``is_lane_j_after_i``/``is_last_writer``: local duplicate
        detection. If two lanes in this program target the same row, only the
        last source lane is kept.
      - ``row_to_lane``: target-row-to-source-lane map, shape ``[V, M]``.
      - ``updates``: projected values for every row in ``base_index``.
      - ``mask``: rows touched by this program.

    The final expression is ``where(mask, updates, old_target_block)`` so
    untouched rows keep their previous value. Duplicate handling is local to
    this Pallas program; duplicate writes from different programs have the same
    unspecified winner semantics as regular parallel stores in other backends.
    """
    oh = _scatter_masked_one_hot(state, plan, name)
    m = f"jnp.shape({oh})[0]"
    eye = f"jnp.eye({m}, dtype=jnp.float32)"
    is_lane_j_after_i = f"jnp.triu(jnp.ones(({m}, {m}), dtype=jnp.float32), k=1)"
    same_output_row = (
        f"jax.lax.dot_general({oh}, jnp.swapaxes({oh}, 0, 1), (((1,), (0,)), ((), ())))"
    )
    is_last_writer = f"(jnp.sum(({same_output_row}) * ({is_lane_j_after_i}), axis=1) == 0).astype(jnp.float32)"
    row_to_lane = (
        f"jax.lax.dot_general(jnp.swapaxes({oh}, 0, 1), "
        f"({eye}) * jnp.expand_dims({is_last_writer}, axis=0), "
        "(((1,), (0,)), ((), ())))"
    )
    updates = expr_from_string(
        "jax.lax.dot_general("
        f"{row_to_lane}, "
        "{value}.astype(jnp.float32), "
        "(((1,), (0,)), "
        "((), ())), "
        "preferred_element_type=jnp.float32, "
        "precision=jax.lax.Precision.HIGHEST"
        f").astype({plan.jnp_dtype})",
        value=value,
    )
    mask_expr = (
        "jax.lax.dot_general("
        f"{row_to_lane}, "
        "jnp.ones_like({value}, dtype=jnp.float32), "
        "(((1,), (0,)), ((), ()))"
        ") > 0"
    )
    return expr_from_string(
        f"jnp.where({mask_expr}, {{updates}}, {name}[{base_index}])",
        updates=updates,
        value=value,
    )


def emit_scatter_add(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
    base_index: str,
    value: ast.AST,
) -> ast.AST:
    """Emit one Pallas program's tensor-indexed add update block.

    ``base_index`` is the target block with the indirect dimension replaced by
    ``:``. The lowering projects all source lanes to target rows with
    ``one_hot(idx).T @ value``. Duplicate source indices within this program are
    summed by the dot, unlike scatter-store's last-writer-wins projection.
    """
    oh = _scatter_masked_one_hot(state, plan, name)
    updates = expr_from_string(
        "jax.lax.dot_general("
        f"jnp.swapaxes({oh}, 0, 1), "
        "{value}.astype(jnp.float32), "
        "(((1,), (0,)), "
        "((), ())), "
        "preferred_element_type=jnp.float32, "
        "precision=jax.lax.Precision.HIGHEST"
        ")",
        value=value,
    )
    return expr_from_string(
        f"({name}[{base_index}].astype(jnp.float32) + {{updates}}).astype("
        f"{plan.jnp_dtype})",
        updates=updates,
    )
