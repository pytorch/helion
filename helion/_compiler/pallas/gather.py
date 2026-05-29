"""Pallas indirect-gather lowering.

No native gather in Pallas. Floating ``table[idx]`` emits
``one_hot(idx, V) @ table``; int32 ``table[idx]`` emits a boolean one-hot
select and reduction. Scatter is rejected at plan_tiling time.
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
    base_index = ", ".join(parts)
    table_expr = f"{name}[{base_index}]"

    if plan.emit_select:
        mask_expr = (
            f"jax.nn.one_hot({idx_name}[...], {name}.shape[0], dtype={plan.jnp_dtype})"
        )
        for _ in range(plan.table_ndim - 1):
            mask_expr = f"jnp.expand_dims({mask_expr}, axis=-1)"
        result = expr_from_string(
            f"jnp.sum({table_expr} * {mask_expr}, "
            f"axis=jnp.ndim({idx_name}[...])"
            f").astype({plan.jnp_dtype})"
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

    p = plan.indirect_pos
    result = expr_from_string(
        "jax.lax.dot_general("
        f"jax.nn.one_hot({idx_name}[...], {name}.shape[{p}], dtype={oh_dtype}), "
        f"{table_dot_expr}, "
        f"(((jnp.ndim({idx_name}[...]),), ({p},)), ((), ())), "
        "preferred_element_type=jnp.float32, "
        f"{precision_arg}"
        f").astype({plan.jnp_dtype})"
    )
    if p > 0:
        n = plan.index_ndim
        src = tuple(range(n, n + p))
        dst = tuple(range(p))
        result = expr_from_string(
            f"jnp.moveaxis({{result}}, {src}, {dst})", result=result
        )
    for dim in plan.none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result
