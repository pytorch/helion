"""Pallas indirect-gather lowering.

No native gather in Pallas. ``table[idx]`` emits ``one_hot(idx, V) @ table``.
Scatter is rejected at plan_tiling time.
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
    gather_axis_size: int
    indirect_pos: int
    none_dims: tuple[int, ...]
    jnp_dtype: str
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
    if indirect_pos != 0:
        raise NotImplementedError(
            "Pallas gather: only dim-0 indirect indexing is supported"
        )
    if not tensor.dtype.is_floating_point:
        raise NotImplementedError(
            f"Pallas gather: table must be floating point, got {tensor.dtype}"
        )

    elements = resident_block_elements(tensor, patterns, config)
    if elements is None:
        raise NotImplementedError(
            "Pallas gather: dynamic-shape tables are not supported; "
            "use @helion.kernel(static_shapes=True)"
        )
    table_bytes = elements * tensor.dtype.itemsize
    if table_bytes > _GATHER_VMEM_THRESHOLD_BYTES:
        raise NotImplementedError(
            f"Pallas gather: resident block is {table_bytes} bytes, exceeds "
            f"the {_GATHER_VMEM_THRESHOLD_BYTES} byte VMEM threshold. The "
            f"current codegen requires the full gather axis in VMEM; reduce "
            f"V, tile the broadcast dims, or use a half-precision dtype."
        )

    # MXU truncates fp32 to bf16 without HIGHEST. For bf16/fp16 the truncation is a no-op.
    use_highest = tensor.dtype not in (torch.bfloat16, torch.float16)

    none_dims = tuple(i for i, idx in enumerate(subscript) if idx is None)
    jnp_dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)

    return GatherPlan(
        gather_axis_size=tensor.shape[0],
        indirect_pos=indirect_pos,
        none_dims=none_dims,
        jnp_dtype=jnp_dtype,
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

    if plan.use_highest_precision:
        oh_dtype = "jnp.float32"
        table_expr = f"{name}[...].astype(jnp.float32)"
        precision_arg = "precision=jax.lax.Precision.HIGHEST, "
    else:
        oh_dtype = plan.jnp_dtype
        table_expr = f"{name}[...]"
        precision_arg = ""

    result = expr_from_string(
        f"jax.lax.dot_general("
        f"jax.nn.one_hot({idx_name}[...], {plan.gather_axis_size}, dtype={oh_dtype}), "
        f"{table_expr}, "
        f"(((jnp.ndim({idx_name}[...]),), (0,)), ((), ())), "
        f"preferred_element_type=jnp.float32, "
        f"{precision_arg}"
        f").astype({plan.jnp_dtype})"
    )
    for dim in plan.none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result
