"""Prove ordinary dense GEMMs with a fresh multiplicative epilogue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .cute_epilogue import _AuxiliaryTensorLoadExpr
from .cute_epilogue import _BinaryTensorExpr
from .cute_epilogue import _CurrentTensorExpr
from .cute_epilogue import _RuntimeScalarExpr
from .tcgen05_schedule_facts import prove_full_tile_schedule_facts

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from .tcgen05_schedule_facts import NativeGemm


@dataclass(frozen=True)
class DenseGemmFacts:
    """Native input and output facts in the frontend's own M,N coordinates.

    These facts prove a single dense MMA and its FP32 multiplicative scales.
    They select neither orientation nor a physical tile; the frontend supplies
    any operand transposes and the output's actual layout remains authoritative.
    """

    block_ids: tuple[int, int, int]
    shape: tuple[int, int, int]
    input_dtype: torch.dtype
    capacity_bytes: int
    output_column_major: bool = False


def _scale_view_supported(
    tensor: torch.Tensor,
    broadcast_axis: int | None,
    shape: tuple[int, int, int],
) -> bool:
    """Admit scalar, row, and column scales without rewriting their addresses."""
    if tensor.dtype != torch.float32 or any(
        type(value) is not int or value < 0
        for value in (*tensor.shape, *tensor.stride())
    ):
        return False
    m, n, _k = shape
    return (
        broadcast_axis == 1
        and tensor.shape == (n,)
        or broadcast_axis == 0
        and tensor.ndim == 2
        and tensor.shape[1] == n
        and (tensor.shape[0] == 1 or (tensor.shape[0] == m and tensor.stride(0) == 0))
        or broadcast_axis == 2
        and tensor.shape == (m, n)
        and tensor.stride(1) == 0
    )


def prove_dense_gemm_facts(
    env: CompileEnvironment, gemm: NativeGemm | None
) -> DenseGemmFacts | None:
    base = prove_full_tile_schedule_facts(gemm)
    if (
        base is None
        or gemm is None
        or not gemm.owns_output_ring
        or not base.rhs_k_major
        or not base.lhs_k_major
        or gemm.output.dtype != torch.bfloat16
    ):
        return None
    for step in gemm.chain.steps:
        expr = step.expr
        if not (
            isinstance(expr, _BinaryTensorExpr)
            and expr.op_name == "mul"
            and expr.op_template == "{lhs} * {rhs}"
            and isinstance(expr.lhs, _CurrentTensorExpr)
        ):
            return None
        scale = expr.rhs
        if isinstance(scale, _AuxiliaryTensorLoadExpr):
            tensor_node = scale.load_node.args[0]
            assert isinstance(tensor_node, torch.fx.Node)
            tensor = tensor_node.meta.get("val")
            if not isinstance(tensor, torch.Tensor) or scale.template != "{aux}":
                return None
            if not _scale_view_supported(tensor, scale.broadcast_axis, base.shape):
                return None
        elif isinstance(scale, _RuntimeScalarExpr) and isinstance(
            scale.source, torch.fx.Node
        ):
            tensor_node = scale.source.args[0]
            assert isinstance(tensor_node, torch.fx.Node)
            tensor = tensor_node.meta.get("val")
            if not (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim == 0
                and tensor.dtype == torch.float32
            ):
                return None
        else:
            return None
    capacity = env.config_spec._cute_tcgen05_config.per_cta_smem_capacity_bytes(
        gemm.candidate.operands.lhs.source_fake.device
    )
    return DenseGemmFacts(
        base.block_ids,
        base.shape,
        base.dtype,
        capacity,
        gemm.candidate.output_column_major,
    )
