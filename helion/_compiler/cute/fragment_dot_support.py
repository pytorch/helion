"""Shared dtype and target eligibility for computed-fragment MMA dots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import matmul_ops
from .mma_support import cute_fp32_dot_uses_tf32

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def _mma_dot_operands(
    node: Node, env: CompileEnvironment
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Get tensor operands with FP32 output/accumulation on an MMA target."""
    if (
        node.op != "call_function"
        or node.target is not matmul_ops.dot
        or len(node.args) < 2
    ):
        return None
    left, right = node.args[:2]
    if not isinstance(left, Node) or not isinstance(right, Node):
        return None
    lhs, rhs, out = left.meta.get("val"), right.meta.get("val"), node.meta.get("val")
    if not (
        isinstance(lhs, torch.Tensor)
        and isinstance(rhs, torch.Tensor)
        and isinstance(out, torch.Tensor)
        and out.dtype == torch.float32
    ):
        return None
    acc = node.args[2] if len(node.args) > 2 else None
    if acc is not None and (
        not isinstance(acc, Node)
        or not isinstance(acc.meta.get("val"), torch.Tensor)
        or acc.meta["val"].dtype != torch.float32
    ):
        return None
    capability = env.config_spec.target_device_capability
    if capability is None or capability[0] < 8:
        return None
    return lhs, rhs


def fragment_half_dot_supported(node: Node, env: CompileEnvironment) -> bool:
    """Matching FP16/BF16 matrices with FP32 accumulation, on SM80 or newer.

    The emitter checks static contraction bounds and predicates partial tiles.
    Split-K and row-walk knobs are exposed by TF32-eligible dots; when such a
    dot shares the kernel, its enabled schedule also applies to half dots.
    """
    operands = _mma_dot_operands(node, env)
    return operands is not None and (
        operands[0].dtype in (torch.float16, torch.bfloat16)
        and operands[1].dtype == operands[0].dtype
        and operands[0].ndim >= 2
        and operands[1].ndim >= 2
    )


def fragment_tf32_dot_supported(node: Node, env: CompileEnvironment) -> bool:
    """FP32 dot/accumulation whose configured precision permits TF32 MMA."""
    operands = _mma_dot_operands(node, env)
    return (
        operands is not None
        and operands[0].dtype == torch.float32
        and operands[1].dtype == torch.float32
        and cute_fp32_dot_uses_tf32()
    )
