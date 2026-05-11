"""Metal MMA (MPP matmul2d) helpers for matmul operations.

This module defines compatibility checks and pseudo-call payloads used by the
owning MPPGraph lowering.  The MPPGraph transform owns the K-loop and emits
the _metal_mpp_setup / _metal_mpp_k_step / cooperative epilogue / store
markers as one unit.

MPP matmul2d supports a variety of input and accumulation dtype combinations.
The accumulation dtype is determined by the output tensor or explicit acc dtype.
For example, half×half can accumulate in either half or float, and bfloat×bfloat
can accumulate in either bfloat or float.  See the full type table in
MPPTensorOpsMatMul2d.h or the MPP Programming Guide:
https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf

After the K-loop the owning MPPGraph stores the cooperative_tensor result to
device memory.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ... import exc
from ..cute.cute_mma import _has_mma_operands
from ..cute.cute_mma import _mma_loop_is_exclusive
from ..cute.cute_mma import _mma_result_can_be_deferred

if TYPE_CHECKING:
    import ast

    from ..aten_lowering import LoweringContext


# ---------------------------------------------------------------------------
# Pseudo-call proxies
#
# The FX-time injected pseudo-calls are placeholders in the
# generated Python (the @metal_jit decorated function); the MSL walker
# (``msl_ast_walker``) recognizes them and emits MPP MSL in their place.
# Modeled as dataclasses so call sites read as keyword constructors and
# the positional layout is centralized in one location.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _MppSetup:
    """``_metal_mpp_setup(...)`` proxy — emitted into ``device_loop.outer_prefix``.

    Carries everything the walker's ``_emit_mpp_setup`` needs to declare
    the MPP tensor handles, ``matmul2d_descriptor``, and cooperative_tensor:

    - ``lhs`` / ``rhs`` / ``bias``: kernel-arg names of the matmul operands
      and (optional) bias tensor; resolved via ``device_function.tensor_arg``
      at FX-codegen time.
    - ``M`` / ``N`` / ``K``: full matmul shape (size hints).
    - ``TILE_M`` / ``TILE_N`` / ``TILE_K``: per-threadgroup tile sizes
      from the user's ``Config.block_sizes``.
    - ``NUM_SG``: number of simdgroups cooperating on each tile
      (``Config.num_warps``).
    - ``in_dtype`` / ``acc_dtype`` / ``bias_dtype``: Metal dtype strings
      (``"float"``, ``"half"``, ``"bfloat"``).  ``acc_dtype`` is the
      cooperative_tensor element type and follows Helion's accumulator
      semantics rather than defaulting to the input dtype.  The output
      tensor's name and dtype are passed through the explicit cooperative
      store marker.
    - ``fx_name``: FX node name of the MMA op (e.g. ``"acc"``); after
      Helion's phi handler renames the placeholder, this is the variable
      the epilogue references.

    Field order is mirrored by :func:`msl_ast_walker._extract_mpp_setup_params`;
    keep them in sync.
    """

    lhs: str
    rhs: str
    M: int
    N: int
    K: int
    TILE_M: int
    TILE_N: int
    TILE_K: int
    NUM_SG: int
    in_dtype: str
    acc_dtype: str
    bias: str
    bias_dtype: str
    fx_name: str

    def __str__(self) -> str:
        return (
            f'_metal_mpp_setup("{self.lhs}", "{self.rhs}", '
            f"{self.M}, {self.N}, {self.K}, "
            f"{self.TILE_M}, {self.TILE_N}, {self.TILE_K}, {self.NUM_SG}, "
            f'"{self.in_dtype}", "{self.acc_dtype}", '
            f'"{self.bias}", "{self.bias_dtype}", '
            f'"{self.fx_name}")'
        )


@dataclasses.dataclass(frozen=True)
class _MppKStep:
    """``_metal_mpp_k_step(setup_var, k_offset)`` proxy — emitted into the K-loop body.

    The walker's ``_emit_mpp_k_step`` translates this into MSL that slices
    the next K-tile of A/B and runs ``_op.run(_Ak, _Bk, _coop)`` to
    accumulate one K-tile's contribution into the cooperative_tensor.
    """

    setup_var: str
    k_offset: str

    def __str__(self) -> str:
        return f"_metal_mpp_k_step({self.setup_var}, {self.k_offset})"


# ---------------------------------------------------------------------------
# MMA compatibility checks
# ---------------------------------------------------------------------------


def _is_mma_compatible(
    node: Node,
    *,
    lhs_idx: int,
    rhs_idx: int,
    acc_idx: int | None,
) -> bool:
    """Check operand compatibility for Metal MMA.

    Operand-position layout differs across FX targets:

    ``aten.mm`` uses ``(lhs, rhs)`` at ``args[0:2]``. ``aten.addmm`` uses
    ``(acc, lhs, rhs)`` at ``args[0:3]``.

    Both variants require lhs/rhs to be 2D MMA-compatible tensor Nodes
    (see :func:`_has_mma_operands`).  When acc is present at the given
    index, it must be a 2D tensor.
    """
    args = node.args
    required_len = max(lhs_idx, rhs_idx) + 1
    if len(args) < required_len:
        return False
    lhs_node, rhs_node = args[lhs_idx], args[rhs_idx]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return False
    if acc_idx is not None and acc_idx < len(args):
        acc_node = args[acc_idx]
        if isinstance(acc_node, Node):
            acc_val = acc_node.meta.get("val")
            if isinstance(acc_val, torch.Tensor) and acc_val.ndim != 2:
                return False
    return _has_mma_operands(lhs_node, rhs_node)


def can_codegen_metal_mma_aten(node: Node, with_acc: bool) -> bool:
    """Return True when an aten mm/addmm node can use Metal MMA.

    Requires all of:
    - Operand compatibility (shape, dtype — see :func:`_is_mma_compatible`)
    - Result can be deferred (no immediate consumers that need the value
      before the K-loop completes)
    - The MMA loop is exclusive (the K reduction loop isn't shared with
      other non-MMA work)
    """
    # addmm: (acc, lhs, rhs) at args[0:3]. mm: (lhs, rhs).
    lhs_idx, rhs_idx, acc_idx = (1, 2, 0) if with_acc else (0, 1, None)
    return (
        _is_mma_compatible(node, lhs_idx=lhs_idx, rhs_idx=rhs_idx, acc_idx=acc_idx)
        and _mma_result_can_be_deferred(node)
        and _mma_loop_is_exclusive(node)
    )


# ---------------------------------------------------------------------------
# Entry point (called from aten_lowering.py)
# ---------------------------------------------------------------------------


def codegen_metal_mma(
    ctx: LoweringContext,
    node: Node,
    with_acc: bool,
) -> ast.AST | None:
    """Reject MPP-compatible aten matmul outside an owning MPPGraph."""
    if not can_codegen_metal_mma_aten(node, with_acc):
        return None
    raise exc.BackendUnsupported(
        "metal",
        "Metal MPP matmul requires an owning MPPGraph. Supported matmul "
        "patterns must load canonical 2D A[M,K] and B[K,N] tiles, reduce "
        "over one K tile loop, optionally apply a fusible pointwise epilogue, "
        "and store C[M,N].",
    )
