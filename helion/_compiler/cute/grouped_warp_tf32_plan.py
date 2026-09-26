"""Typed admission for a grouped warp-MMA TF32 implementation.

This is an explicit approximation policy for mapped ``dot_precision='tf32'``.
Operands retain their FP32 words until the TF32 MMA instruction; this policy
does not promise RN, RNA, or IEEE-equivalent operand conversion. An explicit
TMA-RN configuration remains a separate implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .memory_ops import tensor_has_specialized_base_alignment
from .mma_support import cute_fp32_dot_uses_tf32

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from .tcgen05_flat_grouped_ir import FlatGroupedIRProof


@dataclass(frozen=True)
class GroupedWarpTF32Plan:
    groups: int
    rows: int
    reduction: int
    columns: int
    shared_capacity: int

    def __post_init__(self) -> None:
        limit = (1 << 31) - 1
        if (
            any(
                type(value) is not int
                for value in (
                    self.groups,
                    self.rows,
                    self.reduction,
                    self.columns,
                    self.shared_capacity,
                )
            )
            or not 0 < self.groups <= limit
            or not 0 < self.rows <= limit - 127
            or not 0 < self.reduction <= limit
            or self.reduction % 64
            or not 0 < self.columns <= limit - 127
            or self.columns % 4
            or (self.columns + 63) // 64 > 65535
            or max(
                self.rows * self.reduction,
                self.rows * self.columns,
                self.groups * self.reduction * self.columns,
            )
            >= 1 << 31
            or self.workspace_upper_bound > limit
            or not self.shared_bytes + 1024 <= self.shared_capacity <= 232448
        ):
            raise ValueError("unsupported grouped warp TF32 binding or resource budget")

    @property
    def thread_block(self) -> tuple[int, int, int]:
        return (512, 1, 1)

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.groups, (self.columns + 63) // 64, 1)

    @property
    def shared_bytes(self) -> int:
        # Disjoint A[128,64], output shuffle[128,64], B[64,64], all FP32.
        return 81920

    @property
    def workspace_upper_bound(self) -> int:
        # Retain the existing grouped host allocation/ABI. The warp kernel does
        # not read the descriptor workspace or require a device descriptor fill.
        return self.groups * ((self.rows + 127) // 128) * ((self.columns + 127) // 128)

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "grouped_warp_tf32_raw_v1",
            self.groups,
            self.rows,
            self.reduction,
            self.columns,
            self.thread_block,
            self.grid,
            (128, 64, 64),
            ("source_offset_bits", 64),
            ("raw_fp32_words_to_tf32_mma", True),
            ("accumulator", "ordered_k8_fp32"),
            ("bias", "fp32_after_complete_k"),
            ("shared_spans", ((0, 32768), (32768, 65536), (65536, 81920))),
            ("reserved_shared", 1024),
            self.shared_capacity,
        )


def grouped_warp_tf32_plan(
    env: CompileEnvironment, proof: FlatGroupedIRProof, *, shared_capacity: int
) -> GroupedWarpTF32Plan | None:
    """Check source-width indices, direct-load precision, strides and bind facts."""
    tensors = (proof.offsets, proof.a, proof.b, proof.bias, proof.output)
    if (
        env.config_spec.target_device_capability != (10, 0)
        or not env.settings.static_shapes
        or not cute_fp32_dot_uses_tf32()
        or proof.offset_bits != 64
        or any(tensor.device != env.device for tensor in tensors)
        or not all(
            type(value) is int
            for tensor in tensors
            for value in (*tensor.shape, *tensor.stride())
        )
    ):
        return None
    m, k = proof.a.shape
    g, b_k, n = proof.b.shape
    if (
        b_k != k
        or tuple(proof.a.stride()) != (k, 1)
        or tuple(proof.b.stride()) != (k * n, n, 1)
        or tuple(proof.bias.shape) != (g, n)
        or tuple(proof.bias.stride()) != (n, 1)
        or tuple(proof.offsets.shape) != (g + 1,)
        or tuple(proof.offsets.stride()) != (1,)
        or tuple(proof.output.shape) != (m * n,)
        or tuple(proof.output.stride()) != (1,)
        or not all(
            tensor_has_specialized_base_alignment(env, tensor, 16)
            for tensor in (proof.a, proof.b, proof.bias)
        )
    ):
        return None
    try:
        return GroupedWarpTF32Plan(g, m, k, n, shared_capacity)
    except ValueError:
        return None
