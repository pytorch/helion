"""Typed vector reads from existing full-native shared-memory operand views.

This component proves copy geometry, not graph identity, publication or lifetime.
The caller supplies an already admitted K-major SW128 half-precision tensor (or
its original full-layout row-offset alias). It must not recreate a dense member
layout: the full native height determines the offsets of later K64 panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .chained_vector_ownership import VectorOwnership


@dataclass(frozen=True)
class NativeReadEmission:
    setup: tuple[str, ...]
    copy: str
    values: str


@dataclass(frozen=True)
class NativeVectorRead:
    """Eight unchanged typed values at the caller's original owned coordinates.

    In element units, K_SW128 maps each panel using
    ``column % 64 + 64 * row + (column // 64) * 64 * full_height`` and
    then scales by two bytes before the pointer's ``Swizzle<3,4,3>``. That
    byte-addressed swizzle leaves the low four byte bits (eight halves) intact.
    An aligned eight-column group therefore has contiguous, ordered payloads,
    including after a row-offset alias and at every aligned frame origin.
    The TV ownership selects precisely that group; its repeated row/column
    indices must remain the same as the scalar producer's ownership.

    Logical row tails retain the caller's row guard. This does not authorize
    speculative reads beyond a logical column extent or across publication.
    """

    full_shape: tuple[int, int]
    shape: tuple[int, int]
    row_offset: int
    dtype: torch.dtype
    ownership: VectorOwnership

    def matches(self) -> bool:
        return self == plan_native_vector_read(
            self.full_shape,
            self.shape,
            self.row_offset,
            self.dtype,
            self.ownership,
        )

    def emit(
        self, tensor: str, prefix: str, thread: str, step: str
    ) -> NativeReadEmission:
        """Read the already-bound alias; do not apply its row offset again."""
        if not self.matches():
            raise ValueError("native vector read geometry changed")
        dtype = (
            "cutlass.BFloat16" if self.dtype == torch.bfloat16 else "cutlass.Float16"
        )
        rows, columns = self.ownership.thread_rows, self.ownership.thread_columns
        strides = self.ownership.thread_strides
        return NativeReadEmission(
            (
                f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride={strides!r}), cute.make_layout((1, 8)))",
                f"{prefix}_thread = {prefix}_copy.get_slice({thread})",
                f"{prefix}_source = {prefix}_thread.partition_S({tensor})",
                f"{prefix}_values = cute.make_rmem_tensor({prefix}_source[None, 0, 0].shape, {dtype})",
            ),
            f"cute.copy({prefix}_copy, {prefix}_source[{self.ownership.copy_indices(step)}], {prefix}_values)",
            f"{prefix}_values",
        )


def plan_native_vector_read(
    full_shape: tuple[int, int],
    shape: tuple[int, int],
    row_offset: int,
    dtype: torch.dtype,
    ownership: VectorOwnership,
) -> NativeVectorRead | None:
    """Select the existing half-precision K_SW128 layout, without allocating it.

    ``full_shape`` is the actual native owner; ``shape`` is the original Node's
    logical image. ``row_offset`` is relative to that same owner's layout.
    Source dtype/layout/alias identity must be established by the late caller,
    independently of this geometry proof. MN-major and transposed logical views
    need a different ordered-payload proof and must not call this constructor.
    """
    if (
        type(full_shape) is not tuple
        or type(shape) is not tuple
        or len(full_shape) != 2
        or len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in (*full_shape, *shape))
        or type(row_offset) is not int
        or row_offset < 0
        or row_offset + shape[0] > full_shape[0]
        or shape[1] != full_shape[1]
        or full_shape[0] % 8
        or full_shape[1] % 64
        or dtype not in (torch.bfloat16, torch.float16)
        or not ownership.matches(shape, ownership.threads)
        or ownership.threads < 32
        or ownership.threads > 1024
        or ownership.threads & (ownership.threads - 1)
    ):
        return None
    return NativeVectorRead(full_shape, shape, row_offset, dtype, ownership)
