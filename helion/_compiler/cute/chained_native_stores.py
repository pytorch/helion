"""Raw StMatrix transport into existing transposed native half-typed images.

This proves copy geometry, not graph identity or publication lifetime. The
caller supplies the original transposed K_SW64 member view of a complete native
owner and preserves every original value/zero assignment and role barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .chained_vector_ownership import VectorOwnership


@dataclass(frozen=True)
class NativeStoreEmission:
    setup: tuple[str, ...]
    values: str
    copy: str


@dataclass(frozen=True)
class NativeStMatrixStore:
    """Eight original halves per lane, with a complete 8x32 tile per warp.

    For K_SW64's 32-element K mode, the full owner's unswizzled byte address is
    ``base + 2 * ((member_row + column) * 32 + row)``. The pointer swizzles bits
    7..8 into bits 4..5, leaving each aligned eight-half physical row intact.
    A transposed x4 StMatrix moves the original lane's four ordered half pairs
    to those rows. Full native row origins and frame strides remain caller-
    owned and 128-byte aligned; the member row offset is not applied twice.

    The four-column-lane ownership makes every warp's row guard uniform. The
    fixed K-mode extent is a native layout capability, not a workload shape.
    Other K panels/layouts require a separate ordered transport proof.
    """

    full_shape: tuple[int, int]
    shape: tuple[int, int]
    row_offset: int
    dtype: torch.dtype
    ownership: VectorOwnership

    def matches(self) -> bool:
        return self == plan_native_stmatrix_store(
            self.full_shape, self.shape, self.row_offset, self.dtype, self.ownership
        )

    def emit(
        self, tensor: str, prefix: str, thread: str, step: str
    ) -> NativeStoreEmission:
        if not self.matches():
            raise ValueError("native store geometry changed")
        dtype = (
            "cutlass.BFloat16" if self.dtype == torch.bfloat16 else "cutlass.Float16"
        )
        rows, columns = self.ownership.thread_rows, self.ownership.thread_columns
        return NativeStoreEmission(
            (
                f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True), {dtype}), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
                f"{prefix}_target = {prefix}_copy.get_slice({thread}).partition_D({tensor})",
                f"{prefix}_values = cute.make_rmem_tensor({prefix}_target[None, 0, 0].shape, {dtype})",
            ),
            f"{prefix}_values",
            f"cute.copy({prefix}_copy, {prefix}_values, {prefix}_target[{self.ownership.copy_indices(step)}])",
        )


def plan_native_stmatrix_store(
    full_shape: tuple[int, int],
    shape: tuple[int, int],
    row_offset: int,
    dtype: torch.dtype,
    ownership: VectorOwnership,
) -> NativeStMatrixStore | None:
    """Prove an existing K_SW64 member's transposed, complete vector owners.

    ``shape`` is logical (K, member-N); ``full_shape`` is physical (full-N, K).
    The caller validates native layout/modes and disjoint live destinations.
    Scalar masks may choose zeros, but may not skip individual StMatrix lanes.
    """
    if (
        type(full_shape) is not tuple
        or type(shape) is not tuple
        or len(full_shape) != 2
        or len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in (*full_shape, *shape))
        or type(row_offset) is not int
        or row_offset < 0
        or row_offset % 16
        or full_shape[0] % 16
        or full_shape[1] != 32
        or shape[0] != full_shape[1]
        or row_offset + shape[1] > full_shape[0]
        or dtype not in (torch.bfloat16, torch.float16)
        or not ownership.matches(shape, ownership.threads)
        or ownership.thread_order != "row_major"
        or not 32 <= ownership.threads <= 1024
        or ownership.threads & (ownership.threads - 1)
        or ownership.thread_columns != 4
    ):
        return None
    return NativeStMatrixStore(full_shape, shape, row_offset, dtype, ownership)
