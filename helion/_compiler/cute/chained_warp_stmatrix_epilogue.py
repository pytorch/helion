"""Standard StMatrix publication of an already validated warp epilogue.

The existing register bridge owns source completion, exclusive use and the
original pointwise expression. This adapter owns only the complete row-major
half-precision destination geometry; it introduces no allocation or join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .chained_fragment_epilogue import WarpOperandEpilogue
    from .chained_root_warp_stage import WarpOperandLayout


def render_stmatrix_operand(
    epilogue: WarpOperandEpilogue,
    layout: WarpOperandLayout,
    *,
    producer_threads: int,
    cta_threads: int,
    pointer_alignment: int,
) -> tuple[str, ...] | None:
    """Retain the original expressions, or decline to the original publication.

    Call only for the ordinary (1, warps, 1) warp MMA topology, after validating
    the original bridge and its completion. ``pointer_alignment`` must describe
    the actual destination pointer; an alternate native owner is not admitted
    merely because it has the same logical layout.
    """
    value = epilogue.value
    dtype = {
        torch.bfloat16: "cutlass.BFloat16",
        torch.float16: "cutlass.Float16",
    }.get(layout.dtype)
    if (
        dtype is None
        or value.dtype != dtype
        or layout.node is not value.target
        or layout.role != "a"
        or layout.inner != 1
        or layout.shape != epilogue.shape
        or len(value.coordinates) != 2
        or value.target_coordinates != value.coordinates
        or producer_threads != cta_threads
        or producer_threads < 32
        or producer_threads % 32
        or pointer_alignment < 16
        or pointer_alignment % 16
    ):
        return None
    rows, columns = epilogue.shape
    row_stride, column_stride = layout.stride
    # Whole 16x16 StMatrix4 panels per original warp. Smaller duplicated-panel
    # cases retain autovec rather than extending this geometry proof.
    if (
        rows <= 0
        or rows % 16
        or columns <= 0
        or columns % (16 * (producer_threads // 32))
        or column_stride != 1
        or row_stride < columns
        or row_stride % 8
    ):
        return None

    # This validates the original expression snapshot, coordinate substitution,
    # masks and typed value. No second pointwise renderer or source rewrite.
    original = epilogue.render()
    prefix, previous = epilogue.prefix, epilogue.previous
    return (
        original[0],
        f"{prefix}_values = cute.make_rmem_tensor_like({previous}_acc, {dtype})",
        *original[2:-1],
        f"{prefix}_store_copy = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False), {dtype}), {previous}_mma)",
        f"{prefix}_store_thread = {prefix}_store_copy.get_slice(chain_thread)",
        f"{prefix}_store_source = {prefix}_store_thread.retile({prefix}_values)",
        f"{prefix}_store_target = {prefix}_store_thread.partition_D({epilogue.destination})",
        f"cute.copy({prefix}_store_copy, {prefix}_store_source, {prefix}_store_target)",
    )
