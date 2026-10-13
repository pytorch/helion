"""Typed CuTe TMA wrapper construction for caller-proven compact tiles."""

from __future__ import annotations

import torch


def append_tma_tile(
    body: list[str],
    call_args: list[str],
    *,
    source_index: int,
    atom: str,
    tensor: str,
    rows: int,
    columns: int,
    tile: tuple[int, int],
    dtype: torch.dtype,
) -> None:
    """Append the current-pointer view, matching SMEM layout, and TMA atom.

    The leaf's dtype is independent of any eventual contraction operand dtype.
    Callers own layout/alignment/storage validation, native K_SW tile legality,
    pointer-aware cache behavior, device issue coordinates, and synchronization.
    This helper neither retains tensors nor chooses a transfer schedule.
    """
    dtype_name = {
        torch.bfloat16: "cutlass.BFloat16",
        torch.float16: "cutlass.Float16",
        torch.float32: "cutlass.Float32",
    }[dtype]
    width_bytes = tile[1] * dtype.itemsize
    swizzle = min(128, width_bytes & -width_bytes)
    body.extend(
        [
            f"    {atom}_global = cute.make_tensor(arg{source_index}.iterator.align(16), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)))",
            f"    {atom}_layout = cute.tile_to_shape(cute.nvgpu.tcgen05.make_smem_layout_atom(cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW{swizzle}, {dtype_name}), {tile!r}, order=(0, 1))",
            f"    {atom}, {tensor} = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), {atom}_global, {atom}_layout, {tile!r})",
        ]
    )
    call_args.extend((atom, tensor))
