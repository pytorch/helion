"""Device half of the host remainder fill: every thread stores its share.

``grad[:, -1, :] = 0`` after the launch writes the rows a kernel's complete
tile stores leave unwritten (``helion._compiler.cute.host_remainder_fills``).
Called at the end of the kernel body by every thread of every CTA, this
helper distributes the region's elements over the whole grid (thread
``g`` of ``total`` takes elements ``g, g + total, ...``) and stores the
constant. The region is the tensor with dimension ``dim`` restricted to its
last ``remainder`` coordinates; its shape and strides come from the tensor
argument itself, so dynamic shapes work unchanged.

Every DSL value the loops use is formed before them: a Python scalar
captured inside a staged ``while`` body becomes a staged ``Int32`` and a
``range_constexpr`` bound cannot be formed there, so the coordinate
decomposition is written out per rank under ``const_expr`` branches.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def _cute_remainder_fill(
    tensor: cute.Tensor,
    *,
    dim: int,
    remainder: int,
    value: float,
    is_float: bool,
) -> None:
    rank: int = len(tensor.shape)  # pyrefly: ignore[bad-argument-type]
    extents = [
        cutlass.Int32(remainder) if i == dim else cutlass.Int32(tensor.shape[i])  # pyrefly: ignore[bad-index]
        for i in range(rank)
    ]
    strides = [
        cutlass.Int32(tensor.stride[i])  # pyrefly: ignore[bad-index, missing-attribute]
        for i in range(rank)
    ]
    base = (
        cutlass.Int32(tensor.shape[dim])  # pyrefly: ignore[bad-index]
        - cutlass.Int32(remainder)
    ) * strides[dim]
    count = cutlass.Int32(1)
    for extent in extents:
        count = count * extent
    if cutlass.const_expr(is_float):
        fill = tensor.element_type(cutlass.Float32(value))
    else:
        fill = tensor.element_type(cutlass.Int32(value))
    tx, ty, tz = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    bdx, bdy, bdz = cute.arch.block_dim()
    gdx, gdy, gdz = cute.arch.grid_dim()
    threads = cutlass.Int32(bdx) * cutlass.Int32(bdy) * cutlass.Int32(bdz)
    thread = (
        cutlass.Int32(tx)
        + cutlass.Int32(ty) * cutlass.Int32(bdx)
        + cutlass.Int32(tz) * cutlass.Int32(bdx) * cutlass.Int32(bdy)
    )
    block = (
        cutlass.Int32(bx)
        + cutlass.Int32(by) * cutlass.Int32(gdx)
        + cutlass.Int32(bz) * cutlass.Int32(gdx) * cutlass.Int32(gdy)
    )
    total = threads * cutlass.Int32(gdx) * cutlass.Int32(gdy) * cutlass.Int32(gdz)
    index = block * threads + thread
    if cutlass.const_expr(rank == 1):
        while index < count:
            (tensor.iterator + base + index * strides[0]).store(fill)  # pyrefly: ignore[missing-attribute]
            index = index + total
    elif cutlass.const_expr(rank == 2):
        while index < count:
            c1 = index % extents[1]
            c0 = index // extents[1]
            (tensor.iterator + base + c0 * strides[0] + c1 * strides[1]).store(fill)  # pyrefly: ignore[missing-attribute]
            index = index + total
    elif cutlass.const_expr(rank == 3):
        while index < count:
            c2 = index % extents[2]
            rest = index // extents[2]
            c1 = rest % extents[1]
            c0 = rest // extents[1]
            (
                tensor.iterator
                + base
                + c0 * strides[0]
                + c1 * strides[1]
                + c2 * strides[2]
            ).store(fill)  # pyrefly: ignore[missing-attribute]
            index = index + total
    else:
        while index < count:
            c3 = index % extents[3]
            rest = index // extents[3]
            c2 = rest % extents[2]
            rest = rest // extents[2]
            c1 = rest % extents[1]
            c0 = rest // extents[1]
            (
                tensor.iterator
                + base
                + c0 * strides[0]
                + c1 * strides[1]
                + c2 * strides[2]
                + c3 * strides[3]
            ).store(fill)  # pyrefly: ignore[missing-attribute]
            index = index + total
