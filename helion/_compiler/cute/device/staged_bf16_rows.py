"""Two BF16 row fragments times a common BF16 vector, with 3-stage packed reads.

Generated device code supplies two independent row bases (which may point to
paired rows in one output or corresponding rows of two matrices), the common
vector base, and the first packet inside K for the caller's warp/lane.
Each thread publishes and consumes only its own 2x8 BF16 matrix packets;
collective communication happens in the warp MMA, not in shared memory.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute

from .._flash_row_mma_runtime import mma_bf16
from .ptx import cp_async_16


@cute.jit
def staged_row_dot(
    scratch: cute.Pointer,
    low: cute.Pointer,
    high: cute.Pointer,
    vector: cute.Pointer,
    lane_k: cutlass.Int32,
    panel_k: cutlass.Constexpr,
    num_panels: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> tuple[object, object]:
    lane = cutlass.Int32(cute.arch.lane_idx()) + 32 * cutlass.Int32(
        cute.arch.warp_idx()
    )
    packet = lane * 8
    stride = threads * 16
    half = threads * 8
    c0 = cutlass.Float32(0)
    c1 = cutlass.Float32(0)
    c2 = cutlass.Float32(0)
    c3 = cutlass.Float32(0)
    for panel in cutlass.range_constexpr(2):
        k = cutlass.Int32(panel * panel_k) + lane_k
        target = scratch + panel % 3 * stride + packet
        cp_async_16(cutlass.Int32(target.toint()), cutlass.Int64((low + k).toint()))
        cp_async_16(
            cutlass.Int32((target + half).toint()), cutlass.Int64((high + k).toint())
        )
        cute.arch.cp_async_commit_group()
    for panel in range(num_panels):
        if panel + 2 < num_panels:
            k = cutlass.Int32((panel + 2) * panel_k) + lane_k
            target = scratch + (panel + 2) % 3 * stride + packet
            cp_async_16(cutlass.Int32(target.toint()), cutlass.Int64((low + k).toint()))
            cp_async_16(
                cutlass.Int32((target + half).toint()),
                cutlass.Int64((high + k).toint()),
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(2)
        source = scratch + panel % 3 * stride + packet
        # VectorType is provided by the MLIR extension at runtime.
        packed = ir.VectorType.get([4], cutlass.Uint32.mlir_type)  # pyrefly: ignore[missing-attribute]
        v = cute.arch.load(vector + cutlass.Int32(panel * panel_k) + lane_k, packed)
        a = cute.arch.load(source, packed)
        b = cute.arch.load(source + half, packed)
        c0, c1, c2, c3 = mma_bf16(
            cutlass.Uint32(a[0]),
            cutlass.Uint32(b[0]),
            cutlass.Uint32(a[1]),
            cutlass.Uint32(b[1]),
            cutlass.Uint32(v[0]),
            cutlass.Uint32(v[1]),
            c0,
            c1,
            c2,
            c3,
        )
        c0, c1, c2, c3 = mma_bf16(
            cutlass.Uint32(a[2]),
            cutlass.Uint32(b[2]),
            cutlass.Uint32(a[3]),
            cutlass.Uint32(b[3]),
            cutlass.Uint32(v[2]),
            cutlass.Uint32(v[3]),
            c0,
            c1,
            c2,
            c3,
        )
    cute.arch.cp_async_wait_group(0)
    return c0, c2
