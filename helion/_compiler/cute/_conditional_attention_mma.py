"""Shared-A issue primitive for a separately proved conditional schedule.

The planner supplies descriptors and exact K-step offsets from Tcgen05Tile.
This helper owns one elected issuer in the caller's full issue warp; it does
not publish shared writes, wait for completion, or retire buffers for reuse.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims


@cute.jit
def conditional_mma_ss(
    a: prims.Tcgen05SmemDesc,
    b: prims.Tcgen05SmemDesc,
    destination: cutlass.Int32,
    done: cute.Pointer,
    a_offsets: cutlass.Constexpr,
    b_offsets: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    a_mn_major: cutlass.Constexpr,
    b_mn_major: cutlass.Constexpr,
    accumulate: cutlass.Constexpr,
) -> None:
    assert cutlass.const_expr(len(a_offsets) == len(b_offsets) == 8)
    assert cutlass.const_expr(0 < query_columns <= 64 and query_columns % 8 == 0)
    descriptor = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=query_columns,
        m_dim=128,
        a_major=1 if a_mn_major else 0,
        b_major=1 if b_mn_major else 0,
    )
    output = cutlass.inttoptr(destination, 6, cutlass.Float32)
    for step in cutlass.range_constexpr(8):
        assert cutlass.const_expr(a_offsets[step] % 16 == 0)
        assert cutlass.const_expr(b_offsets[step] % 16 == 0)
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                output,
                a.advance_start_address(a_offsets[step]),
                b.advance_start_address(b_offsets[step]),
                descriptor,
                accumulate or step != 0,
            )
    if prims.elect_sync():
        prims.tcgen05_commit(done, group=prims.CTAGroup.CTA_1)
