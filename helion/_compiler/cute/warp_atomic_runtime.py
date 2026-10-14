"""Warp aggregation that preserves relaxed Int32 fetch-add return values."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cutlass import Int32
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from ._mlir_compat import ir


@dsl_user_op
def warp_fetch_add_int32(
    ptr: ir.Value,
    val: Int32,
    *,
    sem: str = "relaxed",
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Int32:
    """Reserve one contiguous ticket range for each matching address.

    The compiler admits only zero/one increments and relaxed global atomics.
    The active mask comes from inside the ordinary bounds/ownership guard, so
    tails and replicated tile coordinates contribute only their live owners.
    """
    address = llvm.ptrtoint(Int64.mlir_type, ptr, loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [address, val.ir_value(loc=loc, ip=ip)],
            """
            {
              .reg .pred p;
              .reg .b32 active, peers, updates, lower, rank, count;
              .reg .b32 lane, first, reversed, base;
              activemask.b32 active;
              match.any.sync.b64 peers, $1, active;
              setp.ne.s32 p, $2, 0;
              vote.ballot.sync.b32 updates, p, active;
              and.b32 updates, updates, peers;
              popc.b32 count, updates;
              mov.u32 lower, %lanemask_lt;
              and.b32 lower, lower, updates;
              popc.b32 rank, lower;
              brev.b32 reversed, peers;
              bfind.u32 first, reversed;
              sub.u32 first, 31, first;
              mov.u32 lane, %laneid;
              setp.eq.u32 p, lane, first;
              mov.u32 base, 0;
              @p atom.global.relaxed.gpu.add.u32 base, [$1], count;
              shfl.sync.idx.b32 base, base, first, 31, active;
              add.u32 $0, base, rank;
            }
            """,
            "=r,l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
