"""The PTX of ``hl.prefetch``: its lowering emits it, and the cross-loop
scheduler recognizes it (exactly) as a hint that orders nothing."""

from __future__ import annotations

import itertools


def prefetch_asm(*, bulk: bool, scalar: bool, elect: bool) -> str:
    """PTX for one L2 prefetch per element of the pointer operand ($1).

    The plain form issues ``prefetch.global.L2`` unless the int32 predicate
    ($2) is 0.  The bulk form's $2 is a byte count instead, skipped unless
    positive, and widens [$1, $1 + $2) to the 16-byte granules
    ``cp.async.bulk`` requires (a granule never crosses a page).  A scalar
    prefetch is issued once per warp instead of by every thread: with
    ``elect`` (sm_90+) by an elected lane, which keeps ptxas's uniformity
    analysis (selecting a thread by ``%tid.x`` loses it across the whole
    kernel, and no thread of a warp-specialized partition has ``%tid.x ==
    0``); otherwise by thread 0.
    """
    if bulk:
        regs = ".reg .pred p, q; .reg .b32 t, n; .reg .b64 a, e;"
        test = "setp.gt.s32 p, $2, 0;"
        issue = (
            "and.b64 a, $1, -16; cvt.s64.s32 e, $2; add.s64 e, e, $1; "
            "add.s64 e, e, 15; and.b64 e, e, -16; sub.s64 e, e, a; "
            "cvt.u32.u64 n, e; @p cp.async.bulk.prefetch.L2.global [a], n;"
        )
    else:
        regs = ".reg .pred p, q; .reg .b32 t;"
        test = "setp.ne.b32 p, $2, 0;"
        issue = "@p prefetch.global.L2 [$1];"
    if scalar and elect:
        test += " elect.sync _|q, 0xffffffff; and.pred p, p, q;"
    elif scalar:
        test += " mov.u32 t, %tid.x; setp.eq.and.u32 p, t, 0, p;"
    return f"{{ {regs} {test} {issue} }} mov.u32 $0, 0;"


# Every form ``prefetch_asm`` emits.
PREFETCH_ASMS = frozenset(
    prefetch_asm(bulk=bulk, scalar=scalar, elect=elect)
    for bulk, scalar, elect in itertools.product((False, True), repeat=3)
)
