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


def tensor_prefetch_asm(rank: int, warps: int) -> str:
    """PTX for one L2 prefetch of a TMA box: ``$1`` the tensor map (in global
    memory, acquired once by ``TENSOR_MAP_ACQUIRE_ASM``), ``$2``... the box's
    coordinates, innermost dim first.  A single operation takes the whole box,
    where the bulk form takes one contiguous row segment per operation (and
    blocks its issuing warp per operation until the TMA engine accepts it).

    One lane of one warp of the ``warps`` running it issues it: the elected
    lane of the warp whose ``%warpid`` is a multiple of ``warps`` (a power of
    two; one of any ``warps`` consecutive ids), so the TMA engine takes each
    box once instead of once per warp.  The id is read inside the asm (no
    register stays live, unlike a ``%tid.x``-derived warp index).  PTX leaves
    ``%warpid``'s assignment unspecified (it may change under preemption):
    that a CTA's warps hold consecutive ids was measured on B200.  Where it
    fails, or when ``warps`` is not the count of warps running the prefetch
    (a warp-specialized partition of another size), boxes are issued by no
    warp or by several: a hint skipped or repeated, never a wrong result."""
    assert warps & (warps - 1) == 0, warps
    coords = ", ".join(f"${i + 2}" for i in range(rank))
    pick = (
        "elect.sync _|q, 0xffffffff; "
        if warps == 1
        else f"mov.u32 t, %warpid; and.b32 t, t, {warps - 1}; setp.eq.u32 w, t, 0; "
        "elect.sync _|q, 0xffffffff; and.pred q, q, w; "
    )
    return (
        f"{{ .reg .pred q, w; .reg .b32 t; {pick}"
        f"@q cp.async.bulk.prefetch.tensor.{rank}d.L2.global.tile "
        f"[$1, {{{coords}}}]; }} mov.u32 $0, 0;"
    )


# The tensormap-proxy acquire of a tensor map the host copied to global memory
# (hence system scope), once per kernel entry: its result is the map's address,
# which the prefetches read, so they stay after it.  A fence per prefetch would
# make each prefetch wait for the ones before it.
TENSOR_MAP_ACQUIRE_ASM = (
    "{ .reg .pred q; elect.sync _|q, 0xffffffff; "
    "@q fence.proxy.tensormap::generic.acquire.sys [$1], 128; } mov.b64 $0, $1;"
)


# Every form ``prefetch_asm`` and ``tensor_prefetch_asm`` emit.
PREFETCH_ASMS = frozenset(
    [
        *(
            prefetch_asm(bulk=bulk, scalar=scalar, elect=elect)
            for bulk, scalar, elect in itertools.product((False, True), repeat=3)
        ),
        *(
            tensor_prefetch_asm(rank, 1 << log2_warps)
            for rank, log2_warps in itertools.product(range(1, 6), range(6))
        ),
    ]
)
