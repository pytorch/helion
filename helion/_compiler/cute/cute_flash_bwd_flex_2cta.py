"""The 2-CTA FlexAttention backward at head dim 128 (FA4's d128 cluster layout).

The 1-CTA flex backward (``cute_flash_bwd_flex``) reduce-adds a 128 x 128 fp32
dQ partial into global memory per 128 x 128 step: at long walks that traffic
(64 KB per step next to 64 KB of Q / dO loads) is the limiter -- the L2
reduction sectors, not the tensor pipe, bound the kernel, and suppressing the
reduce-adds alone runs the 32K walk 23% faster. A cluster of two CTAs owning
one 256-row KV tile halves it: dQ = dS K is one ``cta_group::2`` MMA whose K
dimension spans both CTAs' KV rows, so one 128 x 128 partial covers 256 KV
rows (each CTA drains its 64 Q rows). This is the dense 2-CTA body
(``cute_flash_bwd.emit_flash_bwd_2cta_device_body``) under the flex entry
point:

* CTA ``r`` of the pair holds KV rows ``[128 r, 128 r + 128)`` of the 256-row
  cluster tile: K / V (its S / dP A operands), the dK / dV accumulators of
  its rows, and the K^T half (its 64 head-dim columns x 256 KV rows, the dQ
  B operand). Per step each CTA loads half of Q / dO in both orientations
  (the pair's split B operands): Q / dO^T halves of 64 Q rows for S / dP,
  Q^T / dO halves of 64 head-dim columns for dK / dV.
* The softmax passes run per CTA on its 128 KV rows x 128 Q columns exactly
  as in the 1-CTA kernel (the same hooks, LSE / delta stagings and exp2
  forms); dS^T is packed into the dP columns of TMEM for the pair's dK MMA
  and staged in shared memory as the dQ A operand: the CTA's own 64 Q rows
  locally, the peer's 64 rows through a DSMEM bulk copy.
* TMEM per CTA: S 0-127 (P packed at 0-63, dQ at 64-127 once S is read),
  dV 128-255, dP 256-383 (dS^T packed at 256 + 64 h), dK 384-511.
* The walk is the flex step table (``fbwd_sSteps``) over the pair's MERGED
  lists: both CTAs must run every ``cta_group::2`` MMA, but the BlockMask
  lists the Q blocks per 128-row KV tile, so the pair walks, per q head,
  the even tile's lists (full, then partial) followed by the Q blocks only
  the odd tile lists, in its order. Each CTA builds the same sequence with
  its own status per entry (bit 24 masked, bit 25 absent: the Q tile is not
  in this CTA's lists, its P and dS are zero and the MMAs run on them;
  ``_merged_prologue``). A walk longer than the table is written whole to
  the CTA's region of a global scratch (``flex_bwd_walk_scratch``, a kernel
  argument the launch allocates) and runs in chunks (``_chunk_loop_2cta``),
  the table refilled by a copy out of the scratch at the working warps'
  barrier between chunks.

Shared memory is at the SM100 limit: the dense 2-CTA buffers (224 KB, the
dQ drain staged as the dense kernel's four (32 x 32) fp32 boxes, one per
reduce warp) leave about 3 KB for the LSE / Delta staging, the barriers, the
merge scratch and the step table, so the table holds 64-128 steps and long
walks run in chunks (cheap: the refill copies a few hundred bytes). Not here
(the config space hides them under the pair): the persistent walk, the
statistics MMAs, staged aux tables and strided maps.
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

from .cute_flash_bwd import _FBWD_2CTA_DS_OFFSETS
from .cute_flash_bwd import _FBWD_2CTA_SEND
from .cute_flash_bwd import _FLASH_BWD_2CTA_KERNEL_PARAMS
from .cute_flash_bwd import _fbwd_epilogue_block
from .cute_flash_bwd_flex import _SM100_SMEM_CAPACITY
from .cute_flash_bwd_flex import _lines
from .cute_flash_bwd_flex import _merge_blocks
from .cute_flash_bwd_flex import _reg_split
from .cute_flash_bwd_flex import _render_hooks
from .cute_flash_bwd_flex import _setmaxreg
from .cute_flash_bwd_flex import additive_bias_region

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from ..device_function import DeviceFunction
    from ..device_function import TensorArg
    from ..generate_ast import GenerateAST
    from .flash_score_mod import FlexHookCode
    from .flex_bwd_match import FlexBwdMatch
    from .flex_bwd_match import FlexBwdPlan

__all__ = [
    "codegen_flex_bwd_2cta",
    "emit_flash_bwd_flex_2cta_device_body",
    "flex_bwd_two_cta_allowed",
    "flex_bwd_two_cta_table_steps",
]

# TMEM columns (per CTA) of the dense 2-CTA plan.
_TMEM_S = 0
_TMEM_DV = 128
_TMEM_DP = 256
_TMEM_DK = 384
_TMEM_DQ = 64

_WAIT = "_helion_flash_rt.mbar_spin_wait({ptr}, {phase}, 10000000)"
_COMMIT = (
    "with cute.arch.elect_one():\n"
    "{ind}    cute_tcgen05_flash.commit({ptr}, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)"
)


def flex_bwd_two_cta_table_steps(
    plan: FlexBwdPlan, device: torch.device, table_cap: int = 0
) -> int:
    """Steps the pair kernel's step table holds (0: the storage does not fit
    even a minimal table): the whole merged walk (at most ``2 * max_steps``,
    both lists of a pair disjoint) when it fits the SM's shared memory, else
    the largest multiple of 32 that does; ``table_cap`` (the
    ``cute_flash_bwd_table_steps`` knob, 0 = automatic) bounds it. A walk
    longer than the table runs in chunks (``_chunk_loop_2cta``)."""
    import cutlass

    from ._flash_runtime import flash_bwd_flex_2cta_shared_storage
    from .tcgen05_config import CuteTcgen05Config

    capacity = (
        CuteTcgen05Config.per_cta_smem_capacity_bytes(device) or _SM100_SMEM_CAPACITY
    )
    dtype = (
        cutlass.BFloat16
        if plan.io_dtype.itemsize == 2 and str(plan.io_dtype).endswith("bfloat16")
        else cutlass.Float16
    )

    def fits(steps: int) -> bool:
        return (
            flash_bwd_flex_2cta_shared_storage(
                dtype, steps, _bits_words(plan)
            ).size_in_bytes()
            <= capacity
        )

    walk_max = 2 * max(plan.max_steps, 1)
    cap = walk_max if table_cap <= 0 else min(table_cap, walk_max)
    if fits(cap):
        return cap
    # The largest multiple of 32 at or below the cap that fits (the storage
    # grows with the table, so ``fits`` is monotone): ``lo * 32`` fits,
    # ``hi * 32`` does not or exceeds the cap.
    lo, hi = 0, cap // 32 + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid * 32):
            lo = mid
        else:
            hi = mid
    return lo * 32


# The smallest automatic step table the pair kernel accepts (a walk shorter
# than that needs only its own length; an explicit cap may go lower).
_FBWD2_TABLE_STEPS_MIN = 32


def _min_table_steps(plan: FlexBwdPlan) -> int:
    return min(_FBWD2_TABLE_STEPS_MIN, 2 * max(plan.max_steps, 1))


def flex_bwd_two_cta_allowed(plan: FlexBwdPlan, device: torch.device) -> bool:
    """Whether the plan can take the 2-CTA kernel: head dims (128, 128), an
    even KV tile count (pairs never straddle a (batch, kv head)), contiguous
    operand maps, 128-row BlockMask KV blocks and a storage that fits a step
    table of at least ``_FBWD2_TABLE_STEPS_MIN`` entries (or the whole walk)."""
    return (
        plan.head_dim == 128
        and plan.head_dim_v == 128
        and plan.num_kv_tiles % 2 == 0
        and plan.operand_strides is None
        and plan.walk.kv_block == 128
        and plan.group <= 255
        and flex_bwd_two_cta_table_steps(plan, device) >= _min_table_steps(plan)
    )


def _table_col(plan: FlexBwdPlan, gi: str) -> str:
    """The BlockMask table's head coordinate of q head ``gi`` of the group."""
    walk = plan.walk
    if walk.table_heads > 1:
        return f"(fbwd_hk * {plan.group} + {gi}) % {walk.table_heads}"
    return "cutlass.Int32(0)"


def _list_load(table: str, col: str, index: str) -> str:
    return f"cutlass.Int32({table}[fbwd_tb, fbwd_th, {col}, {index}])"


def _q_blocks(plan: FlexBwdPlan) -> int:
    """Q blocks of the sequence (the BlockMask's Q granularity)."""
    return -(-plan.q_seq // plan.walk.q_block)


def _bits_words(plan: FlexBwdPlan) -> int:
    """Words of one membership bitmap: one bit per Q block of a head plus a
    sentinel bit for the blocks past the sequence. A BlockMask built for a
    longer Q than the kernel's (legal in FlexAttention) lists blocks beyond
    ``_q_blocks``; every bitmap index is clamped to the sentinel
    (``_bit_index``), so such entries never touch the mbarrier words that
    follow the bitmaps in the shared storage. They still walk as steps: the
    TMA loads of their Q / dO zero-fill and their dQ store is clipped, as in
    the 1-CTA kernel."""
    return (_q_blocks(plan) + 1 + 31) // 32


def _bit_index(plan: FlexBwdPlan, x: str) -> str:
    """``x`` clamped to the bitmaps' sentinel slot."""
    return f"cutlass.min({x}, cutlass.Int32({_q_blocks(plan)}))"


def _head_counts(ind: str) -> str:
    """The four list lengths of the pair's KV tiles for q head ``fbwd_th``."""
    return (
        f"{ind}fbwd_cf0 = cutlass.Int32(_fbwd_full_cnt[fbwd_tb, fbwd_th, fbwd_pt0])\n"
        f"{ind}fbwd_cp0 = cutlass.Int32(_fbwd_part_cnt[fbwd_tb, fbwd_th, fbwd_pt0])\n"
        f"{ind}fbwd_cf1 = cutlass.Int32(_fbwd_full_cnt[fbwd_tb, fbwd_th, fbwd_pt1])\n"
        f"{ind}fbwd_cp1 = cutlass.Int32(_fbwd_part_cnt[fbwd_tb, fbwd_th, fbwd_pt1])\n"
    )


def _bits_build(plan: FlexBwdPlan, ind: str, nthreads: int, barrier: str) -> str:
    """The three membership bitmaps of q head ``fbwd_th`` (its counts
    loaded): the even tile's Q blocks (full and partial lists), the odd
    tile's full list, the odd tile's partial list; ``nthreads`` threads
    clear and set them (shared-memory atomic ORs) between barriers.

    The barrier BEFORE the clear is load-bearing: the previous head's L0
    fill reads the bitmaps (rank 1 derives its status of every even-tile
    entry from bitmaps 1 / 2) in a per-thread loop with no barrier after
    it, and its L1 section (the only other barriers of a head) runs no
    barrier when the odd tile lists nothing (``fbwd_nl1 == 0``). Without it
    the threads that clear the next head's words (``tidx < 3 * words``)
    could zero a bitmap a slower warp was still reading, and that warp wrote
    the entry as absent: the odd tile silently dropped the step (an accuracy
    failure of the 128K GQA-8 causal backward when the chunk refills still
    rebuilt the bitmaps with 448 threads: the clearing threads spanned four
    warps, each refill filled two warps' worth of entries while the other
    twelve ran ahead, 128 times per tile; every launch dropped steps)."""
    words = _bits_words(plan)
    # The loops run over uniform bounds with the thread's share behind a
    # predicate, never ``cutlass.range(tidx, count, nthreads)``: with the
    # shared-memory atomics inside a loop whose trip count differs per
    # thread, ptxas treats the rest of a chunked kernel as divergent and
    # wraps every tcgen05 issue of the MMA warp in VOTEU / R2UR.BROADCAST /
    # BRA.DIV guards (+65% MMA-warp instructions per step, the kernel
    # nearly twice the SASS).
    sets = "".join(
        f"{ind}for fbwd_jb in cutlass.range(0, {cnt}, {nthreads}, unroll=1):\n"
        f"{ind}    fbwd_j = fbwd_jb + tidx\n"
        f"{ind}    if fbwd_j < {cnt}:\n"
        f"{ind}        fbwd_x = {_bit_index(plan, _list_load(tab, col, 'fbwd_j'))}\n"
        f"{ind}        cute.arch.atomic_or(fbwd_sBits.iterator + ({base} + fbwd_x // 32), cutlass.Int32(1) << (fbwd_x % 32))\n"
        for base, cnt, tab, col in (
            ("0", "fbwd_cf0", "_fbwd_full_idx", "fbwd_pt0"),
            ("0", "fbwd_cp0", "_fbwd_part_idx", "fbwd_pt0"),
            (str(words), "fbwd_cf1", "_fbwd_full_idx", "fbwd_pt1"),
            (str(2 * words), "fbwd_cp1", "_fbwd_part_idx", "fbwd_pt1"),
        )
    )
    return (
        f"{ind}{barrier}\n"
        f"{ind}for fbwd_wb in cutlass.range(0, {3 * words}, {nthreads}, unroll=1):\n"
        f"{ind}    if fbwd_wb + tidx < {3 * words}:\n"
        f"{ind}        fbwd_sBits[fbwd_wb + tidx] = cutlass.Int32(0)\n"
        f"{ind}{barrier}\n" + sets + f"{ind}{barrier}\n"
    )


def _bit(plan: FlexBwdPlan, which: int, x: str) -> str:
    """Membership of Q block ``x`` in bitmap ``which`` (0 / 1 / 2)."""
    base = which * _bits_words(plan)
    return f"((cutlass.Int32(fbwd_sBits[{base} + {x} // 32]) >> ({x} % 32)) & 1)"


def _entry_bits(ind: str) -> str:
    """The table entry of Q tile ``fbwd_x`` of head ``fbwd_g`` with this
    CTA's status ``fbwd_st`` (2 full, 1 partial / masked, 0 absent): the
    1-CTA encoding plus the absent flag at bit 25."""
    return (
        f"{ind}fbwd_bits = ((fbwd_st & 1) << 24) | (((2 - fbwd_st) >> 1) << 25)\n"
        f"{ind}fbwd_entry = fbwd_x | (fbwd_g << 16) | fbwd_bits\n"
    )


def _walk_copy(
    plan: FlexBwdPlan,
    *,
    ind: str,
    nthreads: int,
    count: str,
    base: str | None,
) -> str:
    """Copy ``count`` entries of the CTA's walk, from walk position ``base``
    (None: the start) on, out of its global scratch into
    ``fbwd_sSteps[0:count)``: ``nthreads`` threads, a uniform trip count with
    the thread's share behind a predicate (like the bitmap loops), the
    warp's 32 consecutive entries one 128-byte line."""
    position = "fbwd_ci" if base is None else f"{base} + fbwd_ci"
    return (
        f"{ind}for fbwd_ib in cutlass.range(0, {count}, {nthreads}, unroll=1):\n"
        f"{ind}    fbwd_ci = fbwd_ib + tidx\n"
        f"{ind}    if fbwd_ci < {count}:\n"
        f"{ind}        fbwd_sSteps[fbwd_ci] = cutlass.Int32(_fbwd_walk_ws[fbwd_woff + {position}])\n"
    )


def _merged_prologue(plan: FlexBwdPlan, table_steps: int, walk_stride: int = 0) -> str:
    """The merged pair walk's prologue (all 512 threads, indent 4): per q
    head, the head's membership bitmaps, then its entries, L0's directly
    (this CTA's status from its own lists: the even tile's from the sublist,
    the odd tile's from its bitmaps) and the L1-only blocks compacted through
    warp ballots and the warp-count scan in ``fbwd_sScan``; ``fbwd_steps`` is
    the tile's step count.

    The pair's two KV tiles (``fbwd_pt0`` = the even tile, ``fbwd_pt1``) list
    the Q blocks they attend; the walk of q head g is L0_g (the even tile's
    full list, then its partial list) followed by the Q blocks of L1_g (the
    odd tile's lists) that L0_g does not hold, in L1_g's order, every block
    expanded to its ``q_block // 128`` tiles.

    The entries go to the step table itself when the whole walk fits it,
    else (``walk_stride`` > 0) to the CTA's region of the global scratch
    ``_fbwd_walk_ws`` (``fbwd_woff``), whose first ``table_steps`` entries
    the prologue then copies into the table: a chunked walk refills the
    table by the same copy (``_chunk_loop_2cta``), never by rebuilding the
    bitmaps. (The heads' entries interleaved by Q tile instead was measured
    and dropped: it triples the DRAM reads of the GQA-8 cells, since a Q / dO
    tile's reuse comes from neighbouring KV pairs reading it close in time.)

    A walk longer than its storage (the scratch region, or the table when
    the walk fits it) is cut there: the family sizes both for both tiles'
    lists end to end, which a BlockMask with disjoint full / partial lists
    never exceeds."""
    group = plan.group
    sub = plan.walk.q_block // 128
    nthreads, nwarps = 512, 16
    barrier = "cute.arch.sync_threads()"
    ind = "        "
    scratch = walk_stride > 0
    chunked = table_steps < 2 * max(plan.max_steps, 1)
    assert scratch or not chunked
    # Entries past ``cap`` are dropped; the step count is cut to what the
    # roles can read (the scratch when the walk runs in chunks, else the
    # table).
    cap = walk_stride if scratch else table_steps
    steps_cap = walk_stride if chunked else table_steps
    sink = (
        "_fbwd_walk_ws[fbwd_woff + {pos}] = fbwd_entry"
        if scratch
        else "fbwd_sSteps[{pos}] = fbwd_entry"
    )

    def store(ind: str, pos: str) -> str:
        return f"{ind}if {pos} < {cap}:\n{ind}    {sink.format(pos=pos)}\n"

    if sub > 1:
        split_a = (
            f"{ind}    fbwd_jb = fbwd_j // {sub}\n{ind}    fbwd_js = fbwd_j % {sub}\n"
        )
        expand_b = (
            f"{ind}        for fbwd_js in cutlass.range_constexpr({sub}):\n"
            f"{ind}            fbwd_e = fbwd_eb + fbwd_js\n"
            f"{ind}            fbwd_x = fbwd_xb * {sub} + fbwd_js\n"
            + _entry_bits(ind + "            ")
            + store(ind + "            ", "fbwd_e")
        )
    else:
        split_a = f"{ind}    fbwd_jb = fbwd_j\n{ind}    fbwd_js = cutlass.Int32(0)\n"
        expand_b = (
            f"{ind}        fbwd_x = fbwd_xb\n"
            + _entry_bits(ind + "        ")
            + store(ind + "        ", "fbwd_eb")
        )
    if scratch:
        # The first chunk (the whole walk when it fits) into the table.
        copy_in = (
            f"    fbwd_c0 = cutlass.min(fbwd_steps, cutlass.Int32({table_steps}))\n"
            + _walk_copy(
                plan, ind="    ", nthreads=nthreads, count="fbwd_c0", base=None
            )
            + f"    {barrier}\n"
        )
    else:
        copy_in = ""
    return (
        "    fbwd_lane = tidx % 32\n"
        "    fbwd_gbase = cutlass.Int32(0)\n"
        f"    for fbwd_g in cutlass.range({group}, unroll=1):\n"
        f"{ind}fbwd_th = {_table_col(plan, 'fbwd_g')}\n"
        + _head_counts(ind)
        + f"{ind}fbwd_nl0 = (fbwd_cf0 + fbwd_cp0) * {sub}\n"
        f"{ind}fbwd_nl1 = fbwd_cf1 + fbwd_cp1\n"
        + _bits_build(plan, ind, nthreads, barrier)
        + f"{ind}for fbwd_j in cutlass.range(tidx, fbwd_nl0, {nthreads}, unroll=1):\n"
        + split_a
        + f"{ind}    fbwd_xb = cutlass.Int32(0)\n"
        f"{ind}    fbwd_st = cutlass.Int32(2)\n"
        f"{ind}    if fbwd_jb < fbwd_cf0:\n"
        f"{ind}        fbwd_xb = {_list_load('_fbwd_full_idx', 'fbwd_pt0', 'fbwd_jb')}\n"
        f"{ind}    else:\n"
        f"{ind}        fbwd_xb = {_list_load('_fbwd_part_idx', 'fbwd_pt0', 'fbwd_jb - fbwd_cf0')}\n"
        f"{ind}        fbwd_st = cutlass.Int32(1)\n"
        f"{ind}    if fbwd_rank == 1:\n"
        f"{ind}        fbwd_xc = {_bit_index(plan, 'fbwd_xb')}\n"
        f"{ind}        fbwd_st = {_bit(plan, 1, 'fbwd_xc')} * 2 + {_bit(plan, 2, 'fbwd_xc')}\n"
        f"{ind}    fbwd_x = fbwd_xb * {sub} + fbwd_js\n"
        + _entry_bits(ind + "    ")
        + store(ind + "    ", "fbwd_gbase + fbwd_j")
        + f"{ind}fbwd_bbase = fbwd_gbase + fbwd_nl0\n"
        f"{ind}fbwd_run = cutlass.Int32(0)\n"
        f"{ind}fbwd_nblk = (fbwd_nl1 + {nthreads - 1}) // {nthreads}\n"
        f"{ind}for fbwd_blk in cutlass.range(fbwd_nblk, unroll=1):\n"
        f"{ind}    fbwd_j = fbwd_blk * {nthreads} + tidx\n"
        f"{ind}    fbwd_keep = cutlass.Boolean(False)\n"
        f"{ind}    fbwd_xb = cutlass.Int32(0)\n"
        f"{ind}    fbwd_st = cutlass.Int32(0)\n"
        f"{ind}    if fbwd_j < fbwd_nl1:\n"
        f"{ind}        if fbwd_j < fbwd_cf1:\n"
        f"{ind}            fbwd_xb = {_list_load('_fbwd_full_idx', 'fbwd_pt1', 'fbwd_j')}\n"
        f"{ind}            fbwd_st = cutlass.Int32(2)\n"
        f"{ind}        else:\n"
        f"{ind}            fbwd_xb = {_list_load('_fbwd_part_idx', 'fbwd_pt1', 'fbwd_j - fbwd_cf1')}\n"
        f"{ind}            fbwd_st = cutlass.Int32(1)\n"
        f"{ind}        fbwd_xc = {_bit_index(plan, 'fbwd_xb')}\n"
        f"{ind}        fbwd_keep = {_bit(plan, 0, 'fbwd_xc')} == 0\n"
        f"{ind}    fbwd_bal = cute.arch.vote_ballot_sync(fbwd_keep)\n"
        f"{ind}    if fbwd_lane == 0:\n"
        f"{ind}        fbwd_sScan[warp_idx] = cute.arch.popc(fbwd_bal)\n"
        f"{ind}    {barrier}\n"
        f"{ind}    fbwd_pre = cutlass.Int32(0)\n"
        f"{ind}    fbwd_tot = cutlass.Int32(0)\n"
        f"{ind}    for fbwd_w in cutlass.range_constexpr({nwarps}):\n"
        f"{ind}        fbwd_wc = cutlass.Int32(fbwd_sScan[fbwd_w])\n"
        f"{ind}        fbwd_tot = fbwd_tot + fbwd_wc\n"
        f"{ind}        if fbwd_w < warp_idx:\n"
        f"{ind}            fbwd_pre = fbwd_pre + fbwd_wc\n"
        f"{ind}    {barrier}\n"
        f"{ind}    if fbwd_keep:\n"
        f"{ind}        fbwd_eb = fbwd_bbase + (fbwd_run + fbwd_pre + cute.arch.popc(fbwd_bal & ((cutlass.Int32(1) << fbwd_lane) - 1))) * {sub}\n"
        f"{ind}        if fbwd_rank == 0:\n"
        f"{ind}            fbwd_st = cutlass.Int32(0)\n"
        + expand_b
        + f"{ind}    fbwd_run = fbwd_run + fbwd_tot\n"
        f"{ind}fbwd_gbase = fbwd_gbase + fbwd_nl0 + fbwd_run * {sub}\n"
        f"    fbwd_steps = cutlass.min(fbwd_gbase, cutlass.Int32({steps_cap}))\n"
        f"    {barrier}\n" + copy_in
    )


def _pair_counts_head(plan: FlexBwdPlan) -> str:
    """The pair's table coordinates and ``fbwd_any``, the total length of
    its lists (indent 4): non-zero exactly when the merged walk has steps, so
    the load warp issues the tile's K / V / K^T before the prologue."""
    walk = plan.walk
    tb = f"fbwd_b % {walk.table_batch}" if walk.table_batch > 1 else "cutlass.Int32(0)"
    return f"""    fbwd_tb = {tb}
    fbwd_pt0 = fbwd_n_pair * 2
    fbwd_pt1 = fbwd_pt0 + 1
    fbwd_any = cutlass.Int32(0)
    for fbwd_g in cutlass.range({plan.group}, unroll=1):
        fbwd_th = {_table_col(plan, "fbwd_g")}
{_head_counts("        ")}        fbwd_any = fbwd_any + fbwd_cf0 + fbwd_cp0 + fbwd_cf1 + fbwd_cp1
"""


# The chunk boundary barrier of a chunked step table (the 1-CTA kernel's):
# the 14 working warps; the relay warp 14 and the idle warp 15 never read it.
_FBWD2_CHUNK_BARRIER = "_helion_flash_rt.named_barrier_wait_unaligned(7, 448)"


def _chunk_loop_2cta(plan: FlexBwdPlan, table_steps: int, indent: int) -> str:
    """Head of the chunk loop of one warp role (see
    :func:`cute_flash_bwd_flex._chunk_loop`): the tile's steps run in chunks
    of ``table_steps``; every chunk after the first starts at the working
    warps' barrier, copies the chunk's entries from the walk's scratch into
    the table and meets at the barrier again."""
    pad = " " * indent
    copy = _walk_copy(
        plan,
        ind=pad + "        ",
        nthreads=448,
        count="fbwd_csteps",
        base=f"fbwd_chunk * {table_steps}",
    )
    return (
        f"{pad}fbwd_nchunks = (fbwd_steps + {table_steps - 1}) // {table_steps}\n"
        f"{pad}for fbwd_chunk in cutlass.range(fbwd_nchunks, unroll=1):\n"
        f"{pad}    fbwd_csteps = cutlass.min(fbwd_steps - fbwd_chunk * {table_steps}, cutlass.Int32({table_steps}))\n"
        f"{pad}    if fbwd_chunk > 0:\n"
        f"{pad}        {_FBWD2_CHUNK_BARRIER}\n"
        + copy
        + f"{pad}        {_FBWD2_CHUNK_BARRIER}\n"
    )


def _pair_tile_decode(plan: FlexBwdPlan, tile_group: int) -> str:
    """Map the cluster index to ``(fbwd_bhk, fbwd_n_pair)``: the pair's
    256-row KV tile of one (batch, kv head), in the order of
    :func:`cute_flash_bwd_flex.fbwd_tile_group_choices` over pairs."""
    n, bhk = plan.num_kv_tiles // 2, plan.batch * plan.kv_heads
    if tile_group < 2 or bhk % tile_group != 0:
        return (
            f"    fbwd_bhk = fbwd_tile_id // {n}\n"
            f"    fbwd_n_pair = fbwd_tile_id % {n}\n"
        )
    per_group = tile_group * n
    return (
        f"    fbwd_tgrp = fbwd_tile_id // {per_group}\n"
        f"    fbwd_trem = fbwd_tile_id % {per_group}\n"
        f"    fbwd_n_pair = fbwd_trem // {tile_group}\n"
        f"    fbwd_bhk = fbwd_tgrp * {tile_group} + fbwd_trem % {tile_group}\n"
    )


def _step_decode(plan: FlexBwdPlan, ind: str) -> str:
    """The step's Q tile and q head from its table entry (``fbwd_i``)."""
    return (
        f"{ind}fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i])\n"
        f"{ind}fbwd_m_tile = fbwd_e & 65535\n"
        f"{ind}fbwd_bhq = fbwd_bhk * {plan.group} + ((fbwd_e >> 16) & 255)\n"
    )


def _compute_block_2cta(
    *,
    h: int,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    out_scale_expr: str,
    exp2_f32: bool,
    plain: FlexHookCode | None,
    masked: FlexHookCode | None,
    jac: FlexHookCode | None,
    regs_compute: int,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    exp2_emu: int,
    chunked_region: Callable[[str, int], str] | None = None,
) -> str:
    """One compute warpgroup's body (h = column half 0/1 of the CTA's 128 x
    128 S^T / dP^T tile, warps 4-7 / 8-11): the 1-CTA flex passes with the
    dense 2-CTA kernel's P / dS staging. P packs into S columns 0-63 (half 1
    waits for half 0's S read: ``s_read``); dS^T packs into the dP columns
    for the pair's dK MMA and is staged in shared memory as the dQ A
    operand -- the Q half this CTA drains locally, the peer's half through
    the exchange buffer and a DSMEM bulk copy issued by the group's first
    thread. The ``ds_smem_full`` arrive of a step is deferred to the next
    step's S barrier (it then also certifies that S(i+1) left TMEM, whose
    columns 64-127 the dQ(i) MMA overwrites)."""
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    has_score = plan.score_region is not None and not bias_fold
    has_grad = plan.grad_region is not None
    pre_scaled = has_score or has_grad
    p_scale = "fbwd_log2c" if pre_scaled else "fbwd_scale2"
    hooks = [hook for hook in (plain, masked, jac) if hook is not None]
    item_block = _merge_blocks(*(hook.item_prelude for hook in hooks))
    tile_block = _merge_blocks(*(hook.tile_prologue for hook in hooks))
    per_step_item = plan.group > 1
    subs = 32 // hook_cols

    def chunk_prefix(ind: str) -> str:
        lines = [
            f"{ind}fbwd_ch2 = fbwd_ch * {subs} + fbwd_sub + {2 * subs * h}",
            f"{ind}fbwd_off = fbwd_ch * 32 + fbwd_sub * {hook_cols}",
            f"{ind}fbwd_s_chunk = cute.make_tensor(tLDrS[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
            f"{ind}fbwd_cchunk = cute.logical_divide(tLDcS[None, None, fbwd_ch], {hook_cols})[None, fbwd_sub]",
        ]
        if pre_scaled:
            lines.append(
                f"{ind}_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_s_chunk, fbwd_scale)"
            )
        if has_grad:
            lines.extend(
                (
                    f"{ind}fbwd_spre_chunk = cute.make_tensor(fbwd_spre[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
                    f"{ind}cute.autovec_copy(fbwd_s_chunk, fbwd_spre_chunk)",
                )
            )
        return "\n".join(lines) + "\n"

    def bias_lines(ind: str) -> str:
        if not bias_fold or plain is None:
            return ""
        return (
            f"{ind}fbwd_gfrag = cute.make_rmem_tensor(cute.make_layout({hook_cols}), cutlass.Float32)\n"
            + _lines(plain.chunk_body, len(ind))
            + f"{ind}_helion_flash_rt.fbwd_lse_fold(fbwd_lse_sub, fbwd_gfrag, fbwd_neg_log2c)\n"
        )

    def p_pass(hook: FlexHookCode | None, ind: str) -> str:
        return (
            f"{ind}for fbwd_ch in cutlass.range_constexpr(2):\n"
            f"{ind}    fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
            f"{ind}    fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
            f"{ind}    cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
            + (
                ""
                if plan.lse_prescaled
                else f"{ind}    _helion_flash_rt.fbwd_lse_to_log2(fbwd_lse_frg, fbwd_log2c)\n"
            )
            + f"{ind}    for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            + chunk_prefix(ind + "        ")
            + f"{ind}        fbwd_lse_sub = cute.make_tensor(fbwd_lse_frg.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            + bias_lines(ind + "        ")
            + (_lines(hook.chunk_body, len(ind) + 8) if hook is not None else "")
            + f"{ind}        _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_sub, fbwd_off, {hook_cols}, {p_scale}, None, fbwd_off, exp2_f32={exp2_f32}, emu_res={exp2_emu})\n"
        )

    unmasked_hook = None if bias_fold else plain
    if masked is not None:
        p_body = (
            "            if fbwd_masked != 0:\n"
            + p_pass(masked, " " * 16)
            + "            else:\n"
            + p_pass(unmasked_hook, " " * 16)
        )
    else:
        p_body = p_pass(unmasked_hook, " " * 12)
    jac_lines = ""
    if jac is not None:
        jac_lines = (
            f"                for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            f"                    fbwd_ch2 = fbwd_ch * {subs} + fbwd_sub + {2 * subs * h}\n"
            f"                    fbwd_spre_chunk = cute.make_tensor(fbwd_spre[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                    fbwd_cchunk = cute.logical_divide(tLDcS[None, None, fbwd_ch], {hook_cols})[None, fbwd_sub]\n"
            + _lines(jac.chunk_body, 20)
            + f"                    fbwd_ds_sub = cute.make_tensor(tLDrdP.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                    _helion_flash_rt.fbwd_mul_pairs_packed(fbwd_ds_sub, fbwd_spre, fbwd_ch * 32 + fbwd_sub * {hook_cols}, {hook_cols})\n"
        )
    spre_alloc = (
        "            fbwd_spre = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)\n"
        if has_grad
        else ""
    )
    pin_line = (
        "fbwd_m_tile_h = _helion_flash_rt.fbwd_pin_i32(fbwd_m_tile)"
        if hook_pin
        else "fbwd_m_tile_h = fbwd_m_tile"
    )
    item_lines = _lines(item_block, 12 if per_step_item else 8)
    tile_lines = _lines(tile_block, 12)
    item_once = "" if per_step_item else item_lines
    item_each = item_lines if per_step_item else ""

    def stage_lines(ind: str, dst_off: str, col: str) -> str:
        """cp.async staging of a step's LSE / delta columns (threads 0-63 the
        LSE, 64-127 the delta) into a parity slot."""
        return (
            f"{ind}if fbwd_local_tidx < 64:\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + ({dst_off} + fbwd_stat_j), _fbwd_mLSE.iterator + {col})\n"
            f"{ind}else:\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + ({dst_off} + fbwd_stat_j), _fbwd_mDelta.iterator + {col})\n"
            f"{ind}cute.arch.cp_async_commit_group()\n"
        )

    stage_first = (
        "        if fbwd_steps > 0:\n"
        "            fbwd_e0 = cutlass.Int32(fbwd_sSteps[fbwd_tbase])\n"
        f"            fbwd_bhq0 = fbwd_bhk * {plan.group} + ((fbwd_e0 >> 16) & 255)\n"
        f"            fbwd_stage_col0 = fbwd_bhq0 * {plan.q_seq} + cutlass.min((fbwd_e0 & 65535) * 128 + {64 * h} + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
        + stage_lines(" " * 12, f"(fbwd_gstep % 2) * 128 + {64 * h}", "fbwd_stage_col0")
    )
    stage_next = (
        "            if fbwd_i + 1 < fbwd_steps:\n"
        "                fbwd_e1 = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i + 1])\n"
        f"                fbwd_bhq1 = fbwd_bhk * {plan.group} + ((fbwd_e1 >> 16) & 255)\n"
        f"                fbwd_stage_col = fbwd_bhq1 * {plan.q_seq} + cutlass.min((fbwd_e1 & 65535) * 128 + {64 * h} + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
        + stage_lines(" " * 16, f"128 - fbwd_par + {64 * h}", "fbwd_stage_col")
    )
    # Half 0 publishes its S read (P of half 1 lands in its S columns);
    # half 1 waits for it before packing P.
    s_read_arrive = (
        "            cute.arch.sync_warp()\n"
        "            with cute.arch.elect_one():\n"
        "                _helion_flash_rt.mbarrier_arrive(fbwd_s_read_ptr)\n"
        if h == 0
        else ""
    )
    p_store_wait = (
        f"            {_WAIT.format(ptr='fbwd_s_read_ptr', phase='fbwd_phase')}\n"
        if h == 1
        else ""
    )
    epilogue = _fbwd_epilogue_block(
        h=h,
        rows=256,
        d=128,
        epi_off=str(_TMEM_DV if h == 0 else _TMEM_DK),
        epi_scale=(
            "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
            + out_scale_expr
            + "))"
            if h == 1
            else "pass"
        ),
        kv_tile="fbwd_n_tile",
        io_dtype=io_dtype,
        dkv_wait="0",
        v_empty_expr="storage.v_mbar_ptr.data_ptr() + 1",
        gm_expr=("_fbwd_mdVt" if h == 0 else "_fbwd_mdKt") + "[None, None, fbwd_bhk]",
        slots=2,
    ).replace("FBWD_KV_SLOT", "0")
    zero_target = "_fbwd_mdV" if h == 0 else "_fbwd_mdK"
    zero_store = f"""        fbwd_zrow = fbwd_kv_base + fbwd_local_tidx
        if fbwd_zrow < {plan.kv_seq}:
            fbwd_zeros = cute.make_rmem_tensor(cute.make_layout(128), {io_dtype})
            fbwd_zeros.fill({io_dtype}(0.0))
            fbwd_zdst = cute.make_tensor({zero_target}[None, None, fbwd_bhk][fbwd_zrow, None].iterator, cute.make_layout(128))
            cute.autovec_copy(fbwd_zeros, fbwd_zdst)
"""
    setup = f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        {_setmaxreg(regs_compute)}
        fbwd_s_shape = fbwd_sst.partition_shape_C((256, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        fbwd_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 64)))
        tS_h = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_S + 64 * h}, fbwd_half_layout)
        tdP_h = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DP + 64 * h}, fbwd_half_layout)
        # Coordinates of this half's columns within the 128-wide tile (the
        # hooks read the Q position off them); the rows are the pair's.
        cS = cute.domain_offset((0, {64 * h}), cute.make_identity_tensor((256, 128)))
        tScS = fbwd_sst.partition_C(cS)
        fbwd_cs_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 64)))
        tScS_h = cute.make_tensor(tScS.iterator, fbwd_cs_half_layout)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_h)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_h)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_h)
        tLDcS = fbwd_thr_ld.partition_D(tScS_h)
        fbwd_p_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 32)))
        tP_h = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_S + 32 * h}, fbwd_p_half_layout)
        tDS_h = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DP + 64 * h}, fbwd_p_half_layout)
        fbwd_cp_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 32)))
        tPcS_h = cute.make_tensor(tScS.iterator, fbwd_cp_half_layout)
        fbwd_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_st = cute_tcgen05_flash.make_tmem_copy(fbwd_st_atom, tP_h)
        fbwd_thr_st = fbwd_tiled_st.get_slice(fbwd_local_tidx)
        tSTtP = fbwd_thr_st.partition_D(tP_h)
        tSTtDS = fbwd_thr_st.partition_D(tDS_h)
        tSTcP = fbwd_thr_st.partition_S(tPcS_h)
        # dS staging: (q within chunk, chunk, KV half = CTA rank, KV row) of
        # the 64 x 256 dQ A operand; the exchange buffer holds the peer's Q half.
        fbwd_sdS_kv4 = cute.make_tensor(fbwd_sdS_mn.iterator, cute.make_layout((32, 2, 2, 128), stride=(1, 32, 8192, 64)))
        fbwd_sdSx_v2 = cute.make_tensor(fbwd_sdSx_ptr, cute.make_layout((32, 2, 128), stride=(1, 32, 64)))
        fbwd_crow = cutlass.Int32(tLDcS[0][0]) % 128
        fbwd_thr_row = fbwd_crow
        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_gstep = cutlass.Int32(0)
        fbwd_scale = cutlass.Float32({scale_expr})
        fbwd_log2c = cutlass.Float32({plan.log2_factor!r})
        fbwd_neg_log2c = cutlass.Float32({-plan.log2_factor!r})
        fbwd_scale2 = cutlass.Float32(({scale_expr}) * {plan.log2_factor!r})
        fbwd_stat_j = fbwd_local_tidx % 64
"""

    # The step's math and stores after the S load. The absent step (a Q tile
    # only the peer's lists hold) is its own branch with P = 0 and dS = 0
    # stored through the same barrier protocol: a per-element fill merged
    # into the common path became 64 CS2R + 64 SEL per warp-step and broke
    # the f32x2 register pairing of the packed math.
    def step_math(absent: bool) -> str:
        p = "            tLDrS.fill(cutlass.Float32(0.0))\n" if absent else p_body
        ds_math = (
            "                tLDrdP.fill(cutlass.Float32(0.0))\n"
            if absent
            else (
                "                cute.copy(fbwd_tiled_ld, tLDtdP[None, None, fbwd_ch], tLDrdP)\n"
                "                fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
                f"                fbwd_dlt_v = cute.make_tensor(fbwd_sDelta.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
                "                cute.autovec_copy(fbwd_dlt_v, fbwd_dlt_frg)\n"
                "                cute.arch.fence_view_async_tmem_load()\n"
                "                _helion_flash_rt.fbwd_ds_pairs_packed(tLDrdP, tLDrS, fbwd_ch * 32, fbwd_dlt_frg, 32)\n"
                + jac_lines
            )
        )
        return f"""{p}            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
            tSTrP_e = cute.make_tensor(cute.recast_ptr(tSTrP.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrP_e.store(tLDrS.load().to({io_dtype}))
{p_store_wait}            cute.copy(fbwd_tiled_st, tSTrP, tSTtP)
            cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_p_full_ptr, cutlass.Int32(0))
            {_WAIT.format(ptr="fbwd_dp_full_ptr", phase="fbwd_phase")}
            for fbwd_ch in cutlass.range_constexpr(2):
                tLDrdP = cute.make_rmem_tensor(tLDcS[None, None, 0].shape, cutlass.Float32)
{ds_math}                fbwd_ds_bf = cute.make_rmem_tensor(tLDrdP.layout, {io_dtype})
                fbwd_ds_bf.store(tLDrdP.load().to({io_dtype}))
                fbwd_ds_w = cute.make_tensor(cute.recast_ptr(fbwd_ds_bf.iterator, dtype=cutlass.Float32), cute.make_layout(tSTcP[None, None, 0].shape))
                cute.copy(fbwd_tiled_st, fbwd_ds_w, tSTtDS[None, None, fbwd_ch])
                fbwd_ds_flat = cute.make_tensor(fbwd_ds_bf.iterator, cute.make_layout(32))
                if fbwd_ch == 0:
                    # The staging / exchange buffers are dQ(i-1)'s operands.
                    if fbwd_gstep > 0:
                        {_WAIT.format(ptr="fbwd_dq_full_ptr", phase="fbwd_dqf_phase")}
                        fbwd_dqf_phase ^= 1
                if fbwd_rank == {h}:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_kv4[None, fbwd_ch, fbwd_rank, fbwd_crow])
                else:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdSx_v2[None, fbwd_ch, fbwd_crow])
            cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_ds_tmem_full_ptr, cutlass.Int32(0))
            cute.arch.fence_view_async_shared()
            # The group holding the peer's Q half publishes the exchange
            # buffer once all of its lanes wrote it.
            if fbwd_rank != {h}:
                _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
{_FBWD_2CTA_SEND}
"""

    steps_region = f"""{stage_first}        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i])
            fbwd_m_tile = fbwd_e & 65535
            fbwd_gi = (fbwd_e >> 16) & 255
            fbwd_masked = (fbwd_e >> 24) & 1
            fbwd_absent = (fbwd_e >> 25) & 1
            fbwd_bhq = fbwd_bhk * {plan.group} + fbwd_gi
            flash_bh = fbwd_bhq
{item_each}            fbwd_par = (fbwd_gstep % 2) * 128
            {_WAIT.format(ptr="fbwd_s_full_ptr", phase="fbwd_phase")}
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
{spre_alloc}            cute.copy(fbwd_tiled_ld, tLDtS, tLDrS)
            cute.arch.fence_view_async_tmem_load()
{s_read_arrive}            cute.arch.cp_async_wait_group(0)
            _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
            if fbwd_i > 0:
                with cute.arch.elect_one():
                    _helion_flash_rt.mbarrier_arrive(fbwd_ds_smem_full_ptr, cutlass.Int32(0))
            {pin_line}
{tile_lines}{stage_next}            if fbwd_absent != 0:
{textwrap.indent(step_math(True), "    ")}            else:
{textwrap.indent(step_math(False), "    ")}            fbwd_phase ^= 1
            fbwd_gstep = fbwd_gstep + 1
        if fbwd_steps > 0:
            _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_ds_smem_full_ptr, cutlass.Int32(0))
"""
    # The last step's deferred ds_smem_full arrive closes every chunk (the
    # leader's tail dQ waits for it before the chunk barrier).
    if chunked_region is not None:
        steps_region = chunked_region(steps_region, 8)
    tile_region = f"""        flash_bh = fbwd_bhk * {plan.group}
{item_once}{steps_region}        if fbwd_steps > 0:
{textwrap.indent(epilogue.rstrip(), "    ")}
        else:
{textwrap.indent(zero_store, "    ")}"""
    return (
        setup
        + tile_region
        + "        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)\n"
    )


def emit_flash_bwd_flex_2cta_device_body(
    *,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    exp2_f32: bool,
    mask_form: str,
    bindings: list[str],
    device_function: DeviceFunction,
    reg_split: str | None,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    tile_group: int,
    exp2_emu: int,
    table_steps: int,
    walk_stride: int = 0,
) -> str:
    """The 2-CTA flex backward device body (see the module docstring);
    ``walk_stride`` > 0 binds the walk scratch ``_fbwd_walk_ws`` (that many
    entries per CTA), required by a chunked walk."""
    assert plan.head_dim == 128 and plan.head_dim_v == 128
    assert plan.operand_strides is None
    assert hook_cols in (8, 16, 32)
    d = 128
    # A walk longer than the table runs in chunks of ``table_steps``: every
    # role's step region is wrapped in the chunk loop, the table refilled at
    # the working warps' barrier between chunks (``_chunk_loop_2cta``).
    chunked = table_steps < 2 * max(plan.max_steps, 1)
    assert walk_stride > 0 or not chunked
    walk_offset = (
        f"    fbwd_woff = cutlass.Int32(cute.arch.block_idx()[0]) * {walk_stride}\n"
        if walk_stride > 0
        else ""
    )

    def chunked_region(text: str, indent: int) -> str:
        if not chunked:
            return text
        return _chunk_loop_2cta(plan, table_steps, indent) + textwrap.indent(
            re.sub(r"\bfbwd_steps\b", "fbwd_csteps", text), "    "
        )

    bias = additive_bias_region(plan) if bias_fold else None
    _hooks, plain, masked, jac = _render_hooks(
        plan,
        device_function=device_function,
        mask_form=mask_form,
        hook_cols=hook_cols,
        bias=bias,
        aux_smem_budget=0,
    )
    regs_reduce, regs_compute, regs_other = _reg_split(reg_split)
    binding_lines = "".join(f"    {line}\n" for line in bindings)
    compute_blocks = "".join(
        _compute_block_2cta(
            h=h,
            plan=plan,
            io_dtype=io_dtype,
            scale_expr=scale_expr,
            out_scale_expr=scale_expr,
            exp2_f32=exp2_f32,
            plain=plain,
            masked=masked,
            jac=jac,
            regs_compute=regs_compute,
            hook_cols=hook_cols,
            hook_pin=hook_pin,
            bias_fold=bias is not None,
            exp2_emu=exp2_emu,
            chunked_region=chunked_region,
        )
        for h in (0, 1)
    )
    step_decode = _step_decode(plan, " " * 16)
    prefetch = "".join(
        f"        cute_cpasync_flash.prefetch_descriptor({atom})\n"
        for atom in (
            "_fbwd_tma_q",
            "_fbwd_tma_k",
            "_fbwd_tma_v",
            "_fbwd_tma_dot",
            "_fbwd_tma_do2",
            "_fbwd_tma_qt",
            "_fbwd_tma_kt",
            "_fbwd_tma_dq",
            "_fbwd_tma_dv",
            "_fbwd_tma_dk",
        )
    )
    pipeline = (
        "cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages=1, "
        "producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), "
        "consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), "
        "tx_count=fbwd_{bytes}_bytes, barrier_storage=storage.{name}_mbar_ptr.data_ptr(), "
        "cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()"
    )
    pipelines = "".join(
        f"    fbwd_{name}_prod, fbwd_{name}_cons = {pipeline.format(name=name, bytes=nbytes)}\n"
        for name, nbytes in (
            ("q", "q"),
            ("qt", "q"),
            ("kt", "kt"),
            ("k", "k"),
            ("v", "k"),
            ("do", "do"),
        )
    )
    mbar_inits = "".join(
        f"        cute.arch.mbarrier_init(fbwd_{name}_ptr, {count})\n"
        for name, count in (
            ("s_full", 1),
            ("dp_full", 1),
            ("p_full", 16),
            ("ds_tmem_full", 16),
            ("ds_smem_full", 16),
            ("dq_full", 1),
            ("dq_empty", 8),
            ("dkv_done", 1),
            ("dv_done", 1),
            ("ds_cl_full", 1),
            ("ds_cl_leader", 2),
            ("s_read", 4),
        )
    )
    mbar_ptrs = "".join(
        f"    fbwd_{name}_ptr = storage.{field}.data_ptr()\n"
        for name, field in (
            ("s_full", "s_full_mbar"),
            ("dp_full", "dp_full_mbar"),
            ("p_full", "p_full_mbar"),
            ("ds_tmem_full", "ds_tmem_full_mbar"),
            ("ds_smem_full", "ds_smem_full_mbar"),
            ("dq_full", "dq_full_mbar"),
            ("dq_empty", "dq_empty_mbar"),
            ("dkv_done", "dkv_done_mbar"),
            ("dv_done", "dv_done_mbar"),
            ("ds_cl_full", "ds_cluster_full_mbar"),
            ("ds_cl_leader", "ds_cluster_leader_mbar"),
            ("s_read", "s_read_mbar"),
            ("tmem_dealloc", "tmem_dealloc_mbar"),
        )
    )
    reloads = "".join(
        f"{{ind}}fbwd_st_{name} = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + {i})\n"
        for i, name in enumerate(("k", "v", "dsmn", "q", "dot", "do", "qt", "kt"))
    )
    bases = "".join(
        f"            fbwd_mb[{i}] = _helion_flash_ptx.make_smem_desc_start_addr({tensor}[None, None, None, 0].iterator)\n"
        for i, tensor in enumerate(
            ("sK", "sV", "fbwd_sdS_mn", "sQ", "sdOt", "sdO", "sQt", "sKt")
        )
    )
    s_gemm = (
        "_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, "
        'fbwd_st_q, fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)'
    )
    dp_gemm = (
        "_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, "
        'fbwd_st_dot, fbwd_dot_base, tSrDOt[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True, cta_group=2)'
    )

    def dv_gemm(zero: str) -> str:
        return (
            "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, fbwd_st_do, fbwd_do2_base, "
            'tTSrP[None, None, None, 0].layout, tTSrDO[None, None, None, 0].layout, "helion_fbwd_ts_idesc", '
            f"zero_init={zero}, cta_group=2)"
        )

    dk_gemm = (
        "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, fbwd_st_qt, fbwd_qt_base, "
        'tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", '
        f"zero_init=fbwd_dk_zero, cta_group=2, a_offsets={_FBWD_2CTA_DS_OFFSETS})"
    )
    dq_gemm = (
        "_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, "
        'fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True, cta_group=2)'
    )

    def commit(ind: str, ptr: str) -> str:
        return ind + _COMMIT.format(ind=ind, ptr=ptr) + "\n"

    def wait(ind: str, ptr: str, phase: str) -> str:
        return f"{ind}{_WAIT.format(ptr=ptr, phase=phase)}\n{ind}{phase} ^= 1\n"

    i12, i16 = " " * 12, " " * 16
    # The MMA schedule of the dense 2-CTA kernel: S(i+1), dK(i), dP(i+1),
    # dQ(i), dV(i+1) per iteration; S(i+1) waits for dQ(i-1) to leave the S
    # columns it aliases, dQ(i) for both CTAs' dS (local staging through the
    # deferred ds_smem_full, the exchanged half through the relay warp).
    # A chunk's first S overwrites the previous chunk's tail dQ columns; the
    # tail dK of a one-step chunk never reaches the loop's dk_zero reset; dV
    # completes after the last chunk's loop.
    peel_wait = (
        f"{i12}if fbwd_chunk > 0:\n"
        + wait(i12 + "    ", "fbwd_dq_empty_ptr", "fbwd_dqe_phase")
        if chunked
        else ""
    )
    dk_zero_after_tail = (
        f"{i12}fbwd_dk_zero = cutlass.Boolean(False)\n" if chunked else ""
    )
    dv_done_commit = (
        f"{i12}if fbwd_chunk == fbwd_nchunks - 1:\n"
        + commit(i12 + "    ", "fbwd_dv_done_ptr")
        if chunked
        else commit(i12, "fbwd_dv_done_ptr")
    )
    mma_steps = (
        f"{reloads.format(ind=i12)}"
        + peel_wait
        + f"{i12}fbwd_qe = fbwd_q_cons.wait_and_advance()\n"
        f"{i12}{s_gemm}\n"
        + commit(i12, "fbwd_s_full_ptr")
        + f"{i12}fbwd_qe.release()\n"
        f"{i12}fbwd_doe = fbwd_do_cons.wait_and_advance()\n"
        f"{i12}{dp_gemm}\n"
        + commit(i12, "fbwd_dp_full_ptr")
        + wait(i12, "fbwd_p_full_ptr", "fbwd_pf_phase")
        + f"{i12}{dv_gemm('fbwd_dv_zero')}\n"
        f"{i12}fbwd_dv_zero = cutlass.Boolean(False)\n"
        f"{i12}fbwd_doe.release()\n"
        f"{i12}for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):\n"
        f"{reloads.format(ind=i16)}"
        f"{i16}if fbwd_i > 0:\n"
        + wait(i16 + "    ", "fbwd_dq_empty_ptr", "fbwd_dqe_phase")
        + f"{i16}fbwd_qe = fbwd_q_cons.wait_and_advance()\n"
        f"{i16}{s_gemm}\n"
        + commit(i16, "fbwd_s_full_ptr")
        + f"{i16}fbwd_qe.release()\n"
        f"{i16}fbwd_qte = fbwd_qt_cons.wait_and_advance()\n"
        + wait(i16, "fbwd_ds_tmem_full_ptr", "fbwd_dst_phase")
        + f"{i16}{dk_gemm}\n"
        f"{i16}fbwd_dk_zero = cutlass.Boolean(False)\n"
        f"{i16}fbwd_qte.release()\n"
        f"{i16}fbwd_doe = fbwd_do_cons.wait_and_advance()\n"
        f"{i16}{dp_gemm}\n"
        + commit(i16, "fbwd_dp_full_ptr")
        + wait(i16, "fbwd_ds_smem_full_ptr", "fbwd_dss_phase")
        + wait(i16, "fbwd_ds_cl_leader_ptr", "fbwd_dcl_phase")
        + f"{i16}{dq_gemm}\n"
        + commit(i16, "fbwd_dq_full_ptr")
        + wait(i16, "fbwd_p_full_ptr", "fbwd_pf_phase")
        + f"{i16}{dv_gemm('False')}\n"
        f"{i16}fbwd_doe.release()\n"
        f"{reloads.format(ind=i12)}"
        + dv_done_commit
        + f"{i12}fbwd_qte = fbwd_qt_cons.wait_and_advance()\n"
        + wait(i12, "fbwd_ds_tmem_full_ptr", "fbwd_dst_phase")
        + f"{i12}{dk_gemm}\n"
        f"{i12}fbwd_qte.release()\n"
        + dk_zero_after_tail
        + f"{i12}if fbwd_steps > 1:\n"
        + wait(i12 + "    ", "fbwd_dq_empty_ptr", "fbwd_dqe_phase")
        + wait(i12, "fbwd_ds_smem_full_ptr", "fbwd_dss_phase")
        + wait(i12, "fbwd_ds_cl_leader_ptr", "fbwd_dcl_phase")
        + f"{i12}{dq_gemm}\n"
        + commit(i12, "fbwd_dq_full_ptr")
    )
    mma_tail = (
        commit(i12, "fbwd_dkv_done_ptr") + f"{i12}fbwd_ke.release()\n"
        f"{i12}fbwd_kte.release()\n"
    )
    mma_steps_region = chunked_region(mma_steps, 12)
    # The chunk barrier counts the 14 working warps of each CTA: the
    # follower's MMA warp issues nothing but walks the chunk loop (its
    # barriers and refills) with the others.
    follower_chunks = (
        "        if (fbwd_rank != 0) & (fbwd_steps > 0):\n" + chunked_region("", 12)
        if chunked
        else ""
    )
    mma_region = f"""        if fbwd_is_leader & (fbwd_steps > 0):
            fbwd_s_shape = fbwd_sst.partition_shape_C((256, 128))
            tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
            fbwd_dkv_shape = fbwd_tst.partition_shape_C((256, {d}))
            tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
            fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
            tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
            tS_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_S}, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DP}, tStS_frag.layout)
            tDV_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DV}, tDKV_frag.layout)
            tDK_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DK}, tDKV_frag.layout)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DQ}, tDQ_frag.layout)
            tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
            tDS = cute.make_tensor(tdP_t.iterator, _fbwd_ptl.outer)
            tSrK = fbwd_sst.make_fragment_A(sK)
            tSrQ = fbwd_sst.make_fragment_B(sQ)
            tSrV = fbwd_sst.make_fragment_A(sV)
            tSrDOt = fbwd_sst.make_fragment_B(sdOt)
            tTSrP = fbwd_tst.make_fragment_A(tP)
            tTSrDS = fbwd_tst.make_fragment_A(tDS)
            tTSrDO = fbwd_tst.make_fragment_B(sdO)
            tTSrQt = fbwd_tst.make_fragment_B(sQt)
            tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
            tDQrKt = fbwd_dqt.make_fragment_B(sKt)
            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dv_addr = tDV_t.iterator.toint()
            fbwd_dk_addr = tDK_t.iterator.toint()
            fbwd_dqa_addr = tDQ_t.iterator.toint()
            fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
            fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
            fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
            fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.K)
            fbwd_do2_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.MN)
            fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ts_mma.op, "helion_fbwd_ts_idesc")
            fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
            fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
            # Operand start addresses parked in smem; re-read with a volatile
            # load per iteration so the per-k descriptors are never hoisted.
            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
{bases}            fbwd_pf_phase = cutlass.Int32(0)
            fbwd_dst_phase = cutlass.Int32(0)
            fbwd_dss_phase = cutlass.Int32(0)
            fbwd_dcl_phase = cutlass.Int32(0)
            fbwd_dqe_phase = cutlass.Int32(0)
            fbwd_dv_zero = cutlass.Boolean(True)
            fbwd_dk_zero = cutlass.Boolean(True)
            fbwd_ke = fbwd_k_cons.wait_and_advance()
            # V and K^T are loaded once per tile with K: their waits stay
            # outside the (chunked) step region.
            fbwd_ve = fbwd_v_cons.wait_and_advance()
            fbwd_kte = fbwd_kt_cons.wait_and_advance()
{mma_steps_region}{mma_tail}{follower_chunks}"""
    reduce_boxes = "".join(
        f"""
            fbwd_dq_flat = cute.make_tensor(tRDrDQ[None, {c}, 0, 0].iterator, cute.make_layout(32))
            fbwd_sdq_box = fbwd_sdq[None, None, warp_idx]
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            cute.autovec_copy(fbwd_dq_flat, fbwd_sdq_box[fbwd_my_row, None])
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            fbwd_gdq_box = fbwd_gdq[None, None, fbwd_gdq_row, fbwd_my_c0 + {c}]
            tDQsDQ, tDQgDQ = cute_cpasync_flash.tma_partition(_fbwd_tma_dq, 0, cute.make_layout(1), cute.group_modes(fbwd_sdq_box, 0, 2), cute.group_modes(fbwd_gdq_box, 0, 2))
            cute.copy(_fbwd_tma_dq, tDQsDQ, tDQgDQ)
            cute.arch.cp_async_bulk_commit_group()"""
        for c in range(2)
    )
    # The pair's M=128 dQ accumulator: this CTA's 64 Q rows as 128 lanes x
    # 64 columns (the partition's row / first-column coordinates say which).
    reduce_region = f"""        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{_step_decode(plan, " " * 12)}            fbwd_gdq_row = fbwd_m_tile * 4 + fbwd_my_m // 32
            fbwd_gdq = cute.local_tile(_fbwd_mDQt[None, None, fbwd_bhq], (32, 32), (None, None))
            {_WAIT.format(ptr="fbwd_dq_full_ptr", phase="fbwd_phase")}
            fbwd_phase ^= 1
            tRDrDQ = cute.make_rmem_tensor(tRDcDQ.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_dq_empty_ptr, cutlass.Int32(0))
            _helion_flash_rt._scale_fragment_packed_f32x2(tRDrDQ, fbwd_out_scale){reduce_boxes}
"""
    reduce_region = chunked_region(reduce_region, 8)
    tma_loop = f"""            for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{step_decode}                tQgQ = tQgQ_qdl[None, None, 0, fbwd_bhq]
                tDOTgDOT = tDOTgDOT_qdl[None, None, 0, fbwd_bhq]
                tDOgDO = tDOgDO_qdl[None, 0, None, fbwd_bhq]
                tQTgQT = tQTgQT_qdl[None, 0, None, fbwd_bhq]
                fbwd_qe = fbwd_q_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_m_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
                fbwd_doe = fbwd_do_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_dot, tDOTgDOT[None, fbwd_m_tile], tDOTsDOT[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
                cute.copy(_fbwd_tma_do2, tDOgDO[None, fbwd_m_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
                fbwd_qte = fbwd_qt_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_qt, tQTgQT[None, fbwd_m_tile], tQTsQT[None, fbwd_qte.index], tma_bar_ptr=fbwd_qte.barrier)
"""
    tma_steps = (
        "        if fbwd_steps > 0:\n"
        + chunked_region(tma_loop, 12)
        + """            fbwd_q_prod.tail()
            fbwd_qt_prod.tail()
            fbwd_do_prod.tail()
            fbwd_k_prod.tail()
            fbwd_v_prod.tail()
            fbwd_kt_prod.tail()
"""
    )
    body = f"""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    fbwd_local_tidx = tidx % 128
    fbwd_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    fbwd_is_leader = fbwd_rank == cutlass.Int32(0)
    fbwd_tile_id = cutlass.Int32(cute.arch.block_idx()[0]) // 2
{_pair_tile_decode(plan, tile_group)}    fbwd_b = fbwd_bhk // {plan.kv_heads}
    fbwd_hk = fbwd_bhk % {plan.kv_heads}
    fbwd_n_tile = fbwd_n_pair * 2 + fbwd_rank
    fbwd_kv_base = fbwd_n_tile * 128
    fbwd_tbase = cutlass.Int32(0)
{walk_offset}{binding_lines}    if warp_idx == 0:
{prefetch}    _fbwd_storage_cls = _helion_flash_rt.flash_bwd_flex_2cta_shared_storage({io_dtype}, {table_steps}, {_bits_words(plan)})
    smem = cutlass_utils_flash.SmemAllocator()
    storage = smem.allocate(_fbwd_storage_cls)
    fbwd_sSteps = storage.sSteps.get_tensor(cute.make_layout({table_steps}))
    fbwd_sScan = storage.sScan.get_tensor(cute.make_layout(16))
    fbwd_sBits = storage.sBits.get_tensor(cute.make_layout({3 * _bits_words(plan)}))
    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdOt = storage.sdOt.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dotl.outer, swizzle=_fbwd_dotl.inner)
    sQt = storage.sQt.get_tensor(_fbwd_qtl.outer, swizzle=_fbwd_qtl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sKt = storage.sKt.get_tensor(_fbwd_ktl.outer, swizzle=_fbwd_ktl.inner)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdSx_plain = storage.sdSx.get_tensor(cute.make_layout(8192))
    fbwd_sdSx_ptr = cute.recast_ptr(fbwd_sdSx_plain.iterator, _fbwd_dssl.inner)
    fbwd_sLSE = storage.sLSE.get_tensor(cute.make_layout(2 * 128))
    fbwd_sDelta = storage.sDelta.get_tensor(cute.make_layout(2 * 128))
    fbwd_sdq = storage.sdQaccum.get_tensor(_fbwd_dqsl.outer, swizzle=_fbwd_dqsl.inner)
{mbar_ptrs}    if tidx == 0:
{mbar_inits}    cute.arch.mbarrier_init_fence()
    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
    # Every full barrier counts both CTAs' halves of a tile.
    fbwd_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_qsl, mode=[0, 1, 2])) * 2
    fbwd_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ksl, mode=[0, 1, 2])) * 2
    fbwd_kt_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ktl, mode=[0, 1, 2])) * 2
    fbwd_do_bytes = fbwd_q_bytes * 2
{pipelines}    cutlass_pipeline_flash.pipeline_init_arrive(cluster_shape_mn=_fbwd_cluster_vmnk, is_relaxed=True)
    cutlass_pipeline_flash.pipeline_init_wait(cluster_shape_mn=_fbwd_cluster_vmnk)
    fbwd_sst = _fbwd_ss_mma.get_slice(fbwd_rank)
    fbwd_tst = _fbwd_ts_mma.get_slice(fbwd_rank)
    fbwd_dqt = _fbwd_dq_mma.get_slice(fbwd_rank)
    fbwd_mcast = cutlass_pipeline_flash.PipelineUmmaAsync._compute_tmem_sync_mask(_fbwd_cluster_vmnk)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((256, 128, {d}), mode=[0, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((256, 128, {d}), mode=[0, 2]))
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((256, 128, {d}), mode=[1, 2]))
    gdOt = cute.flat_divide(_fbwd_mdOtN, cute.select((256, 128, {d}), mode=[1, 2]))
    gdO = cute.flat_divide(_fbwd_mdOtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gQt = cute.flat_divide(_fbwd_mQtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gKt = cute.flat_divide(_fbwd_mKtT, cute.select((128, {d}, 256), mode=[1, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tSgV = fbwd_sst.partition_A(gV)
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgDOt = fbwd_sst.partition_B(gdOt)
    tTSgDO = fbwd_tst.partition_B(gdO)
    tTSgQt = fbwd_tst.partition_B(gQt)
    tDQgKt = fbwd_dqt.partition_B(gKt)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tDOTsDOT, tDOTgDOT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_dot, 0, cute.make_layout(1), cute.group_modes(sdOt, 0, 3), cute.group_modes(tSgDOt, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do2, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tTSgDO, 0, 3))
    tQTsQT, tQTgQT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_qt, 0, cute.make_layout(1), cute.group_modes(sQt, 0, 3), cute.group_modes(tTSgQt, 0, 3))
    tKTsKT, tKTgKT_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_kt, 0, cute.make_layout(1), cute.group_modes(sKt, 0, 3), cute.group_modes(tDQgKt, 0, 3))
{_pair_counts_head(plan)}    if warp_idx == 13:
        # The tile's K / V / K^T operands need only the tile coordinates:
        # issued before the merged walk is built so they land while it is
        # (an empty pair stores zeros and exits without them).
        if fbwd_any > 0:
            fbwd_ke = fbwd_k_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_k, tKgK_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_pair], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
            fbwd_ve = fbwd_v_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_v, tVgV_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_pair], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
            fbwd_kte = fbwd_kt_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_kt, tKTgKT_kdl[None, 0, None, fbwd_bhk][None, fbwd_n_pair], tKTsKT[None, fbwd_kte.index], tma_bar_ptr=fbwd_kte.barrier)
{_merged_prologue(plan, table_steps, walk_stride)}    if (warp_idx >= 12) & (warp_idx < 16):
        {_setmaxreg(regs_other)}
    if warp_idx == 14:
        # Relay: the peer's exchanged dS half landed here -> the leader's
        # dQ barrier (both CTAs' relays arrive on it).
        if fbwd_steps > 0:
            fbwd_rphase = cutlass.Int32(0)
            for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
                {_WAIT.format(ptr="fbwd_ds_cl_full_ptr", phase="fbwd_rphase")}
                fbwd_rphase ^= 1
                with cute.arch.elect_one():
                    _helion_flash_rt.mbarrier_arrive(fbwd_ds_cl_leader_ptr, cutlass.Int32(0))
    if warp_idx == 13:
{tma_steps}    if warp_idx == 12:
        fbwd_tmem.allocate(512)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
{mma_region}        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
    if warp_idx < 4:
        {_setmaxreg(regs_reduce)}
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {_TMEM_DQ}, tDQ_frag.layout)
        cDQ = cute.make_identity_tensor((128, {d}))
        tDQcDQ = fbwd_dqt.partition_C(cDQ)
        fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
        fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
        tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
        fbwd_out_scale = cutlass.Float32({scale_expr})
        fbwd_my_m = cutlass.Int32(tRDcDQ[0][0])
        fbwd_my_row = fbwd_my_m % 32
        fbwd_my_c0 = cutlass.Int32(tRDcDQ[0][1]) // 32
        fbwd_phase = cutlass.Int32(0)
{reduce_region}        cute.arch.cp_async_bulk_wait_group(0, read=True)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""
    return body  # noqa: RET504


# The once-per-module import of the walk scratch's allocator (the host
# expression of the scratch argument runs inside the generated launch).
_FLASH_FLEX_BWD_WALK_PREAMBLE_IMPORTS = """\
from helion.runtime.cute.flash_flex import flex_bwd_walk_scratch as _helion_flash_flex_bwd_walk_scratch
"""


def _emit_walk_scratch_module_statements(cg: GenerateAST) -> None:
    import ast as ast_module

    if getattr(cg, "_helion_flash_flex_bwd_walk_module_emitted", False):
        return
    cg._helion_flash_flex_bwd_walk_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast_module.parse(_FLASH_FLEX_BWD_WALK_PREAMBLE_IMPORTS).body:
        cg.module_statements.append(stmt)


def _walk_stride(plan: FlexBwdPlan, args: dict[str, TensorArg]) -> int:
    """Entries of one CTA's region of the walk scratch: both tiles' lists of
    every q head of the group end to end (the step table's bound of the
    merged walk, ``2 * max_steps``), over the wider of the kernel's Q blocks
    and the index tables' (a BlockMask built for a longer Q lists blocks past
    the sequence; they walk as steps that touch no valid row)."""
    walk = plan.walk
    width = max(
        _q_blocks(plan),
        *(
            int(args[name].fake_value.shape[-1])
            for name in (walk.full_idx_name, walk.part_idx_name)
        ),
    )
    return 2 * plan.group * width * (walk.q_block // 128)


def _walk_scratch_arg(
    df: DeviceFunction, like_arg: TensorArg, *, ctas: int, stride: int
) -> TensorArg:
    """Bind the walk scratch (``flex_bwd_walk_scratch``: ``ctas * stride``
    int32 entries) as a plain kernel param read by name in the device body;
    the host expression runs inside the generated launch on every call (a
    cached allocation per device and stream)."""
    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg

    env = CompileEnvironment.current()
    with env.fake_mode:
        fake = torch.empty((ctas * stride,), dtype=torch.int32, device=env.device)
    arg = TensorArg(
        df.new_var("fbwd_walk_ws"),
        fake,
        f"_helion_flash_flex_bwd_walk_scratch({like_arg.host_str()}, "
        f"ctas={ctas}, stride={stride})",
    )
    df.arguments.append(arg)
    df._tensor_args[fake] = arg
    return arg


def codegen_flex_bwd_2cta(
    cg: GenerateAST,
    match: FlexBwdMatch,
    *,
    args: dict[str, TensorArg],
    scalar_exprs: list[str],
    scale_expr: str,
    io_dtype_str: str,
    exp2_f32: bool,
    mask_form: str,
    reg_split: object,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    tile_group: int,
) -> bool:
    """Emit the 2-CTA kernel (the arguments were validated by
    :func:`cute_flash_bwd_flex.codegen_attention_flash_bwd_flex`; the caller
    checked :func:`flex_bwd_two_cta_allowed`)."""
    import ast as ast_module

    from ..compile_environment import CompileEnvironment
    from .cute_flash import emit_flash_module_statements

    df = cg.device_function
    plan = match.plan
    env = CompileEnvironment.current()
    batch, heads, kv_heads = plan.batch, plan.heads, plan.kv_heads
    exp2_emu_cfg = df.config.config.get("cute_flash_bwd_exp2_emu", 0)
    exp2_emu = (
        exp2_emu_cfg if isinstance(exp2_emu_cfg, int) and env.settings.fast_math else 0
    )
    # Steps per step table (the ``cute_flash_bwd_table_steps`` knob caps it;
    # a longer walk runs in chunks).
    table_cap_cfg = df.config.config.get("cute_flash_bwd_table_steps", 0)
    table_cap = table_cap_cfg if isinstance(table_cap_cfg, int) else 0
    table_steps = flex_bwd_two_cta_table_steps(plan, env.device, table_cap)
    assert table_steps >= 1, table_steps
    walk = plan.walk
    emit_flash_module_statements(cg)
    bindings = [
        f"_flash_heads = {heads}",
        f"_fbwd_part_cnt = {args[walk.part_cnt_name].name}",
        f"_fbwd_part_idx = {args[walk.part_idx_name].name}",
        f"_fbwd_full_cnt = {args[walk.full_cnt_name].name}",
        f"_fbwd_full_idx = {args[walk.full_idx_name].name}",
    ]
    # The walk scratch: a chunked walk (the table holds less than the whole
    # merged walk) reads the CTA's walk back from global memory; a walk that
    # fits the table is written to it directly.
    total_ctas = batch * kv_heads * plan.num_kv_tiles
    walk_stride = 0
    if table_steps < 2 * max(plan.max_steps, 1):
        walk_stride = _walk_stride(plan, args)
        _emit_walk_scratch_module_statements(cg)
        ws_arg = _walk_scratch_arg(
            df, args[match.q_name], ctas=total_ctas, stride=walk_stride
        )
        df.placeholder_args.add(ws_arg.name)
        bindings.append(f"_fbwd_walk_ws = {ws_arg.name}")
    for index, buffer in enumerate(plan.aux_buffers):
        bindings.append(f"_flash_aux{index} = {args[buffer.name].name}")
    for index, expr in enumerate(scalar_exprs):
        bindings.append(f"_flash_hs{index} = {expr}")
    wrapper_plan: dict[str, object] = {
        "kind": "helion_flash_bwd",
        "flex_two_cta": True,
        "q_name": args[match.q_name].name,
        "k_name": args[match.k_name].name,
        "v_name": args[match.v_name].name,
        "do_name": args[match.do_name].name,
        "lse_name": args[match.lse_name].name,
        "delta_name": args[match.delta_name].name,
        "dq_name": args[match.dq_name].name,
        "dk_name": args[match.dk_name].name,
        "dv_name": args[match.dv_name].name,
        "head_dim": plan.head_dim,
        "dtype": io_dtype_str,
        "total_q_rows": plan.num_bh * plan.q_seq,
        "total_kv_rows": batch * kv_heads * plan.kv_seq,
        "q_stage": 1,
        "do_stage": 1,
        "kv_stage": 1,
        "two_cta": False,
        "persistent": False,
        "total_tiles": total_ctas,
        "topology": "fa4",
        "flex_seq_q": plan.q_seq,
        "flex_bh_q": plan.num_bh,
        "flex_seq_k": plan.kv_seq,
        "flex_bh_k": batch * kv_heads,
    }
    cg.cute_wrapper_plans.append(wrapper_plan)  # type: ignore[attr-defined]
    df.wrapper_only_params.extend(_FLASH_BWD_2CTA_KERNEL_PARAMS)
    df.placeholder_args.update(a.name for a in args.values())
    cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    df.cute_state.attention_flash_threads = 512
    df.cute_state.cluster_shape = (2, 1, 1)
    body_src = emit_flash_bwd_flex_2cta_device_body(
        plan=plan,
        io_dtype=io_dtype_str,
        scale_expr=scale_expr,
        exp2_f32=exp2_f32,
        mask_form=mask_form,
        bindings=bindings,
        device_function=df,
        reg_split=None if reg_split is None else str(reg_split),
        hook_cols=hook_cols,
        hook_pin=hook_pin,
        bias_fold=bias_fold,
        tile_group=tile_group,
        exp2_emu=exp2_emu,
        table_steps=table_steps,
        walk_stride=walk_stride,
    )
    wrapped = ast_module.parse("if True:\n" + body_src)
    assert isinstance(wrapped.body[0], ast_module.If)
    df.body = list(wrapped.body[0].body)
    df.preamble = []
    return True
