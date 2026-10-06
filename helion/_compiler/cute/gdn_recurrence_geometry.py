"""Static geometry shared by the gdn recurrence planner and its SM100 schedule.

Everything here is plain integer arithmetic so both the compile-time matcher
(which must import cheaply) and the CuTe device module can agree on the TMEM
column map, the shared-memory ring budget and the legal knob values.

Admission limits of the schedule (everything else keeps the SIMT lowering):

* ``dhead`` in {16, 32, 64}: one bf16 chunk-tile row of ``w``/``k`` must be a
  single 32/64/128-byte swizzle atom row so both tcgen05 operand majors read
  the same TMA-written tile;
* ``chunk`` a power of two in [16, 256]: the tcgen05 ``N`` of the projection
  and the ``K`` of the update, in K=16 steps, with the fp32 accumulator and
  its bf16 image resident in TMEM;
* the dstate tile (``block_v``) in {16, 32, 64, 128}: the resident state is a
  full tcgen05 tile of ``mma_m`` rows (128, or 64 for tiles of at most 64
  rows); a narrower tile is replicated into the TMEM quadrants it does not
  reach (see :func:`gdn_state_replicas`) and rows past ``block_v`` (and past
  ``dstate`` in the last tile) are padding that is never stored;
* ``dstate`` a multiple of 8: the TMA tensor map of ``u`` needs every
  non-contiguous stride (``dstate * 2`` bytes for the head axis) to be a
  multiple of 16 bytes, and the launcher checks the 16-byte base alignment of
  ``k``/``w``/``u`` at run time;
* the TMEM column map must fit 512 columns and at least one TMA ring stage
  (plus the shared-memory update image of a replicated tile) must fit shared
  memory.
"""

from __future__ import annotations

import dataclasses

GDN_RECURRENCE_KIND = "gdn_recurrence_sm100"
GDN_RECURRENCE_DEVICE_ABI = 5
# The full tcgen05 tile: the widest dstate tile and the default MMA height.
GDN_MMA_M = 128
# tcgen05 ``M`` values of the schedule (the ``cute_gdn_recurrence_mma_m``
# knob), default first.  ``M = 64`` halves the tensor time, the TMEM traffic
# and the update image of a tile of at most 64 rows; its rows live in lanes
# 0-15 of every TMEM quadrant, which the epilogue reads through the 16-lane
# data path (each thread holds two rows), so a warp's state and accumulator
# slices must come in whole eight-column groups.
GDN_MMA_M_CHOICES = (128, 64)
GDN_MMA_K = 16
GDN_ADMITTED_DHEAD = (16, 32, 64)
GDN_ADMITTED_BLOCK_V = (128, 64, 32, 16)
GDN_MIN_CHUNK = 16
GDN_MAX_CHUNK = 256
GDN_EPILOGUE_WARP_CHOICES = (8, 4, 16)
# Pipelined token groups of a chunk (projection N / update K slices),
# preferred first: two groups overlap the first half's update with the second
# half's compute; four add more barriers and commits than they hide.
GDN_TOKEN_GROUP_CHOICES = (2, 1, 4)
GDN_TMEM_QUADRANTS = 4
GDN_TMEM_QUADRANT_LANES = 32
GDN_MAX_STAGES = 4
GDN_ACC_LOAD_COLS = 64
# Tokens whose u values and gates a warp reads into registers ahead of the
# accumulator; wider slices read the rest as they go to bound register use.
GDN_PREFETCH_TOKENS = 16
# Sixteen epilogue warps put five warps on one SM sub-partition (with the
# store and TMA warps), so ptxas caps the kernel at 96 or 80 registers: those
# CTAs take the accumulator in half-size blocks and prefetch half the tokens.
GDN_REGISTER_CAPPED_WARPS = 16
# Update K steps per token group that are issued unrolled.  With a single
# ring stage every step's two operand descriptors are loop-invariant and the
# compiler hoists them out of the chunk loop; sixteen steps (a 256-token
# chunk in one group) exceed the uniform register file and spill, so longer
# groups issue from a rolled loop instead.
GDN_MAX_UNROLLED_K_STEPS = 8
# Columns one lane of the 16-lane TMEM data path moves per ``16x256b``
# repetition: the granularity of every per-warp column slice under M = 64.
GDN_HALF_LANE_COLS = 8
_TMEM_COLUMNS = 512
_TMEM_ALIGN = 32
_SMEM_BUDGET_BYTES = 227 * 1024
_SMEM_RESERVED_BYTES = 1024
_BF16_BYTES = 2
_F32_BYTES = 4
# TMA tensor maps need 16-byte aligned base pointers and non-contiguous
# strides; the ``u`` head stride is ``dstate`` bf16 elements.
GDN_TMA_ALIGNMENT_BYTES = 16
GDN_DSTATE_MULTIPLE = GDN_TMA_ALIGNMENT_BYTES // _BF16_BYTES


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True)
class GdnTmemLayout:
    """Column map of the resident TMEM state and the per-chunk operand images."""

    state_col: int
    state_image_col: int
    acc_col: int
    update_image_col: int
    alloc_cols: int


def gdn_tmem_layout(dhead: int, chunk: int) -> GdnTmemLayout | None:
    """Lay out ``state^T`` (fp32), its bf16 image, the projection accumulator
    and the bf16 update image; ``None`` when they do not fit 512 columns."""

    state_col = 0
    state_image_col = _align_up(state_col + dhead, _TMEM_ALIGN)
    acc_col = _align_up(state_image_col + dhead // 2, _TMEM_ALIGN)
    update_image_col = _align_up(acc_col + chunk, _TMEM_ALIGN)
    total = _align_up(update_image_col + chunk // 2, _TMEM_ALIGN)
    if total > _TMEM_COLUMNS:
        return None
    alloc_cols = _TMEM_ALIGN
    while alloc_cols < total:
        alloc_cols *= 2
    return GdnTmemLayout(
        state_col=state_col,
        state_image_col=state_image_col,
        acc_col=acc_col,
        update_image_col=update_image_col,
        alloc_cols=alloc_cols,
    )


def gdn_half_lanes(mma_m: int) -> bool:
    """Whether the epilogue reads TMEM through the 16-lane data path.

    An ``M = 64`` tcgen05 tile keeps its rows in lanes 0-15 of every quadrant;
    the ``16x256b``/``16x128b``/``16x64b`` load and store shapes spread those
    sixteen lanes over all 32 threads of a warp (two rows per thread).
    """

    return mma_m == 64


def gdn_quadrant_rows(mma_m: int) -> int:
    """tcgen05 rows one TMEM quadrant holds (32, or 16 under ``M = 64``)."""

    return mma_m // GDN_TMEM_QUADRANTS


def gdn_stage_smem_bytes(chunk: int, dhead: int, block_v: int) -> int:
    """Bytes of one ring stage: the ``w`` and ``k`` tiles, the ``u`` tile, the
    fp32 gate vector and the raw ``g`` rows it is computed from."""

    return (2 * chunk * dhead + chunk * block_v) * _BF16_BYTES + 2 * chunk * _F32_BYTES


def gdn_smem_bytes(
    chunk: int, dhead: int, block_v: int, stages: int, mma_m: int
) -> int:
    return (
        stages * gdn_stage_smem_bytes(chunk, dhead, block_v)
        + stages * _F32_BYTES
        + gdn_update_tile_bytes(chunk, block_v, mma_m)
        + _SMEM_RESERVED_BYTES
    )


def gdn_tma_transaction_bytes(chunk: int, dhead: int, block_v: int) -> int:
    return (2 * chunk * dhead + chunk * block_v) * _BF16_BYTES


def gdn_max_stages(chunk: int, dhead: int, block_v: int, mma_m: int) -> int:
    stages = 0
    while (
        stages < GDN_MAX_STAGES
        and gdn_smem_bytes(chunk, dhead, block_v, stages + 1, mma_m)
        <= _SMEM_BUDGET_BYTES
    ):
        stages += 1
    return stages


def gdn_stage_choices(
    chunk: int, dhead: int, block_v: int, mma_m: int
) -> tuple[int, ...]:
    """Legal TMA ring depths, preferred first (three deep when it fits)."""

    max_stages = gdn_max_stages(chunk, dhead, block_v, mma_m)
    if max_stages <= 0:
        return ()
    preferred = min(max_stages, 3)
    return (
        preferred,
        *(stages for stages in range(max_stages, 0, -1) if stages != preferred),
    )


def gdn_epilogue_warp_choices(chunk: int, dhead: int) -> tuple[int, ...]:
    """Epilogue warp counts whose per-warp TMEM column slices stay loadable.

    ``warps // 4`` warps share each 32-lane TMEM quadrant and split the state
    and accumulator columns; both slices must be power-of-two column counts
    of at least one bf16 pair.
    """

    choices = []
    for warps in GDN_EPILOGUE_WARP_CHOICES:
        slices = warps // 4
        if dhead % slices or chunk % slices:
            continue
        state_cols = dhead // slices
        acc_cols = chunk // slices
        if (
            state_cols < 2
            or acc_cols < 2
            or not _is_power_of_two(state_cols)
            or not _is_power_of_two(acc_cols)
        ):
            continue
        choices.append(warps)
    return tuple(choices)


def gdn_mma_m_choices(
    chunk: int, dhead: int, block_v: int, epilogue_warps: int
) -> tuple[int, ...]:
    """tcgen05 ``M`` values legal for a (dstate tile, epilogue warps) pair,
    preferred first.

    ``M = 64`` is preferred for a tile of at most 64 rows: it halves the
    tensor time of both MMAs, the TMEM traffic and the update image, and
    leaves fewer (or no) replicas.  Its 16-lane data path hands each thread
    two-column pairs of eight-column groups, so a warp's slice of the state
    columns has to be a whole number of such groups.  The full 128-row tile
    is always legal (the 128-row dstate tile has no other).
    """

    choices = []
    if (
        block_v <= 64
        and (dhead // (epilogue_warps // GDN_TMEM_QUADRANTS)) % GDN_HALF_LANE_COLS == 0
        and gdn_max_stages(chunk, dhead, block_v, 64) > 0
    ):
        choices.append(64)
    choices.append(GDN_MMA_M)
    return tuple(choices)


def gdn_active_quadrants(block_v: int, mma_m: int) -> int:
    """TMEM quadrants that hold distinct rows of the dstate tile."""

    return -(-block_v // gdn_quadrant_rows(mma_m))


def gdn_state_replicas(block_v: int, mma_m: int) -> int:
    """Copies of the state the ``mma_m``-row tcgen05 tile holds.

    Every tcgen05 row is one dstate index, and every warp may only touch the
    TMEM quadrant ``warp % 4``, which is also its SM sub-partition.  A tile
    narrower than the MMA tile is therefore replicated into the quadrants it
    does not reach, so all four sub-partitions share the per-chunk epilogue:
    each replica's warps handle a slice of the chunk's tokens and the update
    image is assembled in shared memory as the MMA's A operand.
    """

    return GDN_TMEM_QUADRANTS // gdn_active_quadrants(block_v, mma_m)


def gdn_min_tokens_per_split(mma_m: int) -> int:
    """Smallest token slice a warp can turn into update-image rows: one bf16
    pair per lane, or one eight-column repetition of the 16-lane data path."""

    return GDN_HALF_LANE_COLS if gdn_half_lanes(mma_m) else 2


def gdn_token_splits(chunk: int, block_v: int, epilogue_warps: int, mma_m: int) -> int:
    """Warps that share the update computation of one chunk (token slices)."""

    return min(
        gdn_state_replicas(block_v, mma_m) * (epilogue_warps // GDN_TMEM_QUADRANTS),
        chunk // gdn_min_tokens_per_split(mma_m),
    )


def gdn_tokens_per_split(
    chunk: int, block_v: int, epilogue_warps: int, mma_m: int
) -> int:
    """Tokens (accumulator columns) one warp turns into update-image rows."""

    return chunk // gdn_token_splits(chunk, block_v, epilogue_warps, mma_m)


def gdn_token_group_choices(
    chunk: int, block_v: int, epilogue_warps: int, mma_m: int
) -> tuple[int, ...]:
    """Token groups the chunk's projection and update can be split into.

    Each group is a projection MMA over its ``N = chunk // groups`` tokens with
    its own commit and a barrier of its warps, so the update K steps of the
    first groups run while the last groups still compute.  A group needs a
    whole number of K = 16 steps and of computing warps (every epilogue warp
    must compute: a slice-starved tiny chunk keeps the single group).  The
    single group is the plain two-barrier schedule and is always legal.
    """

    splits = gdn_token_splits(chunk, block_v, epilogue_warps, mma_m)
    all_compute = splits == gdn_state_replicas(block_v, mma_m) * (
        epilogue_warps // GDN_TMEM_QUADRANTS
    )
    return tuple(
        groups
        for groups in GDN_TOKEN_GROUP_CHOICES
        if groups == 1
        or (all_compute and splits % groups == 0 and (chunk // groups) % GDN_MMA_K == 0)
    )


def _register_capped(epilogue_warps: int) -> bool:
    return epilogue_warps >= GDN_REGISTER_CAPPED_WARPS


def gdn_acc_load_cols(chunk: int, block_v: int, epilogue_warps: int, mma_m: int) -> int:
    """Lane columns per ``tcgen05.ld`` block of the projection accumulator."""

    cols = GDN_ACC_LOAD_COLS // (2 if _register_capped(epilogue_warps) else 1)
    return min(cols, gdn_tokens_per_split(chunk, block_v, epilogue_warps, mma_m))


def gdn_prefetch_tokens(
    chunk: int, block_v: int, epilogue_warps: int, mma_m: int
) -> int:
    """Lane columns (tokens) of its slice whose ``u`` values and gates a warp
    reads into registers while the projection runs.

    Under ``M = 64`` a thread holds half the columns of two rows, so twice
    the lane columns fill the same registers.
    """

    tokens = GDN_PREFETCH_TOKENS * (2 if gdn_half_lanes(mma_m) else 1)
    tokens //= 2 if _register_capped(epilogue_warps) else 1
    return min(tokens, gdn_tokens_per_split(chunk, block_v, epilogue_warps, mma_m))


def gdn_update_tile_bytes(chunk: int, block_v: int, mma_m: int) -> int:
    """Shared-memory bytes of the bf16 update image when the state is
    replicated (the MMA reads it from shared memory); zero otherwise."""

    if gdn_state_replicas(block_v, mma_m) == 1:
        return 0
    return mma_m * chunk * _BF16_BYTES


def gdn_store_warps(block_v: int, mma_m: int) -> int:
    """``h`` store warps: one per TMEM quadrant when the state is replicated.

    Each then stores only its replica's share of the columns, off the
    epilogue's critical path.  An unreplicated tile has no spare replica, and
    its epilogue warps store their own columns while the update MMA runs
    instead: four extra warps storing the columns would only contend with
    them for the sub-partitions' issue slots.
    """

    if gdn_state_replicas(block_v, mma_m) == 1:
        return 0
    return GDN_TMEM_QUADRANTS


def gdn_cta_warps(block_v: int, epilogue_warps: int, mma_m: int) -> int:
    """Warps the CTA launches with: the epilogue warps, the ``h`` store warps
    and the TMA warp."""

    return epilogue_warps + gdn_store_warps(block_v, mma_m) + 1


def gdn_candidate_block_sizes(dstate: int) -> tuple[int, ...]:
    """dstate tiles worth trying for ``dstate``, preferred first.

    The preferred tile is the largest power of two not above ``dstate`` (capped
    at the full tcgen05 ``M``); smaller tiles trade padded TMEM rows for more
    CTAs.  Each candidate still has to pass :func:`gdn_shape_admitted`.
    """

    largest = GDN_ADMITTED_BLOCK_V[-1]
    while largest * 2 <= min(dstate, GDN_MMA_M):
        largest *= 2
    return tuple(block_v for block_v in GDN_ADMITTED_BLOCK_V if block_v <= largest)


def gdn_shape_admitted(*, dhead: int, chunk: int, dstate: int, block_v: int) -> bool:
    """Admission limits of the tcgen05 schedule (everything else stays SIMT).

    Admission is judged with the full 128-row MMA, whose replicated update
    image is the largest; ``M = 64`` only ever needs less shared memory.
    """

    return (
        dhead in GDN_ADMITTED_DHEAD
        and block_v in GDN_ADMITTED_BLOCK_V
        and GDN_MIN_CHUNK <= chunk <= GDN_MAX_CHUNK
        and _is_power_of_two(chunk)
        and dstate > 0
        and dstate % GDN_DSTATE_MULTIPLE == 0
        and gdn_tmem_layout(dhead, chunk) is not None
        and bool(gdn_stage_choices(chunk, dhead, block_v, GDN_MMA_M))
        and bool(gdn_epilogue_warp_choices(chunk, dhead))
    )


__all__ = [
    "GDN_ACC_LOAD_COLS",
    "GDN_ADMITTED_BLOCK_V",
    "GDN_ADMITTED_DHEAD",
    "GDN_DSTATE_MULTIPLE",
    "GDN_EPILOGUE_WARP_CHOICES",
    "GDN_HALF_LANE_COLS",
    "GDN_MAX_STAGES",
    "GDN_MAX_UNROLLED_K_STEPS",
    "GDN_MMA_K",
    "GDN_MMA_M",
    "GDN_MMA_M_CHOICES",
    "GDN_PREFETCH_TOKENS",
    "GDN_RECURRENCE_DEVICE_ABI",
    "GDN_RECURRENCE_KIND",
    "GDN_REGISTER_CAPPED_WARPS",
    "GDN_TMA_ALIGNMENT_BYTES",
    "GDN_TMEM_QUADRANTS",
    "GDN_TMEM_QUADRANT_LANES",
    "GdnTmemLayout",
    "gdn_acc_load_cols",
    "gdn_active_quadrants",
    "gdn_candidate_block_sizes",
    "gdn_cta_warps",
    "gdn_epilogue_warp_choices",
    "gdn_half_lanes",
    "gdn_max_stages",
    "gdn_min_tokens_per_split",
    "gdn_mma_m_choices",
    "gdn_prefetch_tokens",
    "gdn_quadrant_rows",
    "gdn_shape_admitted",
    "gdn_smem_bytes",
    "gdn_stage_choices",
    "gdn_stage_smem_bytes",
    "gdn_state_replicas",
    "gdn_store_warps",
    "gdn_tma_transaction_bytes",
    "gdn_tmem_layout",
    "gdn_token_group_choices",
    "gdn_token_splits",
    "gdn_tokens_per_split",
    "gdn_update_tile_bytes",
]
