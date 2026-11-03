"""The measured flash-attention seed bodies, as data.

``flash_attention_seed_configs`` (``cute_flash``) starts every search from the
surface's per-family defaults and the bodies here: each template pins the knob
values a measured winner, or a structure the search cannot assemble axis by
axis, was recorded with.  A template is written against the surface that
measured it: a polynomial exp2 packet it names exists on the fast_math surface
(the seed builder substitutes the exact packet without the setting), its K/V
ring depth is the measured one (the builder caps the list-walk bodies to the
shape's shared-memory budget), and the knobs it leaves unstated take the
surface's defaults or the facts' values (the staged output tile count, the
softmax lowering, the local-TMA packet and threshold).  Every key is a
registered knob (``test_cute_flash_knobs`` pins that); whether a value is
legal on a surface is decided per shape by the builder's ``_flash_seed_set_all``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .flash_knobs import (
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET,
)
from .flash_knobs import (
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET,
)
from .flash_knobs import (
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET,
)
from .flash_knobs import FLASH_CAUSAL_HEAD_PAIR_KEY
from .flash_knobs import FLASH_CAUSAL_KV_ORDER_KEY
from .flash_knobs import FLASH_CAUSAL_LOOP_SPLIT_KEY
from .flash_knobs import FLASH_CAUSAL_LPT_L2_SHIFT_KEY
from .flash_knobs import FLASH_CAUSAL_LPT_SWIZZLE_KEY
from .flash_knobs import FLASH_CAUSAL_SKIP_DEAD_TILE_KEY
from .flash_knobs import FLASH_CAUSAL_STAGGER_KEY
from .flash_knobs import FLASH_CORR_REGS_KEY
from .flash_knobs import FLASH_CORR_TILE_SIZE_KEY
from .flash_knobs import FLASH_DISC_PIPE_KEY
from .flash_knobs import FLASH_E2E_OFFSET0_KEY
from .flash_knobs import FLASH_E2E_OFFSET_KEY
from .flash_knobs import FLASH_E2E_SCHEDULE_KEY
from .flash_knobs import FLASH_EARLY_PROLOGUE_LOADS_KEY
from .flash_knobs import FLASH_EPI_STG_KEY
from .flash_knobs import FLASH_EPI_TMA_KEY
from .flash_knobs import FLASH_EPI_TMA_SETUP_KEY
from .flash_knobs import FLASH_EXP2_PACKET_KEY
from .flash_knobs import FLASH_FIRST_LOAD_ORDER_KEY
from .flash_knobs import FLASH_FLEX_HOOK_FAST_FULL_KEY
from .flash_knobs import FLASH_FLEX_ROLLED_CHUNKS_KEY
from .flash_knobs import FLASH_KV_STAGE_KEY
from .flash_knobs import FLASH_MASKED_E2E_SCHEDULE_KEY
from .flash_knobs import FLASH_MMA_INTERLEAVE_KEY
from .flash_knobs import FLASH_O_OVER_Q_KEY
from .flash_knobs import FLASH_O_STAGE_KEY
from .flash_knobs import FLASH_OTHER_REGS_KEY
from .flash_knobs import FLASH_P_STORE_REP_KEY
from .flash_knobs import FLASH_PERSISTENT_CTAS_PER_SM_KEY
from .flash_knobs import FLASH_PERSISTENT_KEY
from .flash_knobs import FLASH_PERSISTENT_LOOP_KEY
from .flash_knobs import FLASH_PIPELINE_FAMILY_KEY
from .flash_knobs import FLASH_PRECOMPUTE_QK_DESC_KEY
from .flash_knobs import FLASH_RESCALE_CHUNK_COLS_KEY
from .flash_knobs import FLASH_RESCALE_THRESHOLD_KEY
from .flash_knobs import FLASH_ROLE_CHAIN_KEY
from .flash_knobs import FLASH_ROLE_MAP_KEY
from .flash_knobs import FLASH_ROW_EPILOGUE_WARPS_KEY
from .flash_knobs import FLASH_ROWMAX_KEY
from .flash_knobs import FLASH_S_LOAD_REP_KEY
from .flash_knobs import FLASH_S_RESIDENT_KEY
from .flash_knobs import FLASH_SOFTMAX_DISC_KEY
from .flash_knobs import FLASH_SOFTMAX_REGION_KEY
from .flash_knobs import FLASH_SOFTMAX_REGS_KEY
from .flash_knobs import FLASH_SOFTMAX_SETUP_KEY
from .flash_knobs import FLASH_SP_ROW_SUM_KEY
from .flash_knobs import FLASH_SPLIT_P_ARRIVE_KEY
from .flash_knobs import FLASH_STAT_TRANSPORT_KEY
from .flash_knobs import FLASH_UNEVEN_KV_RING_KEY
from .flash_knobs import FLASH_WAIT_HINT_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

# The causal pipe-split schedule for the modifier body: the hd64 fp16 causal
# winner's fa4 schedule (16/8 cadence, descending loop split, two-deep PASS2
# pipeline, split P arrival, TMA output) with its hybrid packet replaced by
# the plain 8x2 packet (the manual packets exist only for the standard causal
# body). Without the hoisted bias this point loses to the ascending 1x1
# schedule (2.02 vs 1.61 ms at s8192); with it the point measured 1.06 ms
# against the 1.25 ms the search otherwise converges on.
SEED_MODIFIER_CAUSAL_SPLIT: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_EXP2_PACKET_KEY: "8x2",
    FLASH_E2E_SCHEDULE_KEY: "16/8",
    FLASH_MASKED_E2E_SCHEDULE_KEY: "inherit",
    FLASH_E2E_OFFSET_KEY: 12,
    FLASH_E2E_OFFSET0_KEY: 7,
    FLASH_KV_STAGE_KEY: 2,
    FLASH_SOFTMAX_DISC_KEY: True,
    FLASH_DISC_PIPE_KEY: 2,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_P_STORE_REP_KEY: 16,
    FLASH_S_LOAD_REP_KEY: 32,
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_EPI_TMA_KEY: True,
}

# The asymmetric pair has no measured tuning table. Its fa4 body measured
# best with FA4's structure for the pair: the uneven three-slot K/V ring, the
# TMA output staged in the dead Q stages and the causal resident packet
# (cadence, registers and threshold from the hd128 causal winners: 0.419 ms
# at s4096 vs 0.429 for the same structure without the packet and 0.433 for
# the direct-store ring); the descending loop-split schedule alone (default
# and XU cadence) is the direct-store fallback.
_ASYMMETRIC_PAIR_STAGED_STRUCTURE: Mapping[str, object] = {
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_UNEVEN_KV_RING_KEY: True,
    FLASH_O_OVER_Q_KEY: True,
    FLASH_EPI_TMA_KEY: True,
    FLASH_CORR_TILE_SIZE_KEY: 16,
}
# The hd128 causal winners' register split, the fragment default one
# (200/64/48) and FA4's row for the shape (192/72/56) are the measured
# splits of the staged structure.
_HD128_WINNER_REGS: Mapping[str, object] = {
    FLASH_SOFTMAX_REGS_KEY: 160,
    FLASH_CORR_REGS_KEY: 112,
    FLASH_OTHER_REGS_KEY: 48,
}
# The 192-wide QK MMA is half again as long as hd128's, which leaves the
# softmax warps the slack to run most exp2 on the XU pipe: on the
# power-capped B200 the 16/2 resident cadence measured 1.5% and the
# hardware-only one 1.2% faster than the 16/6 packet at s8192 (paired
# alternation), the same direction as FA4's tuning row for the shape (one
# emulated pair in 16, 192/72 registers).  The polynomial cadences are
# fast_math-only; the builder seeds the same structures on the hardware-only
# sibling without the setting.
SEED_ASYMMETRIC_PAIR_BODIES: tuple[Mapping[str, object], ...] = (
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET
        ),
        **_HD128_WINNER_REGS,
        FLASH_RESCALE_THRESHOLD_KEY: 32.0,
        FLASH_WAIT_HINT_KEY: 0,
        FLASH_E2E_OFFSET_KEY: 4,
        FLASH_E2E_OFFSET0_KEY: 7,
    },
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET
        ),
    },
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET
        ),
        **_HD128_WINNER_REGS,
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_WAIT_HINT_KEY: 0,
    },
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET
        ),
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_WAIT_HINT_KEY: 10_000_000,
    },
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET
        ),
        FLASH_SOFTMAX_REGS_KEY: 192,
        FLASH_CORR_REGS_KEY: 72,
        FLASH_OTHER_REGS_KEY: 56,
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_WAIT_HINT_KEY: 0,
        # Half of FA4's L2-sized launch group: 16 batch-heads at s2048
        # measured one harness step (1.7%) faster than FA4's 32 (in-kernel
        # spans 94.5 vs 98.6 us at boost clocks) and is neutral at s4096 /
        # s8192.
        FLASH_CAUSAL_LPT_L2_SHIFT_KEY: 1,
    },
    {
        **_ASYMMETRIC_PAIR_STAGED_STRUCTURE,
        FLASH_EXP2_PACKET_KEY: (
            _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET
        ),
        **_HD128_WINNER_REGS,
        FLASH_RESCALE_THRESHOLD_KEY: 8.0,
        FLASH_WAIT_HINT_KEY: 0,
    },
    {**_ASYMMETRIC_PAIR_STAGED_STRUCTURE, FLASH_E2E_SCHEDULE_KEY: "xu"},
    {
        FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    },
    {
        FLASH_CAUSAL_KV_ORDER_KEY: "descending",
        FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        FLASH_E2E_SCHEDULE_KEY: "xu",
    },
)

# The staggered one-head schedule needs the descending split walk and a
# four-deep aliased ring (the builder adds the head size's seed depth, and
# the output in the dead Q stages where the ring does not fit beside it).
SEED_CAUSAL_STAGGER_STRUCTURE: Mapping[str, object] = {
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_CAUSAL_STAGGER_KEY: True,
    FLASH_CAUSAL_HEAD_PAIR_KEY: False,
}
# The head pair (two consecutive query heads of one K/V head at one tile)
# needs the descending split walk and the TMA output store.
SEED_CAUSAL_HEAD_PAIR_STRUCTURE: Mapping[str, object] = {
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_EPI_TMA_KEY: True,
    FLASH_EPI_STG_KEY: False,
    FLASH_CAUSAL_HEAD_PAIR_KEY: True,
    FLASH_CAUSAL_SKIP_DEAD_TILE_KEY: False,
    FLASH_CAUSAL_STAGGER_KEY: False,
}

# The family loop seeds ``fa4_local_tma`` (per-CTA TMA partitioning) with the
# neutral packet and the packet loop varies the base family only, yet the
# dense hd64 shapes are won by that family with a wide packet: 4x2 or 8x2
# packet, 16/4 emulation schedule, six-stage K/V ring, direct epilogue, one
# persistent counted CTA per SM, FA4's role map, and the plain
# (non-discounted) softmax with the split P arrival, one-deep pipe and
# 16-column rescale (threshold 16, or the largest the dtype allows). It
# matched cuDNN at s4096 where an hour of search from the generic seeds ended
# 3% behind it; with the family's default softmax path instead it is 9%
# slower than that search's winner, so the softmax knobs are part of the seed.
SEED_DENSE_HD64_LOCAL_TMA_PACKETS: tuple[str, ...] = ("4x2", "8x2")
SEED_DENSE_HD64_LOCAL_TMA_RESCALE_THRESHOLD = 16.0
SEED_DENSE_HD64_LOCAL_TMA: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4_local_tma",
    FLASH_E2E_SCHEDULE_KEY: "16/4",
    FLASH_KV_STAGE_KEY: 6,
    FLASH_EPI_TMA_KEY: False,
    FLASH_PERSISTENT_CTAS_PER_SM_KEY: 1,
    FLASH_PERSISTENT_LOOP_KEY: "counted",
    FLASH_ROLE_MAP_KEY: "fa4",
    FLASH_SOFTMAX_DISC_KEY: False,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_DISC_PIPE_KEY: 1,
    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
    FLASH_RESCALE_THRESHOLD_KEY: SEED_DENSE_HD64_LOCAL_TMA_RESCALE_THRESHOLD,
    FLASH_WAIT_HINT_KEY: 0,
}

# The causal hd64 shapes are won by three fa4 bodies that the family and
# packet loops never seed. The paired 16-column score-load body (4x1 packet,
# 8/2 schedule with hardware-exp2 masked tiles, serial PASS2, 168/104/56
# registers, 32/32 rescale, TMA-store epilogue, descending KV order with the
# loop split, K-then-V-then-Q load order) wins at s2048; the hybrid degree-1
# body with the two-deep chunked PASS2, 200/64/48 registers, 8-column rescale
# chunks and threshold 8 wins at s4096; and the 8x2 packet at the 16/6
# cadence on the paired score-load body with a seven-stage K/V ring and the
# two-deep PASS2 wins at s8192. The nearest generic seeds are 6-8 knobs away
# and a cold search's own pick re-measured 1-2% behind the campaign winners,
# so each body is seeded as measured. Neither the dead stage-0 tile skip, the
# halved L2 launch group nor the staggered walk pays at head_dim 64 (skip:
# equal at s2048, -0.4% at s4096, -0.6% at s8192; shift: -0.4% to -1.2%;
# stagger: at the winner at s2048, 0.5-1% behind at s4096 / s8192, and the
# s8192 body's seven-stage ring would otherwise take the resolver's
# on-where-legal default), so the seeds pin all three off; the fragments
# still search each of them.
SEED_CAUSAL_HD64_FA4_PINS: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_EPI_TMA_KEY: True,
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_PRECOMPUTE_QK_DESC_KEY: True,
    FLASH_CAUSAL_SKIP_DEAD_TILE_KEY: False,
    FLASH_CAUSAL_LPT_L2_SHIFT_KEY: 0,
    FLASH_CAUSAL_STAGGER_KEY: False,
    FLASH_CAUSAL_HEAD_PAIR_KEY: False,
    FLASH_EARLY_PROLOGUE_LOADS_KEY: True,
}
SEED_CAUSAL_HD64_BODIES: tuple[Mapping[str, object], ...] = (
    # s2048: paired 16-column score loads, serial PASS2, 4x1 packet.
    {
        FLASH_EXP2_PACKET_KEY: "4x1",
        FLASH_E2E_SCHEDULE_KEY: "8/2",
        FLASH_MASKED_E2E_SCHEDULE_KEY: "xu",
        FLASH_E2E_OFFSET_KEY: 4,
        FLASH_E2E_OFFSET0_KEY: 4,
        FLASH_KV_STAGE_KEY: 2,
        FLASH_S_LOAD_REP_KEY: 16,
        FLASH_DISC_PIPE_KEY: 1,
        FLASH_SOFTMAX_REGS_KEY: 168,
        FLASH_CORR_REGS_KEY: 104,
        FLASH_OTHER_REGS_KEY: 56,
        FLASH_RESCALE_THRESHOLD_KEY: 32.0,
        FLASH_RESCALE_CHUNK_COLS_KEY: 32,
        FLASH_CORR_TILE_SIZE_KEY: 32,
        FLASH_FIRST_LOAD_ORDER_KEY: 2,
        FLASH_WAIT_HINT_KEY: 10000000,
    },
    # s8192: the same score-load body with the 8x2 packet at the 16/6
    # cadence, a seven-stage K/V ring and the two-deep chunked PASS2.
    {
        FLASH_EXP2_PACKET_KEY: "8x2",
        FLASH_E2E_SCHEDULE_KEY: "16/6",
        FLASH_MASKED_E2E_SCHEDULE_KEY: "xu",
        FLASH_E2E_OFFSET_KEY: 14,
        FLASH_E2E_OFFSET0_KEY: 2,
        FLASH_KV_STAGE_KEY: 7,
        FLASH_S_LOAD_REP_KEY: 16,
        FLASH_DISC_PIPE_KEY: 2,
        FLASH_SOFTMAX_REGS_KEY: 168,
        FLASH_CORR_REGS_KEY: 104,
        FLASH_OTHER_REGS_KEY: 56,
        FLASH_RESCALE_THRESHOLD_KEY: 16.0,
        FLASH_RESCALE_CHUNK_COLS_KEY: 16,
        FLASH_CORR_TILE_SIZE_KEY: 32,
        FLASH_FIRST_LOAD_ORDER_KEY: 4,
        FLASH_WAIT_HINT_KEY: 0,
    },
)
# The s4096 body: the hybrid packet's structure (fa4, the 16/8 cadence on
# both loop segments, the two-deep chunked PASS2 with the split P arrival,
# the descending loop split; ``_flash_hybrid_exp2_packet_structure``) with
# its measured knobs.  The packet itself is fast_math-only; without the
# setting the same body is seeded on the surface's exact default packet with
# the structure the packet pins.
SEED_CAUSAL_HD64_HYBRID_BODY: Mapping[str, object] = {
    FLASH_EPI_TMA_KEY: True,
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    FLASH_PRECOMPUTE_QK_DESC_KEY: True,
    FLASH_CAUSAL_SKIP_DEAD_TILE_KEY: False,
    FLASH_CAUSAL_LPT_L2_SHIFT_KEY: 0,
    FLASH_CAUSAL_STAGGER_KEY: False,
    FLASH_CAUSAL_HEAD_PAIR_KEY: False,
    FLASH_EARLY_PROLOGUE_LOADS_KEY: True,
    FLASH_SOFTMAX_REGS_KEY: 200,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 48,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_RESCALE_CHUNK_COLS_KEY: 8,
    FLASH_E2E_OFFSET_KEY: 0,
    FLASH_E2E_OFFSET0_KEY: 4,
    FLASH_FIRST_LOAD_ORDER_KEY: 2,
}

# The two-tile fa4 body of the dense hd128 shapes (half the K/V traffic of
# the single-tile alternating body, so the cheaper one under a power cap):
# the fa4 xu seed with the serial PASS2 (disc_pipe 1), the whole-row sum and
# the softmax lowering the surface offers (the builder adds it: the resident
# value graph on a dense FP16/BF16 surface -- output only, or out + lse --
# auto where the surface has neither fact).
SEED_DENSE_HD128_TWO_TILE: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_E2E_SCHEDULE_KEY: "xu",
    FLASH_E2E_OFFSET_KEY: 0,
    FLASH_E2E_OFFSET0_KEY: 0,
    FLASH_EXP2_PACKET_KEY: "1x1",
    FLASH_STAT_TRANSPORT_KEY: "single",
    FLASH_SOFTMAX_DISC_KEY: False,
    FLASH_DISC_PIPE_KEY: 1,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_SP_ROW_SUM_KEY: "whole",
    FLASH_KV_STAGE_KEY: 3,
    FLASH_SOFTMAX_REGS_KEY: 200,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 48,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
    FLASH_CORR_TILE_SIZE_KEY: 16,
    FLASH_ROW_EPILOGUE_WARPS_KEY: "softmax",
    FLASH_EPI_TMA_KEY: True,
    FLASH_EPI_STG_KEY: False,
    FLASH_WAIT_HINT_KEY: 10_000_000,
}

# Each emitter-level policy seeded once so quick tuning cannot miss a source
# variant merely because it is conditionally active with another field:
# ``(key, value, the body that hosts it)``.
SEED_SOURCE_VARIANTS: tuple[tuple[str, object, Mapping[str, object]], ...] = (
    # The flat ws_overlap grid is a distinct source variant (prologue TMA
    # loads issued before the TMEM barrier, TMEM freed under the output
    # drain) from the persistent ws_overlap loop the family seed carries.
    # It is seeded with the staged O epilogue the measured flat winners
    # use; the fragment default follows the surface's default family,
    # which is the direct store wherever that family is fa4.
    (
        FLASH_PERSISTENT_KEY,
        False,
        {FLASH_PIPELINE_FAMILY_KEY: "ws_overlap", FLASH_EPI_STG_KEY: True},
    ),
    (FLASH_ROWMAX_KEY, "tmem", {FLASH_S_LOAD_REP_KEY: 32}),
    (
        FLASH_STAT_TRANSPORT_KEY,
        "single",
        {FLASH_SOFTMAX_DISC_KEY: False, FLASH_EXP2_PACKET_KEY: "1x1"},
    ),
    (
        FLASH_STAT_TRANSPORT_KEY,
        "single",
        {
            FLASH_SOFTMAX_DISC_KEY: False,
            FLASH_EXP2_PACKET_KEY: "1x1",
            FLASH_E2E_SCHEDULE_KEY: "xu",
            FLASH_E2E_OFFSET_KEY: 0,
            FLASH_E2E_OFFSET0_KEY: 0,
        },
    ),
    (FLASH_PERSISTENT_LOOP_KEY, "counted", {}),
    (
        FLASH_SP_ROW_SUM_KEY,
        "whole",
        {FLASH_SOFTMAX_DISC_KEY: False},
    ),
    (FLASH_SOFTMAX_SETUP_KEY, "stage_local", {}),
    (
        FLASH_EPI_TMA_SETUP_KEY,
        "role_local",
        {FLASH_EPI_TMA_KEY: True},
    ),
)

# KV-tile-pruned masks also run the fa4 pair body: seeded once with the knobs
# the measured causal fa4 winners share (two-deep chunked PASS2, split P
# arrival, TMA epilogue, K-first loads, interleaved MMA) and the 8/2 exp2
# schedule, which the pinned sweep on the sliding-window and prefix-LM shapes
# preferred over the causal winners' 16/8 (0.303 vs 0.329 ms, 0.933 vs 1.015
# ms at s8192).  The hd64 hybrid packet (the causal winners' structure:
# descending three-segment split, 16/8 cadence) carries the same knobs; the
# builder adds it only where the packet is a searchable choice (it is a
# polynomial exp2 seed and fast_math-only: where the fast_math surface offers
# it and the setting is off, the same structure is seeded on the surface's
# default packet; a body that never offers the packet -- hd128, the
# document-id mask -- gets neither variant).  Measured at s8192 on B200:
# prefix-LM 0.930 -> 0.876 ms (6% faster than 8/2, ahead of flex-cute),
# sliding-window 0.300 -> 0.316 ms (the 16-iteration cadence does not fit its
# ~10-tile work items), so both structures are seeded and measured.
SEED_PRUNED_FA4_STRUCTURE: Mapping[str, object] = {
    FLASH_DISC_PIPE_KEY: 2,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_EPI_TMA_KEY: True,
    FLASH_FIRST_LOAD_ORDER_KEY: 3,
    FLASH_MMA_INTERLEAVE_KEY: True,
}

# The per-row KV lower bound (document attention): the two campaign winners'
# structures on B200 (b1 h16 hd64 bf16; s16384: the XU cadence with the
# single-stage ring and a five-deep K/V ring; s8192: the 8/2 cadence with a
# three-deep ring, chained roles and the 8.0 rescale threshold), each seeded
# with the global grid and with the segment-aligned tiling
# (``cute_flash_row_bound_tiling``), which loads every document at its own
# row offset and keeps the TMA store for whole tiles. The pinned pruned seed
# above measured 0.172 ms against the found 0.156 at s16384 without this
# structure.
_ROW_BOUND_COMMON: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_EXP2_PACKET_KEY: "1x1",
    FLASH_STAT_TRANSPORT_KEY: "ring2",
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_DISC_PIPE_KEY: 2,
    FLASH_EPI_TMA_KEY: True,
    FLASH_MMA_INTERLEAVE_KEY: True,
    FLASH_PRECOMPUTE_QK_DESC_KEY: True,
    FLASH_ROLE_MAP_KEY: "fa4",
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
}
SEED_ROW_BOUND_LONG: Mapping[str, object] = {
    **_ROW_BOUND_COMMON,
    FLASH_E2E_SCHEDULE_KEY: "xu",
    FLASH_KV_STAGE_KEY: 5,
    FLASH_CAUSAL_LPT_SWIZZLE_KEY: 16,
    FLASH_FIRST_LOAD_ORDER_KEY: 2,
    FLASH_RESCALE_THRESHOLD_KEY: 32.0,
    FLASH_SOFTMAX_REGS_KEY: 176,
    FLASH_CORR_REGS_KEY: 120,
    FLASH_OTHER_REGS_KEY: 40,
    FLASH_CORR_TILE_SIZE_KEY: 16,
    FLASH_ROLE_CHAIN_KEY: False,
}
SEED_ROW_BOUND_SHORT: Mapping[str, object] = {
    **_ROW_BOUND_COMMON,
    FLASH_E2E_SCHEDULE_KEY: "8/2",
    FLASH_E2E_OFFSET_KEY: 4,
    FLASH_KV_STAGE_KEY: 3,
    FLASH_FIRST_LOAD_ORDER_KEY: 3,
    FLASH_WAIT_HINT_KEY: 10_000_000,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
    FLASH_SOFTMAX_REGS_KEY: 200,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 48,
    FLASH_CORR_TILE_SIZE_KEY: 8,
    FLASH_ROLE_CHAIN_KEY: True,
}

# The FlexAttention list walk: the fa4 pair body's defaults come from the
# family seed; the pruned-interval campaign structure is seeded on it too
# (8x2 packets, chunked body with the two-deep PASS2 pipe, split P arrival,
# descending walk, TMA epilogue, K-first loads, interleaved MMA) at both
# measured cadences: the causal winners' 16/8 and the 8/2 the pinned pruned
# sweep preferred on short work items. No loop split (an owned no-op here)
# and the inherited masked cadence.  Persistent with a deeper ring than the
# fragment default: the 2026-10-12 attention-gym campaign's list-walk winners
# were all persistent with kv_stage 4-8 (TMA epilogue); the split-KV knob
# stays at the fragment default, the grid-derived seeded split
# (``_flash_kv_split_seed``: 1 once the pair items fill the device).  Every
# list-walk body's ring is capped by the builder to the shape's aliased
# budget; the staged output tile count a body leaves unstated is the shared
# tile where the slot it frees deepens the seeded ring (head dim 128: 3 -> 4),
# two tiles otherwise.
SEED_FLEX_FA4_STRUCTURE: Mapping[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_PERSISTENT_KEY: True,
    FLASH_KV_STAGE_KEY: 5,
    FLASH_EXP2_PACKET_KEY: "8x2",
    FLASH_SOFTMAX_DISC_KEY: True,
    FLASH_DISC_PIPE_KEY: 2,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_P_STORE_REP_KEY: 16,
    FLASH_S_LOAD_REP_KEY: 32,
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_EPI_TMA_KEY: True,
    FLASH_FIRST_LOAD_ORDER_KEY: 3,
    FLASH_MMA_INTERLEAVE_KEY: True,
}
# The campaign winners' bodies as seeds: the short walk (flamingo 8K, about 7
# blocks per item: 16/4 cadence, 4x2 packets, four-deep PASS2 pipe,
# 160/128/64 registers, ring 7) and the medium walk (sta2d, 18 blocks per
# item: 8/2 cadence, split P arrival, 8-row correction tiles, ring 8).
SEED_FLEX_SHORT_WALK: Mapping[str, object] = {
    **SEED_FLEX_FA4_STRUCTURE,
    FLASH_E2E_SCHEDULE_KEY: "16/4",
    FLASH_E2E_OFFSET_KEY: 13,
    FLASH_E2E_OFFSET0_KEY: 6,
    FLASH_EXP2_PACKET_KEY: "4x2",
    FLASH_DISC_PIPE_KEY: 4,
    FLASH_SPLIT_P_ARRIVE_KEY: False,
    FLASH_P_STORE_REP_KEY: 32,
    FLASH_FIRST_LOAD_ORDER_KEY: 0,
    FLASH_WAIT_HINT_KEY: 10_000_000,
    FLASH_KV_STAGE_KEY: 7,
    FLASH_RESCALE_THRESHOLD_KEY: 16.0,
    FLASH_SOFTMAX_REGS_KEY: 160,
    FLASH_CORR_REGS_KEY: 128,
    FLASH_OTHER_REGS_KEY: 64,
    FLASH_CORR_TILE_SIZE_KEY: 32,
}
SEED_FLEX_MEDIUM_WALK: Mapping[str, object] = {
    **SEED_FLEX_FA4_STRUCTURE,
    FLASH_E2E_SCHEDULE_KEY: "8/2",
    FLASH_E2E_OFFSET_KEY: 4,
    FLASH_E2E_OFFSET0_KEY: 1,
    FLASH_DISC_PIPE_KEY: 1,
    FLASH_P_STORE_REP_KEY: 32,
    FLASH_KV_STAGE_KEY: 8,
    FLASH_RESCALE_THRESHOLD_KEY: 4.0,
    FLASH_SOFTMAX_REGS_KEY: 160,
    FLASH_CORR_REGS_KEY: 112,
    FLASH_OTHER_REGS_KEY: 80,
    FLASH_CORR_TILE_SIZE_KEY: 8,
}
# The same bodies with the register-resident PASS2 (one TMEM read of S per
# step): FA4's register split that holds the row, the mixed Rep32/Rep16 store
# with the 3/4 release, and 8-column correction tiles. With 64 correction
# registers the 32-column correction epilogue spills next to a resident
# softmax (STACK 16, flamingo 8K 218 -> 212 us at 8 columns, GPU 5,
# 2026-10-14); 8 columns do not.
SEED_FLEX_RESIDENT_REGS: Mapping[str, object] = {
    FLASH_S_RESIDENT_KEY: True,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_SOFTMAX_REGS_KEY: 200,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 48,
    FLASH_CORR_TILE_SIZE_KEY: 8,
    FLASH_RESCALE_CHUNK_COLS_KEY: 8,
}
# FA4's own block-sparse configuration (flash_attn.cute at hd64: a 10-slot
# K/V ring, split P arrival at 3/4, rescale threshold 8, 200/64/48 registers,
# Q loaded with the first K/V): pinned on the flamingo 8K winner with the
# item words it measured 213.0 us vs 224.3 (GPU 6, 2026-10-13), and its
# resident twin.
SEED_FLEX_FA4_TEMPLATE: Mapping[str, object] = {
    **SEED_FLEX_SHORT_WALK,
    FLASH_KV_STAGE_KEY: 10,
    FLASH_SPLIT_P_ARRIVE_KEY: True,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_SOFTMAX_REGS_KEY: 200,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 48,
    FLASH_FIRST_LOAD_ORDER_KEY: 3,
}
# The long walk (the 2026-10-16 campaign's causal 128K hd128 winner, 1024 KV
# tiles per item): one CTA per pair item (the persistent strided order
# streams every head's K/V at once and thrashes L2 on long rows: 45 vs 34
# ms), a two-slot ring, the 16-column cadence with a one-deep PASS2 pipe and
# Rep16 loads/stores, split P arrival, 168/80/32 registers and the ascending
# order; the cadence emulates 2 of 16 pairs (the MUFU pipe is not the bound
# on long rows: 16/6 measured 1-1.5% behind 16/2 and the hardware-only
# cadence there).
SEED_FLEX_LONG_WALK: Mapping[str, object] = {
    **SEED_FLEX_FA4_STRUCTURE,
    FLASH_PERSISTENT_KEY: False,
    FLASH_KV_STAGE_KEY: 2,
    FLASH_E2E_SCHEDULE_KEY: "16/2",
    FLASH_E2E_OFFSET_KEY: 2,
    FLASH_E2E_OFFSET0_KEY: 9,
    FLASH_EXP2_PACKET_KEY: "4x2",
    FLASH_DISC_PIPE_KEY: 1,
    FLASH_S_LOAD_REP_KEY: 16,
    FLASH_P_STORE_REP_KEY: 16,
    FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
    FLASH_FIRST_LOAD_ORDER_KEY: 0,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_SOFTMAX_REGS_KEY: 168,
    FLASH_CORR_REGS_KEY: 80,
    FLASH_OTHER_REGS_KEY: 32,
    FLASH_CORR_TILE_SIZE_KEY: 8,
    FLASH_RESCALE_CHUNK_COLS_KEY: 32,
}
# The pass-2c causal 128K hd128 winner (2026-10-17, 33.14 ms vs FA4 32.89
# once the shared region was searchable): the long walk's one CTA per pair
# item and Rep16 loads/stores with a three-slot ring, the 8/2 cadence, the
# shared softmax region, the rolled chunk loop, 184/88/32 registers and
# 16-row correction tiles.  Pinned on the two 32K GQA cells it runs the
# steady step 1.6-3% faster than their pass-2b cold winners (gqa4-causal 2781
# vs 2805-2845 cycles per pair step, sliding-window 2749-2762 vs 2798-2805,
# mid-kernel stamps), so every list-walk search seeds it, and its persistent
# twin (the sliding-window winner's walk form).
SEED_FLEX_P2C_WALK: Mapping[str, object] = {
    **SEED_FLEX_LONG_WALK,
    FLASH_KV_STAGE_KEY: 3,
    FLASH_E2E_SCHEDULE_KEY: "8/2",
    FLASH_E2E_OFFSET_KEY: 1,
    FLASH_E2E_OFFSET0_KEY: 1,
    FLASH_SOFTMAX_REGION_KEY: "shared",
    FLASH_FLEX_ROLLED_CHUNKS_KEY: True,
    FLASH_SOFTMAX_REGS_KEY: 184,
    FLASH_CORR_REGS_KEY: 88,
    FLASH_OTHER_REGS_KEY: 32,
    FLASH_CORR_TILE_SIZE_KEY: 16,
}
# The causal 128K hd128 winner of the shared-region autotune (W, 2026-10-17,
# 33.84 ms cold, 32.74 pinned vs FA4 33.4 in the same run): the long walk's
# one CTA per pair item and Rep16 loads/stores with a three-slot ring, the
# 16/6 cadence, the shared softmax region, 152/88/56 registers, the
# descending order and a 32 rescale threshold.  Pinned on the 32K GQA cells
# it runs the steady step 0.5-3% faster than their pass-2b winners
# (gqa4-causal 2761, sliding-window 2781-2791 cycles per pair step).
SEED_FLEX_W_WALK: Mapping[str, object] = {
    **SEED_FLEX_LONG_WALK,
    FLASH_KV_STAGE_KEY: 3,
    FLASH_E2E_SCHEDULE_KEY: "16/6",
    FLASH_E2E_OFFSET_KEY: 13,
    FLASH_E2E_OFFSET0_KEY: 13,
    FLASH_EXP2_PACKET_KEY: "4x1",
    FLASH_SOFTMAX_REGION_KEY: "shared",
    FLASH_CAUSAL_KV_ORDER_KEY: "descending",
    FLASH_FIRST_LOAD_ORDER_KEY: 5,
    FLASH_RESCALE_THRESHOLD_KEY: 32.0,
    FLASH_SOFTMAX_REGS_KEY: 152,
    FLASH_CORR_REGS_KEY: 88,
    FLASH_OTHER_REGS_KEY: 56,
    FLASH_CORR_TILE_SIZE_KEY: 8,
    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
}
# The mid-length grouped-query walk (the 2026-10-16 campaign's pass-3
# gqa-causal-bshd winner: b4 h32 kv8 s4096 hd128, 32 KV tiles per item): the
# persistent walk with the shared softmax region and the rolled chunk loop,
# the 8/2 cadence on 8x1 packets with the two-deep PASS2 pipe, Rep32 S loads,
# the descending order, a three-slot ring with two output tiles, 184/64/40
# registers and 8-row correction tiles, without the hooked PASS1's all-kept
# vote.  Pinned on the cell it runs 0.628 ms where the nearest seed above
# (the p2c walk's persistent twin) runs 0.655 and the per-family default
# 0.661 (GPU 6, 2026-10-18); the two cold searches that lacked it ended at
# 0.646 / 0.654.
SEED_FLEX_MID_WALK: Mapping[str, object] = {
    **SEED_FLEX_FA4_STRUCTURE,
    FLASH_FLEX_HOOK_FAST_FULL_KEY: False,
    FLASH_KV_STAGE_KEY: 3,
    FLASH_O_STAGE_KEY: 2,
    FLASH_E2E_SCHEDULE_KEY: "8/2",
    FLASH_E2E_OFFSET_KEY: 0,
    FLASH_E2E_OFFSET0_KEY: 1,
    FLASH_EXP2_PACKET_KEY: "8x1",
    FLASH_SOFTMAX_REGION_KEY: "shared",
    FLASH_FLEX_ROLLED_CHUNKS_KEY: True,
    FLASH_RESCALE_THRESHOLD_KEY: 8.0,
    FLASH_RESCALE_CHUNK_COLS_KEY: 16,
    FLASH_SOFTMAX_REGS_KEY: 184,
    FLASH_CORR_REGS_KEY: 64,
    FLASH_OTHER_REGS_KEY: 40,
    FLASH_CORR_TILE_SIZE_KEY: 8,
}

# A grid of fewer 128-row tiles than SMs is bound by per-CTA fixed latency:
# the 64-row tile of the flat ws_overlap grid (twice the CTAs, half the
# softmax per CTA), with the one-pass softmax where the sequence is exactly
# two KV tiles (the builder adds it where it is a searchable choice). The
# 64-row tile is legal only with the staged O epilogue (a direct-store
# default would canonicalize the tile back to 128 rows and the one-pass
# softmax off); the two-stage K/V ring measured best for it.
SEED_SMALL_GRID_WS: Mapping[str, object] = {
    FLASH_PERSISTENT_KEY: False,
    FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
    FLASH_EPI_STG_KEY: True,
    FLASH_KV_STAGE_KEY: 2,
}

# Every template by name (the registry the knob tests validate).
FLASH_SEED_TEMPLATES: dict[str, Mapping[str, object]] = {
    "modifier_causal_split": SEED_MODIFIER_CAUSAL_SPLIT,
    **{
        f"asymmetric_pair_{index}": body
        for index, body in enumerate(SEED_ASYMMETRIC_PAIR_BODIES)
    },
    "causal_stagger_structure": SEED_CAUSAL_STAGGER_STRUCTURE,
    "causal_head_pair_structure": SEED_CAUSAL_HEAD_PAIR_STRUCTURE,
    "dense_hd64_local_tma": SEED_DENSE_HD64_LOCAL_TMA,
    "causal_hd64_fa4_pins": SEED_CAUSAL_HD64_FA4_PINS,
    "causal_hd64_s2048": SEED_CAUSAL_HD64_BODIES[0],
    "causal_hd64_s8192": SEED_CAUSAL_HD64_BODIES[1],
    "causal_hd64_hybrid": SEED_CAUSAL_HD64_HYBRID_BODY,
    "dense_hd128_two_tile": SEED_DENSE_HD128_TWO_TILE,
    **{
        f"source_variant_{index}": body
        for index, (_key, _value, body) in enumerate(SEED_SOURCE_VARIANTS)
    },
    "pruned_fa4_structure": SEED_PRUNED_FA4_STRUCTURE,
    "row_bound_long": SEED_ROW_BOUND_LONG,
    "row_bound_short": SEED_ROW_BOUND_SHORT,
    "flex_fa4_structure": SEED_FLEX_FA4_STRUCTURE,
    "flex_short_walk": SEED_FLEX_SHORT_WALK,
    "flex_medium_walk": SEED_FLEX_MEDIUM_WALK,
    "flex_resident_regs": SEED_FLEX_RESIDENT_REGS,
    "flex_fa4_template": SEED_FLEX_FA4_TEMPLATE,
    "flex_long_walk": SEED_FLEX_LONG_WALK,
    "flex_p2c_walk": SEED_FLEX_P2C_WALK,
    "flex_w_walk": SEED_FLEX_W_WALK,
    "flex_mid_walk": SEED_FLEX_MID_WALK,
    "small_grid_ws": SEED_SMALL_GRID_WS,
}
