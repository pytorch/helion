"""The CuTe flash-attention knob registry.

Every flash config key is an importable constant here; ``FLASH_KNOBS`` is the
ordered registry of the searched knobs (a knob's position is its coverage
axis), ``FLASH_KNOB_WITNESSES`` the ordered legality witnesses the structural
coverage design reads through the generated ``FLASH_AUTOTUNE_VALUE_DEPENDENCIES``,
and the key partitions (``FLASH_AUTOTUNE_CONFIG_KEYS``, the legacy and derived
keys, ``FLASH_CONFIG_KEYS``, the backward keys) and the choice tables of the
knobs whose values are names (pipeline families, exp2 packets, schedules,
statistics transports, role maps) are declared once.  The module is an
import leaf: ``cute_flash`` (the resolver, surface and seeds),
``flash_tuning`` (the target policy records validate against the choice
tables) and the autotuner read it; it reads nothing of theirs.  The legacy spellings a fixed
config may still carry are folded by the adapters here
(``upgrade_legacy_flash_config``, ``flash_legacy_e2e_params``,
``fold_flash_ws_inactive_values``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from .cute_flash import FlashAttentionConfig

# The register-MMA row programs' pipeline family (``cute_flash_row_mma``).
ROW_MMA_FAMILY = "row_mma"


class FlashPipelineFamilyFlags(NamedTuple):
    topology: str
    separate_kv_rings: bool = False
    causal_two_cta: bool = False
    use_2cta_instrs: bool = False
    use_cga2_local_cta: bool = False
    use_clc_scheduler: bool = False
    local_tma_partition: bool = False
    tensor_4d_tma: bool = False
    # One query tile per work item, two softmax warpgroups on alternating KV
    # steps, separate QK/PV issue warps (see ``cute_flash_alt``).
    alternating_warpgroups: bool = False


FLASH_PIPELINE_FAMILY_FLAGS: dict[str, FlashPipelineFamilyFlags] = {
    "ws_overlap": FlashPipelineFamilyFlags("ws_overlap"),
    "fa4": FlashPipelineFamilyFlags("fa4"),
    # Register-MMA row programs (mma.sync, cp.async, no TMA/TMEM): 8- or
    # 16-row query tiles whose warps split the keys; the latency-bound
    # structure for grids too small to fill the device.
    ROW_MMA_FAMILY: FlashPipelineFamilyFlags(ROW_MMA_FAMILY),
    "fa4_deep_1cta": FlashPipelineFamilyFlags("fa4", separate_kv_rings=True),
    "fa4_2cta_causal": FlashPipelineFamilyFlags(
        "fa4", causal_two_cta=True, use_2cta_instrs=True
    ),
    "fa4_tma_4d": FlashPipelineFamilyFlags("fa4", tensor_4d_tma=True),
    "fa4_local_tma": FlashPipelineFamilyFlags("fa4", local_tma_partition=True),
    "fa4_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4", local_tma_partition=True, tensor_4d_tma=True
    ),
    "fa4_cga2_local": FlashPipelineFamilyFlags("fa4", use_cga2_local_cta=True),
    "fa4_cga2_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_cga2_local_cta=True, tensor_4d_tma=True
    ),
    "fa4_2cta": FlashPipelineFamilyFlags("fa4", use_2cta_instrs=True),
    "fa4_2cta_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_2cta_instrs=True, tensor_4d_tma=True
    ),
    "fa4_clc": FlashPipelineFamilyFlags("fa4", use_clc_scheduler=True),
    "fa4_clc_tma_4d": FlashPipelineFamilyFlags(
        "fa4", use_clc_scheduler=True, tensor_4d_tma=True
    ),
    "fa4_clc_local_tma": FlashPipelineFamilyFlags(
        "fa4", use_clc_scheduler=True, local_tma_partition=True
    ),
    "fa4_clc_local_tma_4d": FlashPipelineFamilyFlags(
        "fa4",
        use_clc_scheduler=True,
        local_tma_partition=True,
        tensor_4d_tma=True,
    ),
    # New families go last: autotune seeds and coverage rows index the family
    # list by position.
    "fa4_alt": FlashPipelineFamilyFlags("fa4", alternating_warpgroups=True),
    # The single-query-tile body with the cooperative two-CTA MMA: each CTA
    # of the pair holds 128 query rows and half of every K/V tile, the leader
    # issues M=256 MMAs over both (the structure of FA4's hd256 kernel).
    "ws_2cta": FlashPipelineFamilyFlags("ws_overlap", use_2cta_instrs=True),
}
FLASH_PIPELINE_FAMILIES = tuple(FLASH_PIPELINE_FAMILY_FLAGS)
FLASH_AUTOTUNE_PIPELINE_FAMILIES = FLASH_PIPELINE_FAMILIES


def _flash_pipeline_family_flags(
    family: object,
) -> FlashPipelineFamilyFlags | None:
    return FLASH_PIPELINE_FAMILY_FLAGS.get(family) if isinstance(family, str) else None


_FLASH_EXP2_PACKET_PARAMS: dict[str, tuple[int, int]] = {
    "1x1": (1, 1),
    "4x1": (4, 1),
    "4x2": (4, 2),
    "8x1": (8, 1),
    "8x2": (8, 2),
}
# Polynomial exp2 packets.  The degree-2 minimax pass (``_POLY_EX2_DEG2``;
# the ``deg1`` routes are aliases of it) is an approximation of
# ``ex2.approx``, not a bitwise reordering: its bf16 probabilities and fp32
# row statistics are not those of the exact pass (the bf16 output stays at
# its rounding floor; an fp32 LSE output moves by up to 2.5e-3).  Under the
# numerics rule an approximation is not a searchable autotuner choice without
# the user's opt-in, so every packet that runs the polynomial
# (``_FLASH_FAST_MATH_EXP2_PACKETS`` below: the dense fp16 hd64 packets, the
# causal hd64 degree-2 and hybrid packets, the pruned-interval hd64 hybrid
# packet, the dense bf16 hd128 degree-2 packet and the three polynomial
# cadences of the causal hd128 / (192, 128) resident packet) reaches the
# flash surface only under the kernel's ``fast_math`` setting.  The setting
# is threaded as an explicit ``fast_math`` keyword through every site that
# must agree: the fragment choices and search choices and the target-policy
# value widening (``flash_autotune_fragments``), the compound packet
# overrides (``_flash_compound_exp2_packet_overrides``: ``{}`` means "not
# effective"), the resolver (``resolve_flash_config``: a request resolves to
# the exact default packet, which also covers ``HELION_CUTE_FLASH_EXP2_PACKET``),
# the seeds (generic, resident pair, causal hd64 hybrid, pruned source
# variant and the sm_103 tables, each keeping its structure with an exact
# substitute packet where legal), ``flash_config_from_config`` and the
# codegen entry.  ``ConfigSpec`` records the setting when the flash search is
# enabled and passes it to every call it makes.  The hardware-only resident
# sibling (``..._prefetch2_xu_early_acquire``, residue zero: every exp2 on
# the XU pipe, ``degree2`` false in ``_flash_disc_exp2_codegen_params``) runs
# no polynomial and is exempt.
_FLASH_DEG2_EXP2_PACKET = "deg2_16x6"
_FLASH_HYBRID_EXP2_PACKET = "hybrid_deg1_16x8"
_FLASH_DEG1_EXP2_PACKET = "deg1_16x8"
_FLASH_DEG1_SHORT_CORR10_EXP2_PACKET = "deg1_8x2_corr10"
_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET = (
    "causal_hd128_resident3_013_prefetch2_deg2_early_acquire"
)
# Cadence siblings of the causal resident schedule: the same resident
# softmax/correction structure with fewer degree-2 polynomial pairs per
# 16-element period (two or one instead of three) or none at all (every exp2
# on the hardware XU pipe, FA4's choice at head_dim 192 where the longer QK
# MMA gives the softmax warps the slack).  The cadence is part of the packet,
# as for ``deg1_16x8`` / ``deg1_8x2_corr10``.
_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X4_EARLY_ACQUIRE_EXP2_PACKET = (
    "causal_hd128_resident3_013_prefetch2_deg2_16x4_early_acquire"
)
_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET = (
    "causal_hd128_resident3_013_prefetch2_deg2_16x2_early_acquire"
)
_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET = (
    "causal_hd128_resident3_013_prefetch2_xu_early_acquire"
)
_FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKET_CHOICES: tuple[str, ...] = (
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET,
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X4_EARLY_ACQUIRE_EXP2_PACKET,
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET,
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET,
)
_FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKETS = frozenset(
    _FLASH_CAUSAL_HD128_RESIDENT_EXP2_PACKET_CHOICES
)
_FLASH_DEG1_EXP2_PACKETS = frozenset(
    (_FLASH_DEG1_EXP2_PACKET, _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET)
)
_FLASH_DEG1_EXP2_OFFSET = 0
_FLASH_DEG1_EXP2_OFFSET0 = 10
_FLASH_MANUAL_EXP2_PACKET_PARAMS: dict[str, tuple[int, int]] = {
    _FLASH_DEG2_EXP2_PACKET: (8, 3),
    _FLASH_HYBRID_EXP2_PACKET: (8, 4),
    _FLASH_DEG1_EXP2_PACKET: (8, 4),
    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET: (8, 2),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET: (
        8,
        3,
    ),
    # The emulation batch is the residue pairs of one eight-pair batch.
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X4_EARLY_ACQUIRE_EXP2_PACKET: (
        8,
        2,
    ),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET: (
        8,
        1,
    ),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET: (8, 1),
}
# ``(e2e_freq, e2e_res)``: a residue of zero is the hardware-only cadence
# (``cute_flash_e2e_schedule`` name ``xu``; its phase offsets are dead).
_FLASH_MANUAL_EXP2_PACKET_SCHEDULES: dict[str, tuple[int, int]] = {
    _FLASH_DEG2_EXP2_PACKET: (16, 6),
    _FLASH_HYBRID_EXP2_PACKET: (16, 8),
    _FLASH_DEG1_EXP2_PACKET: (16, 8),
    _FLASH_DEG1_SHORT_CORR10_EXP2_PACKET: (8, 2),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_EARLY_ACQUIRE_EXP2_PACKET: (
        16,
        6,
    ),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X4_EARLY_ACQUIRE_EXP2_PACKET: (
        16,
        4,
    ),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_DEG2_16X2_EARLY_ACQUIRE_EXP2_PACKET: (
        16,
        2,
    ),
    _FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET: (16, 0),
}
# The compound packets that run the polynomial exp2 on some pairs of every
# period (a residue greater than zero): reachable under ``fast_math`` only.
# Every compound packet but the hardware-only resident sibling.
_FLASH_FAST_MATH_EXP2_PACKETS: frozenset[str] = frozenset(
    packet
    for packet, (_, e2e_res) in _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.items()
    if e2e_res > 0
)
assert (
    frozenset(_FLASH_MANUAL_EXP2_PACKET_PARAMS)
    - {_FLASH_CAUSAL_HD128_RESIDENT3_013_PREFETCH2_XU_EARLY_ACQUIRE_EXP2_PACKET}
    == _FLASH_FAST_MATH_EXP2_PACKETS
)


def _flash_e2e_schedule_name(exp2_impl: str, e2e_freq: int, e2e_res: int) -> str:
    if exp2_impl != "split" or e2e_res == 0:
        return "xu"
    return f"{e2e_freq}/{e2e_res}"


def _flash_manual_exp2_packet_schedule_name(packet: str) -> str:
    """Return the canonical ``cute_flash_e2e_schedule`` name of a compound packet."""
    e2e_freq, e2e_res = _FLASH_MANUAL_EXP2_PACKET_SCHEDULES[packet]
    return _flash_e2e_schedule_name("split", e2e_freq, e2e_res)


def _flash_exp2_packet_admitted(packet: object, fast_math: bool) -> bool:
    """Whether ``packet`` may appear on a surface with the given setting.

    The polynomial packets (``_FLASH_FAST_MATH_EXP2_PACKETS``) run an
    approximation of the hardware exp2: every surface that offers, seeds,
    canonicalizes or resolves a packet asks this predicate and withholds
    them unless the kernel's ``fast_math`` setting is on.
    """
    return fast_math or packet not in _FLASH_FAST_MATH_EXP2_PACKETS


def flash_exp2_packet_is_compound(packet: object) -> bool:
    """Return whether a packet owns structural schedule requirements."""
    return packet in _FLASH_MANUAL_EXP2_PACKET_PARAMS


# The exp2 packets a config may name: the plain packed pairs and the
# compound (polynomial / resident) packets.
FLASH_EXP2_PACKET_CHOICES: tuple[str, ...] = (
    *_FLASH_EXP2_PACKET_PARAMS,
    *_FLASH_MANUAL_EXP2_PACKET_PARAMS,
)
# The exp2 schedule names the surface offers (``flash_autotune_fragments``):
# the generic cadences and the compound packets' own.
FLASH_E2E_SCHEDULE_CHOICES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "16/4",
            "8/2",
            "16/2",
            "xu",
            *(
                _flash_manual_exp2_packet_schedule_name(packet)
                for packet in _FLASH_MANUAL_EXP2_PACKET_SCHEDULES
            ),
        )
    )
)
# The masked-tile cadence names of a variable-trip-count plan: inherit the
# main cadence, or one of the schedules above.
FLASH_MASKED_E2E_SCHEDULE_CHOICES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            "inherit",
            "xu",
            "16/4",
            "8/2",
            *(
                _flash_manual_exp2_packet_schedule_name(packet)
                for packet in _FLASH_MANUAL_EXP2_PACKET_SCHEDULES
            ),
        )
    )
)
FLASH_STAT_TRANSPORT_CHOICES: tuple[str, ...] = ("ring2", "single", "single_final")
FLASH_ROLE_MAP_CHOICES: tuple[str, ...] = ("helion", "fa4")

# ---------------------------------------------------------------------------
# Autotune surface for the flash-attention config.
#
# Mirrors the ``Tcgen05WarpSpec`` pattern in ``strategies.py`` /
# ``tcgen05_config.py``. Independent arithmetic fields retain individual keys;
# structural scheduling is represented by one compound pipeline-family key.
# ``FLASH_CONFIG_KEYS`` aggregates the active and legacy inputs, and
# ``flash_config_from_config`` reconstructs the effective dataclass.
#
# Gating: these keys are ONLY inserted into the search surface when
# ``ConfigSpec.cute_flash_search_enabled`` is True (set when the flash detector
# fires; default False). With the flag off the keys never appear, so the config
# Mapping never carries them and ``resolve_flash_config`` behaves byte-identical
# to the pre-autotune env-only path.
# ---------------------------------------------------------------------------
FLASH_S_STAGE_KEY = "cute_flash_s_stage"
FLASH_KV_STAGE_KEY = "cute_flash_kv_stage"
FLASH_PERSISTENT_KEY = "cute_flash_persistent"
FLASH_PERSISTENT_CTAS_PER_SM_KEY = "cute_flash_persistent_ctas_per_sm"
FLASH_RECOMPUTE_TILE_COORDS_KEY = "cute_flash_recompute_tile_coords"
FLASH_E2E_SCHEDULE_KEY = "cute_flash_e2e_schedule"
FLASH_MASKED_E2E_SCHEDULE_KEY = "cute_flash_masked_e2e_schedule"
FLASH_E2E_OFFSET_KEY = "cute_flash_e2e_offset"
FLASH_E2E_OFFSET0_KEY = "cute_flash_e2e_offset0"
# Legacy fixed-config override keys. The autotuner uses the paired schedule key above.
FLASH_EXP2_IMPL_KEY = "cute_flash_exp2_impl"
FLASH_E2E_FREQ_KEY = "cute_flash_e2e_freq"
FLASH_E2E_RES_KEY = "cute_flash_e2e_res"
FLASH_MMA_INTERLEAVE_KEY = "cute_flash_mma_interleave"
FLASH_Q_TILE_COUNT_KEY = "cute_flash_q_tile_count"
FLASH_STAT_TRANSPORT_KEY = "cute_flash_stat_transport"
FLASH_WAIT_HINT_KEY = "cute_flash_wait_hint"
FLASH_EXP2_PACKET_KEY = "cute_flash_exp2_packet"
# Compound schedule family. Legacy topology/cluster/TMA keys normalize into it.
FLASH_PIPELINE_FAMILY_KEY = "cute_flash_pipeline_family"
# fa4 win (commit 38ff4d1a): the topology selector + the two fa4 perf levers.
FLASH_TOPOLOGY_KEY = "cute_flash_topology"
FLASH_SOFTMAX_DISC_KEY = "cute_flash_softmax_disc"
FLASH_DISC_PIPE_KEY = "cute_flash_disc_pipe"
FLASH_SPLIT_P_ARRIVE_KEY = "cute_flash_split_p_arrive"
FLASH_P_CHUNK_ARRIVE_KEY = "cute_flash_p_chunk_arrive"
FLASH_UNEVEN_KV_RING_KEY = "cute_flash_uneven_kv_ring"
FLASH_O_OVER_Q_KEY = "cute_flash_o_over_q"
FLASH_ROW_BOUND_TILING_KEY = "cute_flash_row_bound_tiling"
# Tilings of a per-row KV lower-bound plan (``FlashAttentionConfig.row_bound_tiling``).
_FLASH_ROW_BOUND_TILINGS = ("global", "segment")
FLASH_P_STORE_REP_KEY = "cute_flash_p_store_rep"
FLASH_S_LOAD_REP_KEY = "cute_flash_s_load_rep"
FLASH_PRECOMPUTE_QK_DESC_KEY = "cute_flash_precompute_qk_desc"
FLASH_FIRST_LOAD_ORDER_KEY = "cute_flash_first_load_order"
# 0..4 permute the first work item's Q0/K0/Q1/V0 TMA issue order; 5 is the
# staged order: K0 and Q0 first, then (first work item only) wait for the MMA
# warp's signal that they landed before issuing Q1 and V0, so the kernel-start
# HBM burst carries only the bytes the first score tile needs.
FLASH_FIRST_LOAD_ORDER_CHOICES = (0, 1, 2, 3, 4, 5)
FLASH_FIRST_LOAD_ORDER_STAGED = 5
FLASH_KV_ORDER_KEY = "cute_flash_kv_order"
FLASH_EPI_TMA_KEY = "cute_flash_epi_tma"
FLASH_O_STAGE_KEY = "cute_flash_o_stage"
# Staged output tiles of the TMA epilogue: one per query stage, or one shared.
FLASH_O_STAGE_CHOICES: tuple[int, ...] = (2, 1)
FLASH_EPI_STG_KEY = "cute_flash_epi_stg"
FLASH_EPI_STG_STORE_KEY = "cute_flash_epi_stg_store"
FLASH_EPI_STG_GMEM_KEY = "cute_flash_epi_stg_gmem"
FLASH_RESCALE_THRESHOLD_KEY = "cute_flash_rescale_threshold"
FLASH_SKIP_RESCALE_STATS_KEY = "cute_flash_skip_rescale_stats"
FLASH_RESCALE_CHUNK_COLS_KEY = "cute_flash_rescale_chunk_cols"
FLASH_SOFTMAX_REGS_KEY = "cute_flash_softmax_regs"
FLASH_CORR_REGS_KEY = "cute_flash_corr_regs"
_FLASH_SOFTMAX_REGS_VALUES = (152, 160, 168, 176, 184, 192, 200)
_FLASH_CORR_REGS_VALUES = (64, 72, 80, 88, 96, 104, 112, 120, 128)
FLASH_OTHER_REGS_KEY = "cute_flash_other_regs"
FLASH_CORR_TILE_SIZE_KEY = "cute_flash_corr_tile_size"
FLASH_ROW_EPILOGUE_WARPS_KEY = "cute_flash_row_epilogue_warps"
FLASH_PACKED_REDUCE_KEY = "cute_flash_packed_reduce"
FLASH_BIAS_FORM_KEY = "cute_flash_bias_form"
# Lowerings of the affine (ALiBi / relative-position) score biases: the
# per-element integer form, or the hoisted per-fragment base with packed
# FFMA/FADD pairs or scalar FFMA-immediate + FADD per element.
_FLASH_BIAS_FORMS = ("direct", "packed", "scalar")
# The two hoisted forms share one rounding (FFMA then FADD per element) and
# are the only values the autotuner may choose between; ``direct`` rounds
# differently and is never searched.
_FLASH_SEARCHED_BIAS_FORMS = ("packed", "scalar")
FLASH_BIAS_STAGE_KEY = "cute_flash_bias_stage"
# Where the fa4 chunked softmax reads an additive tensor bias: global memory
# in the softmax warps (``direct``) or a shared-memory ring the load warp
# fills by TMA one KV step ahead (``tma``); see ``_flash_bias_stage_depth``.
_FLASH_BIAS_STAGES = ("direct", "tma")
# Ring depths tried for the staged bias, deepest first: each stage holds both
# query tiles' 128 x kv_tile boxes (64 KB of a 16-bit bias at 128 columns).
_FLASH_BIAS_STAGE_DEPTHS = (2, 1)
FLASH_SMALL_BIASED_KEY = "cute_flash_small_biased"
FLASH_CAUSAL_LPT_SWIZZLE_KEY = "cute_flash_causal_lpt_swizzle"
FLASH_CAUSAL_LPT_L2_SHIFT_KEY = "cute_flash_causal_lpt_l2_shift"
FLASH_PAIR_CLUSTER_KEY = "cute_flash_pair_cluster"
# Legal halvings of the L2 budget behind the L2-sized causal launch group.
_FLASH_CAUSAL_LPT_L2_SHIFTS = (0, 1)
FLASH_CAUSAL_KV_ORDER_KEY = "cute_flash_causal_kv_order"
FLASH_ROLE_MAP_KEY = "cute_flash_role_map"
FLASH_ROLE_CHAIN_KEY = "cute_flash_role_chain"
FLASH_USE_2CTA_KEY = "cute_flash_use_2cta"
FLASH_CGA2_LOCAL_KEY = "cute_flash_cga2_local"
FLASH_CLC_KEY = "cute_flash_clc"
FLASH_CLC_HEADS_PER_BATCH_KEY = "cute_flash_clc_heads_per_batch"
FLASH_CLC_PDL_KEY = "cute_flash_clc_pdl"
FLASH_CLC_STAGES_KEY = "cute_flash_clc_stages"
FLASH_LOCAL_TMA_PARTITION_KEY = "cute_flash_local_tma_partition"
FLASH_TENSOR_4D_TMA_KEY = "cute_flash_tensor_4d_tma"
FLASH_CAUSAL_LOOP_SPLIT_KEY = "cute_flash_causal_loop_split"
FLASH_CAUSAL_MASK_STOREBACK_KEY = "cute_flash_causal_mask_storeback"
FLASH_FLEX_MASK_FORM_KEY = "cute_flash_flex_mask_form"
FLASH_FLEX_AUX_SMEM_KEY = "cute_flash_flex_aux_smem"
FLASH_FLEX_PREFETCH_KEY = "cute_flash_flex_prefetch"
FLASH_FLEX_COALESCE_KEY = "cute_flash_flex_coalesce"
FLASH_FLEX_ROLLED_CHUNKS_KEY = "cute_flash_flex_rolled_chunks"
FLASH_FLEX_ITEM_WORDS_KEY = "cute_flash_flex_item_words"
FLASH_S_RESIDENT_KEY = "cute_flash_s_resident"
FLASH_SOFTMAX_REGION_KEY = "cute_flash_softmax_region"
FLASH_FLEX_HOOK_FAST_FULL_KEY = "cute_flash_flex_hook_fast_full"
FLASH_CAUSAL_SKIP_DEAD_TILE_KEY = "cute_flash_causal_skip_dead_tile"
FLASH_CAUSAL_STAGGER_KEY = "cute_flash_causal_stagger"
# The staggered walk holds each K/V slot across two steps: the shallowest
# aliased ring it runs on (the hd64 seeds' depth; ``cute_flash`` keeps the
# per-head-size seed depths beside the stagger legality rule).
FLASH_CAUSAL_STAGGER_MIN_KV_STAGE = 4
FLASH_CAUSAL_HEAD_PAIR_KEY = "cute_flash_causal_head_pair"
FLASH_EARLY_PROLOGUE_LOADS_KEY = "cute_flash_early_prologue_loads"
FLASH_FLEX_SKIP_ABSENT_KEY = "cute_flash_flex_skip_absent"
FLASH_FLEX_DEAD_STAGE_SPLIT_KEY = "cute_flash_flex_dead_stage_split"
FLASH_FLEX_EARLY_QK_KEY = "cute_flash_flex_early_qk"
FLASH_FLEX_TAIL_SPLIT_KEY = "cute_flash_flex_tail_split"
FLASH_KV_SPLIT_KEY = "cute_flash_kv_split"
# Conditional children of the structural coverage design: a searched value
# that renders only under its parents' values.  A design whose overrides pin
# any parent elsewhere reports the child as uncovered telemetry, not a defect
# (the shared staged output tile needs the TMA epilogue without the staged
# vector store, the output over the dead Q stages, the uneven ring, the
# segment tiling or a split merge: ``_flash_o_stage_legal``).
FLASH_COVERAGE_CHILD_PARENTS: dict[
    tuple[str, object], tuple[tuple[str, object], ...]
] = {
    (FLASH_O_STAGE_KEY, 1): (
        (FLASH_EPI_TMA_KEY, True),
        (FLASH_EPI_STG_KEY, False),
        (FLASH_O_OVER_Q_KEY, False),
        (FLASH_UNEVEN_KV_RING_KEY, False),
        (FLASH_ROW_BOUND_TILING_KEY, "global"),
        (FLASH_KV_SPLIT_KEY, 1),
        (FLASH_FLEX_TAIL_SPLIT_KEY, 0),
    ),
}
FLASH_PERSISTENT_LOOP_KEY = "cute_flash_persistent_loop"
FLASH_SP_ROW_SUM_KEY = "cute_flash_sp_row_sum"
FLASH_SOFTMAX_LOWERING_KEY = "cute_flash_softmax_lowering"
FLASH_ROW_SUM_SCHEDULE_KEY = "cute_flash_row_sum_schedule"
FLASH_SOFTMAX_SETUP_KEY = "cute_flash_softmax_setup"
FLASH_KV_TILE_N_KEY = "cute_flash_kv_tile_n"
FLASH_Q_TILE_M_KEY = "cute_flash_q_tile_m"
FLASH_WS_ONE_PASS_KEY = "cute_flash_ws_one_pass"
FLASH_WS_LOADER_WARP_KEY = "cute_flash_ws_loader_warp"
FLASH_WS_MASK_PHASES_KEY = "cute_flash_ws_mask_phases"
FLASH_WS_K_STAGE_EXTRA_KEY = "cute_flash_ws_k_stage_extra"
FLASH_ROW_WARPS_KEY = "cute_flash_row_warps"
FLASH_ROW_TILE_M_KEY = "cute_flash_row_tile_m"
FLASH_EPI_TMA_SETUP_KEY = "cute_flash_epi_tma_setup"
FLASH_ROWMAX_KEY = "cute_flash_rowmax"


class FlashKnob(NamedTuple):
    """One searched flash knob: its config key and the ``FlashAttentionConfig``
    field ``flash_effective_config_values`` projects it from."""

    key: str
    field: str


# The ordered knob registry.  A knob's position is its coverage axis: the
# structural coverage design derives its row pattern from the axis positions,
# so new knobs go last (an inserted axis would reshuffle every later axis's
# rows; the 64-row query tile's witness depends on them).
FLASH_KNOBS: tuple[FlashKnob, ...] = (
    FlashKnob(FLASH_S_STAGE_KEY, "s_stage"),
    FlashKnob(FLASH_KV_STAGE_KEY, "kv_stage"),
    FlashKnob(FLASH_PERSISTENT_KEY, "persistent"),
    FlashKnob(FLASH_PERSISTENT_CTAS_PER_SM_KEY, "persistent_ctas_per_sm"),
    FlashKnob(FLASH_RECOMPUTE_TILE_COORDS_KEY, "recompute_tile_coords"),
    FlashKnob(FLASH_E2E_SCHEDULE_KEY, "e2e_schedule"),
    FlashKnob(FLASH_MASKED_E2E_SCHEDULE_KEY, "masked_e2e_schedule"),
    FlashKnob(FLASH_E2E_OFFSET_KEY, "e2e_offset"),
    FlashKnob(FLASH_E2E_OFFSET0_KEY, "e2e_offset0"),
    FlashKnob(FLASH_EXP2_PACKET_KEY, "exp2_packet"),
    FlashKnob(FLASH_MMA_INTERLEAVE_KEY, "mma_interleave"),
    FlashKnob(FLASH_WAIT_HINT_KEY, "wait_hint"),
    FlashKnob(FLASH_STAT_TRANSPORT_KEY, "stat_transport"),
    FlashKnob(FLASH_PIPELINE_FAMILY_KEY, "pipeline_family"),
    FlashKnob(FLASH_SOFTMAX_DISC_KEY, "softmax_disc"),
    FlashKnob(FLASH_DISC_PIPE_KEY, "disc_pipe_depth"),
    FlashKnob(FLASH_SPLIT_P_ARRIVE_KEY, "split_p_arrive"),
    FlashKnob(FLASH_P_STORE_REP_KEY, "p_store_repetition"),
    FlashKnob(FLASH_S_LOAD_REP_KEY, "s_load_repetition"),
    FlashKnob(FLASH_PRECOMPUTE_QK_DESC_KEY, "precompute_qk_desc"),
    FlashKnob(FLASH_FIRST_LOAD_ORDER_KEY, "first_load_order"),
    FlashKnob(FLASH_KV_ORDER_KEY, "kv_order"),
    FlashKnob(FLASH_EPI_TMA_KEY, "epi_tma"),
    FlashKnob(FLASH_O_STAGE_KEY, "o_stage"),
    FlashKnob(FLASH_EPI_STG_KEY, "epi_stg"),
    FlashKnob(FLASH_EPI_STG_STORE_KEY, "epi_stg_store"),
    FlashKnob(FLASH_EPI_STG_GMEM_KEY, "epi_stg_gmem"),
    FlashKnob(FLASH_RESCALE_THRESHOLD_KEY, "rescale_threshold"),
    FlashKnob(FLASH_SKIP_RESCALE_STATS_KEY, "skip_rescale_stats"),
    FlashKnob(FLASH_RESCALE_CHUNK_COLS_KEY, "rescale_chunk_cols"),
    FlashKnob(FLASH_SOFTMAX_REGS_KEY, "softmax_regs"),
    FlashKnob(FLASH_CORR_REGS_KEY, "corr_regs"),
    FlashKnob(FLASH_OTHER_REGS_KEY, "other_regs"),
    FlashKnob(FLASH_CORR_TILE_SIZE_KEY, "corr_tile_size"),
    FlashKnob(FLASH_ROW_EPILOGUE_WARPS_KEY, "row_epilogue_warps"),
    FlashKnob(FLASH_PACKED_REDUCE_KEY, "packed_reduce"),
    FlashKnob(FLASH_SMALL_BIASED_KEY, "small_biased"),
    FlashKnob(FLASH_CAUSAL_LPT_SWIZZLE_KEY, "causal_lpt_swizzle"),
    FlashKnob(FLASH_CAUSAL_LPT_L2_SHIFT_KEY, "causal_lpt_l2_shift"),
    FlashKnob(FLASH_PAIR_CLUSTER_KEY, "pair_cluster"),
    FlashKnob(FLASH_CAUSAL_KV_ORDER_KEY, "causal_kv_order"),
    FlashKnob(FLASH_ROLE_MAP_KEY, "role_map"),
    FlashKnob(FLASH_ROLE_CHAIN_KEY, "role_chain"),
    FlashKnob(FLASH_CLC_HEADS_PER_BATCH_KEY, "clc_heads_per_batch"),
    FlashKnob(FLASH_CLC_PDL_KEY, "clc_use_pdl"),
    FlashKnob(FLASH_CLC_STAGES_KEY, "clc_stages"),
    FlashKnob(FLASH_CAUSAL_LOOP_SPLIT_KEY, "causal_loop_split"),
    FlashKnob(FLASH_PERSISTENT_LOOP_KEY, "persistent_loop"),
    FlashKnob(FLASH_SP_ROW_SUM_KEY, "sp_row_sum"),
    FlashKnob(FLASH_ROWMAX_KEY, "rowmax"),
    FlashKnob(FLASH_SOFTMAX_LOWERING_KEY, "softmax_lowering"),
    FlashKnob(FLASH_ROW_SUM_SCHEDULE_KEY, "row_sum_schedule"),
    FlashKnob(FLASH_SOFTMAX_SETUP_KEY, "softmax_setup"),
    FlashKnob(FLASH_EPI_TMA_SETUP_KEY, "epi_tma_setup"),
    FlashKnob(FLASH_KV_TILE_N_KEY, "kv_tile_n"),
    FlashKnob(FLASH_Q_TILE_M_KEY, "q_tile_m"),
    FlashKnob(FLASH_WS_ONE_PASS_KEY, "ws_one_pass"),
    FlashKnob(FLASH_ROW_WARPS_KEY, "row_warps"),
    FlashKnob(FLASH_ROW_TILE_M_KEY, "row_tile_m"),
    # New axes go last: the structural coverage design derives its row pattern
    # from the axis positions, and an inserted axis would reshuffle every later
    # axis's rows (the 64-row query tile's witness depends on them).
    FlashKnob(FLASH_P_CHUNK_ARRIVE_KEY, "p_chunk_arrive"),
    FlashKnob(FLASH_BIAS_FORM_KEY, "bias_form"),
    FlashKnob(FLASH_UNEVEN_KV_RING_KEY, "uneven_kv_ring"),
    FlashKnob(FLASH_O_OVER_Q_KEY, "o_over_q"),
    FlashKnob(FLASH_CAUSAL_MASK_STOREBACK_KEY, "causal_mask_storeback"),
    FlashKnob(FLASH_ROW_BOUND_TILING_KEY, "row_bound_tiling"),
    FlashKnob(FLASH_CAUSAL_SKIP_DEAD_TILE_KEY, "causal_skip_dead_tile"),
    FlashKnob(FLASH_CAUSAL_STAGGER_KEY, "causal_stagger"),
    FlashKnob(FLASH_CAUSAL_HEAD_PAIR_KEY, "causal_head_pair"),
    FlashKnob(FLASH_EARLY_PROLOGUE_LOADS_KEY, "early_prologue_loads"),
    FlashKnob(FLASH_FLEX_SKIP_ABSENT_KEY, "flex_skip_absent"),
    FlashKnob(FLASH_FLEX_DEAD_STAGE_SPLIT_KEY, "flex_dead_stage_split"),
    FlashKnob(FLASH_FLEX_EARLY_QK_KEY, "flex_early_qk"),
    FlashKnob(FLASH_FLEX_TAIL_SPLIT_KEY, "flex_tail_split"),
    FlashKnob(FLASH_FLEX_MASK_FORM_KEY, "flex_mask_form"),
    FlashKnob(FLASH_KV_SPLIT_KEY, "kv_split"),
    FlashKnob(FLASH_FLEX_AUX_SMEM_KEY, "flex_aux_smem"),
    FlashKnob(FLASH_FLEX_PREFETCH_KEY, "flex_prefetch"),
    FlashKnob(FLASH_FLEX_COALESCE_KEY, "flex_coalesce"),
    FlashKnob(FLASH_FLEX_ROLLED_CHUNKS_KEY, "flex_rolled_chunks"),
    FlashKnob(FLASH_FLEX_ITEM_WORDS_KEY, "flex_item_words"),
    FlashKnob(FLASH_S_RESIDENT_KEY, "s_resident"),
    FlashKnob(FLASH_SOFTMAX_REGION_KEY, "softmax_region"),
    FlashKnob(FLASH_WS_LOADER_WARP_KEY, "ws_loader_warp"),
    FlashKnob(FLASH_WS_MASK_PHASES_KEY, "ws_mask_phases"),
    FlashKnob(FLASH_WS_K_STAGE_EXTRA_KEY, "ws_k_stage_extra"),
    FlashKnob(FLASH_FLEX_HOOK_FAST_FULL_KEY, "flex_hook_fast_full"),
    FlashKnob(FLASH_BIAS_STAGE_KEY, "bias_stage"),
)

FLASH_AUTOTUNE_CONFIG_KEYS: tuple[str, ...] = tuple(knob.key for knob in FLASH_KNOBS)

FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_TOPOLOGY_KEY,
    FLASH_USE_2CTA_KEY,
    FLASH_CGA2_LOCAL_KEY,
    FLASH_CLC_KEY,
    FLASH_LOCAL_TMA_PARTITION_KEY,
    FLASH_TENSOR_4D_TMA_KEY,
)

# The paired schedule key's three-way spelling (``cute_flash_e2e_schedule``
# names the same cadence).
FLASH_LEGACY_E2E_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_EXP2_IMPL_KEY,
    FLASH_E2E_FREQ_KEY,
    FLASH_E2E_RES_KEY,
)

FLASH_LEGACY_CONFIG_KEYS: tuple[str, ...] = (
    *FLASH_LEGACY_E2E_CONFIG_KEYS,
    *FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS,
)


def flash_legacy_e2e_params(
    read: Callable[[str], object | None],
    exp2_impl: str,
    e2e_freq: int,
    e2e_res: int,
) -> tuple[str, int, int]:
    """Apply a config's legacy cadence keys over a parsed schedule.

    ``read`` looks a key up in the config (the resolver's override-aware
    reader or ``config.get``); a present key replaces its component of the
    ``(exp2_impl, e2e_freq, e2e_res)`` cadence.
    """
    exp2_impl_cfg = read(FLASH_EXP2_IMPL_KEY)
    if exp2_impl_cfg is not None:
        exp2_impl = str(exp2_impl_cfg)
    e2e_freq_cfg = read(FLASH_E2E_FREQ_KEY)
    if e2e_freq_cfg is not None:
        e2e_freq = int(e2e_freq_cfg)  # type: ignore[arg-type]
    e2e_res_cfg = read(FLASH_E2E_RES_KEY)
    if e2e_res_cfg is not None:
        e2e_res = int(e2e_res_cfg)  # type: ignore[arg-type]
    return exp2_impl, e2e_freq, e2e_res


def upgrade_legacy_flash_config(
    config: dict[str, object],
    resolve: Callable[[Mapping[str, object]], FlashAttentionConfig],
) -> None:
    """Fold a config's legacy structural keys into the compound family key.

    A config that names a topology, cluster or TMA flag
    (``FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS``) without
    ``cute_flash_pipeline_family`` is resolved -- with its persistence, which
    selects among the single-tile families -- and carries the resolved family
    from here on.  The resolver reads a named family's flags and ignores the
    per-flag keys, so they are dropped; a config that already names a family
    is left alone (normalization drops its stale legacy keys at the end).
    """
    if FLASH_PIPELINE_FAMILY_KEY in config:
        return
    legacy = {
        key: config[key]
        for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
        if config.get(key) is not None
    }
    if not legacy:
        return
    if FLASH_PERSISTENT_KEY in config:
        legacy[FLASH_PERSISTENT_KEY] = config[FLASH_PERSISTENT_KEY]
    config[FLASH_PIPELINE_FAMILY_KEY] = resolve(legacy).pipeline_family
    for key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
        config.pop(key, None)


def fold_flash_ws_inactive_values(config: dict[str, object]) -> None:
    """Fold recognized fixed-config aliases outside the WS search domain.

    A fixed config may carry old FA4 values for the fields the single-tile
    ``ws_overlap`` body does not read (a masked cadence, an exp2 packet, a
    wait hint); each folds to the WS surface's one value so the config names
    the WS body's identity (unknown values are left for strict validation).
    """
    masked_schedule = config.get(FLASH_MASKED_E2E_SCHEDULE_KEY)
    if isinstance(masked_schedule, str) and (
        masked_schedule in ("inherit", "xu", "16/4", "8/2")
        or masked_schedule
        in (
            f"{freq}/{res}"
            for freq, res in _FLASH_MANUAL_EXP2_PACKET_SCHEDULES.values()
        )
    ):
        config[FLASH_MASKED_E2E_SCHEDULE_KEY] = "inherit"
    packet = config.get(FLASH_EXP2_PACKET_KEY)
    if isinstance(packet, str) and packet in FLASH_EXP2_PACKET_CHOICES:
        config[FLASH_EXP2_PACKET_KEY] = "1x1"
    wait_hint = config.get(FLASH_WAIT_HINT_KEY)
    if type(wait_hint) is int and wait_hint in (0, 10_000_000):
        config[FLASH_WAIT_HINT_KEY] = 10_000_000


# Accepted as an input and retained in normalized configs, but not sampled:
# changing the local query-slot count requires selecting another structural
# family rather than mutating an arithmetic child.
FLASH_DERIVED_CONFIG_KEYS: tuple[str, ...] = (FLASH_Q_TILE_COUNT_KEY,)

FLASH_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_AUTOTUNE_CONFIG_KEYS + FLASH_LEGACY_CONFIG_KEYS + FLASH_DERIVED_CONFIG_KEYS
)
# These fields form one staged-output schedule. Marginal coverage is insufficient:
# all effective combinations emit different producer/consumer code.
FLASH_AUTOTUNE_INTERACTION_KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        FLASH_EPI_TMA_KEY,
        FLASH_EPI_STG_KEY,
        FLASH_EPI_STG_STORE_KEY,
        FLASH_EPI_STG_GMEM_KEY,
    ),
)


class FlashKnobWitness(NamedTuple):
    """Knob values legal only together with other knob values.  The structural
    coverage design adds one witness context per value (the requirements plus
    the value) so the value is reachable by construction instead of by a
    lucky combination of covering rows."""

    key: str
    values: tuple[object, ...]
    requires: Mapping[str, object]


# The chunked fa4 softmax body: the flex hook and its knobs live there.
_FA4_CHUNKED_BODY: dict[str, object] = {
    FLASH_PIPELINE_FAMILY_KEY: "fa4",
    FLASH_SOFTMAX_DISC_KEY: True,
}

# In the order the coverage design visits them (the witness rows' order is
# part of the design): new witnesses go last.
FLASH_KNOB_WITNESSES: tuple[FlashKnobWitness, ...] = (
    # The 64-row query tile and its one-pass softmax exist only on the flat,
    # staged, two-stage ``ws_overlap`` body.
    FlashKnobWitness(
        FLASH_Q_TILE_M_KEY,
        (64,),
        {
            FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            FLASH_PERSISTENT_KEY: False,
            FLASH_EPI_STG_KEY: True,
            FLASH_KV_STAGE_KEY: 2,
        },
    ),
    FlashKnobWitness(
        FLASH_WS_ONE_PASS_KEY,
        (True,),
        {
            FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
            FLASH_PERSISTENT_KEY: False,
            FLASH_EPI_STG_KEY: True,
            FLASH_KV_STAGE_KEY: 2,
            FLASH_Q_TILE_M_KEY: 64,
        },
    ),
    # The single-tile bodies' issue-stream knobs live on the two-stage
    # warp-specialized body (not the single score stage's prefetch/drain
    # loops nor the small biased plan's SIMT body).
    FlashKnobWitness(
        FLASH_WS_LOADER_WARP_KEY,
        (True,),
        {FLASH_S_STAGE_KEY: 2, FLASH_SMALL_BIASED_KEY: False},
    ),
    FlashKnobWitness(
        FLASH_WS_MASK_PHASES_KEY,
        (True,),
        {FLASH_S_STAGE_KEY: 2, FLASH_SMALL_BIASED_KEY: False},
    ),
    FlashKnobWitness(
        FLASH_WS_K_STAGE_EXTRA_KEY,
        (1,),
        {FLASH_S_STAGE_KEY: 2, FLASH_SMALL_BIASED_KEY: False},
    ),
    # Output staged in the dead Q stages: a staged output on a
    # non-persistent FA4 body.
    FlashKnobWitness(
        FLASH_O_OVER_Q_KEY,
        (True,),
        {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_PERSISTENT_KEY: False,
            FLASH_EPI_TMA_KEY: True,
        },
    ),
    # The single-application causal mask lives on the fa4 chunked softmax.
    FlashKnobWitness(FLASH_CAUSAL_MASK_STOREBACK_KEY, (True,), _FA4_CHUNKED_BODY),
    # The flex hook lives on the chunked fa4 softmax body.
    FlashKnobWitness(FLASH_FLEX_MASK_FORM_KEY, ("element",), _FA4_CHUNKED_BODY),
    # The split-KV walk lives on the list walk's fa4 pair body.
    FlashKnobWitness(FLASH_KV_SPLIT_KEY, (2, 4), {FLASH_PIPELINE_FAMILY_KEY: "fa4"}),
    FlashKnobWitness(FLASH_FLEX_AUX_SMEM_KEY, (4096, 16384), _FA4_CHUNKED_BODY),
    FlashKnobWitness(FLASH_FLEX_PREFETCH_KEY, (32, 64, 128), _FA4_CHUNKED_BODY),
    FlashKnobWitness(FLASH_FLEX_COALESCE_KEY, (32, 64, 128), _FA4_CHUNKED_BODY),
    FlashKnobWitness(FLASH_FLEX_ROLLED_CHUNKS_KEY, (True,), _FA4_CHUNKED_BODY),
    FlashKnobWitness(FLASH_FLEX_HOOK_FAST_FULL_KEY, (True,), _FA4_CHUNKED_BODY),
    FlashKnobWitness(FLASH_FLEX_ITEM_WORDS_KEY, (True,), _FA4_CHUNKED_BODY),
    FlashKnobWitness(
        FLASH_S_RESIDENT_KEY,
        (True,),
        {**_FA4_CHUNKED_BODY, FLASH_S_LOAD_REP_KEY: 32},
    ),
    # One softmax region for both stages: the list walk's chunked fa4 body.
    FlashKnobWitness(FLASH_SOFTMAX_REGION_KEY, ("shared",), _FA4_CHUNKED_BODY),
    # Segment-aligned tiling of the per-row KV lower bound: the
    # non-persistent FA4 pair body with a staged epilogue in its own ``sO``
    # (the item loop reloads Q, so the output cannot live in the Q stages).
    FlashKnobWitness(
        FLASH_ROW_BOUND_TILING_KEY,
        ("segment",),
        {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_PERSISTENT_KEY: False,
            FLASH_EPI_TMA_KEY: True,
            FLASH_O_OVER_Q_KEY: False,
        },
    ),
    # The dead stage-0 tile is a property of the descending two-stage fa4 walk.
    FlashKnobWitness(
        FLASH_CAUSAL_SKIP_DEAD_TILE_KEY,
        (True,),
        {FLASH_PIPELINE_FAMILY_KEY: "fa4", FLASH_CAUSAL_KV_ORDER_KEY: "descending"},
    ),
    # The staggered walk holds each K/V slot across two steps: the descending
    # fa4 pair with a ring of at least four slots (the hd128 pair reaches that
    # depth only with the output staged in the dead Q stages).
    FlashKnobWitness(
        FLASH_CAUSAL_STAGGER_KEY,
        (True,),
        {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_CAUSAL_KV_ORDER_KEY: "descending",
            FLASH_KV_STAGE_KEY: FLASH_CAUSAL_STAGGER_MIN_KV_STAGE,
            FLASH_EPI_TMA_KEY: True,
            FLASH_O_OVER_Q_KEY: True,
        },
    ),
    # The head pair addresses two heads' tiles through the TMA output store
    # on the descending fa4 pair with the chunked softmax.
    FlashKnobWitness(
        FLASH_CAUSAL_HEAD_PAIR_KEY,
        (True,),
        {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_CAUSAL_KV_ORDER_KEY: "descending",
            FLASH_EPI_TMA_KEY: True,
            FLASH_SOFTMAX_DISC_KEY: True,
        },
    ),
    # The pre-sync prologue loads exist only on the non-persistent one-CTA fa4
    # ring (the resolver clears them elsewhere).
    FlashKnobWitness(
        FLASH_EARLY_PROLOGUE_LOADS_KEY,
        (True,),
        {FLASH_PIPELINE_FAMILY_KEY: "fa4", FLASH_PERSISTENT_KEY: False},
    ),
    # The absent-stage skip is a property of the fa4 pair body's list walk
    # (the plan pins the family; the witness names the body).
    FlashKnobWitness(
        FLASH_FLEX_SKIP_ABSENT_KEY, (True,), {FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    ),
    # The dead-stage split rides on the absent-stage skip.
    FlashKnobWitness(
        FLASH_FLEX_DEAD_STAGE_SPLIT_KEY,
        (True,),
        {FLASH_PIPELINE_FAMILY_KEY: "fa4", FLASH_FLEX_SKIP_ABSENT_KEY: True},
    ),
    # The early next-item QK0 is a property of the fa4 pair body's
    # persistent item loop: both of its values are witnessed there (the
    # resolver pins the knob off on the grid, so a grid row says nothing
    # about either form of the persistent loop).
    FlashKnobWitness(
        FLASH_FLEX_EARLY_QK_KEY,
        (True, False),
        {FLASH_PIPELINE_FAMILY_KEY: "fa4", FLASH_PERSISTENT_KEY: True},
    ),
    # The tail split is a property of the fa4 pair body's one-item grid.
    FlashKnobWitness(
        FLASH_FLEX_TAIL_SPLIT_KEY,
        (2, 4, 8),
        {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_PERSISTENT_KEY: False,
            FLASH_KV_SPLIT_KEY: 1,
        },
    ),
)

# ``(key, value) -> requirements`` in witness order, one entry per value.
FLASH_AUTOTUNE_VALUE_DEPENDENCIES: dict[tuple[str, object], dict[str, object]] = {
    (witness.key, value): dict(witness.requires)
    for witness in FLASH_KNOB_WITNESSES
    for value in witness.values
}


# The order ``flash_effective_config_values`` writes a resolved config's
# values: the active autotune keys and the derived query-slot count, in the
# order the normalized configs and the seeds have always carried them (the
# first knobs in axis order with the query-slot count after the MMA
# interleave, then each newer knob beside its siblings).  The autotune cache
# identity and ``Config`` equality are order-insensitive; ``Config.to_json``
# and the seed goldens are not, so the order is pinned here rather than
# derived from the registry.  New knobs go at the end.
FLASH_EFFECTIVE_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_S_STAGE_KEY,
    FLASH_KV_STAGE_KEY,
    FLASH_PERSISTENT_KEY,
    FLASH_PERSISTENT_CTAS_PER_SM_KEY,
    FLASH_RECOMPUTE_TILE_COORDS_KEY,
    FLASH_E2E_SCHEDULE_KEY,
    FLASH_MASKED_E2E_SCHEDULE_KEY,
    FLASH_E2E_OFFSET_KEY,
    FLASH_E2E_OFFSET0_KEY,
    FLASH_EXP2_PACKET_KEY,
    FLASH_MMA_INTERLEAVE_KEY,
    FLASH_Q_TILE_COUNT_KEY,
    FLASH_WAIT_HINT_KEY,
    FLASH_STAT_TRANSPORT_KEY,
    FLASH_PIPELINE_FAMILY_KEY,
    FLASH_SOFTMAX_DISC_KEY,
    FLASH_DISC_PIPE_KEY,
    FLASH_SPLIT_P_ARRIVE_KEY,
    FLASH_P_CHUNK_ARRIVE_KEY,
    FLASH_P_STORE_REP_KEY,
    FLASH_S_LOAD_REP_KEY,
    FLASH_PRECOMPUTE_QK_DESC_KEY,
    FLASH_FIRST_LOAD_ORDER_KEY,
    FLASH_KV_ORDER_KEY,
    FLASH_EPI_TMA_KEY,
    FLASH_O_STAGE_KEY,
    FLASH_EPI_STG_KEY,
    FLASH_EPI_STG_STORE_KEY,
    FLASH_EPI_STG_GMEM_KEY,
    FLASH_RESCALE_THRESHOLD_KEY,
    FLASH_SKIP_RESCALE_STATS_KEY,
    FLASH_RESCALE_CHUNK_COLS_KEY,
    FLASH_SOFTMAX_REGS_KEY,
    FLASH_CORR_REGS_KEY,
    FLASH_OTHER_REGS_KEY,
    FLASH_CORR_TILE_SIZE_KEY,
    FLASH_ROW_EPILOGUE_WARPS_KEY,
    FLASH_PACKED_REDUCE_KEY,
    FLASH_BIAS_FORM_KEY,
    FLASH_BIAS_STAGE_KEY,
    FLASH_SMALL_BIASED_KEY,
    FLASH_CAUSAL_LPT_SWIZZLE_KEY,
    FLASH_CAUSAL_LPT_L2_SHIFT_KEY,
    FLASH_PAIR_CLUSTER_KEY,
    FLASH_CAUSAL_KV_ORDER_KEY,
    FLASH_ROLE_MAP_KEY,
    FLASH_ROLE_CHAIN_KEY,
    FLASH_CLC_HEADS_PER_BATCH_KEY,
    FLASH_CLC_PDL_KEY,
    FLASH_CLC_STAGES_KEY,
    FLASH_CAUSAL_LOOP_SPLIT_KEY,
    FLASH_PERSISTENT_LOOP_KEY,
    FLASH_SP_ROW_SUM_KEY,
    FLASH_ROWMAX_KEY,
    FLASH_SOFTMAX_LOWERING_KEY,
    FLASH_ROW_SUM_SCHEDULE_KEY,
    FLASH_SOFTMAX_SETUP_KEY,
    FLASH_EPI_TMA_SETUP_KEY,
    FLASH_KV_TILE_N_KEY,
    FLASH_Q_TILE_M_KEY,
    FLASH_WS_ONE_PASS_KEY,
    FLASH_WS_LOADER_WARP_KEY,
    FLASH_WS_MASK_PHASES_KEY,
    FLASH_WS_K_STAGE_EXTRA_KEY,
    FLASH_ROW_WARPS_KEY,
    FLASH_ROW_TILE_M_KEY,
    FLASH_UNEVEN_KV_RING_KEY,
    FLASH_O_OVER_Q_KEY,
    FLASH_CAUSAL_MASK_STOREBACK_KEY,
    FLASH_ROW_BOUND_TILING_KEY,
    FLASH_CAUSAL_SKIP_DEAD_TILE_KEY,
    FLASH_CAUSAL_STAGGER_KEY,
    FLASH_CAUSAL_HEAD_PAIR_KEY,
    FLASH_EARLY_PROLOGUE_LOADS_KEY,
    FLASH_FLEX_SKIP_ABSENT_KEY,
    FLASH_FLEX_DEAD_STAGE_SPLIT_KEY,
    FLASH_FLEX_EARLY_QK_KEY,
    FLASH_FLEX_TAIL_SPLIT_KEY,
    FLASH_FLEX_MASK_FORM_KEY,
    FLASH_KV_SPLIT_KEY,
    FLASH_FLEX_AUX_SMEM_KEY,
    FLASH_FLEX_PREFETCH_KEY,
    FLASH_FLEX_COALESCE_KEY,
    FLASH_FLEX_ROLLED_CHUNKS_KEY,
    FLASH_FLEX_ITEM_WORDS_KEY,
    FLASH_S_RESIDENT_KEY,
    FLASH_SOFTMAX_REGION_KEY,
    FLASH_FLEX_HOOK_FAST_FULL_KEY,
)
assert sorted(FLASH_EFFECTIVE_CONFIG_KEYS) == sorted(
    (*FLASH_AUTOTUNE_CONFIG_KEYS, *FLASH_DERIVED_CONFIG_KEYS)
)

_FLASH_EFFECTIVE_CONFIG_FIELDS: dict[str, str] = {
    **{knob.key: knob.field for knob in FLASH_KNOBS},
    FLASH_Q_TILE_COUNT_KEY: "q_tile_count",
}


def flash_effective_config_values(
    config: FlashAttentionConfig,
) -> dict[str, object]:
    """Project a resolved flash config onto the active autotune schema (the
    registry's knobs and the derived query-slot count) in
    ``FLASH_EFFECTIVE_CONFIG_KEYS`` order."""
    return {
        key: getattr(config, _FLASH_EFFECTIVE_CONFIG_FIELDS[key])
        for key in FLASH_EFFECTIVE_CONFIG_KEYS
    }


# ---------------------------------------------------------------------------
# Backward (attention gradient) knobs, read by the dense, flex and row-MMA
# backward emitters and listed once for the config schema.
# ---------------------------------------------------------------------------
FLASH_BWD_PERSISTENT_KEY = "cute_flash_bwd_persistent"
FLASH_BWD_TWO_CTA_KEY = "cute_flash_bwd_two_cta"
FLASH_BWD_EXP2_F32_KEY = "cute_flash_bwd_exp2_f32"
FLASH_BWD_MASK_FORM_KEY = "cute_flash_bwd_mask_form"
FLASH_BWD_REGS_KEY = "cute_flash_bwd_regs"
FLASH_BWD_S_FIRST_KEY = "cute_flash_bwd_s_first"
FLASH_BWD_KV_STAGE_KEY = "cute_flash_bwd_kv_stage"
FLASH_BWD_HOOK_COLS_KEY = "cute_flash_bwd_hook_cols"
FLASH_BWD_HOOK_PIN_KEY = "cute_flash_bwd_hook_pin"
FLASH_BWD_BIAS_FOLD_KEY = "cute_flash_bwd_bias_fold"
FLASH_BWD_TILE_GROUP_KEY = "cute_flash_bwd_tile_group"
FLASH_BWD_STATS_MMA_KEY = "cute_flash_bwd_stats_mma"
FLASH_BWD_AUX_SMEM_KEY = "cute_flash_bwd_aux_smem"
FLASH_BWD_DS_TMEM_KEY = "cute_flash_bwd_ds_tmem"
FLASH_BWD_EXP2_EMU_KEY = "cute_flash_bwd_exp2_emu"
FLASH_BWD_TABLE_STEPS_KEY = "cute_flash_bwd_table_steps"
FLASH_BWD_FAMILY_KEY = "cute_flash_bwd_family"
FLASH_BWD_ROW_KV_KEY = "cute_flash_bwd_row_kv"
FLASH_BWD_ROW_Q_SPLIT_KEY = "cute_flash_bwd_row_q_split"
FLASH_BWD_ROW_Q_CHUNK_KEY = "cute_flash_bwd_row_q_chunk"
FLASH_BWD_CONFIG_KEYS: tuple[str, ...] = (
    FLASH_BWD_PERSISTENT_KEY,
    FLASH_BWD_TWO_CTA_KEY,
    FLASH_BWD_EXP2_F32_KEY,
    FLASH_BWD_MASK_FORM_KEY,
    FLASH_BWD_REGS_KEY,
    FLASH_BWD_S_FIRST_KEY,
    FLASH_BWD_KV_STAGE_KEY,
    FLASH_BWD_HOOK_COLS_KEY,
    FLASH_BWD_HOOK_PIN_KEY,
    FLASH_BWD_BIAS_FOLD_KEY,
    FLASH_BWD_TILE_GROUP_KEY,
    FLASH_BWD_STATS_MMA_KEY,
    FLASH_BWD_AUX_SMEM_KEY,
    FLASH_BWD_DS_TMEM_KEY,
    FLASH_BWD_EXP2_EMU_KEY,
    FLASH_BWD_TABLE_STEPS_KEY,
    FLASH_BWD_FAMILY_KEY,
    FLASH_BWD_ROW_KV_KEY,
    FLASH_BWD_ROW_Q_SPLIT_KEY,
    FLASH_BWD_ROW_Q_CHUNK_KEY,
)
