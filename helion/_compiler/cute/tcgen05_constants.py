from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

# Validated CtaGroup.TWO autotune/runtime envelope for the B200 CuTe path.
# Re-verify the K-cap runtime and guard-boundary tests before raising the
# K-tile threshold or broadening the tile shape.
TCGEN05_TWO_CTA_BLOCK_M = 256
TCGEN05_TWO_CTA_BLOCK_N = 256
# The 5000x5000x5000 bias_residual_gelu CLC + aux-TMA scheduler family
# measures faster with a narrower N tile while keeping the validated
# CtaGroup.TWO M tile and K-tail shape. Autotune can admit the same
# narrow tile for other shapes that still have an N edge at this size.
TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N = 128
# The narrow-N Target8 row has enough SMEM headroom at ab=2 to double the
# K tile. B200 measurements put this ahead of the older narrow-N K=128 row.
TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K = 256
# With partial-output TMA store and row-vector aux staging, the same narrow-N
# CLC + aux-TMA row runs faster with two accumulator stages than the historical
# edge-family default.
TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES = 2
TCGEN05_TWO_CTA_MAX_K_TILES = 256
# The double-output-edge + K-tail CtaGroup.TWO search family is validated
# only for large square-ish GEMMs where persistent clustered scheduling has
# enough work to amortize its edge handling.
TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM = 4096
# That same family is validated only at this K/stage point. The output-edge
# full-tile TMA-store epilogue needs an extra SMEM tile, so the edge seed must
# stay within ``TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES``. Larger explicit
# AB-stage configs remain legal only through the SIMT-store fallback when they
# otherwise fit.
TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K = 128
TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES = 2
TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES = 1
# Half-depth K tile for the deep-AB edge family: bk=64 halves the per-stage
# AB SMEM so 16-bit operands fit a 5-stage AB pipeline within the B200 budget
# (bf16 5000^3 measures 948 vs 815 TFLOP/s against the bk=128 edge seed).
TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K = 64
# nvjet's ``64x6`` staging for the full-tile 256x256 cluster-2x2 family: six
# bk=64 stages hold the same 192 KB as three bk=128 stages, so the deep seed
# only trades stage granularity (earlier first MMA, shorter drain).
TCGEN05_TWO_CTA_DEEP_AB_STAGES = 6
# One-wave two-CTA tiles: 256 x {128, 64} CtaGroup.TWO tiles for GEMMs whose
# 256x256 grid leaves most SMs idle (cuBLAS's nvjet picks 2-CTA 128x64 tiles
# in 2x2 clusters with a 64x10 K ring for fp16 1024^3).  Admitted by the
# search projection and seeded when the 256x256 grid is below the SM count.
TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS = (128, 64)
# The N tiles the tcgen05 CTA pair (``tcgen05_cluster_m=2`` on the 256-row
# pair tile) runs its MMA at throughout the backend: the paired pipeline, the
# materialized two-CTA regions and the cluster_m=2 search projection all take
# ``bn in (64, 128, 256)``.
TCGEN05_TWO_CTA_PAIR_BLOCK_NS = (256, 128, 64)
# The post row-vector-staging Target8 CLC + aux-TMA edge rows now measure
# fastest with two accumulator stages. Keep the historical edge-family default
# above for non-CLC/non-aux variants. Narrow-N keeps a separate value above.
TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES = 2
# The 5000x5000x5000 bias_residual_gelu edge+K-tail reference sweep on B200
# showed the hybrid TMA/SIMT store running faster with two C-store stages than
# with the earlier four-stage seed.
TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES = 2
# The same reference sweep converged on a smaller L2 scheduler group plus the
# four-tile L2 swizzle. Keep these with the edge+K-tail stage tuple so the
# seeded/fixed CtaGroup.TWO family starts at the measured production row.
TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING = 2
# The wide-N CLC + aux-TMA Target8 row prefers an intermediate tile scheduler
# group. Narrow-N keeps its separate grouping below.
TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING = 8
# CLC + aux-TMA edge rows measure faster when the matmul K loop is flattened,
# has accumulator multi-buffering disabled, and is warp specialized.
TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN = True
TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER = False
TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE = True
TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING = 16
TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE = 4
# The scheduler full/edge split scans the same static tile space twice. On the
# 5000x5000x5000 bias_residual_gelu target, keeping the split path at size=1
# (no scheduler swizzle) measured faster than the monolithic edge seed's size=4.
TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE = 1
TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES = 2
# Plain 16-bit two-CTA epilogues take the (128, 32) subtile (and the
# explicit-store family's no-unroll K loop) on this N tile, where CuTe's
# with-source rule would give (128, 64); narrower tiles already get (128, 32)
# from the default rule. Measured in ``cute_mma`` next to the decision
# (``tcgen05_plain_narrow_subtile``).
TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_N = 256
assert (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES
    <= TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES
), "edge K-tail seed must fit the full-tile TMA-store AB-stage limit"
# fp8 small-grid CtaGroup.TWO family. The bm=256 full-tile cluster_m=2 row
# produces only ``(M/256)*(N/256)`` output clusters, which badly underfills the
# device on the small/wave-limited fp8 serving GEMMs this kernel targets (e.g.
# 512x2048x4096 -> 16 clusters on a 148-SM B200).
TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M = 128
TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N = 128
# Best measured seed L2 grouping for the full-tile CtaGroup.TWO row.
TCGEN05_TWO_CTA_SEED_L2_GROUPING = 4
# This pid order stays slightly ahead of persistent_blocked in the validated
# CtaGroup.TWO envelope.
TCGEN05_TWO_CTA_SEED_PID_TYPE = "persistent_interleaved"

# SMEM budget reserved for non-AB allocations on B200 when sampling the
# ``tcgen05_ab_stages=3`` search arm. The role-local persistent kernel also
# allocates space for the C accumulator stages, pipeline mailboxes, and TMEM
# bookkeeping; the canonical 256x256x128 cluster_m=2 ab=3 path measured
# 196 608 bytes (192 KiB) AB + ~24 KiB other under the 227 KiB optin cap.
# Hold 28 KiB back from the per-CTA budget so candidates we admit keep a
# comfortable margin against ptxas's hard limit. This is intentionally
# conservative because the search space cannot enumerate every C-stage /
# acc-stage variant cheaply and ``ptxas: shared > 232KB`` is bad UX during
# tuning.
TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES = 28 * 1024
# Static SMEM model of the DEFAULT-layout role-local GEMM
# (``CuteTcgen05Config.default_layout_smem_bytes``), measured on B200 from the
# compiled cubins and the kept PTX. Every ``cute.arch.alloc_smem`` becomes one
# PTX ``.shared`` variable; ptxas places them in reverse allocation order,
# each at the next offset rounded up to its alignment, and the arena may fill
# the opt-in capacity exactly (a probe allocating the 232 448 B opt-in
# compiles and launches; 8 B more fails in NVVM). ``cuobjdump`` reports the
# arena plus the 1 KiB the system reserves per CTA. The kernel allocates, in
# order: the TMEM holding buffer (4 B), the two-CTA dealloc mbarrier (8 B,
# dropped from one-CTA kernels), the accumulator barriers (16 B per acc
# stage), the A and B rings (128 B aligned), the AB mbarriers (16 B per
# stage), the C ring (1 KiB aligned) and the ``pre_acc_wait`` row stages
# (128 B aligned). Laid out in reverse, the row stages are padded to the C
# ring's alignment and the AB mbarriers to the B ring's, so the arena is
#   round_up(row stages, 1 KiB) + C ring + round_up(16 B * ab_stages, 128 B)
#   + AB ring + 16 B * acc_stages + 4 B (+ 8 B on a CTA pair),
# which matches every calibrated kernel to the byte: 215 204 B for the
# one-CTA 128x128x64 ab=6 fp16 GEMM with a 2 KiB fp32 row stage
# (2 048 + 16 384 + 128 + 196 608 + 36), 231 596 B for the 2x1 256x128x64
# ab=8 c=4 twin (2 048 + 32 768 + 128 + 196 608 + 44), 214 180 B with a
# promoted bn=128 row (512 B staged, 1 KiB occupied), 231 716 B at ab=12
# (256 B of mbarriers).
TCGEN05_SMEM_MBARRIER_BYTES = 8
# The B ring's 128 B alignment follows the AB mbarriers in the arena.
TCGEN05_SMEM_AB_MBARRIER_CHUNK_BYTES = 128
# The C ring's 1 KiB alignment follows the row stages in the arena.
TCGEN05_SMEM_ROW_STAGE_CHUNK_BYTES = 1024
TCGEN05_SMEM_TMEM_HOLDING_BUFFER_BYTES = 4
TCGEN05_SMEM_TWO_CTA_TMEM_DEALLOC_MBARRIER_BYTES = 8
# Small allocations the model does not enumerate, all 8/16 B aligned and
# placed after the A ring: the scheduler-warp mailbox barriers (16 B per
# stage, at most two stages), the CLC response buffer and mbarrier (24 B at
# 16 B alignment) and the DSL's own bookkeeping; measured at +40 B on a
# scheduler-warp kernel and +68 B on a CLC kernel. One 128 B chunk covers
# them.
TCGEN05_SMEM_SMALL_ALLOCATION_ALLOWANCE_BYTES = 128
# Hard floor on the per-CTA SMEM optin cap required to admit
# ``tcgen05_ab_stages=3`` into autotune search. B200's optin reports
# 232 448 bytes (= 227 * 1024). Devices below this threshold sit
# outside the warp-specialized tcgen05 envelope this gate is designed
# for, so admission stays off — explicit user configs still go through
# the validation surface and get the loud cute_dsl ptxas error.
TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN = 227 * 1024


def tcgen05_ab_smem_bytes_per_cta(
    *,
    bm: int,
    bn: int,
    bk: int,
    dtype_bytes: int,
    ab_stages: int,
    cluster_m: int,
) -> int:
    """Return per-CTA AB-SMEM bytes for a tcgen05 matmul tile / staging.

    The role-local lowering allocates the tcgen05 A/B SMEM staging via
    ``cutlass.utils.blackwell_helpers.make_smem_layout_a/b`` whose per-CTA
    ``cosize`` is the partitioned tile-shape times ``num_stages``. For
    ``cluster_m=2`` (CtaGroup.TWO) the tiled MMA partitions A and B across
    the two cluster CTAs so each CTA holds half of each operand. For
    ``cluster_m=1`` (CtaGroup.ONE) each CTA holds the full A/B operand.

    The autotune search-space gate uses this helper to admit
    ``tcgen05_ab_stages=3`` candidates only when the per-CTA cost fits
    the B200 SMEM optin budget after the non-AB reservation in
    ``TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES``. The numbers were
    cross-checked against ``cute.cosize`` for the canonical tile shapes
    (256x256x128 CtaGroup.TWO ab=3 = 196 608 bytes; 128x256x128
    CtaGroup.ONE ab=3 = 294 912 bytes / overflows the 232 KB optin cap).
    """
    assert ab_stages >= 1
    assert cluster_m in (1, 2)
    a_per_stage = bm * bk * dtype_bytes
    b_per_stage = bn * bk * dtype_bytes
    if cluster_m == 2:
        # CtaGroup.TWO: tiled MMA partitions both operands across the two
        # cluster CTAs. Each CTA's SMEM holds half of A and half of B.
        a_per_stage //= 2
        b_per_stage //= 2
    return ab_stages * (a_per_stage + b_per_stage)


def tcgen05_round_up_smem_bytes(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment`` (both positive)."""
    assert alignment > 0
    return -(-value // alignment) * alignment


def tcgen05_fixed_smem_overhead_bytes(
    *, ab_stages: int, acc_stages: int, cluster_m: int
) -> int:
    """Static SMEM the role-local GEMM allocates besides its rings and stages.

    The AB mbarriers as the arena lays them out (padded to the B ring's
    128 B alignment: 128 B up to eight stages, 256 B up to sixteen), the
    accumulator barriers, the TMEM holding buffer and, on a CTA pair, the
    dealloc mbarrier. Exact for the measured kernels (164 B one-CTA / 172 B
    two-CTA up to eight AB stages, 292 B / 300 B at nine to sixteen).
    """
    assert ab_stages > 0 and acc_stages > 0
    assert cluster_m in (1, 2)
    ab_mbarriers = tcgen05_round_up_smem_bytes(
        2 * ab_stages * TCGEN05_SMEM_MBARRIER_BYTES,
        TCGEN05_SMEM_AB_MBARRIER_CHUNK_BYTES,
    )
    acc_mbarriers = 2 * acc_stages * TCGEN05_SMEM_MBARRIER_BYTES
    dealloc = TCGEN05_SMEM_TWO_CTA_TMEM_DEALLOC_MBARRIER_BYTES if cluster_m == 2 else 0
    return (
        ab_mbarriers + acc_mbarriers + TCGEN05_SMEM_TMEM_HOLDING_BUFFER_BYTES + dealloc
    )


# Workstream A Stage 2 (cycle 90): the residual full-tile family gets a deeper
# C-store ring (foundation for the Stage-4 store-warp split, which drains tile N
# while the epi warps run tile N+1). The validated deeper depth is 4 — the only
# other ``EnumFragment((2, 4))`` choice — and it fits at ab=2 under the 232 KB
# B200 SMEM cap (256x256x128 CtaGroup.TWO bf16: 128 KB AB + 64 KB C = 192 KB).
# ab=3 + c=4 overflows (192 KB AB + 64 KB C = 256 KB → raw ``ptxas: too much
# shared``), so the deeper ring is admitted only behind the c-stages SMEM budget
# gate in ``CuteTcgen05Config``.
TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES = 4


def tcgen05_default_epilogue_tile_size(
    bm: int,
    bn: int,
    *,
    elem_width_d: int,
    elem_width_c: int | None,
) -> tuple[int, int]:
    """Return the DEFAULT (auto) epilogue subtile ``(tile_m, tile_n)`` in elems.

    The role-local ``Tcgen05LayoutStrategy.DEFAULT`` path sizes its epilogue
    subtile via ``cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape``
    (``tcgen05_default_epilogue_tile_expr``), whose tile choice is the pure
    Python ``compute_epilogue_tile_size``. The tile depends on whether the
    epilogue reads a source-C tensor: ``compute_epilogue_tile_size`` shrinks N
    when a C tile competes for SMEM. For a 256x256 CTA tile with 16-bit elements
    it is ``(128, 64)`` WITH a source C (residual family, ``elem_width_c`` set)
    but ``(128, 32)`` WITHOUT one (plain matmul, ``elem_width_c=None``) — and in
    both cases NOT the ``(128, 32)`` EXPLICIT_EPI_TILE direct-entry (TVM-FFI)
    tile, which is a separate codepath. Pass ``elem_width_c`` matching the
    config's real source-C presence so the SMEM budget gate matches codegen (and
    tracks any future CuTe tile-rule change rather than a hard-coded guess).
    """
    # Lazy import: ``cutlass`` is only loaded on the cute backend, while this
    # module is imported on the general autotuner path too.
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    # The role-local epilogue D/C tensors are row-major (LayoutEnum.ROW_MAJOR,
    # see ``cute_mma.py`` ``tcgen05_c_layout``), i.e. NOT M-major.
    tile_m, tile_n = compute_epilogue_tile_size(
        bm,
        bn,
        False,
        elem_width_d,
        elem_width_c,
        d_is_m_major=False,
        c_is_m_major=False,
    )
    return tile_m, tile_n


def tcgen05_c_smem_bytes_per_cta(
    *,
    epi_tile_m: int,
    epi_tile_n: int,
    dtype_bytes: int,
    c_stages: int,
) -> int:
    """Return per-CTA C-store-ring SMEM bytes for a tcgen05 epilogue.

    The role-local TMA-store epilogue allocates a multistage SMEM ring whose
    per-stage cosize is ``epi_tile_m * epi_tile_n * dtype_bytes``. The autotune
    search-space gate sums this against the AB pipeline cost to admit a deeper
    C ring (``tcgen05_c_stages=4``) only when the combined AB+C SMEM fits the
    B200 optin budget after the same non-AB reservation the
    ``tcgen05_ab_stages=3`` gate uses. For the residual full-tile family
    (256x256 CTA tile, 16-bit, DEFAULT epi tile ``(128, 64)``): each C stage is
    128*64*2 = 16 KB, so ab=2 + c=4 = 128 KB AB + 64 KB C = 192 KB (fits the
    232 KB cap), while ab=3 + c=4 = 192 KB AB + 64 KB C = 256 KB (overflows).
    """
    assert c_stages >= 1
    assert epi_tile_m > 0 and epi_tile_n > 0 and dtype_bytes > 0
    return c_stages * epi_tile_m * epi_tile_n * dtype_bytes


def tcgen05_aux_ring_smem_bytes_per_cta(
    *,
    epi_tile_m: int,
    epi_tile_n: int,
    dtype_bytes: int,
    aux_stages: int,
    ring_count: int,
) -> int:
    """Per-CTA SMEM of the staged exact-shape auxiliary rings plus barriers.

    The staged aux load modes (``tcgen05_aux_load_mode`` in
    ``TCGEN05_AUX_LOAD_MODES_STAGED``) allocate one ``make_smem_layout_epi``
    ring of ``aux_stages`` epilogue subtiles per exact-shape operand
    (``cute_mma._emit_tcgen05_aux_pipeline_setup``), each 1 KiB aligned, and
    one full/empty mbarrier pair per stage. Zero without staged operands.
    """
    assert aux_stages >= 1 and ring_count >= 0
    if ring_count == 0:
        return 0
    stage_bytes = tcgen05_round_up_smem_bytes(
        epi_tile_m * epi_tile_n * dtype_bytes, TCGEN05_SMEM_ROW_STAGE_CHUNK_BYTES
    )
    return ring_count * aux_stages * stage_bytes + 2 * aux_stages * (
        TCGEN05_SMEM_MBARRIER_BYTES
    )


# C-store epilogue placement knobs. Keeping these in Config makes generated-code
# changes visible to BoundKernel's Config-keyed compile cache; most values are
# used for diagnostics, while the output-edge + K-tail seed uses the measured
# faster first-in-loop / before-subtile-loop pair.
TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY = "tcgen05_c_acquire_placement"
TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP = "pre_loop"
TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE = "before_store"
TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP = "first_in_loop"
TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER = "later_before_barrier"
TCGEN05_C_ACQUIRE_PLACEMENTS = (
    TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
    TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER,
    TCGEN05_C_ACQUIRE_PLACEMENT_BEFORE_STORE,
)
TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY = "tcgen05_acc_wait_placement"
TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP = "subtile_loop"
TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP = "before_subtile_loop"
TCGEN05_ACC_WAIT_PLACEMENTS = (
    TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)
# Placement of independent per-subtile auxiliary-tensor loads (for example,
# scales, bias, or residuals) relative to the accumulator pipeline's consumer
# wait. Both TMA-store and full-tile SIMT epilogues honor this setting. The
# pre-wait form issues the auxiliary loads while tcgen05 MMA may still be
# producing the accumulator, hiding their latency; the TMEM-to-register copy
# and all accumulator-dependent epilogue arithmetic remain after the wait.
# Keeping auxiliary fragments live across the wait can increase register
# pressure, so post-wait is the conservative default in most cases.
# SIMT edge-only fallback loads are scheduled early
# independently of this setting because that path has been validated as a
# family.
TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY = "tcgen05_aux_load_placement"
TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT = "pre_acc_wait"
TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT = "post_acc_wait"
TCGEN05_AUX_LOAD_PLACEMENTS = (
    TCGEN05_AUX_LOAD_PLACEMENT_POST_ACC_WAIT,
    TCGEN05_AUX_LOAD_PLACEMENT_PRE_ACC_WAIT,
)
# The pid type of each GEMM launch of an ordered ``hl.barrier()`` chain (one
# entry per ``materialized_matmul_block_ids`` triple, like
# ``tcgen05_region_ab_stages``).  The chain's own ``pid_type`` is ``flat``
# (one ordered launch per phase, the SIMT phases flat); ``inherit`` keeps a
# GEMM launch flat, while a persistent value runs it as the static persistent
# two-CTA pipeline (``tcgen05_cta_group="two"``, ``tcgen05_cluster_m=2``) of
# the standalone single-GEMM surface, its ring depths per region
# (``tcgen05_region_ab_stages`` / ``tcgen05_region_c_stages``).  Pure
# scheduling: the same tiles, operands and epilogue arithmetic.
TCGEN05_REGION_PID_TYPES_CONFIG_KEY = "tcgen05_region_pid_types"
TCGEN05_REGION_PID_TYPE_INHERIT = "inherit"
TCGEN05_REGION_PID_TYPE_CHOICES: tuple[str, ...] = (
    TCGEN05_REGION_PID_TYPE_INHERIT,
    "persistent_blocked",
    "persistent_interleaved",
)
# The recipe a persistent GEMM launch of a chain runs under, projected onto
# its launch config by ``materialized_fission_codegen._stage_config`` (the
# standalone paired pipeline's fixed coordinates, ``_paired_pipeline_parameters``).
# The chain's whole-kernel config keeps the flat recipe of its ``inherit``
# launches (a scheduler warp, a one-slot accumulator ring, direct stores ...),
# which these launches do not see.
TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION: tuple[tuple[str, object], ...] = (
    ("tcgen05_persistence_model", "static_persistent"),
    ("tcgen05_strategy", "role_local_monolithic"),
    ("tcgen05_layout_strategy", "default"),
    ("tcgen05_warp_spec_mma_warps", 1),
    ("tcgen05_warp_spec_ab_load_warps", 1),
    ("tcgen05_warp_spec_epi_load_warps", 0),
    ("tcgen05_warp_spec_scheduler_warps", 0),
    ("tcgen05_warp_spec_c_input_warps", 0),
    ("tcgen05_warp_spec_store_warps", 0),
    ("tcgen05_num_epi_warps", 4),
    ("tcgen05_acc_stages", 2),
    ("tcgen05_c_store_mode", "normal"),
    ("tcgen05_aux_load_mode", "simt"),
    ("tcgen05_acc_wait_placement", "subtile_loop"),
    ("tcgen05_epilogue_layout", "normal"),
    ("tcgen05_tvm_ffi_launch", False),
    ("tcgen05_flat_role_coordinates", False),
    ("tcgen05_layout_overrides_epi_tile_m", None),
    ("tcgen05_layout_overrides_epi_tile_n", None),
    ("tcgen05_layout_overrides_smem_swizzle_a", None),
    ("tcgen05_layout_overrides_smem_swizzle_b", None),
    ("tcgen05_layout_overrides_d_store_box_n", None),
)
# ``pre_acc_wait`` on an exact-shape auxiliary operand (a full ``(m, n)``
# residual / gate input) normally issues each subtile's global read just ahead
# of that subtile's accumulator wait; the named barriers between subtiles
# expose every read's latency. With the whole-fragment hoist on, the epilogue
# warps read the tile's entire auxiliary fragment into registers once, before
# the accumulator wait, where the latency hides under the tile's MMA (the
# register hoist the two-CTA row-vector family already takes). Off keeps the
# per-subtile reads; the budget below bounds the live fragment so the hoist
# cannot spill (64 32-bit registers per thread at the 256-register epilogue
# ceiling).
TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY = "tcgen05_aux_fragment_hoist"
TCGEN05_AUX_FRAGMENT_HOIST_MAX_BYTES_PER_THREAD = 256


def tcgen05_aux_fragment_bytes_per_thread(
    *, bm: int, bn: int, itemsize: int, epi_warps: int
) -> int:
    """Bytes of one exact-shape auxiliary tile fragment held per epilogue thread.

    The epilogue partitions the ``(bm, bn)`` tile evenly over its
    ``epi_warps * 32`` threads; the hoist keeps every element of the thread's
    share live across the accumulator wait.
    """
    assert bm > 0 and bn > 0 and itemsize > 0 and epi_warps > 0
    return -(-bm * bn * itemsize // (epi_warps * 32))


class Tcgen05RowvecAuxRow(NamedTuple):
    """One N-broadcast (row-vector) aux operand of a tcgen05 epilogue."""

    itemsize: int
    # The store lowering stages the row converted to FP32 once per tile
    # (``memory_ops`` ``rowvec_stage_promotes_to_f32``): a 16-bit row that only
    # the FP32 root op ``acc <op> row`` consumes.
    promoted: bool


class Tcgen05RowvecAuxFacts(NamedTuple):
    """Pre-codegen facts about the row-vector aux rows of a matmul's stores.

    ``output_itemsize`` is the element size of the stored output(s); ``rows``
    lists every N-broadcast aux leaf the analyzed epilogue chains load. The
    autotune seeds use them to model the SMEM the ``pre_acc_wait`` row stages
    add (``CuteTcgen05Config.rowvec_aux_stage_fits``). ``output_count`` is the
    number of analyzed output stores: the store lowering allocates one
    TMA-store C ring per output, so the SMEM model charges the ring that many
    times (two fresh outputs of one accumulator keep two rings, with or
    without the shared fanout epilogue).
    """

    output_itemsize: int
    rows: tuple[Tcgen05RowvecAuxRow, ...]
    output_count: int = 1
    # Exact-shape rank-2 auxiliary operands of the output's dtype on the
    # store that reads the most of them: the staged aux load modes keep one
    # ``tcgen05_aux_stages``-deep ring of epilogue subtiles per operand.
    exact_aux_count: int = 0


def tcgen05_rowvec_stage_smem_bytes(
    *, rows: tuple[Tcgen05RowvecAuxRow, ...], bn: int, epi_warps: int
) -> int:
    """Per-CTA SMEM of the ``pre_acc_wait`` row stages for an N tile of ``bn``.

    Charges every stage the store lowering (``memory_ops``) may take: a
    promoted 16-bit row becomes one CTA-shared FP32 row (``bn * 4`` B, rows a
    multiple of 32 wide); a 32-bit row is staged once per epilogue warp
    (``epi_warps * bn * 4`` B) when it covers whole 128-bit copy tiles
    (``bn % 128 == 0``), is at least 64 wide and the stage stays within 4 KiB;
    every other row is read from GMEM per subtile and needs no SMEM.
    """
    total = 0
    for row in rows:
        if row.promoted and bn % 32 == 0:
            total += bn * 4
        elif row.itemsize == 4 and bn % 128 == 0 and bn >= 64:
            stage_bytes = epi_warps * bn * 4
            if stage_bytes <= 4 * 1024:
                total += stage_bytes
    return total


def tcgen05_two_cta_edge_k_tail_seed_overrides() -> dict[str, object]:
    """Return the measured CtaGroup.TWO edge+K-tail seed/fixup knobs."""
    return {
        "tcgen05_ab_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES,
        "tcgen05_acc_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_ACC_STAGES,
        "tcgen05_c_stages": TCGEN05_TWO_CTA_EDGE_K_TAIL_C_STAGES,
        "l2_groupings": [TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_GROUPING],
        "tcgen05_l2_swizzle_size": TCGEN05_TWO_CTA_EDGE_K_TAIL_L2_SWIZZLE_SIZE,
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY: (
            TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP
        ),
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY: (
            TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP
        ),
    }


TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY = "tcgen05_epilogue_layout"
TCGEN05_EPILOGUE_LAYOUT_NORMAL = "normal"
# Diagnostic-only layout split for the role-local TMA-store epilogue.
TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R = "split_first_t2r"
TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL = "split_acc_t2r_store_tail"
TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R = "module_helper_acc_t2r"
TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL = "module_helper_store_tail"
TCGEN05_EPILOGUE_LAYOUTS = (
    TCGEN05_EPILOGUE_LAYOUT_NORMAL,
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_FIRST_T2R,
    TCGEN05_EPILOGUE_LAYOUT_SPLIT_ACC_T2R_STORE_TAIL,
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_ACC_T2R,
    TCGEN05_EPILOGUE_LAYOUT_MODULE_HELPER_STORE_TAIL,
)
TCGEN05_C_STORE_MODE_CONFIG_KEY = "tcgen05_c_store_mode"
TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY = "tcgen05_diagnostic_invalid_output"
TCGEN05_C_STORE_MODE_NORMAL = "normal"
# Invalid-output diagnostic modes intentionally change correctness. General
# config validation requires the explicit diagnostic-invalid-output opt-in above.
TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE = "skip_epilogue_store"
# Register-direct output store: the epilogue warps write the converted
# accumulator fragments straight to global memory (the SIMT ``CopyR2GOp``
# body) instead of staging them through the C ring for a TMA store.  Exact
# (same conversion, same values); it trades the TMA store's coalescing for a
# shorter per-tile chain (no named barriers, no async-proxy fence, no
# bulk-store drain), which pays off when a CTA's whole epilogue is exposed.
TCGEN05_C_STORE_MODE_DIRECT = "direct"
# Small-grid seeds up to this N tile get a register-direct twin: at 16-bit
# output widths a thread's chunk is at most two 16-byte stores per tile.
TCGEN05_SMALL_GRID_DIRECT_STORE_MAX_BN = 32
TCGEN05_C_STORE_MODES = (
    TCGEN05_C_STORE_MODE_NORMAL,
    TCGEN05_C_STORE_MODE_DIRECT,
    TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE,
)
TCGEN05_AUX_LOAD_MODE_CONFIG_KEY = "tcgen05_aux_load_mode"
TCGEN05_AUX_LOAD_MODE_SIMT = "simt"
TCGEN05_AUX_LOAD_MODE_TMA = "tma"
# Exact-shape auxiliary tiles of the epilogue chain (a residual, a gate, a
# mask of the output's shape and dtype) staged per subtile through a TMA ring
# that the epilogue warps fill themselves: warp 0 issues the first
# ``tcgen05_aux_stages`` subtiles' bulk loads before the accumulator wait
# (they land under the tile's MMA) and refills the ring one subtile ahead of
# the reads. Unlike ``tma`` it needs no producer warp or scheduler, so the
# monolithic strategy, the static two-CTA pair and the fanout family take it,
# and the AB ring keeps every stage the SMEM model (``pipeline_smem``,
# ``default_layout_smem_bytes``) proves fits beside the aux ring.
TCGEN05_AUX_LOAD_MODE_EPI_TMA = "epi_tma"
TCGEN05_AUX_LOAD_MODES = (
    TCGEN05_AUX_LOAD_MODE_SIMT,
    TCGEN05_AUX_LOAD_MODE_TMA,
    TCGEN05_AUX_LOAD_MODE_EPI_TMA,
)
# Modes that stage exact-shape auxiliary tiles through the per-subtile SMEM
# aux rings (one ring per staged operand, ``tcgen05_aux_stages`` deep).
TCGEN05_AUX_LOAD_MODES_STAGED = (
    TCGEN05_AUX_LOAD_MODE_TMA,
    TCGEN05_AUX_LOAD_MODE_EPI_TMA,
)
# Both staged modes bulk-load every exact-shape auxiliary operand through a
# row-major TensorMap; codegen refuses them with this reason unless each
# operand proves a 16-byte base, 16-byte outer strides and a unit last stride
# (``cute_mma._tcgen05_tma_aux_tensor_is_legal``), and the normalizer /
# search surface mirror the bind-time proof (``exact_aux_tma_provable``).
TCGEN05_AUX_TMA_ILLEGAL_TENSOR_REASON = (
    "requires TMA-legal auxiliary tensors (a 16-byte base, 16-byte outer "
    "strides and a contiguous last axis)"
)
TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY = "tcgen05_acc_producer_mode"
TCGEN05_ACC_PRODUCER_MODE_NORMAL = "normal"
# Skips UMMA fence/issue while keeping AB and accumulator pipeline handshakes.
TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA = "skip_umma"
TCGEN05_ACC_PRODUCER_MODES = (
    TCGEN05_ACC_PRODUCER_MODE_NORMAL,
    TCGEN05_ACC_PRODUCER_MODE_SKIP_UMMA,
)
TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY = "tcgen05_acc_producer_advance_mode"
TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# removes only the accumulator producer PipelineState advance edge.
TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP = "skip"
TCGEN05_ACC_PRODUCER_ADVANCE_MODES = (
    TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL,
    TCGEN05_ACC_PRODUCER_ADVANCE_MODE_SKIP,
)
TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY = "tcgen05_ab_producer_acquire_mode"
TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# removes only AB producer acquire/try-acquire edges.
TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP = "skip"
TCGEN05_AB_PRODUCER_ACQUIRE_MODES = (
    TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL,
    TCGEN05_AB_PRODUCER_ACQUIRE_MODE_SKIP,
)
TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY = (
    "tcgen05_ab_initial_producer_acquire_mode"
)
TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# removes only the first initial-prefetch AB producer acquire edge.
TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST = "skip_first"
TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES = (
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_SKIP_FIRST,
)
TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY = "tcgen05_ab_producer_advance_mode"
TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# removes only the AB producer PipelineState advance edge.
TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP = "skip"
TCGEN05_AB_PRODUCER_ADVANCE_MODES = (
    TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL,
    TCGEN05_AB_PRODUCER_ADVANCE_MODE_SKIP,
)
TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY = "tcgen05_ab_consumer_wait_mode"
TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# removes only AB consumer try-wait/wait edges.
TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP = "skip"
TCGEN05_AB_CONSUMER_WAIT_MODES = (
    TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL,
    TCGEN05_AB_CONSUMER_WAIT_MODE_SKIP,
)
TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY = "tcgen05_ab_consumer_phase_mode"
TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL = "normal"
# Invalid-output diagnostic for the guarded clustered CtaGroup.ONE bridge:
# initializes the AB consumer pipeline state with phase 1 instead of phase 0.
TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1 = "phase1"
TCGEN05_AB_CONSUMER_PHASE_MODES = (
    TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL,
    TCGEN05_AB_CONSUMER_PHASE_MODE_PHASE1,
)
TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY = "tcgen05_sched_consumer_wait_mode"
TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL = "normal"
# Diagnostic-only scheduler-broadcast wait topology. Lane 0 waits on the
# sched-pipeline full barrier, then the warp reconverges before all lanes fence
# and read the SMEM mailbox. B200 profiling measured this slower than the
# normal whole-warp wait path; keep it opt-in for scheduler-wait experiments.
TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER = "warp_leader"
TCGEN05_SCHED_CONSUMER_WAIT_MODES = (
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL,
    TCGEN05_SCHED_CONSUMER_WAIT_MODE_WARP_LEADER,
)
TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY = "tcgen05_sched_stage_count"
TCGEN05_SCHED_STAGE_COUNTS = (1, 2)
TCGEN05_CUBIN_LINEINFO_CONFIG_KEY = "tcgen05_cubin_lineinfo"
TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY = "tcgen05_tvm_ffi_launch"

# Per-config knob for the C-input warp's auxiliary-tensor SMEM-ring pipeline
# stage count. Exposed as an autotune knob for ablation. Cycle-10 ablation
# (``cute_plan.md`` §6 Target 8) measured the 3-stage variant ~2% slower
# than the 2-stage default on T8, so the default 2 stays the autotuner's
# measured optimum; the knob remains in the search surface so future
# targets can re-explore. ``{2, 3}`` fit structurally on B200 — see
# ``cute_plan.md`` for the SMEM-budget audit. The autotune gate
# (``aux_stages_autotune_fragments`` in ``tcgen05_config.py``) admits
# the knob only for the ``_aux_tma_edge_search_enabled`` family so other
# paths stay byte-identical at the default. (Cycle 46 widened
# ``_aux_tma_search_enabled`` to also admit the full-tile cluster_m=2
# family, but this stage-count knob remains scoped to the edge gate
# because its choices were measured only on the T8/CLC edge rows.)
TCGEN05_AUX_STAGES_CONFIG_KEY = "tcgen05_aux_stages"
TCGEN05_AUX_STAGE_COUNT_DEFAULT = 2
TCGEN05_AUX_STAGE_COUNT_CHOICES = (2, 3)
# The epilogue-issued ring (``tcgen05_aux_load_mode="epi_tma"``) may also
# hold a whole four-subtile per-CTA tile ahead of the accumulator wait; its
# SMEM model (``CuteTcgen05Config.epi_tma_ab_stage_limit``) prices the depth.
TCGEN05_EPI_TMA_AUX_STAGE_COUNT_CHOICES = (2, 3, 4)
TCGEN05_AUX_STAGE_COUNT_VALIDATION_CHOICES = tuple(
    dict.fromkeys(
        (*TCGEN05_AUX_STAGE_COUNT_CHOICES, *TCGEN05_EPI_TMA_AUX_STAGE_COUNT_CHOICES)
    )
)

# Cycle 15 hypothesis 2 (``cute_plan.md`` §6 Target 8): per-config
# knob for the consumer-warp ``setmaxregister_increase`` ceiling. The
# default (256) preserves cycle-14 baseline emission; lower values
# cap the per-thread register budget that the consumer warps request,
# which forces ``ptxas`` to spill rather than reserve at the full
# 255-reg peak. The cycle-13 Deep Replan ranking flagged this as the
# most direct lever after H1 (rmem-allocation fold) was falsified in
# cycle 14 (NCU reg count stayed at 255). Surface: the call site is
# ``cute_mma.py`` ``_emit_mma_pipeline`` consumer branch; the
# autotune gate (``consumer_regs_autotune_fragments`` in
# ``tcgen05_config.py``) admits the knob only for the
# ``_aux_tma_edge_search_enabled`` family (mirroring the aux_stages
# gate) so other paths stay byte-identical at the 256 default. (Cycle
# 46 widened ``_aux_tma_search_enabled`` to also admit the full-tile
# cluster_m=2 family, but this register-cap knob stays scoped to the
# edge gate because its choices were measured only on the T8/CLC edge
# rows.) The choices include 256 so the default-with-knob configuration
# matches the default-without-knob emission byte-for-byte.
TCGEN05_CONSUMER_REGS_CONFIG_KEY = "tcgen05_consumer_regs"
TCGEN05_CONSUMER_REGS_DEFAULT = 256
TCGEN05_CONSUMER_REGS_CHOICES = (224, 232, 240, 256)

# Independent single-CTA grouped worklists can use ptxas's static register
# allocation instead of reserving a separate producer/consumer warpgroup budget.
# Keep the existing reallocation by default; the static alternative changes no
# MMA, pipeline, or memory ownership and remains subject to compiled resources.
TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY = "tcgen05_warp_register_reallocation"

# Stage tuples that the TVM-FFI flat-role seed accepts, keyed by ``bk``. The
# general FFI seed eligibility (``tcgen05_config.py``) consumes this table via
# ``tcgen05_direct_entry_stage_tuple_allowed``. ``bk=64`` admits the shallow
# ``(3, 2)`` and deep ``(6, 4)`` A/B pipelines; ``bk=128`` admits the
# ``(3, 2)`` pipeline. The admission is otherwise structural (any CtaGroup.TWO
# ``bm=bn=256`` shape) — these stage tuples are the only per-``bk`` constraint.
TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK: dict[int, tuple[tuple[int, int], ...]] = {
    64: ((3, 2), (6, 4)),
    128: ((3, 2),),
}


def tcgen05_direct_entry_stage_tuple_allowed(
    *, bk: int, ab_stage_count: int, c_stage_count: int
) -> bool:
    """Return True iff ``(ab_stage_count, c_stage_count)`` is admitted for ``bk``."""
    return (
        ab_stage_count,
        c_stage_count,
    ) in TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK.get(bk, ())


# Diagnostic-only G4 admission proof. This key lets tests exercise the
# smallest larger-BN tcgen05 codegen candidate without broadening production
# selector/search defaults.
TCGEN05_LARGE_BN_PROOF_CONFIG_KEY = "tcgen05_large_bn_proof"
# Diagnostic Target1 topology probe: keep role predicates warp-based while
# deriving logical MMA coordinates from flat threadIdx.x / warp / lane values.
TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY = "tcgen05_flat_role_coordinates"
# Grouped persistent scheduler/codegen mode for rank3 RHS grouped GEMM.
# Codegen owns the narrow envelope check before admitting generated kernels.
TCGEN05_GROUPED_MODE_CONFIG_KEY = "tcgen05_grouped_mode"
TCGEN05_GROUPED_FULL_COVERAGE_CONFIG_KEY = "tcgen05_grouped_full_coverage"
TCGEN05_GROUPED_FULL_COVERAGE_OFF = "off"
TCGEN05_GROUPED_FULL_COVERAGE_DENSE = "fixed_tma_dense"
TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL = "fixed_tma_dense_local"
TCGEN05_GROUPED_FULL_COVERAGE_MODES = (
    TCGEN05_GROUPED_FULL_COVERAGE_OFF,
    TCGEN05_GROUPED_FULL_COVERAGE_DENSE,
    TCGEN05_GROUPED_FULL_COVERAGE_DENSE_LOCAL,
)
TCGEN05_GROUPED_MODE_STATIC = "static"
TCGEN05_GROUPED_MODE_DYNAMIC = "dynamic"
TCGEN05_GROUPED_MODE_DIRECT = "direct"
TCGEN05_GROUPED_MODE_WORKLIST_NM = "worklist_nm"
TCGEN05_GROUPED_MODES = (
    TCGEN05_GROUPED_MODE_STATIC,
    TCGEN05_GROUPED_MODE_DYNAMIC,
    TCGEN05_GROUPED_MODE_DIRECT,
    TCGEN05_GROUPED_MODE_WORKLIST_NM,
)
TCGEN05_GROUPED_DYNAMIC_MODES = (
    TCGEN05_GROUPED_MODE_DYNAMIC,
    TCGEN05_GROUPED_MODE_DIRECT,
    TCGEN05_GROUPED_MODE_WORKLIST_NM,
)
TCGEN05_GROUPED_STATIC_BLOCK_K_CHOICES = (16, 32, 64, 128)
TCGEN05_GROUPED_STATIC_COMMON_K_BLOCK_PAIRS = (
    (16, 16),
    (32, 32),
    (64, 64),
    (96, 32),
    (160, 32),
    (192, 64),
)
TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY = "tcgen05_grouped_static_reserved_sms"
TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES = (0, 2, 3, 4, 5, 8, 16)
TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX = 1024
# Exact per-group M/N/K values embedded by an AOT specialization.  The runtime
# validates these values against the scheduler metadata before launch; codegen
# can then replace the generic warp-wide group search with constant branches.
TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY = (
    "tcgen05_grouped_static_problem_signature"
)
# The specialized scheduler emits one constant branch per group in each
# warp-specialized role. Larger signatures use the generic scheduler to bound
# generated source and compile time.
TCGEN05_GROUPED_STATIC_SPECIALIZATION_MAX_GROUPS = 8
# Direct pointer/stride metadata for grouped dynamic TensorMap updates. This
# keeps A/B/D payload tensors in place and only passes small per-group metadata
# tensors to generated tcgen05 code.
TCGEN05_GROUPED_EXTERNAL_DIRECT_POINTERS_CONFIG_KEY = (
    "tcgen05_grouped_external_direct_pointers"
)
TCGEN05_GROUPED_EXTERNAL_DIRECT_STRIDES_CONFIG_KEY = (
    "tcgen05_grouped_external_direct_strides"
)
# Validated source-row tiles for compact N,M worklist schedules.  The selected
# tile is carried explicitly by the schedule plan; it is independent of the
# CTA K tile and must not leak into general grouped-GEMM tiling decisions.
TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY = (
    "tcgen05_grouped_worklist_source_m_tile"
)
# The physical UMMA M dimension is the logical output-column tile in N,M
# orientation. None retains the established CTA-group-dependent geometry.
TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY = "tcgen05_grouped_worklist_mma_m"
TCGEN05_GROUPED_WORKLIST_MMA_M_CHOICES = (None, 64, 128, 256)
# Independent AB staging may combine two pure logical K64 iterations. The
# registered logical block and its caller-declared search ceiling stay intact.
TCGEN05_GROUPED_AB_TILE_K_CONFIG_KEY = "tcgen05_grouped_ab_tile_k"
TCGEN05_GROUPED_D_SHIFT_CONFIG_KEY = "tcgen05_grouped_d_shift"
TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY = "tcgen05_grouped_balanced_workers"
# The existing opt-in selects a validated paired width recipe from the
# configured physical source M tile. Absence/False keeps the native path.
TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY = "tcgen05_grouped_prefix_adaptive_n"
TCGEN05_GROUPED_FULL_ALLOCATION_A_CONFIG_KEY = "tcgen05_grouped_full_allocation_a"
# Reviewed runtime-variable-M profiles may opt into the host-expanded direct
# tile table.  Absence or explicit ``False`` retains the established
# scheduler-warp/SMEM-mailbox path, so profiles can choose between the two
# generic schedulers without specializing on the runtime M vector.
TCGEN05_GROUPED_RUNTIME_DIRECT_CONFIG_KEY = "tcgen05_grouped_runtime_direct"
# CUDA exposes grid.z as an unsigned 16-bit launch dimension. Runtime-direct
# CLC emits one exact tile record per cluster into that dimension.
TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS = (1 << 16) - 1
# Compact source rows avoid padding small-M experts to the historical 224-row
# default while retaining the 32-row granularity of the validated store wave.
TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE = 32
TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE = 128
TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT = 224
TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE = 256
TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES = (64, 128)
TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES = (
    TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
    TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
    TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE,
)
# N,M orientation maps the packed source-M tile onto the physical MMA-N mode.
# Keep the role-specific alias so descriptor assertions do not appear to be
# validating an MMA dimension against an unrelated packing knob.
# External worklists retain their established packing domain. Device layouts
# additionally admit source128 on ONE and TWO; source64 remains admitted only
# by the dense-local proof.
TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE = 64
TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES = (
    8,
    16,
    TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE,
    TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE,
    TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE,
)
TCGEN05_GROUPED_WORKLIST_TWO_CTA_SOURCE_M_TILE_CHOICES = (
    *TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
    TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE,
)
TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES = (
    8,
    16,
    *TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES,
    TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE,
    TCGEN05_GROUPED_WORKLIST_MEDIUM_SOURCE_M_TILE,
)
TCGEN05_GROUPED_WORKLIST_MMA_N_CHOICES = (
    TCGEN05_GROUPED_WORKLIST_TWO_CTA_SOURCE_M_TILE_CHOICES
)
TCGEN05_GROUPED_WORKLIST_MMA_M_TILE = 256
TCGEN05_GROUPED_WORKLIST_STORE_SHAPE = (128, 32, 32)


def tcgen05_grouped_worklist_store_shape(
    mma_m: int, mma_n: int
) -> tuple[int, int, int]:
    """Epilogue tile and store box for one physical MMA.

    The physical N dimension is routed rows. Small N tiles must keep their
    actual routed extent through TMEM loads, C staging and the TMA store box.
    """
    return (
        min(mma_m, TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[0]),
        min(mma_n, TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[1]),
        min(mma_n, TCGEN05_GROUPED_WORKLIST_STORE_SHAPE[2]),
    )


# The grouped scheduler mailbox reserves slots for CTA M/N, validity,
# metadata/group indices, problem M/N/K, and the packed-output row start.
# Generated consumers determine which non-validity slots are published.
TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT = 9


# Runtime-direct N,M scheduling expands one row per logical output tile on the
# host. It deliberately contains no validity sentinel: the runtime
# ``total_clusters`` scalar bounds every role-local loop. This shared enum is
# the schema contract between the host table builder and generated device loads.
class Tcgen05GroupedRuntimeTileField(IntEnum):
    CTA_M = 0
    CTA_N = 1
    METADATA_IDX = 2
    GROUP_IDX = 3
    PROBLEM_M = 4
    PROBLEM_N = 5
    PROBLEM_K = 6
    GLOBAL_M_START = 7
    VALID_M = 8
    STORE_M = 9


TCGEN05_GROUPED_RUNTIME_TILE_FIELD_COUNT = len(Tcgen05GroupedRuntimeTileField)


TCGEN05_LARGE_BN_PROOF_PROBLEM_SHAPE = (64, 512, 16)
TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES = (64, 512, 16)
TCGEN05_LARGE_BN_PROOF_CLUSTER_M = 1
TCGEN05_LARGE_BN_PROOF_PID_TYPE = "flat"
TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS = (
    ("tcgen05_ab_stages", 2),
    ("tcgen05_acc_stages", 1),
    ("tcgen05_c_stages", 2),
)
# Diagnostic-only codegen proof for the guarded clustered CtaGroup.ONE bridge.
TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY = (
    "tcgen05_cluster_m2_one_cta_role_local"
)

# CtaGroup.ONE tcgen05 MMA covers 64/128 M tiles; 256 M tiles are validated only
# after projecting onto the CtaGroup.TWO path.
TCGEN05_ONE_CTA_MAX_BLOCK_M = 128


def resolve_tcgen05_grouped_worklist_mma_shape(
    *,
    cluster_m: object,
    block_k: object,
    source_m_tile: object,
    mma_m: object | None = None,
) -> tuple[int, int] | None:
    """Resolve the physical ``(M, N)`` UMMA tile for a worklist profile.

    The public DSL tile remains 256x128 for every grouped N,M worklist.  The
    physical collective is instead 256x``source_m_tile`` for CtaGroup.TWO, or
    M128x8/16/32/128, M64x32, or the BK128-only M128x64 CtaGroup.ONE profile.
    Keep this shared by MMA selection, codegen, and SMEM validation so logical
    tile dimensions do not leak into physical resource accounting.
    """
    if (
        type(cluster_m) is not int
        or type(block_k) is not int
        or type(source_m_tile) is not int
        or block_k not in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
        or source_m_tile not in TCGEN05_GROUPED_WORKLIST_DEVICE_SOURCE_M_TILE_CHOICES
    ):
        return None
    if cluster_m == 1:
        if source_m_tile not in TCGEN05_GROUPED_WORKLIST_ONE_CTA_SOURCE_M_TILE_CHOICES:
            return None
        if (
            source_m_tile == TCGEN05_GROUPED_WORKLIST_WIDE_SOURCE_M_TILE
            and block_k != 128
        ):
            return None
        if mma_m is None:
            mma_m = TCGEN05_ONE_CTA_MAX_BLOCK_M
        if type(mma_m) is not int or mma_m not in (64, 128):
            return None
        if (
            mma_m == 64
            and source_m_tile != TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
        ):
            return None
        return mma_m, source_m_tile
    if (
        cluster_m == 2
        and source_m_tile in TCGEN05_GROUPED_WORKLIST_TWO_CTA_SOURCE_M_TILE_CHOICES
        and (mma_m is None or mma_m == TCGEN05_GROUPED_WORKLIST_MMA_M_TILE)
    ):
        return TCGEN05_GROUPED_WORKLIST_MMA_M_TILE, source_m_tile
    return None


class Tcgen05GroupedWorklistMmaProfile(NamedTuple):
    cluster_m: int
    source_m_tile: int
    mma_m: int
    mma_n: int


def resolve_tcgen05_grouped_worklist_mma_profile(
    config: Mapping[str, object],
    *,
    block_k: object,
) -> Tcgen05GroupedWorklistMmaProfile | None:
    """Resolve one normalized worklist config to its physical MMA profile."""
    if config.get(TCGEN05_GROUPED_MODE_CONFIG_KEY) != TCGEN05_GROUPED_MODE_WORKLIST_NM:
        return None
    if config.get("tcgen05_grouped_worklist_orientation", "nm") == "mn":
        source_m = config.get(TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY, 128)
        mma_n = config.get("tcgen05_grouped_worklist_mma_n", 256)
        if (
            type(source_m) is int
            and source_m in (128, 256)
            and type(mma_n) is int
            and mma_n in (128, 192, 224, 256)
            and config.get("tcgen05_cluster_m", 1) == 2
            and config.get("tcgen05_cluster_n", 1) in (1, 2)
            and block_k in (64, 128)
            and config.get(TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY)
            in (None, source_m)
        ):
            return Tcgen05GroupedWorklistMmaProfile(2, source_m, source_m, mma_n)
        return None
    cluster_m = config.get("tcgen05_cluster_m", 1)
    source_m_tile = config.get(
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CONFIG_KEY,
        TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_DEFAULT,
    )
    shape = resolve_tcgen05_grouped_worklist_mma_shape(
        cluster_m=cluster_m,
        block_k=block_k,
        source_m_tile=source_m_tile,
        mma_m=config.get(TCGEN05_GROUPED_WORKLIST_MMA_M_CONFIG_KEY),
    )
    if shape is None:
        return None
    assert type(cluster_m) is int and type(source_m_tile) is int
    return Tcgen05GroupedWorklistMmaProfile(
        cluster_m,
        source_m_tile,
        *shape,
    )


def _append_aligned_tcgen05_smem(offset: int, size: int, alignment: int) -> int:
    assert offset >= 0 and size >= 0 and alignment > 0
    return ((offset + alignment - 1) // alignment) * alignment + size


def tcgen05_grouped_worklist_epilogue_shape(
    *,
    bm: int,
    bn: int,
    cluster_m: int,
    output_dtype_bytes: int,
    orientation: str = "nm",
    explicit_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve the same epilogue dimensions as the device and D wrapper.

    N,M worklists keep their established explicit store tile. M,N worklists
    use the CuTe helper, including its divisibility and TWO/M128 permutation
    rules. Its pure-Python sizing entry point needs neither MLIR nor CUDA.
    """
    if explicit_shape is not None:
        return explicit_shape
    if orientation == "nm":
        return tcgen05_grouped_worklist_store_shape(bm, bn)[:2]
    assert orientation == "mn" and cluster_m in (1, 2)
    if not (cluster_m == 2 and bm == 128):
        return tcgen05_default_epilogue_tile_size(
            bm,
            bn,
            elem_width_d=output_dtype_bytes * 8,
            elem_width_c=output_dtype_bytes * 8,
        )
    # CuTe remains a lazy dependency for the backend-neutral config schema.
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    return compute_epilogue_tile_size(
        bm // 2,
        bn,
        True,
        output_dtype_bytes * 8,
        None,
        False,
        False,
    )


def tcgen05_grouped_worklist_smem_bytes(
    *,
    group_count: int,
    device_split_sizes: bool,
    sched_stage_count: int,
    bm: int,
    bn: int,
    bk: int,
    dtype_bytes: int,
    ab_stages: int,
    acc_stages: int,
    c_stages: int,
    cluster_m: int,
    output_dtype_bytes: int | None = None,
    orientation: str = "nm",
    epi_tile_shape: tuple[int, int] | None = None,
    immutable_ab_tensormaps: bool = False,
    immutable_d_tensormap: bool = False,
) -> int:
    """Return the worklist allocation bound for the resolved physical modes.

    The defaults retain the conservative dynamic-TensorMap bound. Descriptor
    discounts require the caller's input ownership/padding and output masking
    proofs, and must match descriptors actually omitted by codegen.
    Keep allocation ordering synchronized with codegen because alignment padding
    makes reordering observable at the admission boundary.
    """
    assert group_count > 0 and sched_stage_count > 0 and cluster_m in (1, 2)
    assert bm > 0 and bn > 0 and bk > 0 and dtype_bytes > 0
    assert ab_stages > 0 and acc_stages > 0 and c_stages > 0
    if output_dtype_bytes is None:
        output_dtype_bytes = dtype_bytes
    assert output_dtype_bytes > 0
    a_stage_bytes = bm * bk * dtype_bytes
    b_stage_bytes = bn * bk * dtype_bytes
    assert a_stage_bytes % cluster_m == 0
    assert b_stage_bytes % cluster_m == 0

    offset = 0
    # Scheduler work-tile mailbox, followed by optional device-derived problem
    # metadata. Host worklists supply problem sizes and starts from global
    # wrapper tensors, so they do not allocate either metadata array in SMEM.
    offset = _append_aligned_tcgen05_smem(
        offset,
        TCGEN05_GROUPED_WORKLIST_MAILBOX_FIELD_COUNT * sched_stage_count * 4,
        16,
    )
    if device_split_sizes:
        offset = _append_aligned_tcgen05_smem(offset, group_count * 4 * 4, 16)
        offset = _append_aligned_tcgen05_smem(offset, group_count * 4, 16)
    # TMEM allocator state and accumulator/scheduler pipeline mbarriers.
    offset = _append_aligned_tcgen05_smem(offset, 4, 4)
    offset = _append_aligned_tcgen05_smem(offset, 8, 8)
    offset = _append_aligned_tcgen05_smem(offset, acc_stages * 2 * 8, 8)
    offset = _append_aligned_tcgen05_smem(offset, sched_stage_count * 2 * 8, 8)
    # A/B stages, mutable input TensorMaps, and AB barriers.
    offset = _append_aligned_tcgen05_smem(
        offset, ab_stages * a_stage_bytes // cluster_m, 128
    )
    offset = _append_aligned_tcgen05_smem(
        offset, ab_stages * b_stage_bytes // cluster_m, 128
    )
    if not immutable_ab_tensormaps:
        offset = _append_aligned_tcgen05_smem(offset, 2 * 128, 128)
    # PipelineTmaUmma owns a full and an empty barrier for every AB stage.
    offset = _append_aligned_tcgen05_smem(offset, ab_stages * 2 * 8, 8)
    # Mutable output TensorMap and the aligned TMA-store ring.
    if not immutable_d_tensormap:
        offset = _append_aligned_tcgen05_smem(offset, 128, 128)
    epi_tile_m, epi_tile_n = tcgen05_grouped_worklist_epilogue_shape(
        bm=bm,
        bn=bn,
        cluster_m=cluster_m,
        output_dtype_bytes=output_dtype_bytes,
        orientation=orientation,
        explicit_shape=epi_tile_shape,
    )
    c_smem_bytes = tcgen05_c_smem_bytes_per_cta(
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        dtype_bytes=output_dtype_bytes,
        c_stages=c_stages,
    )
    return _append_aligned_tcgen05_smem(offset, c_smem_bytes, 1024)
