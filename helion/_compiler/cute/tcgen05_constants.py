from __future__ import annotations

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
assert (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_AB_STAGES
    <= TCGEN05_TWO_CTA_EDGE_TMA_STORE_MAX_AB_STAGES
), "edge K-tail seed must fit the full-tile TMA-store AB-stage limit"
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


# C-store epilogue placement knobs. Keeping these in Config makes generated-code
# changes visible to BoundKernel's Config-keyed compile cache; most values are
# used for diagnostics, while the output-edge + K-tail seed uses the measured
# faster first-in-loop / before-subtile-loop pair.
TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY = "tcgen05_c_acquire_placement"
TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP = "pre_loop"
TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP = "first_in_loop"
TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER = "later_before_barrier"
TCGEN05_C_ACQUIRE_PLACEMENTS = (
    TCGEN05_C_ACQUIRE_PLACEMENT_PRE_LOOP,
    TCGEN05_C_ACQUIRE_PLACEMENT_FIRST_IN_LOOP,
    TCGEN05_C_ACQUIRE_PLACEMENT_LATER_BEFORE_BARRIER,
)
TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY = "tcgen05_acc_wait_placement"
TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP = "subtile_loop"
TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP = "before_subtile_loop"
TCGEN05_ACC_WAIT_PLACEMENTS = (
    TCGEN05_ACC_WAIT_PLACEMENT_SUBTILE_LOOP,
    TCGEN05_ACC_WAIT_PLACEMENT_BEFORE_SUBTILE_LOOP,
)


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
TCGEN05_C_STORE_MODES = (
    TCGEN05_C_STORE_MODE_NORMAL,
    TCGEN05_C_STORE_MODE_SKIP_EPILOGUE_STORE,
)
TCGEN05_AUX_LOAD_MODE_CONFIG_KEY = "tcgen05_aux_load_mode"
TCGEN05_AUX_LOAD_MODE_SIMT = "simt"
TCGEN05_AUX_LOAD_MODE_TMA = "tma"
TCGEN05_AUX_LOAD_MODES = (
    TCGEN05_AUX_LOAD_MODE_SIMT,
    TCGEN05_AUX_LOAD_MODE_TMA,
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

# Validated G2.2 Target1 diagnostic envelope for promoting the TVM-FFI launch
# path into normal autotune without enabling it broadly.
TCGEN05_TARGET1_TVM_FFI_SHAPE = (1024, 4096, 1024)
TCGEN05_TARGET1_TVM_FFI_BLOCK_K = 64
TCGEN05_TARGET1_TVM_FFI_AB_STAGES = 3
# Diagnostic-only G4 admission proof. This key lets tests exercise the
# smallest larger-BN tcgen05 codegen candidate without broadening production
# selector/search defaults.
TCGEN05_LARGE_BN_PROOF_CONFIG_KEY = "tcgen05_large_bn_proof"
# Diagnostic Target1 topology probe: keep role predicates warp-based while
# deriving logical MMA coordinates from flat threadIdx.x / warp / lane values.
TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY = "tcgen05_flat_role_coordinates"
# Diagnostic Target1 scheduler-object probe: use a Helion-owned pure-matmul
# CLC scheduler object to publish work tiles for the validated Target1
# TVM-FFI/flat-role seed. This stays explicit until forced diagnostics show
# it can replace the selected static scheduler path.
TCGEN05_PURE_CLC_SCHEDULER_OBJECT_CONFIG_KEY = "tcgen05_pure_clc_scheduler_object"
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
