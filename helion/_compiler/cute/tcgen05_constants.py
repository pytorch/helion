from __future__ import annotations

# Validated CtaGroup.TWO autotune/runtime envelope for the B200 CuTe path.
# Re-verify the K-cap runtime and guard-boundary tests before raising the
# K-tile threshold or broadening the tile shape.
TCGEN05_TWO_CTA_BLOCK_M = 256
TCGEN05_TWO_CTA_BLOCK_N = 256
TCGEN05_TWO_CTA_MAX_K_TILES = 256
# Best measured seed L2 grouping for the validated 4096^3 CtaGroup.TWO row.
TCGEN05_TWO_CTA_SEED_L2_GROUPING = 4
# Same-session forced 4096^3 sweeps keep this pid order slightly ahead of
# persistent_blocked while staying in the validated CtaGroup.TWO envelope.
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


# Diagnostic-only C-store acquire placement knob. Keeping this in Config makes
# generated-code changes visible to BoundKernel's Config-keyed compile cache.
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
TCGEN05_CUBIN_LINEINFO_CONFIG_KEY = "tcgen05_cubin_lineinfo"
# Diagnostic-only G4 admission proof. This key lets tests exercise the
# smallest larger-BN tcgen05 codegen candidate without broadening production
# selector/search defaults.
TCGEN05_LARGE_BN_PROOF_CONFIG_KEY = "tcgen05_large_bn_proof"
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
