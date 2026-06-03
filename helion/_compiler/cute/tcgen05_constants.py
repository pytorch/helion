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
