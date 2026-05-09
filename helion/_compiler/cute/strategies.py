"""Data model for tcgen05 lowering strategy and warp-spec records.

The CuTe matmul lowering picks its kernel *shape* from a small,
named enum (``Tcgen05Strategy``) rather than from a flat bag of
boolean knobs. Within a chosen strategy the autotuner explores
structured records (``Tcgen05WarpSpec``, ``Tcgen05LayoutOverrides``).
This file is the single source of truth for what those types look
like and what their per-strategy invariants are.

See ``cute_plan.md`` §3 (three-axis framing) and §4 (data model).
G2-A introduces these types and wires them through ``ConfigSpec``
without changing generated code; later G2 sub-steps consume the
fields in codegen.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class Tcgen05Strategy(str, enum.Enum):
    """Structural kernel shape for the tcgen05 lowering.

    Strategies pick the *control flow / warp role inventory* of the
    generated kernel. They are *named* — they don't compose. Within
    a chosen strategy the autotuner explores records (warp split,
    layout overrides) that the strategy declares.

    - ``ROLE_LOCAL_MONOLITHIC`` (default, byte-identity-pinned).
      6 specialized warps (1 TMA-load + 1 MMA-exec + 4 epilogue).
      Each role-local ``while`` loop carries its own
      ``StaticPersistentTileScheduler``.
    - ``ROLE_LOCAL_WITH_SCHEDULER``. 7 specialized warps: adds a
      dedicated scheduler warp that publishes ``(virtual_pid,
      tile_coord_mnkl, is_valid)`` into a per-CTA SMEM mailbox via
      a ``PipelineAsync``. The C-input epilogue-load warp Quack
      uses (Quack's 8th warp) is not present in cycle 33 because
      Helion does not yet have a productive C-input warp (cycle
      34 widens ``Tcgen05WarpSpec.c_input_warps`` to 1 to occupy
      the slot); ``CuteTcgen05MatmulPlan.launched_warp_count``
      rounds to 8 launched warps (one inert padding warp) so
      warpgroup ``setmaxregister`` semantics are uniform.
      Validated at ``cluster_m`` ∈ {1, 2} and ``cluster_n``
      ∈ {1, 2}: each CTA in the cluster runs its own scheduler
      that publishes locally and each CTA's consumers release
      locally (no peer-CTA broadcast). Both CTAs converge on the
      same cluster-level virtual_pid because the consumer
      ``virtual_pid = work_tile_smem[0] // cluster_m + ...``
      formula collapses the per-CTA ``cta_id_in_cluster`` offset.
      cycle 33 lifted the cluster_n=2 restriction by widening the
      sched_pipeline ``cluster_size`` argument to the full
      ``cluster_m * cluster_n`` envelope so the deferred-init
      protocol participates in the cluster-wide barrier-init.
    """

    ROLE_LOCAL_MONOLITHIC = "role_local_monolithic"
    ROLE_LOCAL_WITH_SCHEDULER = "role_local_with_scheduler"


class Tcgen05PersistenceModel(str, enum.Enum):
    """Persistence axis for the tcgen05 lowering.

    Orthogonal to ``Tcgen05Strategy`` — the same warp-spec shape can
    run static or dynamic persistent. Today's ``pid_type=persistent_*``
    Helion config maps to ``STATIC_PERSISTENT`` by default.

    - ``CLC_PERSISTENT``: Blackwell sm_100+ hardware tile-scheduler
      driven persistent kernel via ``nvvm.clusterlaunchcontrol_try_cancel``
      (CLC). Quack-best uses this path whenever
      ``arch >= 100 and use_clc_persistence``; the CLC instruction is
      issued from a dedicated scheduler warp and writes the next
      cluster's CTA id into a SMEM response buffer (or a "canceled"
      sentinel when the wave finishes). G2-H (cute_plan.md) wires
      this through ``ROLE_LOCAL_WITH_SCHEDULER`` so the existing
      sched-warp role becomes the CLC issuer. Validator requires
      ``arch >= 100`` AND ``ROLE_LOCAL_WITH_SCHEDULER``.
    - ``DYNAMIC_PERSISTENT``: atomic-counter / ``tile_count_semaphore``
      driven dynamic persistent. Quack's fallback when CLC is
      unavailable. Helion has no codegen for this today; validator
      rejects it until a strategy consumes it.
    """

    NON_PERSISTENT = "non_persistent"
    STATIC_PERSISTENT = "static_persistent"
    CLC_PERSISTENT = "clc_persistent"
    DYNAMIC_PERSISTENT = "dynamic_persistent"


class Tcgen05LayoutStrategy(str, enum.Enum):
    """Axis-3 layout strategy: how epi_tile/swizzle/d-store choices
    are sourced.

    - ``DEFAULT``: rely on CuTe helpers (``compute_epilogue_tile_shape``,
      A/B major-mode swizzle inference). The autotuner cannot override
      these.
    - ``EXPLICIT_EPI_TILE``: user/autotune controls ``epi_tile_*`` and
      ``smem_swizzle_*`` via ``Tcgen05LayoutOverrides``. Fields whose
      override is ``None`` fall back to the CuTe-derived value.

    G2-A only declares the enum; ``EXPLICIT_EPI_TILE`` is wired in
    G2-E once warp-spec strategies have moved perf.
    """

    DEFAULT = "default"
    EXPLICIT_EPI_TILE = "explicit_epi_tile"


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------

# Default ``Tcgen05WarpSpec`` for ``ROLE_LOCAL_MONOLITHIC``. Pinned to the
# current 6-warp layout that the role-local lowering emits today. Keep these
# numbers in lockstep with ``program_id.py``'s warp-role accounting and
# ``cute_mma.py``'s register-split call.
ROLE_LOCAL_MONOLITHIC_AB_LOAD_WARPS = 1
ROLE_LOCAL_MONOLITHIC_MMA_WARPS = 1
ROLE_LOCAL_MONOLITHIC_EPI_WARPS = 4
ROLE_LOCAL_MONOLITHIC_EPI_LOAD_WARPS = 0
ROLE_LOCAL_MONOLITHIC_SCHEDULER_WARPS = 0
# ``c_input_warps`` slot for the dedicated C-input / auxiliary-tensor warp
# that drives a TMA-loaded SMEM-ring producer pipeline (G3.1-C step-2 in
# ``cute_plan.md`` §7.5.3.2). Default 0 keeps the historical inert-padding
# behavior under ``ROLE_LOCAL_WITH_SCHEDULER``; the validator allows the
# value 1 only under ``ROLE_LOCAL_WITH_SCHEDULER`` once the TMA producer +
# SMEM ring + role-local while loop land. The autotune surface stays
# narrowed to 0 until perf is characterized.
ROLE_LOCAL_MONOLITHIC_C_INPUT_WARPS = 0
# Today's role-local lowering uses a (decrease, increase) register split of
# (120, 256). The decrease side runs on TMA-load + scheduler warps; the
# increase side runs on MMA-exec + epilogue + epilogue-load warps.
ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT = (120, 256)


@dataclasses.dataclass(frozen=True)
class Tcgen05WarpSpec:
    """Structured record describing the warp-role split.

    Each field is independent so the autotuner can permute them, but
    cross-field invariants (sums, total warps, strategy compatibility)
    are validated together in ``ConfigSpec.normalize`` via
    ``validate_tcgen05_strategy_invariants``.

    Field meanings:

    - ``ab_load_warps``: warps issuing TMA loads for the A/B operands.
      Today ``ROLE_LOCAL_MONOLITHIC`` uses 1 (single TMA producer).
    - ``mma_warps``: warps issuing the tcgen05 MMA. Today's tcgen05
      atom contracts force this to 1 — a single warp issues the UMMA.
    - ``epi_warps``: warps reading TMEM and writing the output tile.
      ``tcgen05.ld`` is per-warp and CUTLASS's
      ``tmem_warp_shape_mn=(4,1)`` requires exactly 4 warps for
      correctness today (see §9.3 of ``cute_plan.md``).
    - ``epi_load_warps``: warps loading C input for the epilogue.
      0 today (no C input fused); 1 in the planned 8-warp shape.
    - ``scheduler_warps``: 0 in ``ROLE_LOCAL_MONOLITHIC`` (each role
      runs its own scheduler), 1 in ``ROLE_LOCAL_WITH_SCHEDULER``
      (dedicated scheduler warp drives a broadcasting pipeline).
    - ``c_input_warps``: dedicated C-input warp count. Currently
      narrowed to ``{0}`` for both strategies (the data-model slot
      is plumbed through normalize / round-trip but the validator
      rejects nonzero); cycle 34 widens
      ``ROLE_LOCAL_WITH_SCHEDULER`` to ``{0, 1}`` once the dedicated
      TMA producer + SMEM ring + role-local while loop land
      (``cute_plan.md`` §7.5.3.2). Setting this to 1 is intended
      to convert the 8-warp shape's currently-inert padding warp
      under ``ROLE_LOCAL_WITH_SCHEDULER`` into a productive
      C-input TMA producer; ``ROLE_LOCAL_MONOLITHIC`` has no such
      warp slot to occupy. The autotune surface stays narrowed to
      0 until cycle 34 perf-validates the productive C-input warp.
    - ``register_split``: ``(decrease, increase)`` ``setmaxregister``
      counts. The current 6-warp shape uses ``(120, 256)``. Each
      entry's range is enforced by its per-field fragment in
      ``ConfigSpec._tcgen05_strategy_autotune_fragments``; the
      surface stays narrowed to the implemented strategy's values
      until a strategy consumes the field. No cross-fragment
      invariant applies.
    """

    ab_load_warps: int
    mma_warps: int
    epi_warps: int
    epi_load_warps: int
    scheduler_warps: int
    register_split: tuple[int, int]
    # ``c_input_warps`` is keyword-only (via ``KW_ONLY``) so adding
    # the field after ``register_split`` doesn't shift positional
    # constructors that existing callers rely on; the validator and
    # lowering use the field name, not position.
    _: dataclasses.KW_ONLY
    c_input_warps: int = 0

    @property
    def total_warps(self) -> int:
        return (
            self.ab_load_warps
            + self.mma_warps
            + self.epi_warps
            + self.epi_load_warps
            + self.scheduler_warps
            + self.c_input_warps
        )


# Today's role-local lowering corresponds to the ``ROLE_LOCAL_MONOLITHIC``
# strategy with the warp split below. Keep this constant and its enum
# sibling in sync.
ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC = Tcgen05WarpSpec(
    ab_load_warps=ROLE_LOCAL_MONOLITHIC_AB_LOAD_WARPS,
    mma_warps=ROLE_LOCAL_MONOLITHIC_MMA_WARPS,
    epi_warps=ROLE_LOCAL_MONOLITHIC_EPI_WARPS,
    epi_load_warps=ROLE_LOCAL_MONOLITHIC_EPI_LOAD_WARPS,
    scheduler_warps=ROLE_LOCAL_MONOLITHIC_SCHEDULER_WARPS,
    register_split=ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT,
    c_input_warps=ROLE_LOCAL_MONOLITHIC_C_INPUT_WARPS,
)


@dataclasses.dataclass(frozen=True)
class Tcgen05LayoutOverrides:
    """Axis-3 overrides for layout choices that default to axis-1.

    Each field's ``None`` default means "use the value the analysis
    pass computed" (CuTe helper output, atom contract, etc.).

    G2-A introduces the slot. Today only ``Tcgen05LayoutStrategy.DEFAULT``
    is wired through codegen; ``EXPLICIT_EPI_TILE`` (G2-E) is the
    first consumer of the override fields. Validation today checks
    only the structural shape (types + ranges) — atom-contract checks
    against the active problem shape happen in lowering once the
    strategy is consumed there.

    ``smem_swizzle_a`` / ``smem_swizzle_b`` (user-config exposure only):
        Selects the A/B operand SMEM atom kind
        (``cute.nvgpu.tcgen05.SmemLayoutAtomKind``) by byte value
        (one of ``TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES``: ``0`` /
        ``32`` / ``64`` / ``128``). ``None`` ⇒ delegate to CuTe's
        ``get_smem_layout_atom_ab`` greedy auto-inference (the
        canonical-seed byte-identity path).

        **The autotuner does not sample swizzle overrides.** The
        canonical seed's auto-inference picks ``SW128`` (the
        optimal value), so adding the swizzle to the autotune
        fragment surface would only sample regressions. A future
        cycle that finds a per-shape win (e.g. tiny ``bk`` shapes
        where ``major_mode_bytes < 128`` gates ``SW128``) can lift
        this and add the swizzle to the autotune fragment surface.
    """

    epi_tile_m: int | None = None
    epi_tile_n: int | None = None
    smem_swizzle_a: int | None = None
    smem_swizzle_b: int | None = None
    d_store_box_n: int | None = None


# Legal SMEM atom swizzle byte values for ``Tcgen05LayoutOverrides.smem_swizzle_*``.
# Maps to ``cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind`` as follows:
#
# - ``0``  → ``{K|MN}_INTER`` (no swizzle, 16-byte interleave atom)
# - ``32`` → ``{K|MN}_SW32``  (32-byte swizzle pattern)
# - ``64`` → ``{K|MN}_SW64``  (64-byte swizzle pattern)
# - ``128``→ ``{K|MN}_SW128`` (128-byte swizzle pattern)
#
# The K/MN prefix is determined by the operand's major mode (A is K-major,
# B is MN-major in Helion's tcgen05 lowering today). ``MN_SW128_32B``
# (a fp32-only variant) is intentionally not exposed: Helion's tcgen05
# path only runs on bf16/fp16 today, and exposing the variant without a
# matching dtype gate would produce ``ValueError`` at codegen time.
TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES: tuple[int, ...] = (0, 32, 64, 128)

# Mapping from the swizzle byte choice to the smallest legal major-mode
# bytes-per-row. CuTe's auto-inference (``get_smem_layout_atom_ab``)
# requires ``major_mode_size_bits % num_contiguous_bits == 0`` where
# ``num_contiguous_bits`` is ``128 / 256 / 512 / 1024`` for INTER /
# SW32 / SW64 / SW128. We expose the bytes equivalent so the codegen
# validator can compute ``major_mode_bytes`` from the active tile shape
# + dtype and reject illegal user overrides with a clean error.
_TCGEN05_SMEM_SWIZZLE_BYTE_TO_MIN_MAJOR_BYTES: dict[int, int] = {
    0: 16,  # INTER: 128 contiguous bits = 16 bytes
    32: 32,  # SW32: 256 contiguous bits = 32 bytes
    64: 64,  # SW64: 512 contiguous bits = 64 bytes
    128: 128,  # SW128: 1024 contiguous bits = 128 bytes
}


def smem_swizzle_min_major_mode_bytes(swizzle_bytes: int) -> int:
    """Smallest legal major-mode bytes-per-row for the given swizzle.

    Used by ``cute_mma.py`` codegen-time validation: a user-provided
    ``smem_swizzle_a/b`` override must agree with the active tile
    shape + dtype's major-mode bytes-per-row, otherwise CuTe's
    ``make_smem_layout_atom`` would build a layout that the TMA
    contract rejects at runtime.
    """
    if swizzle_bytes not in _TCGEN05_SMEM_SWIZZLE_BYTE_TO_MIN_MAJOR_BYTES:
        raise ValueError(
            f"smem_swizzle_min_major_mode_bytes: {swizzle_bytes!r} is not "
            f"a legal swizzle byte choice; expected one of "
            f"{TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r}"
        )
    return _TCGEN05_SMEM_SWIZZLE_BYTE_TO_MIN_MAJOR_BYTES[swizzle_bytes]


def smem_swizzle_atom_kind_suffix(swizzle_bytes: int) -> str:
    """Map swizzle bytes to the SmemLayoutAtomKind suffix.

    Returns ``"INTER"``, ``"SW32"``, ``"SW64"``, or ``"SW128"`` so the
    codegen call site can build ``cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_<suffix>``
    or ``...MN_<suffix>``. Caller picks the K/MN prefix from the operand's
    major mode.
    """
    if swizzle_bytes not in _TCGEN05_SMEM_SWIZZLE_BYTE_TO_MIN_MAJOR_BYTES:
        raise ValueError(
            f"smem_swizzle_atom_kind_suffix: {swizzle_bytes!r} is not "
            f"a legal swizzle byte choice; expected one of "
            f"{TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r}"
        )
    if swizzle_bytes == 0:
        return "INTER"
    return f"SW{swizzle_bytes}"


def tcgen05_smem_layout_expr(
    *,
    tiled_mma: str,
    bm: int,
    bn: int,
    bk: int,
    dtype_str: str,
    num_stages: int,
    operand: str,
    swizzle_override: int | None,
) -> str:
    """Emit the CuTe expression that builds the staged SMEM layout for A or B.

    Single source of truth for the device-side
    (``cute_mma._make_tcgen05_layout_plan_setup``) and host-side
    (``runtime._append_cute_wrapper_plan``) atom expression: the two
    sides must agree byte-for-byte or the TMA descriptor mismatches
    the SMEM staging at runtime, so both codegen paths call this
    helper rather than emitting their own strings.

    With ``swizzle_override=None`` (the default) we delegate the atom
    kind selection to CuTe's ``make_smem_layout_a`` / ``make_smem_layout_b``
    helpers — that is the byte-identity behavior, kept unchanged on
    the canonical 4096³ seed.

    With an explicit ``swizzle_override`` (one of
    ``TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES`` — 0/32/64/128) we inline the
    body of ``make_smem_layout_a/b`` and substitute the chosen
    ``SmemLayoutAtomKind`` so the user-facing
    ``Tcgen05LayoutOverrides.smem_swizzle_*`` knob becomes load-bearing.
    The path mirrors CuTe's helper exactly except for the atom-kind
    selection: partition_shape → append num_stages → make atom →
    tile_to_mma_shape with the major-mode-determined ``order``.

    Helion's tcgen05 lowering wires ``OperandMajorMode.K`` for A and
    ``OperandMajorMode.MN`` for B (see ``make_trivial_tiled_mma``
    call in ``runtime._append_cute_wrapper_plan``), so the ``order``
    and atom-kind prefix below are hard-coded to that contract.
    """
    if operand == "a":
        if swizzle_override is None:
            return (
                "cutlass.utils.blackwell_helpers.make_smem_layout_a("
                f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, {num_stages})"
            )
        atom_kind = (
            f"cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_"
            f"{smem_swizzle_atom_kind_suffix(swizzle_override)}"
        )
        # A is K-major: partition shape uses dice (1, None, 1); the
        # tile_to_mma_shape ``order`` is (1, 2, 3) (matches the
        # is_k_major=True branch of ``make_smem_layout_a``).
        return (
            "cute.nvgpu.tcgen05.tile_to_mma_shape("
            f"cute.nvgpu.tcgen05.make_smem_layout_atom({atom_kind}, {dtype_str}), "
            f"cute.append({tiled_mma}.partition_shape_A("
            f"cute.dice(({bm}, {bn}, {bk}), (1, None, 1))), {num_stages}), "
            "order=(1, 2, 3))"
        )
    assert operand == "b", f"unexpected operand {operand!r}"
    if swizzle_override is None:
        return (
            "cutlass.utils.blackwell_helpers.make_smem_layout_b("
            f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, {num_stages})"
        )
    atom_kind = (
        f"cute.nvgpu.tcgen05.SmemLayoutAtomKind.MN_"
        f"{smem_swizzle_atom_kind_suffix(swizzle_override)}"
    )
    # B is MN-major: partition shape uses dice (None, 1, 1); the
    # tile_to_mma_shape ``order`` is (2, 1, 3) (matches the
    # is_k_major=False branch of ``make_smem_layout_b``).
    return (
        "cute.nvgpu.tcgen05.tile_to_mma_shape("
        f"cute.nvgpu.tcgen05.make_smem_layout_atom({atom_kind}, {dtype_str}), "
        f"cute.append({tiled_mma}.partition_shape_B("
        f"cute.dice(({bm}, {bn}, {bk}), (None, 1, 1))), {num_stages}), "
        "order=(2, 1, 3))"
    )


# ---------------------------------------------------------------------------
# Public config keys for the autotuner
# ---------------------------------------------------------------------------

# Top-level strategy choice. The autotune surface is narrowed to the
# implemented set in ``ConfigSpec._tcgen05_strategy_scalar_fragments``.
TCGEN05_STRATEGY_CONFIG_KEY = "tcgen05_strategy"

# Persistence model. The default is *derived from* ``pid_type`` so
# serialized configs cannot encode contradictions like
# ``pid_type=flat`` paired with ``static_persistent``.
TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY = "tcgen05_persistence_model"

# Layout strategy. Today only ``DEFAULT`` is wired through codegen.
TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY = "tcgen05_layout_strategy"

# ``Tcgen05WarpSpec`` field config keys. Each field is its own knob
# so the autotuner can permute them independently; cross-field
# invariants are checked by ``validate_tcgen05_strategy_invariants``.
#
# ``epi_warps`` is *not* exposed as a separate ``tcgen05_warp_spec_*``
# key — it is read from the existing ``tcgen05_num_epi_warps`` config
# field so there is a single source of truth (mismatched values
# cannot exist in serialized configs by construction).
TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY = "tcgen05_warp_spec_ab_load_warps"
TCGEN05_WARP_SPEC_MMA_WARPS_KEY = "tcgen05_warp_spec_mma_warps"
TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY = "tcgen05_warp_spec_epi_load_warps"
TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY = "tcgen05_warp_spec_scheduler_warps"
# ``c_input_warps``: dedicated C-input / auxiliary-tensor warp count.
# G3.1-C step-2 in ``cute_plan.md`` §7.5.3.2; today narrowed to 0 in
# the autotune surface, validated to ``{0, 1}`` only under
# ``ROLE_LOCAL_WITH_SCHEDULER`` in the user-config validation surface.
TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY = "tcgen05_warp_spec_c_input_warps"
# Register-split is exposed as two scalar keys (decrease, increase)
# rather than a tuple value because flat config values are scalars.
TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY = "tcgen05_warp_spec_register_decrease"
TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY = "tcgen05_warp_spec_register_increase"

# Source of truth for ``epi_warps`` lives in the existing
# ``tcgen05_num_epi_warps`` field, narrowed to ``(4,)`` at validation
# time by ``narrow_tcgen05_autotune_to_validated_configs``.
TCGEN05_NUM_EPI_WARPS_CONFIG_KEY = "tcgen05_num_epi_warps"

# L2 tile-scheduler swizzle size (Quack ``max_swizzle_size`` equivalent).
# Threaded into ``cutlass.utils.PersistentTileSchedulerParams`` as the
# ``swizzle_size`` kwarg. ``1`` means no swizzle (current default,
# byte-identity-pinned); larger values group consecutive cluster
# linear-IDs along the slow raster axis to improve L2 reuse on
# bandwidth-bound shapes. The CuTe-DSL scheduler already implements the
# grouping math; this knob just routes the user/autotuner choice into
# the constructor instead of relying on the ``swizzle_size=1`` default.
TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY = "tcgen05_l2_swizzle_size"

# Legal L2 tile-scheduler swizzle sizes. ``1`` means no swizzle (current
# default), the others mirror Quack's ``max_swizzle_size`` envelope (the
# upstream knob accepts powers of two from ``1`` up). The accept set is
# intentionally narrowed to powers of two ``<= 32`` so the autotuner
# does not waste budget exploring values that exceed any practical
# raster cluster count on B200; users who need a larger value can
# extend this set explicitly.
TCGEN05_LEGAL_L2_SWIZZLE_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

# Default L2 swizzle size (no swizzle = byte-identity-preserved).
TCGEN05_L2_SWIZZLE_SIZE_DEFAULT: int = 1

TCGEN05_WARP_SPEC_KEYS: tuple[str, ...] = (
    TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY,
    TCGEN05_WARP_SPEC_MMA_WARPS_KEY,
    TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY,
    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
    TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY,
    TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY,
    TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY,
)

# ``Tcgen05LayoutOverrides`` field config keys. Each defaults to None
# meaning "use Tcgen05DerivedShape default". Concrete values are only
# legal under ``Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE`` and require
# atom-contract validation that runs in lowering — at the data-model
# layer we only check value ranges.
TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY = "tcgen05_layout_overrides_epi_tile_m"
TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY = "tcgen05_layout_overrides_epi_tile_n"
TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY = "tcgen05_layout_overrides_smem_swizzle_a"
TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY = "tcgen05_layout_overrides_smem_swizzle_b"
TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY = "tcgen05_layout_overrides_d_store_box_n"

TCGEN05_LAYOUT_OVERRIDES_KEYS: tuple[str, ...] = (
    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY,
    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY,
    TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY,
    TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY,
    TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY,
)

# Aggregate of every config key the strategy data model adds.
TCGEN05_STRATEGY_CONFIG_KEYS: tuple[str, ...] = (
    TCGEN05_STRATEGY_CONFIG_KEY,
    TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY,
    TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
    *TCGEN05_WARP_SPEC_KEYS,
    *TCGEN05_LAYOUT_OVERRIDES_KEYS,
)


def l2_swizzle_size_from_config(config: Mapping[str, object]) -> int:
    """Read ``tcgen05_l2_swizzle_size`` out of a normalized config.

    Defaults to ``TCGEN05_L2_SWIZZLE_SIZE_DEFAULT`` (= ``1`` = no
    swizzle). Returns the raw integer; the codegen call site is
    responsible for emitting ``swizzle_size=`` as a kwarg to
    ``cutlass.utils.PersistentTileSchedulerParams(...)`` and for
    suppressing the kwarg when the value is ``1`` so the no-swizzle
    path stays byte-identical to pre-cycle-42. ``ConfigSpec.normalize``
    has already validated this knob against ``EnumFragment(int)`` so
    the value is guaranteed to be a positive integer at this point.
    """
    value = config.get(
        TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, TCGEN05_L2_SWIZZLE_SIZE_DEFAULT
    )
    return int(value)  # type: ignore[arg-type]


# Default values for the role-local-monolithic warp spec, keyed for
# config insertion. ``epi_warps`` is sourced from
# ``tcgen05_num_epi_warps`` — see TCGEN05_NUM_EPI_WARPS_CONFIG_KEY.
TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY: dict[str, int] = {
    TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_AB_LOAD_WARPS,
    TCGEN05_WARP_SPEC_MMA_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_MMA_WARPS,
    TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_EPI_LOAD_WARPS,
    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_SCHEDULER_WARPS,
    TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_C_INPUT_WARPS,
    TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY: ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT[0],
    TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY: ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT[1],
}


def warp_spec_from_config(config: Mapping[str, object]) -> Tcgen05WarpSpec:
    """Read warp-spec fields out of a normalized config.

    Accepts any ``Mapping[str, object]`` (e.g. ``dict`` or
    ``helion.Config``) so the codegen path can pass ``df.config``
    directly without unwrapping its inner dict.

    ``epi_warps`` is sourced from ``tcgen05_num_epi_warps`` — the
    existing single source of truth — so a user cannot simultaneously
    pass ``tcgen05_num_epi_warps`` and a separate
    ``tcgen05_warp_spec_epi_warps`` with mismatched values. Caller
    must have already passed *config* through ``ConfigSpec.normalize``
    so every key is present with a valid value. Raises ``KeyError``
    otherwise — there is no fallback at this layer because a missing
    key indicates a normalize() bug we want to surface loudly.
    """

    def _as_int(value: object) -> int:
        return int(value)  # type: ignore[arg-type]

    return Tcgen05WarpSpec(
        ab_load_warps=_as_int(config[TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY]),
        mma_warps=_as_int(config[TCGEN05_WARP_SPEC_MMA_WARPS_KEY]),
        epi_warps=_as_int(config[TCGEN05_NUM_EPI_WARPS_CONFIG_KEY]),
        epi_load_warps=_as_int(config[TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY]),
        scheduler_warps=_as_int(config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY]),
        register_split=(
            _as_int(config[TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY]),
            _as_int(config[TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY]),
        ),
        # ``c_input_warps`` was added in cycle 33 as the foundation
        # for G3.1-C step-2 (``cute_plan.md`` §7.5.3.2). The
        # validator below restricts its accept set per-strategy; the
        # default is 0 so configs serialized before cycle 33 (which
        # never carry the key) round-trip via ``ConfigSpec.normalize``
        # picking up ``ROLE_LOCAL_MONOLITHIC_C_INPUT_WARPS``.
        c_input_warps=_as_int(config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY]),
    )


def layout_overrides_from_config(
    config: Mapping[str, object],
) -> Tcgen05LayoutOverrides:
    """Read layout-override fields out of a normalized config."""

    def _as_optional_int(value: object) -> int | None:
        if value is None:
            return None
        return int(value)  # type: ignore[arg-type]

    return Tcgen05LayoutOverrides(
        epi_tile_m=_as_optional_int(
            config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY)
        ),
        epi_tile_n=_as_optional_int(
            config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY)
        ),
        smem_swizzle_a=_as_optional_int(
            config.get(TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY)
        ),
        smem_swizzle_b=_as_optional_int(
            config.get(TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY)
        ),
        d_store_box_n=_as_optional_int(
            config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY)
        ),
    )


# Set of ``pid_type`` values this helper knows how to map. Kept in
# sync with ``VALID_PID_TYPES`` in ``config_spec.py``; widening that
# tuple without revisiting this helper is a contract drift the assert
# below catches loudly.
_KNOWN_PID_TYPES_FOR_PERSISTENCE_MODEL: frozenset[str] = frozenset(
    {"flat", "xyz", "persistent_blocked", "persistent_interleaved"}
)


def derive_persistence_model_from_pid_type(
    pid_type: object,
) -> Tcgen05PersistenceModel:
    """Map the existing ``pid_type`` knob onto a persistence model.

    Used during ``ConfigSpec.normalize`` when the user did not supply
    an explicit ``tcgen05_persistence_model``. The mapping is
    intentionally conservative: only the existing knob values are
    recognized, and ``DYNAMIC_PERSISTENT`` / ``CLC_PERSISTENT`` never
    fall out of this function — they are reachable only via an
    explicit user opt-in. Helion has no codegen for dynamic
    persistence today, and CLC requires explicit user opt-in so the
    benchmarking surface stays under explicit control until perf is
    characterized.

    ``pid_type`` is validated upstream by ``ConfigSpec.normalize``
    against ``VALID_PID_TYPES`` so this helper never sees an unknown
    value in practice — assert that contract so a future widening of
    ``VALID_PID_TYPES`` that forgets to revisit this mapping fails
    loudly instead of silently mapping the new value to
    ``NON_PERSISTENT``.
    """
    assert pid_type in _KNOWN_PID_TYPES_FOR_PERSISTENCE_MODEL, (
        f"derive_persistence_model_from_pid_type: unknown pid_type "
        f"{pid_type!r}; update _KNOWN_PID_TYPES_FOR_PERSISTENCE_MODEL "
        "to include the new value (or extend the mapping if the new "
        "pid_type implies a different persistence model)."
    )
    if pid_type in ("persistent_blocked", "persistent_interleaved"):
        return Tcgen05PersistenceModel.STATIC_PERSISTENT
    return Tcgen05PersistenceModel.NON_PERSISTENT


def _persistence_model_pid_type_compatible(
    persistence_model: Tcgen05PersistenceModel, pid_type: object
) -> bool:
    """Return whether ``persistence_model`` is consistent with ``pid_type``.

    ``CLC_PERSISTENT`` overlays a runtime CLC query on a
    ``pid_type=persistent_*`` launch (the kernel still runs in
    persistent-grid mode; CLC just replaces the static launch-grid
    distribution with a hardware-driven canceller). So the agreement
    is "CLC and STATIC both require ``pid_type=persistent_*``", not
    "the persistence model derived from pid_type matches exactly".
    """
    derived = derive_persistence_model_from_pid_type(pid_type)
    if persistence_model == derived:
        return True
    if persistence_model is Tcgen05PersistenceModel.CLC_PERSISTENT:
        return derived is Tcgen05PersistenceModel.STATIC_PERSISTENT
    return False


# ---------------------------------------------------------------------------
# Strategy-conditional cross-fragment invariants
# ---------------------------------------------------------------------------

# A strategy's *capability* table: which persistence models the
# generated codegen can actually produce. Keep this aligned with the
# strategies enumerated above and update when a new lowering lands.
_STRATEGY_SUPPORTED_PERSISTENCE: dict[
    Tcgen05Strategy, frozenset[Tcgen05PersistenceModel]
] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: frozenset(
        {
            Tcgen05PersistenceModel.NON_PERSISTENT,
            Tcgen05PersistenceModel.STATIC_PERSISTENT,
        }
    ),
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: frozenset(
        {
            Tcgen05PersistenceModel.NON_PERSISTENT,
            Tcgen05PersistenceModel.STATIC_PERSISTENT,
            # G2-H: CLC-based dynamic persistent scheduling. The
            # sched-warp role doubles as the CLC query issuer; the
            # consumer warps receive the next cluster's CTA id via
            # the same SMEM mailbox + ``PipelineAsync`` topology as
            # the static path. Arch-gated separately.
            Tcgen05PersistenceModel.CLC_PERSISTENT,
        }
    ),
}

# Minimum CUDA compute capability (major version) for each persistence
# model. Used by the validator to reject CLC selection on architectures
# without the ``clusterlaunchcontrol`` family of instructions. Only
# ``CLC_PERSISTENT`` is gated today; the static / non-persistent paths
# work on every supported arch.
_PERSISTENCE_MIN_ARCH_MAJOR: dict[Tcgen05PersistenceModel, int] = {
    Tcgen05PersistenceModel.CLC_PERSISTENT: 10,
}

# Strategy-conditional warp-count requirements. Keys are strategy
# values. Each entry lists the *required* warp counts for that
# strategy. ``None`` means "any value in the validated range".
_STRATEGY_REQUIRED_SCHEDULER_WARPS: dict[Tcgen05Strategy, int] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: 0,
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: 1,
}

# Strategy-conditional ``c_input_warps`` accept set. Per ``cute_plan.md``
# §7.5.3.2, only ``ROLE_LOCAL_WITH_SCHEDULER`` can host a productive
# C-input warp (the inert padding warp under the WITH_SCHEDULER 8-warp
# shape becomes the C-input TMA producer). ``ROLE_LOCAL_MONOLITHIC``
# has no such warp slot — its 6-warp shape is fully populated by the
# TMA-load + MMA-exec + 4 epi warps. Values outside the per-strategy
# accept set are rejected so user configs cannot reach an unsupported
# combination silently. cycle 33: validator accepts ``{0}`` for both
# strategies; cycle 34 widens ``ROLE_LOCAL_WITH_SCHEDULER`` to ``{0, 1}``
# once the dedicated TMA producer + SMEM ring + role-local while
# loop land. This staging keeps the data-model slot stable for
# config serialization round-trips while keeping user configs from
# reaching a not-yet-implemented codegen path.
_STRATEGY_SUPPORTED_C_INPUT_WARPS: dict[Tcgen05Strategy, frozenset[int]] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: frozenset({0}),
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: frozenset({0}),
}

# When the strategy demands warpgroup-aligned splits (every 4 warps
# == one warpgroup, ``setmaxregister`` is warpgroup-uniform), the
# total warp count must be a multiple of 4. ``ROLE_LOCAL_MONOLITHIC``
# (1+1+4 = 6 role warps) and ``ROLE_LOCAL_WITH_SCHEDULER`` (1+1+4+1 =
# 7 role warps) both rely on the launched-warp padding in
# ``CuteTcgen05MatmulPlan.launched_warp_count`` to round to a
# multiple of 4 at the launch boundary, so neither strategy needs
# this validator-level check today. The set is intentionally empty
# but the branch below stays live: a future strategy whose role
# count *itself* must be a multiple of 4 (e.g. when ``register_split``
# becomes per-warpgroup and the role assignment is rigid) can opt
# in by adding itself to this set, and the validator will reject
# misconfigured ``Tcgen05WarpSpec`` records loudly.
_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL: frozenset[Tcgen05Strategy] = frozenset()

# Strategy-conditional cluster-shape capability. Each entry lists
# the cluster_m values the strategy's lowering is currently known
# to run correctly. ``ROLE_LOCAL_MONOLITHIC`` runs at cluster_m 1
# and 2 (the validated cluster_m=2 ONE-CTA bridge plus the
# default cluster_m=1 path). ``ROLE_LOCAL_WITH_SCHEDULER`` runs
# correctly at cluster_m 1 and 2 — both CTAs in the cluster run
# their own scheduler that publishes locally and consumers
# release locally (see ``cute_mma._codegen_cute_mma`` for the
# ``consumer_mask_to_leader=False`` justification). Setting
# ``cluster_m`` outside the supported set is rejected by the
# cross-fragment validator so a user config can't reach a
# hanging runtime.
_STRATEGY_SUPPORTED_CLUSTER_M: dict[Tcgen05Strategy, frozenset[int]] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: frozenset({1, 2}),
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: frozenset({1, 2}),
}

# Strategy-conditional cluster_n capability. cluster_n=2 is now validated
# under both ``ROLE_LOCAL_MONOLITHIC`` (the existing Quack-canonical 4-CTA
# cluster path) and ``ROLE_LOCAL_WITH_SCHEDULER`` (cycle 33: the precursor
# for G3.1-C step-2's productive C-input warp). The WITH_SCHEDULER
# scheduler topology is "every CTA in the cluster runs its own scheduler
# that publishes locally" (see ``cute_mma._codegen_cute_mma`` ``consumer
# _mask_to_leader=False`` branch); generalizing to cluster_n=2 keeps the
# per-CTA-local pattern with the cluster_size at the sched_pipeline level
# updated to ``cluster_m * cluster_n`` so the deferred-pipeline cluster-
# barrier sync still spans the full V=2 cluster_n=2 4-CTA cluster envelope.
# Setting outside the supported set is rejected so user configs cannot
# reach an untested cluster shape.
_STRATEGY_SUPPORTED_CLUSTER_N: dict[Tcgen05Strategy, frozenset[int]] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: frozenset({1, 2}),
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: frozenset({1, 2}),
}

# (strategy, persistence_model)-conditional cluster_n accept set. Used to
# reject combinations where the (strategy, persistence) pair has a
# cluster-broadcast topology that has not been generalized to cluster_n>1.
# Today's only entry: ``CLC_PERSISTENT`` under ``ROLE_LOCAL_WITH_SCHEDULER``
# is cluster_m-only — the CLC scheduler-warp body iterates lanes
# ``< cluster_m`` to publish to peer CTAs (see ``program_id.py``
# ``_build_scheduler_warp_role_local_while_clc``); cluster_n>1 CTAs would
# never receive the CLC mailbox publish and would hang on the
# ``producer_acquire`` wait. Generalizing the CLC broadcast to cluster_n>1
# is out of scope for cycle 33 (deferred to a future cycle alongside the
# ``cluster_n>1`` C-input pipeline work). When a (strategy, persistence)
# pair appears here, it overrides the per-strategy
# ``_STRATEGY_SUPPORTED_CLUSTER_N`` entry; missing entries fall back to
# the per-strategy capability.
_STRATEGY_PERSISTENCE_SUPPORTED_CLUSTER_N: dict[
    tuple[Tcgen05Strategy, Tcgen05PersistenceModel], frozenset[int]
] = {
    (
        Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
        Tcgen05PersistenceModel.CLC_PERSISTENT,
    ): frozenset({1}),
}


def validate_tcgen05_strategy_invariants(
    *,
    strategy: Tcgen05Strategy,
    persistence_model: Tcgen05PersistenceModel,
    layout_strategy: Tcgen05LayoutStrategy,
    warp_spec: Tcgen05WarpSpec,
    layout_overrides: Tcgen05LayoutOverrides,
    pid_type: object,
    cluster_m: int,
    cluster_n: int = 1,
    arch_major: int | None = None,
) -> list[str]:
    """Cross-fragment invariants for the tcgen05 strategy data model.

    Returns the list of failure messages rather than raising so the
    caller can decide how to surface them (raise, log, silently
    drop). The matmul path uses raise; ``_fix_invalid`` callers may
    discard.

    Per-field range checks (``ab_load_warps >= 1``, register-split
    1..256, etc.) are *not* duplicated here — those are already
    enforced by the ``EnumFragment`` / ``IntegerFragment`` ranges in
    ``ConfigSpec._tcgen05_strategy_scalar_fragments``. ``epi_warps``
    is *also* not checked here — it is owned by the existing
    ``tcgen05_num_epi_warps`` validation
    (``restrict_tcgen05_num_epi_warps_validation``); duplicating the
    rule across two layers makes the lifting story confusing.

    The invariants here are the *cross-fragment* ones the strategy
    enum determines:

    - persistence model must be supported by the chosen strategy,
      and must agree with the active ``pid_type`` knob;
    - persistence model arch gate (e.g. ``CLC_PERSISTENT`` requires
      ``arch_major >= 10``);
    - scheduler-warp count is strategy-determined;
    - the total warp count must be warpgroup-aligned when the
      strategy demands it;
    - the active ``cluster_m`` must be in the strategy's supported
      cluster-shape set;
    - layout-override values are only legal under
      ``Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE``.

    ``arch_major`` is the major compute capability (e.g. 10 for
    sm_100). The runtime call site (``ConfigSpec.normalize`` when
    CUDA is available) is the only one that supplies a real value;
    the unit-test / serialization round-trip paths that have no GPU
    pass ``None`` so the arch-gate check skips and the rest of the
    invariants still validate. A CPU-only CI replay would therefore
    accept ``CLC_PERSISTENT`` without rejection — that's intentional
    so config serialization round-trips don't depend on the host
    GPU, but it also means CPU CI never catches an arch-gated
    misconfiguration on its own; the runtime path catches it on
    first compile.

    Mirrors the pattern of
    ``narrow_tcgen05_autotune_to_validated_configs`` — call from
    ``ConfigSpec.normalize`` after every fragment has been resolved.
    """
    errors: list[str] = []

    # Persistence model must be supported by the chosen strategy.
    supported = _STRATEGY_SUPPORTED_PERSISTENCE.get(strategy, frozenset())
    if persistence_model not in supported:
        errors.append(
            f"persistence model {persistence_model.value!r} is not "
            f"supported by tcgen05 strategy {strategy.value!r}; "
            f"valid choices are {sorted(m.value for m in supported)!r}"
        )

    # The persistence model must agree with the active pid_type so
    # that serialized configs do not carry contradictory state. Today
    # ``pid_type=flat|xyz`` implies ``NON_PERSISTENT`` and
    # ``pid_type=persistent_*`` implies ``STATIC_PERSISTENT`` (or
    # ``CLC_PERSISTENT`` when the user explicitly opts in — CLC
    # overlays the same persistent-grid launch with a runtime CLC
    # query). ``DYNAMIC_PERSISTENT`` has no codegen path so it never
    # agrees with any current ``pid_type`` value.
    if not _persistence_model_pid_type_compatible(persistence_model, pid_type):
        derived = derive_persistence_model_from_pid_type(pid_type)
        errors.append(
            f"tcgen05_persistence_model={persistence_model.value!r} "
            f"contradicts pid_type={pid_type!r} (which implies "
            f"{derived.value!r}); set both consistently or drop one "
            "to take the derived default"
        )

    # Arch-gated persistence models. ``CLC_PERSISTENT`` requires
    # sm_100+ because ``nvvm.clusterlaunchcontrol_try_cancel`` is a
    # Blackwell hardware feature.
    min_arch = _PERSISTENCE_MIN_ARCH_MAJOR.get(persistence_model)
    if min_arch is not None and arch_major is not None and arch_major < min_arch:
        errors.append(
            f"tcgen05_persistence_model={persistence_model.value!r} "
            f"requires CUDA compute capability major >= {min_arch} "
            f"(arch sm_{min_arch}0+); current arch major is "
            f"{arch_major}"
        )

    # scheduler_warps is strategy-determined.
    required_scheduler = _STRATEGY_REQUIRED_SCHEDULER_WARPS.get(strategy)
    if (
        required_scheduler is not None
        and warp_spec.scheduler_warps != required_scheduler
    ):
        errors.append(
            f"tcgen05 strategy {strategy.value!r} requires "
            f"scheduler_warps={required_scheduler}, got "
            f"{warp_spec.scheduler_warps}"
        )

    # ``c_input_warps`` accept set is per-strategy; today narrows
    # both strategies to ``{0}`` until the dedicated TMA producer +
    # SMEM ring + role-local while loop lands. The data-model slot
    # is plumbed through normalize / round-trip paths so the
    # accept set can widen without further config-shape churn.
    supported_c_input = _STRATEGY_SUPPORTED_C_INPUT_WARPS.get(strategy, frozenset())
    if supported_c_input and warp_spec.c_input_warps not in supported_c_input:
        errors.append(
            f"tcgen05 strategy {strategy.value!r} only accepts "
            f"c_input_warps in {sorted(supported_c_input)!r}; got "
            f"c_input_warps={warp_spec.c_input_warps}"
        )

    # ``epi_warps`` is intentionally NOT checked here — its accept-
    # set is owned by ``tcgen05_num_epi_warps`` validation
    # (``restrict_tcgen05_num_epi_warps_validation``) so there is one
    # place to lift the gate when G2 fixes the multi-warp store.

    # Total warp count must form clean warpgroups when the strategy
    # requires it.
    if (
        strategy in _STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL
        and warp_spec.total_warps % 4 != 0
    ):
        errors.append(
            f"tcgen05 strategy {strategy.value!r} requires the total "
            f"warp count to be a multiple of 4 (warpgroup-aligned); "
            f"got total_warps={warp_spec.total_warps}"
        )

    # Active cluster_m must be in the strategy's supported set.
    # See ``_STRATEGY_SUPPORTED_CLUSTER_M`` for per-strategy
    # capability and the dict-level comment for the topology
    # rationale. Rejecting unsupported values here keeps a user-set
    # ``helion.Config(tcgen05_strategy=..., tcgen05_cluster_m=...)``
    # from compiling onto a runtime path that has not been
    # validated.
    supported_cluster_m = _STRATEGY_SUPPORTED_CLUSTER_M.get(strategy, frozenset())
    if supported_cluster_m and cluster_m not in supported_cluster_m:
        errors.append(
            f"tcgen05 strategy {strategy.value!r} only runs correctly "
            f"at tcgen05_cluster_m in {sorted(supported_cluster_m)!r}; "
            f"got tcgen05_cluster_m={cluster_m}"
        )

    # Active cluster_n must be in the strategy's supported set.
    # See ``_STRATEGY_SUPPORTED_CLUSTER_N`` for per-strategy
    # capability. cluster_n=2 also requires cluster_m=2 (the V=2
    # 4-CTA cluster); enforce both checks since a config that sets
    # cluster_n=2 on cluster_m=1 would silently drop in codegen and
    # mislead the user.
    supported_cluster_n = _STRATEGY_SUPPORTED_CLUSTER_N.get(strategy, frozenset())
    if supported_cluster_n and cluster_n not in supported_cluster_n:
        errors.append(
            f"tcgen05 strategy {strategy.value!r} only runs correctly "
            f"at tcgen05_cluster_n in {sorted(supported_cluster_n)!r}; "
            f"got tcgen05_cluster_n={cluster_n}"
        )
    if cluster_n > 1 and cluster_m != 2:
        errors.append(
            f"tcgen05_cluster_n={cluster_n} requires tcgen05_cluster_m=2 "
            f"with use_2cta=True (the validated 4-CTA cluster envelope; "
            f"cute_plan.md §6.12); got tcgen05_cluster_m={cluster_m}"
        )

    # (strategy, persistence_model) paired cluster_n accept set.
    # Some persistence models have cluster-broadcast topology that
    # has not been generalized to cluster_n>1; they must reject
    # cluster_n>1 even when the per-strategy capability allows it.
    # See ``_STRATEGY_PERSISTENCE_SUPPORTED_CLUSTER_N`` for the rationale.
    paired_supported_cluster_n = _STRATEGY_PERSISTENCE_SUPPORTED_CLUSTER_N.get(
        (strategy, persistence_model)
    )
    if (
        paired_supported_cluster_n is not None
        and cluster_n not in paired_supported_cluster_n
    ):
        errors.append(
            f"tcgen05 strategy {strategy.value!r} with persistence model "
            f"{persistence_model.value!r} only runs correctly at "
            f"tcgen05_cluster_n in {sorted(paired_supported_cluster_n)!r}; "
            f"got tcgen05_cluster_n={cluster_n}"
        )

    # Layout overrides may only carry concrete values under the
    # explicit-epi-tile layout strategy *for fields that depend on the
    # epi-tile shape*. ``smem_swizzle_a`` / ``smem_swizzle_b`` are
    # orthogonal to ``compute_epilogue_tile_shape`` — they control the
    # A/B operand SMEM atom selection, not the D-output epilogue tile —
    # so the swizzle codegen wiring accepts the swizzle overrides under
    # both ``DEFAULT`` and ``EXPLICIT_EPI_TILE``. Other override fields
    # (``epi_tile_m``, ``epi_tile_n``, ``d_store_box_n``) still gate on
    # ``EXPLICIT_EPI_TILE`` because they only have meaning when the
    # explicit-epi-tile codegen path consumes them; under ``DEFAULT``
    # they would be silently ignored.
    _SWIZZLE_OVERRIDE_FIELDS = frozenset({"smem_swizzle_a", "smem_swizzle_b"})
    if layout_strategy is Tcgen05LayoutStrategy.DEFAULT:
        for field_name, value in dataclasses.asdict(layout_overrides).items():
            if value is None:
                continue
            if field_name in _SWIZZLE_OVERRIDE_FIELDS:
                # Swizzle override is wired under DEFAULT. Validate the
                # value is a legal swizzle byte choice here so
                # user-facing errors fire at config-time rather than
                # crashing inside CuTe's atom builder.
                if value not in TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES:
                    errors.append(
                        f"layout_overrides.{field_name}={value!r} must be "
                        f"one of {TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r} "
                        "(swizzle bytes; 0 = no swizzle / INTER)"
                    )
                continue
            errors.append(
                f"layout_overrides.{field_name}={value!r} requires "
                f"tcgen05_layout_strategy="
                f"{Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value!r}; "
                f"got {layout_strategy.value!r}"
            )
    else:
        # ``EXPLICIT_EPI_TILE``: still validate the swizzle byte choice.
        # The atom-contract check against the active tile shape happens
        # in lowering once codegen consumes the value.
        for field_name in _SWIZZLE_OVERRIDE_FIELDS:
            value = getattr(layout_overrides, field_name)
            if value is None:
                continue
            if value not in TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES:
                errors.append(
                    f"layout_overrides.{field_name}={value!r} must be "
                    f"one of {TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r} "
                    "(swizzle bytes; 0 = no swizzle / INTER)"
                )

    return errors
