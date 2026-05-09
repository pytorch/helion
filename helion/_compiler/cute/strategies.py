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


class Tcgen05Strategy(str, enum.Enum):
    """Structural kernel shape for the tcgen05 lowering.

    Strategies pick the *control flow / warp role inventory* of the
    generated kernel. They are *named* — they don't compose. Within
    a chosen strategy the autotuner explores records (warp split,
    layout overrides) that the strategy declares.

    - ``ROLE_LOCAL_MONOLITHIC`` (default, current). 6 specialized
      warps (1 TMA-load + 1 MMA-exec + 4 epilogue). Each role-local
      ``while`` loop carries its own ``StaticPersistentTileScheduler``.
    - ``ROLE_LOCAL_WITH_SCHEDULER``. 8 specialized warps: adds a
      dedicated scheduler warp that broadcasts ``(virtual_pid,
      tile_coord_mnkl, is_valid)`` over a ``PipelineAsync`` and a
      dedicated epilogue-load warp for C input. Targets Quack-best
      shape parity. Not yet implemented in codegen (G2-C); the enum
      slot exists so the data model can be exercised end-to-end at
      G2-A.
    """

    ROLE_LOCAL_MONOLITHIC = "role_local_monolithic"
    ROLE_LOCAL_WITH_SCHEDULER = "role_local_with_scheduler"


class Tcgen05PersistenceModel(str, enum.Enum):
    """Persistence axis for the tcgen05 lowering.

    Orthogonal to ``Tcgen05Strategy`` — the same warp-spec shape can
    run static or dynamic persistent. Today's ``pid_type=persistent_*``
    Helion config maps to ``STATIC_PERSISTENT``. ``DYNAMIC_PERSISTENT``
    (atomic-increment driven tile distribution from a global counter)
    is the Quack-best path; Helion has no codegen for it today and
    the strategy validation rejects it until a strategy that supports
    it lands (G2-F if needed).
    """

    NON_PERSISTENT = "non_persistent"
    STATIC_PERSISTENT = "static_persistent"
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

    @property
    def total_warps(self) -> int:
        return (
            self.ab_load_warps
            + self.mma_warps
            + self.epi_warps
            + self.epi_load_warps
            + self.scheduler_warps
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
    """

    epi_tile_m: int | None = None
    epi_tile_n: int | None = None
    smem_swizzle_a: int | None = None
    smem_swizzle_b: int | None = None
    d_store_box_n: int | None = None


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
# Register-split is exposed as two scalar keys (decrease, increase)
# rather than a tuple value because flat config values are scalars.
TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY = "tcgen05_warp_spec_register_decrease"
TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY = "tcgen05_warp_spec_register_increase"

# Source of truth for ``epi_warps`` lives in the existing
# ``tcgen05_num_epi_warps`` field, narrowed to ``(4,)`` at validation
# time by ``narrow_tcgen05_autotune_to_validated_configs``.
TCGEN05_NUM_EPI_WARPS_CONFIG_KEY = "tcgen05_num_epi_warps"

TCGEN05_WARP_SPEC_KEYS: tuple[str, ...] = (
    TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY,
    TCGEN05_WARP_SPEC_MMA_WARPS_KEY,
    TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY,
    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY,
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

# Default values for the role-local-monolithic warp spec, keyed for
# config insertion. ``epi_warps`` is sourced from
# ``tcgen05_num_epi_warps`` — see TCGEN05_NUM_EPI_WARPS_CONFIG_KEY.
TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY: dict[str, int] = {
    TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_AB_LOAD_WARPS,
    TCGEN05_WARP_SPEC_MMA_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_MMA_WARPS,
    TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_EPI_LOAD_WARPS,
    TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: ROLE_LOCAL_MONOLITHIC_SCHEDULER_WARPS,
    TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY: ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT[0],
    TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY: ROLE_LOCAL_MONOLITHIC_REGISTER_SPLIT[1],
}


def warp_spec_from_config(config: dict[str, object]) -> Tcgen05WarpSpec:
    """Read warp-spec fields out of a normalized config.

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
    )


def layout_overrides_from_config(
    config: dict[str, object],
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
    recognized, and ``DYNAMIC_PERSISTENT`` never falls out of this
    function — it is reachable only via an explicit user opt-in,
    because Helion has no codegen for dynamic persistence today.

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
        }
    ),
}

# Strategy-conditional warp-count requirements. Keys are strategy
# values. Each entry lists the *required* warp counts for that
# strategy. ``None`` means "any value in the validated range".
_STRATEGY_REQUIRED_SCHEDULER_WARPS: dict[Tcgen05Strategy, int] = {
    Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC: 0,
    Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER: 1,
}

# When the strategy demands warpgroup-aligned splits (every 4 warps
# == one warpgroup, ``setmaxregister`` is warpgroup-uniform), the
# total warp count must be a multiple of 4. Today's
# ``ROLE_LOCAL_MONOLITHIC`` uses 6 warps (1+1+4) so it intentionally
# does not require warpgroup alignment — the existing role-local
# lowering tolerates a non-warpgroup total. ``ROLE_LOCAL_WITH_SCHEDULER``
# uses 8 warps (2 warpgroups) and depends on warpgroup-uniform
# ``setmaxregister`` semantics (cute_plan.md §2). Adding a future
# strategy is a deliberate edit here.
_STRATEGY_REQUIRES_WARPGROUP_ALIGNED_TOTAL: frozenset[Tcgen05Strategy] = frozenset(
    {
        Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER,
    }
)


def validate_tcgen05_strategy_invariants(
    *,
    strategy: Tcgen05Strategy,
    persistence_model: Tcgen05PersistenceModel,
    layout_strategy: Tcgen05LayoutStrategy,
    warp_spec: Tcgen05WarpSpec,
    layout_overrides: Tcgen05LayoutOverrides,
    pid_type: object,
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
    - scheduler-warp count is strategy-determined;
    - the total warp count must be warpgroup-aligned when the
      strategy demands it;
    - layout-override values are only legal under
      ``Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE``.

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
    # ``pid_type=persistent_*`` implies ``STATIC_PERSISTENT``;
    # ``DYNAMIC_PERSISTENT`` has no codegen path so it never agrees
    # with any current ``pid_type`` value.
    derived = derive_persistence_model_from_pid_type(pid_type)
    if persistence_model != derived:
        errors.append(
            f"tcgen05_persistence_model={persistence_model.value!r} "
            f"contradicts pid_type={pid_type!r} (which implies "
            f"{derived.value!r}); set both consistently or drop one "
            "to take the derived default"
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

    # Layout overrides may only carry concrete values under the
    # explicit-epi-tile layout strategy. Today only the DEFAULT
    # strategy is wired through codegen, so any non-None override
    # under DEFAULT would silently be ignored — surface that loudly.
    if layout_strategy is Tcgen05LayoutStrategy.DEFAULT:
        for field_name, value in dataclasses.asdict(layout_overrides).items():
            if value is not None:
                errors.append(
                    f"layout_overrides.{field_name}={value!r} requires "
                    f"tcgen05_layout_strategy="
                    f"{Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value!r}; "
                    f"got {layout_strategy.value!r}"
                )

    return errors
